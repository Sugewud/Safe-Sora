import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from timm.models.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange
import math
from typing import  Callable
from einops import rearrange, repeat
from functools import partial
from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)

class SS2D_map(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
       
        self.prepro = nn.Conv2d(7,self.d_inner,1,1,0)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, C ,H, W = x.shape
        # x = x.permute(0, 3, 1, 2).contiguous()
        x = self.prepro(x)
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class SS2D_local(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    def local_scan(
        self,
        x: torch.Tensor,
        H: int = 14,
        W: int = 14,
        w_h: int = 7,
        w_w: int = 7,
        flip: bool = False,
        column_first: bool = False
    ):
        B, C, _, _ = x.shape
        x = x.view(B, C, H, W)

        Hg = math.floor(H / w_h)
        Wg = math.floor(W / w_w)

        if (H % w_h != 0) or (W % w_w != 0):
            newH = Hg * w_h
            newW = Wg * w_w
            x = x[:, :, :newH, :newW]

        H = Hg * w_h
        W = Wg * w_w

        if column_first:
            # [B, C, Hg, w_h, Wg, w_w] -> [B, C, Wg, Hg, w_w, w_h]
            x_reshaped = x.view(B, C, Hg, w_h, Wg, w_w) \
                        .permute(0, 1, 4, 2, 5, 3) \
                        .reshape(B, C, -1)
        else:
            # [B, C, Hg, w_h, Wg, w_w] -> [B, C, Hg, Wg, w_h, w_w]
            x_reshaped = x.view(B, C, Hg, w_h, Wg, w_w) \
                        .permute(0, 1, 2, 4, 3, 5) \
                        .reshape(B, C, -1)

        if flip:
            x_reshaped = x_reshaped.flip([-1])

        return x_reshaped
    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        
        x1 = self.local_scan(x, H, W, w_h=H//4, w_w=W//4) # x.shape, H, W torch.Size([2, 1536, 20, 32]) 20 32 x1.shape torch.Size([2, 1536, 600])
        x2 = self.local_scan(x, H, W, w_h=H//4, w_w=W//4, column_first = True)
        x3 = self.local_scan(x, H, W, w_h=H//4, w_w=W//4, flip=True)
        x4 = self.local_scan(x, H, W, w_h=H//4, w_w=W//4, column_first = True, flip=True)
        xs = torch.stack([x1,x2,x3,x4],dim=1) # xs.shape torch.Size([2, 4, 1536, 600])

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) 
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        # x.shape ([2, 1536, 20, 32])
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 1,
            expand: float = 1.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))

        self.ln_11 = norm_layer(hidden_dim)
        self.self_attention1 = SS2D_local(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path1 = DropPath(drop_path)
        self.skip_scale1= nn.Parameter(torch.ones(hidden_dim))

        self.block = nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True))

        self.linear_out = nn.Linear(hidden_dim * 2,hidden_dim)

    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        device = input.device

        prepare = rearrange(input, "b h w c -> b c h w").contiguous().to(device)
        xfm = DWTForward(J=2, mode='zero', wave='haar').to(device)
        ifm = DWTInverse(mode='zero', wave='haar').to(device)

        Yl, Yh = xfm(prepare)
        h00 = torch.zeros(prepare.shape).float().to(device)
        for i in range(len(Yh)):
          if i == len(Yh) - 1:
            h00[:, :, :Yl.size(2), :Yl.size(3)] = Yl
            h00[:, :, :Yl.size(2), Yl.size(3):Yl.size(3) * 2] = Yh[i][:, :, 0, :, :]
            h00[:, :, Yl.size(2):Yl.size(2) * 2, :Yl.size(3)] = Yh[i][:, :, 1, :, :]
            h00[:, :, Yl.size(2):Yl.size(2) * 2, Yl.size(3):Yl.size(3) * 2] = Yh[i][:, :, 2, :, :]
          else:
            h00[:, :, :Yh[i].size(3), Yh[i].size(4):] = Yh[i][:, :, 0, :, :h00.shape[3] - Yh[i].size(4)]
            h00[:, :, Yh[i].size(3):, :Yh[i].size(4)] = Yh[i][:, :, 1, :h00.shape[2] - Yh[i].size(3), :]
            h00[:, :, Yh[i].size(3):, Yh[i].size(4):] = Yh[i][:, :, 2, :h00.shape[2] - Yh[i].size(3), :h00.shape[3] - Yh[i].size(4)]
        
        h00 = rearrange(h00, "b c h w -> b h w c").contiguous()
        h11 = self.ln_11(h00)
        h11 = h00*self.skip_scale1 + self.drop_path1(self.self_attention1(h11))
        h11 = rearrange(h11, "b h w c -> b c h w").contiguous()

        for i in range(len(Yh)):
          if i == len(Yh) - 1:
            Yl = h11[:, :, :Yl.size(2), :Yl.size(3)] 
            Yh[i][:, :, 0, :, :] = h11[:, :, :Yl.size(2), Yl.size(3):Yl.size(3) * 2] 
            Yh[i][:, :, 1, :, :] = h11[:, :, Yl.size(2):Yl.size(2) * 2, :Yl.size(3)] 
            Yh[i][:, :, 2, :, :] = h11[:, :, Yl.size(2):Yl.size(2) * 2, Yl.size(3):Yl.size(3) * 2] 
          else:
            Yh[i][:, :, 0, :, :h11.shape[3] - Yh[i].size(4)] = h11[:, :, :Yh[i].size(3), Yh[i].size(4):] 
            Yh[i][:, :, 1, :h11.shape[2] - Yh[i].size(3), :] = h11[:, :, Yh[i].size(3):, :Yh[i].size(4)] 
            Yh[i][:, :, 2, :h11.shape[2] - Yh[i].size(3), :h11.shape[3] - Yh[i].size(4)] = h11[:, :, Yh[i].size(3):, Yh[i].size(4):] 
        Yl = Yl.to(device)
        temp = ifm((Yl, [Yh[1]]))
        recons2 = ifm((temp, [Yh[0]])).to(device)
        recons2 = rearrange(recons2, "b c h w -> b h w c").contiguous()

        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        
        x = x.view(B, -1, C).contiguous()

        # wave trans
        x_dwt = recons2.view(B, -1, C).contiguous()

        # wave trans. The shapes may not match slightly due to the wavelet transform
        if x.shape != x_dwt.shape:
            x_dwt = x_dwt[:,:x.shape[1],:]

        # # wave trans
        x_final = torch.cat((x,x_dwt),2)
        x_final = self.linear_out(x_final)

        return x_final


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.conv3d = nn.Conv3d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.pixel_unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x, H, W):
        B_T, HW, C = x.shape
        T = 8
        B = B_T // T

        # [B*T, HW, C] -> [B, T, C, H, W]
        x = rearrange(x, "(b t) (h w) c -> b t c h w", b=B, t=T, h=H, w=W).contiguous()
        x = rearrange(x, "b t c h w -> b c t h w")

        # Conv3d
        x = self.conv3d(x)

        # PixelUnshuffle via reshaping to 2D
        x = rearrange(x, "b c t h w -> (b t) c h w").contiguous()
        x = self.pixel_unshuffle(x)  # 2D op
        x = rearrange(x, "(b t) c h w -> b c t h w", b=B, t=T)

        # Flatten back
        x = rearrange(x, "b c t h w -> (b t) (h w) c").contiguous()
        return x

class Downsample_input(nn.Module):
    def __init__(self, n_feat):
        super(Downsample_input, self).__init__()
        self.conv3d = nn.Conv3d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.pixel_unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x, H, W):
        B_T, H_, W_, C = x.shape
        T = 8
        B = B_T // T

        # [B*T, H, W, C] -> [B, T, C, H, W]
        x = rearrange(x, "(b t) h w c -> b c t h w", b=B, t=T)

        x = self.conv3d(x)

        # PixelUnshuffle
        x = rearrange(x, "b c t h w -> (b t) c h w").contiguous()
        x = self.pixel_unshuffle(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=B, t=T)

        # -> [B*T, HW, C]
        x = rearrange(x, "b c t h w -> (b t) (h w) c").contiguous()
        return x

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.conv3d = nn.Conv3d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x, H, W):
        B_T, HW, C = x.shape
        T = 8
        B = B_T // T

        # -> [B, C, T, H, W]
        x = rearrange(x, "(b t) (h w) c -> b c t h w", b=B, t=T, h=H, w=W)

        x = self.conv3d(x)

        # PixelShuffle
        x = rearrange(x, "b c t h w -> (b t) c h w").contiguous()
        x = self.pixel_shuffle(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=B, t=T)

        # -> [B*T, HW, C]
        x = rearrange(x, "b c t h w -> (b t) (h w) c").contiguous()
        return x

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

from SFMamba_3D import SFMamba_3D
from DWT_3d_index import wavelet_3d_index

class SFMambaUNet(nn.Module):
    def __init__(self,
                 inp_channels=7,
                 out_channels=3,
                 dim=4,
                 num_blocks=[1, 1, 1, 1],
                 mlp_ratio=1.,
                 num_refinement_blocks=0,
                 drop_path_rate=0.,
                 bias=False,
                 batch_size=1,
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super().__init__()
        self.mlp_ratio = mlp_ratio
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        base_d_state = 1
        self.encoder_level1 = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=base_d_state,
            )
            for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 3),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 3),
            )
            for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(num_blocks[0])])

        self.refinement = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = zero_module(nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias))
        self.map1 = SS2D_map(
                d_model=dim,
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=base_d_state,)
        self.map2 = SS2D_map(
                d_model=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),)
        self.map3 = SS2D_map(
                d_model=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),)
        self.process1 = nn.Linear(dim,dim*2)
        self.process2 = nn.Linear(dim * 2,dim * 4)
        self.process3 = nn.Linear(dim * 4,dim * 8)

        self.conv0 = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=1)

        self.Refine_3D = nn.ModuleList([
            SFMamba_3D(dim=8, scan_index = self.wavelet_index(torch.ones(batch_size, 8, 3, 320, 512))),
            SFMamba_3D(dim=8, scan_index = self.wavelet_index(torch.ones(batch_size, 8, 3, 320, 512))),
        ])
    
    def wavelet_index(self, x):
        B, nf, C, H, W = x.shape
        idx_tensor = wavelet_3d_index(nf, H, W)
        return idx_tensor

    def forward(self, inp_img, vae_features, multi_scale_watermark):

        _, _, H, W = inp_img.shape
        inp_enc_level1 = self.patch_embed(inp_img)  # b,hw,c
        # print(inp_enc_level1.shape,'come')
        out_enc_level1 = inp_enc_level1
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, [H, W])
        inp_enc_level2 = self.down1_2(out_enc_level1, H, W)  # b, hw//4, 2c

        inp_img1 = torch.cat((multi_scale_watermark[1], self.conv2(vae_features[2])), dim=1)
        inp_img2 = torch.cat((multi_scale_watermark[2], self.conv1(vae_features[1])), dim=1)
        inp_img3 = torch.cat((multi_scale_watermark[3], self.conv0(vae_features[0])), dim=1)

        map1 = self.map1(inp_img1)
        map1 = self.process1(map1)
        map1 = rearrange(map1,"b h w c-> b (h w) c")
        map2 = self.map2(inp_img2)
        map2 = self.process2(map2)
        map2 = rearrange(map2,"b h w c-> b (h w) c")
        map3 = self.map3(inp_img3)
        map3 = self.process3(map3)
        map3 = rearrange(map3,"b h w c-> b (h w) c")

        out_enc_level2 = inp_enc_level2 + map1
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, [H // 2, W // 2])

        inp_enc_level3 = self.down2_3(out_enc_level2, H // 2, W // 2)  # b, hw//16, 4c
        out_enc_level3 = inp_enc_level3 + map2
        for layer in self.encoder_level3:
            out_enc_level3 = layer(out_enc_level3, [H // 4, W // 4])
        # print(out_enc_level3.shape)
        inp_enc_level4 = self.down3_4(out_enc_level3, H // 4, W // 4)  # b, hw//64, 8c
        
        latent = inp_enc_level4 + map3
        for layer in self.latent:
            latent = layer(latent, [H // 8, W // 8])

        inp_dec_level3 = self.up4_3(latent, H // 8, W // 8)  # b, hw//16, 4c
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 2)
        inp_dec_level3 = rearrange(inp_dec_level3, "b (h w) c -> b c h w", h=H // 4, w=W // 4).contiguous()
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = rearrange(inp_dec_level3, "b c h w -> b (h w) c").contiguous()  # b, hw//16, 4c
        out_dec_level3 = inp_dec_level3
        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_dec_level3, [H // 4, W // 4])

        inp_dec_level2 = self.up3_2(out_dec_level3, H // 4, W // 4)  # b, hw//4, 2c
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b (h w) c -> b c h w", h=H // 2, w=W // 2).contiguous()
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b c h w -> b (h w) c").contiguous()  # b, hw//4, 2c
        out_dec_level2 = inp_dec_level2
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2, [H // 2, W // 2])

        inp_dec_level1 = self.up2_1(out_dec_level2, H // 2, W // 2)  # b, hw, c
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)
        out_dec_level1 = inp_dec_level1
        

        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1, [H, W])
            # print(out_dec_level1,'========111111111111111111111111111')
            # exit()
        for layer in self.refinement:
            out_dec_level1 = layer(out_dec_level1, [H, W])

        out_dec_level1 = rearrange(out_dec_level1, "b (h w) c -> b c h w", h=H, w=W).contiguous()

        d = 8  #8

        b = out_dec_level1.shape[0] // d
        out_dec_level1 = out_dec_level1.view(b, d, *out_dec_level1.shape[1:])
        residual = out_dec_level1
        for layer in self.Refine_3D:
            out_dec_level1 = layer(out_dec_level1)
        out_dec_level1 = out_dec_level1 + residual
        out_dec_level1 = out_dec_level1.view(-1, *out_dec_level1.shape[2:])

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:, -3:, :, :]
        
        # exit()
        return torch.tanh(out_dec_level1)

