import torch
import torch.nn as nn

class Router(nn.Module):
    def __init__(self, frame_channels=3, patch_channels=4, hidden_dim=128):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(frame_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(patch_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, frames, patches):
        
        B, T, C, H, W   = frames.shape
        _, P, C2, hp, wp = patches.shape

        f_enc = self.frame_encoder(frames.view(-1, C, H, W)).view(B, T, -1)  # [B,8,D]
        p_enc = self.patch_encoder(patches.view(-1, C2, hp, wp)).view(B, P, -1)  # [B,256,D]

        scores  = torch.einsum('btd,bpd->btp', f_enc, p_enc)          # [B,8,256]
        probs   = torch.softmax(scores.transpose(1, 2), dim=-1)       # [B,256,8]
        return probs, scores



class LocalMatcher(nn.Module):
    
    def __init__(self, hidden=64):
        super().__init__()
        
        self.v_encoder = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.p_encoder = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def split_video(self, v):  # v: [4,40,64]  -> [32,4,10,8]
        v = v.view(4, 4, 10, 8, 8)               # C, row4, h=10, col8, w=8
        v = v.permute(1, 3, 0, 2, 4)             # row4, col8, C, 10, 8
        return v.reshape(32, 4, 10, 8)

    def forward(self, v_latent, p_tensor, p_old_idx):
        
        k = p_tensor.size(0)

        
        v_emb = self.v_encoder(self.split_video(v_latent).to(p_tensor.device)).squeeze(-1).squeeze(-1)  # [32,hidden]
        p_emb = self.p_encoder(p_tensor).squeeze(-1).squeeze(-1)                                        # [k,hidden]

        
        scores = torch.matmul(p_emb, v_emb.t())

        
        _, pref = torch.sort(scores, dim=-1, descending=True)  
        assign_pos = p_tensor.new_full((k,), -1, dtype=torch.long)  
        taken = torch.zeros(32, dtype=torch.bool, device=p_tensor.device)

        for r in range(32):                      
            want = pref[:, r]                   
            for pid in range(k):
                if assign_pos[pid] >= 0:        
                    continue
                pos = want[pid].item()
                if not taken[pos]:
                    taken[pos]      = True
                    assign_pos[pid] = pos
        
        if (assign_pos < 0).any():
            raise RuntimeError("A patch was not assigned to an intra-frame position (this should not normally happen)")

        return assign_pos  # [k] ∈ [0,31]

class Expert_Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)

class Adaptive_Embedding(nn.Module):
    def __init__(self, patch_dim=1024, num_frames=8, cap_per_frame=32):
        super().__init__()
        self.num_frames       = num_frames
        self.cap_per_frame    = cap_per_frame     # 32
        self.router           = Router(frame_channels=4, patch_channels=4)
        self.local_matcher    = LocalMatcher()
        self.expert           = Expert_Model(patch_dim)

    def forward(self, patches, video_latent):
        """
        patches      : [B,4,256,16,16]
        video_latent : [B,8,4,40,64]
        return       : out_x  [B,4,256,16,16]
                       idxmap [B,256]  old‑>new
        """
        B, _, _, ph, pw = patches.shape
        device = patches.device

        
        p5d    = patches.permute(0, 2, 1, 3, 4).contiguous()   # [B,256,4,16,16]
        p_flat = p5d.view(B, 256, -1)                          # [B,256,D]

        
        probs, _ = self.router(video_latent, p5d)              # probs:[B,256,8]
        top2, idx2 = torch.sort(probs, dim=-1, descending=True)

        assigned_frame = p_flat.new_full((B, 256), -1, dtype=torch.long)
        capacity       = torch.full((B, self.num_frames),
                                    self.cap_per_frame, dtype=torch.long, device=device)

        for r in range(self.num_frames):
            cand = idx2[:, :, r]            # [B,256]
            mask_un = assigned_frame.lt(0)
            un_bi   = mask_un.nonzero(as_tuple=False)         
            if un_bi.numel() == 0:
                break
            for b in range(B):
                mask_b = (un_bi[:,0]==b)
                if not mask_b.any(): continue
                i_list = un_bi[mask_b][:,1]                    
                f_list = cand[b, i_list]
                for f in range(self.num_frames):
                    mask_f = (f_list==f)
                    cnt = mask_f.sum().item()
                    if cnt==0 or capacity[b,f]==0: continue
                    take = min(cnt, capacity[b,f].item())
                    chosen = i_list[mask_f][:take]
                    assigned_frame[b, chosen] = f
                    capacity[b,f]-=take

        
        out_x   = p_flat.new_zeros(B, 256, p_flat.size(-1))
        idx_map = p_flat.new_empty(B, 256, dtype=torch.long)

        for f in range(self.num_frames):
            for b in range(B):
                idxs = (assigned_frame[b]==f).nonzero(as_tuple=False).squeeze(-1)  # [k]
                k = idxs.numel()
                if k==0: continue

                p_tensor = p5d[b, idxs]                           # [k,4,16,16]

                assign_pos = self.local_matcher(
                    video_latent[b,f], p_tensor, idxs)            # [k]

                p_vec   = p_flat[b, idxs]                         # [k,D]
                p_out   = self.expert(p_vec)                      # [k,D]

                base = f * self.cap_per_frame
                for j in range(k):
                    pos            = assign_pos[j].item()
                    global_new_idx = base + pos
                    out_x[b, global_new_idx] = p_out[j]
                    idx_map[b, idxs[j]]     = global_new_idx

        out_x = out_x.view(B, 256, 4, ph, pw).permute(0,2,1,3,4).contiguous()
        return out_x, idx_map

def revert_order(y_2d, idx_map):
    B,N,D = y_2d.shape
    return torch.gather(y_2d, 1, idx_map.unsqueeze(-1).expand(-1,-1,D))
