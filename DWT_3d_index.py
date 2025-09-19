import torch

def wavelet_3d_index(D, H, W, subband_orders='w'):
    

    d = D // 2
    h = H // 2
    w = W // 2

    
    subband_offsets = [
        (0,    0,    0   ),  # LLL
        (0,    0,    w   ),  # LLH
        (0,    h,    0   ),  # LHL
        (d,    0,    0   ),  # HLL
        (0,    h,    w   ),  # LHH
        (d,    0,    w   ),  # HLH
        (d,    h,    0   ),  # HHL
        (d,    h,    w   ),  # HHH
    ]

    idx_list = []

    def subband_traverse(d_, h_, w_, zd, yd, xd, order_in):
        if order_in == 'w':
            for z_ in range(d_):
                for y_ in range(h_):
                    for x_ in range(w_):
                        yield (zd + z_, yd + y_, xd + x_)
        elif order_in == 'h':
            for z_ in range(d_):
                for x_ in range(w_):
                    for y_ in range(h_):
                        yield (zd + z_, yd + y_, xd + x_)
        else:
            for x_ in range(w_):
                for y_ in range(h_):
                    for z_ in range(d_):
                        yield (zd + z_, yd + y_, xd + x_)

    def coords_to_linear(z, y, x, H, W):
        return z * (H * W) + y * W + x

    for (zd, yd, xd) in subband_offsets:
        for (Z, Y, X) in subband_traverse(d, h, w, zd, yd, xd, subband_orders):
            linear_idx = coords_to_linear(Z, Y, X, H, W)
            idx_list.append(linear_idx)

    return torch.tensor(idx_list, dtype=torch.long)


def wavelet_3d_index_ablation(D, H, W):
    idx_list = []

    def coords_to_linear(z, y, x, H, W):
        return z * (H * W) + y * W + x

    for z in range(D):
        for y in range(H):
            for x in range(W):
                idx_list.append(coords_to_linear(z, y, x, H, W))

    return torch.tensor(idx_list, dtype=torch.long)
