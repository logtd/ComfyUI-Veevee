import torch
import torch.nn.functional as F


@torch.no_grad()
def get_frame_pair_trajectories(backward_flow, backward_occlusions, imgs, scale=1.0):
    flows = F.interpolate(backward_flow, scale_factor=1./scale, mode='bilinear')[0][[1,0]] / scale
    _, H, W = flows.shape
    masks = torch.logical_not(F.interpolate(backward_occlusions[None], scale_factor=1./scale, mode='bilinear') > 0.5)[0]
    frames = F.interpolate(imgs, scale_factor=1./scale, mode='bilinear').view(2, 3, -1)
    grid = torch.stack(torch.meshgrid([torch.arange(H), torch.arange(W)]), dim=0).to(flows.device)
    warp_grid = torch.round(grid + flows)
    mask = torch.logical_and(torch.logical_and(torch.logical_and(torch.logical_and(warp_grid[0] >= 0, warp_grid[0] < H),
                         warp_grid[1] >= 0), warp_grid[1] < W), masks[0]).view(-1)
    warp_grid = warp_grid.view(2, -1)
    warp_ind = (warp_grid[0] * W + warp_grid[1]).to(torch.long)
    frame_pair_trajectories = torch.zeros_like(warp_ind) - 1
    
    for outer_idx, inner_idx in enumerate(warp_ind):
        if mask[outer_idx]:
            if frame_pair_trajectories[inner_idx] == -1:
                frame_pair_trajectories[inner_idx] = outer_idx
            else:
                target_pixel = frames[0,:,inner_idx]
                preouter_idx = frame_pair_trajectories[inner_idx]
                prev_pixel = frames[1,:,preouter_idx]
                current_pixel = frames[1,:,outer_idx]
                if ((prev_pixel - target_pixel)**2).mean() > ((current_pixel - target_pixel)**2).mean():
                    mask[preouter_idx] = False 
                    frame_pair_trajectories[inner_idx] = outer_idx
                else:
                    mask[outer_idx] = False
                    
    unused_idxs = torch.arange(len(mask)).to(mask.device)[~mask]
    disconnected_pixels = frame_pair_trajectories == -1
    frame_pair_trajectories[disconnected_pixels] = unused_idxs
    return frame_pair_trajectories, disconnected_pixels


@torch.no_grad()
def get_full_frames_trajectories(backward_flows, backward_occlusions, imgs, scale=1.0):
    B, H, W = imgs.shape[0], int(imgs.shape[2] // scale), int(imgs.shape[3] // scale)
    attn_mask = torch.ones(H*W, B, B, dtype=torch.bool).to(imgs.device) 
    forward_trajectories = [torch.arange(H*W).to(imgs.device)]
    backward_trajectories = [torch.arange(H*W).to(imgs.device)]
    for i in range(len(imgs)-1):
        frame_mask = torch.ones(B, B, dtype=torch.bool).to(imgs.device)
        frame_mask[:i+1,i+1:] = False
        frame_mask[i+1:,:i+1] = False
        frame_pair_trajectories, disconnected_pixels = get_frame_pair_trajectories(backward_flows[i:i+1], backward_occlusions[i:i+1], imgs[i:i+2], scale)
        attn_mask[disconnected_pixels[forward_trajectories[-1]]] = torch.logical_and(attn_mask[disconnected_pixels[forward_trajectories[-1]]], frame_mask)
        forward_trajectories += [frame_pair_trajectories[forward_trajectories[-1]]]
        backward_trajectories += [torch.sort(forward_trajectories[-1])[1]]
    forward_trajectoriess = torch.stack(forward_trajectories, dim=0).unsqueeze(1)
    backward_trajectoriess = torch.stack(backward_trajectories, dim=0).unsqueeze(1)
    return forward_trajectoriess, backward_trajectoriess, attn_mask.unsqueeze(1)


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)],
                          )
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid


def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]


def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode,
                           return_mask=mask)


def forward_backward_consistency_check(fwd_flow, bwd_flow,
                                       alpha=0.01,
                                       beta=0.5
                                       ):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ
