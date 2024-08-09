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