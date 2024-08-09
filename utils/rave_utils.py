import random

import torch
from einops import rearrange

from comfy.ldm.modules.attention import optimized_attention


def padding_count(n_frames, grid_frame_count):
    remainder = n_frames % grid_frame_count
    if remainder == 0:
        return 0
    else:
        difference = grid_frame_count - remainder
        return difference


def rave_attention(q, k, v, grid_size, seed, extra_options):
    batch_size, sequence_length, dim = q.shape
    kv_sequence_length = k.shape[1]
    n_heads = extra_options['n_heads']
    len_conds = len(extra_options['cond_or_uncond'])
    n_frames = batch_size // len_conds
    original_n_frames = n_frames

    grid_frame_count = grid_size * grid_size
    n_padding_frames = padding_count(n_frames, grid_frame_count)
    if n_padding_frames > 0:
        random.seed(seed)
        cond_qs = []
        cond_ks = []
        cond_vs = []
        padding_frames = [random.randint(
            0, n_frames-1) for _ in range(n_padding_frames)]
        for cond_idx in range(len_conds):
            start, end = cond_idx*n_frames, (cond_idx+1)*n_frames
            cond_q = q[start:end]
            cond_q = torch.cat([cond_q, cond_q[padding_frames]])
            cond_qs.append(cond_q)
            cond_k = k[start:end]
            cond_k = torch.cat([cond_k, cond_k[padding_frames]])
            cond_ks.append(cond_k)
            cond_v = v[start:end]
            cond_v = torch.cat([cond_v, cond_v[padding_frames]])
            cond_vs.append(cond_v)

        q = torch.cat(cond_qs)
        k = torch.cat(cond_ks)
        v = torch.cat(cond_vs)

    n_frames = n_frames + n_padding_frames

    # get h,w
    shape = extra_options['original_shape']
    oh, ow = shape[-2:]
    ratio = oh/ow
    d = sequence_length
    w = int((d/ratio)**(0.5))
    h = int(d/w)

    p = kv_sequence_length // sequence_length
    q = rearrange(q, 'b (h w) c -> b h w c', h=h, w=w)
    k = rearrange(k, 'b (h w p) c -> p b h w c', h=h, w=w,  p=p)
    v = rearrange(v, 'b (h w p) c -> p b h w c', h=h, w=w,  p=p)

    target_indexes = shuffle_indices(n_frames, seed=seed)

    original_indexes = list(range(n_frames))
    qs = []
    ks = [[] for _ in range(p)]
    vs = [[] for _ in range(p)]

    for i in range(len_conds):
        start, end = i*n_frames, (i+1)*n_frames
        q[start:end] = shuffle_tensors2(
            q[start:end], original_indexes, target_indexes)
        qs.append(list_to_grid(q[start:end], grid_size))
    for kv_idx in range(p):
        for i in range(len_conds):
            k[kv_idx, start:end] = shuffle_tensors2(
                k[kv_idx, start:end], original_indexes, target_indexes)
            ks[kv_idx].append(list_to_grid(k[kv_idx, start:end], grid_size))
            v[kv_idx, start:end] = shuffle_tensors2(
                v[kv_idx, start:end], original_indexes, target_indexes)
            vs[kv_idx].append(list_to_grid(v[kv_idx, start:end], grid_size))

    for kv_idx in range(p):
        ks[kv_idx] = torch.cat(ks[kv_idx])
        vs[kv_idx] = torch.cat(vs[kv_idx])

    q = torch.cat(qs)
    k = torch.stack(ks)
    v = torch.stack(vs)

    q = rearrange(q, 'b h w c -> b (h w) c')
    k = rearrange(k, 'p b h w c -> b (h w p) c')
    v = rearrange(v, 'p b h w c -> b (h w p) c')

    out = optimized_attention(q, k, v, n_heads, None)

    gh, gw = grid_size*h, grid_size*w
    out = rearrange(out, 'b (h w) c -> b h w c', h=gh, w=gw)
    out = grid_to_list(out, grid_size)
    out = rearrange(out, 'b h w c -> b (h w) c')

    outs = []
    for i in range(len_conds):
        start, end = i*n_frames, (i+1)*n_frames
        cond_out = shuffle_tensors2(
            out[start:end], target_indexes, original_indexes)
        cond_out = cond_out[:original_n_frames]
        outs.append(cond_out)

    return torch.cat(outs)


def shuffle_indices(size, seed=None):
    if seed is not None:
        random.seed(seed)
    indices = list(range(size))
    random.shuffle(indices)
    return indices


def shuffle_tensors2(tensor, current_indices, target_indices):
    tensor_dict = {current_idx: t for current_idx,
                   t in zip(current_indices, tensor)}
    shuffled_tensors = [tensor_dict[current_idx]
                        for current_idx in target_indices]
    return torch.stack(shuffled_tensors)


def grid_to_list(tensor, grid_size):
    frame_count = len(tensor) * grid_size * grid_size
    flattened_list = [flatten_grid(grid.unsqueeze(
        0), [grid_size, grid_size]) for grid in tensor]
    list_tensor = torch.cat(flattened_list, dim=-2)
    return torch.cat(torch.chunk(list_tensor, frame_count, dim=-2), dim=0)


def list_to_grid(tensor, grid_size):
    grid_frame_count = grid_size * grid_size
    grid_count = len(tensor) // grid_frame_count
    flat_grids = [torch.cat([a for a in tensor[i * grid_frame_count:(i + 1)
                            * grid_frame_count]], dim=-2).unsqueeze(0) for i in range(grid_count)]
    unflattened_grids = [unflatten_grid(
        flat_grid, [grid_size, grid_size]) for flat_grid in flat_grids]
    return torch.cat(unflattened_grids, dim=0)


def flatten_grid(x, grid_shape):
    B, H, W, C = x.size()
    hs, ws = grid_shape
    img_h = H // hs
    flattened = torch.cat(torch.split(x, img_h, dim=1), dim=2)
    return flattened


def unflatten_grid(x, grid_shape):
    ''' 
    x: B x C x H x W
    '''
    B, H, W, C = x.size()
    hs, ws = grid_shape
    img_w = W // (ws)

    unflattened = torch.cat(torch.split(x, img_w, dim=2), dim=1)

    return unflattened