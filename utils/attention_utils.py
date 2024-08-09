import math
import random
import torch
import torch.nn.functional as F
from einops import rearrange


def reshape_heads_to_batch_dim3(tensor, head_size):
    batch_size1, batch_size2, seq_len, dim = tensor.shape
    tensor = tensor.reshape(batch_size1, batch_size2,
                            seq_len, head_size, dim // head_size)
    tensor = tensor.permute(0, 3, 1, 2, 4)
    return tensor


def trajectory_hidden_states(query,
               key,
               value,
               trajectories,
               use_old_qk,
               extra_options,
               n_heads):
    if not use_old_qk:
        query = value
        key = value
    # TODO: Hardcoded for SD1.5
    _,_, oh, ow = extra_options['original_shape']
    height = 64 #  int(value.shape[1]**0.5)
    width = height
    cond_size = len(extra_options['cond_or_uncond'])
    video_length = len(query) // cond_size

    sub_idxs = extra_options.get('ad_params', {}).get('sub_idxs', None)
    idx = 0
    if sub_idxs is not None:
        idx = sub_idxs[0]

    traj_window = trajectories['trajectory_windows'][idx]
    if f'traj{height}' not in traj_window:
        return value
    trajs = traj_window[f'traj{height}']
    traj_mask = traj_window[f'mask{height}']

    start = -video_length+1
    end = trajs.shape[2]

    traj_key_sequence_inds = torch.cat(
        [trajs[:, :, 0, :].unsqueeze(-2), trajs[:, :, start:end, :]], dim=-2)
    traj_mask = torch.cat([traj_mask[:, :, 0].unsqueeze(-1),
                           traj_mask[:, :, start:end]], dim=-1)

    t_inds = traj_key_sequence_inds[:, :, :, 0]
    x_inds = traj_key_sequence_inds[:, :, :, 1]
    y_inds = traj_key_sequence_inds[:, :, :, 2]

    query_tempo = query.unsqueeze(-2)
    _key = rearrange(key, '(b f) (h w) d -> b f h w d',
                     b=cond_size,  h=height)
    _value = rearrange(value, '(b f) (h w) d -> b f h w d',
                       b=cond_size, h=height)
    key_tempo = _key[:, t_inds, x_inds, y_inds]
    value_tempo = _value[:, t_inds, x_inds, y_inds]
    key_tempo = rearrange(key_tempo, 'b f n l d -> (b f) n l d')
    value_tempo = rearrange(value_tempo, 'b f n l d -> (b f) n l d')

    traj_mask = rearrange(torch.stack(
        [traj_mask] * cond_size),  'b f n l -> (b f) n l')
    traj_mask = traj_mask[:, None].repeat(
        1, n_heads, 1, 1).unsqueeze(-2)
    attn_bias = torch.zeros_like(
        traj_mask, dtype=key_tempo.dtype, device=query.device)  # regular zeros_like
    attn_bias[~traj_mask] = -torch.inf

    # flow attention
    query_tempo = reshape_heads_to_batch_dim3(query_tempo, n_heads)
    key_tempo = reshape_heads_to_batch_dim3(key_tempo, n_heads)
    value_tempo = reshape_heads_to_batch_dim3(value_tempo, n_heads)

    attn_matrix2 = query_tempo @ key_tempo.transpose(-2, -1) / math.sqrt(
        query_tempo.size(-1)) + attn_bias
    attn_matrix2 = F.softmax(attn_matrix2, dim=-1)
    out = (attn_matrix2@value_tempo).squeeze(-2)

    hidden_states = rearrange(out, 'b k r d -> b r (k d)')

    return hidden_states


def select_pivot_indexes(length, batch_size, seed=None):
    # Create a new Random object with the given seed
    rng = random.Random(seed)
    
    # Use the seeded Random object to generate the random index
    rnd_idx = rng.randint(0, batch_size - 1)
    
    return [min(i, length-1) for i in range(rnd_idx, length, batch_size)] + [length-1]