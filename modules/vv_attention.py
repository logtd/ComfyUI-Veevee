from einops import rearrange, repeat
import torch
import torch.nn.functional as F

from comfy.ldm.modules.attention import CrossAttention as ComfyCrossAttention, optimized_attention
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel


from ..utils.module_utils import isinstance_str
from ..utils.attention_utils import select_pivot_indexes
from ..utils.rave_utils import rave_attention
from ..configs.pivot_config import PivotAttentionConfig
from ..configs.rave_config import RaveAttentionConfig
from ..configs.sca_config import SparseCasualAttentionConfig, SCADirection
from ..configs.flow_config import FlowAttentionConfig


class VeeveeAttention(ComfyCrossAttention):
    def init_module(self, block, idx):
        self.block = block
        self.idx = idx
        self.block_idx = (self.block, self.idx)

    def _perform_rave_attention(self, transformer_options):
        step_percent = transformer_options['STEP_PERCENT']
        rave_config: RaveAttentionConfig = transformer_options.get('RAVE_CONFIG', None)
        if rave_config is None or self.block_idx not in rave_config.targets or not (rave_config.start_percent <= step_percent <= rave_config.end_percent):
            return False
        return True
    
    def _rave_attn(self, q, k, v, transformer_options):
        rave_config: RaveAttentionConfig = transformer_options['RAVE_CONFIG']
        return rave_attention(q, k, v, rave_config.grid_size, rave_config.seed, transformer_options)
        
    def _sca_attn(self, q, k, v, transformer_options):
        step_percent = transformer_options['STEP_PERCENT']
        sca_config: SparseCasualAttentionConfig = transformer_options.get('SCA_CONFIG', None)
        if sca_config is None or self.block_idx not in sca_config.targets or not (sca_config.start_percent <= step_percent <= sca_config.end_percent):
            return q,k,v
        
        len_conds = len(transformer_options['cond_or_uncond'])
        n_frames = q.shape[0] // len_conds
        
        ks = []
        vs = []
        for idx in range(n_frames):
            for cond_idx in range(len_conds):
                relative_idx = idx - (cond_idx * n_frames)
                if sca_config.direction == SCADirection.PREVIOUS or sca_config.direction == SCADirection.BOTH:
                    prev_idx = max(relative_idx-1, 0) + (cond_idx * n_frames)
                    ks.append(torch.cat([k[idx], k[prev_idx]]))
                    vs.append(torch.cat([v[idx], v[prev_idx]]))
                if sca_config.direction == SCADirection.NEXT or sca_config.direction == SCADirection.BOTH:
                    next_idx = min(relative_idx+1, len(q)-1) + (cond_idx * n_frames)
                    ks.append(torch.cat([k[idx], k[next_idx]]))
                    vs.append(torch.cat([v[idx], v[next_idx]]))

        return q, torch.stack(ks), torch.stack(vs)
    
    def _pivot_attn(self, q, k, v, transformer_options):
        step_percent = transformer_options['STEP_PERCENT']
        pivot_config: PivotAttentionConfig = transformer_options.get('PIVOT_CONFIG', None)
        if pivot_config is None or self.block_idx not in pivot_config.targets or not (pivot_config.start_percent <= step_percent <= pivot_config.end_percent):
            return q,k,v
        
        len_conds = len(transformer_options['cond_or_uncond'])
        n_frames = q.shape[0] // len_conds

        pivot_idxs = select_pivot_indexes(n_frames, pivot_config.batch_size, pivot_config.seed)
        
        for idx in range(n_frames):
            pivot_idx = pivot_idxs[idx//pivot_config.batch_size]
            for cond_idx in range(len_conds):
                k[idx+(n_frames*cond_idx)] = k[pivot_idx+(n_frames*cond_idx)]
                v[idx+(n_frames*cond_idx)] = v[pivot_idx+(n_frames*cond_idx)]

        return q, k, v

    def _interframe_alignment(self, query, key, hidden_states, transformer_options, direction):
        step_percent = transformer_options['STEP_PERCENT']
        flow_config: FlowAttentionConfig = transformer_options.get('FLOW_CONFIG', None)
        if flow_config is None or self.block_idx not in flow_config.targets or not (flow_config.start_percent <= step_percent <= flow_config.end_percent):
            return hidden_states

        cond_list = transformer_options['cond_or_uncond']
        batch_size = len(hidden_states) // len(cond_list)
        flow = None
        flows = []
        if direction == 'forward':
            flows = transformer_options['FLOW']['forward_flows']
        if direction == 'backward':
            flows = transformer_options['FLOW']['backward_flows']

        for possible_flow in flows:
            if possible_flow['forward_trajectory'].shape[2] == query.shape[1]:
                flow = possible_flow
                break

        if not flow:
            return hidden_states

        backward_trajectory = flow['backward_trajectory']
        forward_trajectory = flow['forward_trajectory']
        attn_mask = flow['attn_masks']

        key = rearrange(key, "(b f) d c -> f (b c) d", f=batch_size)
        query = rearrange(query, "(b f) d c -> f (b c) d", f=batch_size)
        hidden_states = rearrange(hidden_states, "(b f) d c -> f (b c) d", f=batch_size)

        key = torch.gather(key, 2, forward_trajectory.expand(-1,key.shape[1],-1))
        query = torch.gather(query, 2, forward_trajectory.expand(-1,query.shape[1],-1))
        hidden_states = torch.gather(hidden_states, 2, forward_trajectory.expand(-1,hidden_states.shape[1],-1))

        key = rearrange(key, "f (b c) d -> (b d) f c", b=len(cond_list))
        query = rearrange(query, "f (b c) d -> (b d) f c", b=len(cond_list))
        hidden_states = rearrange(hidden_states, "f (b c) d -> (b d) f c", b=len(cond_list))

        query = query.view(-1, batch_size, self.heads, self.dim_head).transpose(1, 2).detach()
        key = key.view(-1, batch_size, self.heads, self.dim_head).transpose(1, 2).detach()
        hidden_states = hidden_states.view(-1, batch_size, self.heads, self.dim_head).transpose(1, 2).detach()

        hidden_states = F.scaled_dot_product_attention(
            query, key, hidden_states, 
            attn_mask = (attn_mask.repeat(len(cond_list),1,1,1))
        )

        hidden_states = rearrange(hidden_states, "(b d) h f c -> f (b h c) d", b=len(cond_list))
        hidden_states = torch.gather(hidden_states, 2, backward_trajectory.expand(-1,hidden_states.shape[1],-1)).detach()
        hidden_states = rearrange(hidden_states, "f (b h c) d -> (b f) h d c", b=len(cond_list), h=self.heads)
        hidden_states = rearrange(hidden_states, 'b d q f -> b q (d f)')
        
        return hidden_states

    def forward(self, x, extra_options=None):
        # [step][attn/res][(block, idx)][sub_idx[0]]
        ad_params = extra_options.get('ad_params', {})
        sub_idxs = ad_params.get('sub_idxs', None)
        if sub_idxs is None:
            sub_idx = 0
        else:
            sub_idx = sub_idxs[0]

        step = extra_options['STEP']
        qk_state = x
        inj_config = extra_options.get('INJ_CONFIG', None)
        if inj_config is not None and self.block_idx in inj_config.attn_map:

            attn_inj_steps = extra_options.get('ATTN_INJECTION_STEPS', 0)
            if extra_options['INJECTION_KEY'] == 'SAMPLING' and step < attn_inj_steps and step < inj_config.attn_save_steps:
                len_cond = len(extra_options['cond_or_uncond'])
                sigma_key = inj_config.attn_sigmas[step]
                attn_inj = inj_config.attn_injections[sigma_key][self.block_idx]
                if sub_idxs is not None:
                    qk_state = attn_inj[sub_idxs].to(x.device)
                else:
                    qk_state = attn_inj.to(x.device)
                if len_cond > 1:
                    qk_state = torch.cat([qk_state]*len_cond)
            elif extra_options['INJECTION_KEY'] == 'UNSAMPLING' and step < inj_config.attn_save_steps:
                overlap = extra_options.get('OVERLAP', None)
                sigma_key = inj_config.attn_sigmas[step]
                attn_inj = inj_config.attn_injections[sigma_key][self.block_idx]
                if overlap is None or sub_idx == 0:
                    attn_inj.append(x.clone().detach().cpu())
                else:
                    attn_inj.append(x[overlap:].clone().detach().cpu())
            
        q = self.to_q(qk_state)
        k = self.to_k(qk_state)
        v = self.to_v(x)

        q, k, v = self._pivot_attn(q, k, v, extra_options)
        q, k, v = self._sca_attn(q, k, v, extra_options)

        if self._perform_rave_attention(extra_options):
            attn_out = self._rave_attn(q, k, v, extra_options)
        else:
            attn_out = optimized_attention(q, k, v, self.heads, attn_precision=self.attn_precision)

        attn_out = self._interframe_alignment(attn_out, attn_out, attn_out, extra_options, 'forward') 
        attn_out = self._interframe_alignment(attn_out, attn_out, attn_out, extra_options, 'backward') 

        return self.to_out(attn_out)
    

def _get_attn_modules(module):
    blocks = list(filter(lambda x: isinstance_str(x[1], 'BasicTransformerBlock'), module.named_modules()))
    return [block.attn1 for _, block in blocks]


def inject_vv_atn(diffusion_model: UNetModel):
    input = _get_attn_modules(diffusion_model.input_blocks)
    middle = _get_attn_modules(diffusion_model.middle_block)
    output = _get_attn_modules(diffusion_model.output_blocks)

    for i, attn in enumerate(input):
        attn.__class__ = VeeveeAttention
        attn.init_module('input', i)

    for i, attn in enumerate(middle):
        attn.__class__ = VeeveeAttention
        attn.init_module('middle', i)

    for i, attn in enumerate(output):
        attn.__class__ = VeeveeAttention
        attn.init_module('output', i)