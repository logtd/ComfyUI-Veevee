import torch

from comfy.ldm.modules.diffusionmodules.openaimodel import ResBlock as ComfyResBlock, UNetModel

from ..utils.module_utils import isinstance_str


class ResBlock(ComfyResBlock):
    def init_module(self, block, idx):
        self.block = block
        self.idx = idx
        self.block_idx = (block, idx)

    def forward(self, x, emb, extra_options):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = None
        if not self.skip_t_emb:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_norm(h)
            if emb_out is not None:
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                h *= (1 + scale)
                h += shift
            h = out_rest(h)
        else:
            if emb_out is not None:
                if self.exchange_temb_dims:
                    emb_out = emb_out.movedim(1, 2)
                h = h + emb_out
            h = self.out_layers(h)

        ad_params = extra_options.get('ad_params', {})
        sub_idxs = ad_params.get('sub_idxs', [0])
        if sub_idxs is None:
            sub_idx = 0
        else:
            sub_idx = sub_idxs[0]

        res_inj_steps = extra_options.get('RES_INJECTION_STEPS', 0)
        step = extra_options['STEP']
        inj_config = extra_options.get('INJ_CONFIG', None)
        if inj_config and self.block_idx in inj_config.res_map:
            if extra_options['INJECTION_KEY'] == 'SAMPLING' and step < res_inj_steps and step < inj_config.res_save_steps:
                if extra_options['INJECTION_KEY'] == 'SAMPLING':
                    len_cond = len(extra_options['cond_or_uncond'])
                    sigma_key = inj_config.res_sigmas[step]
                    res_inj = inj_config.res_injections[sigma_key][self.block_idx]
                    if sub_idxs is not None:
                        h = res_inj[sub_idxs].to(x.device)
                    else:
                        h = res_inj.to(x.device)
                    if len_cond > 1:
                        h = torch.cat([h]*len_cond)
            elif extra_options['INJECTION_KEY'] == 'UNSAMPLING' and step < inj_config.res_save_steps:
                overlap = extra_options.get('OVERLAP', None)
                sigma_key = inj_config.res_sigmas[step]
                res_inj = inj_config.res_injections[sigma_key][self.block_idx]
                if overlap == 0 or sub_idx == 0:
                    res_inj.append(h.clone().detach().cpu())
                else:
                    res_inj.append(h[overlap:].clone().detach().cpu())
    
        return self.skip_connection(x) + h



def _get_resnet_modules(module):
    blocks = list(filter(lambda x: isinstance_str(x[1], 'ResBlock'), module.named_modules()))
    return [block for _, block in blocks]


def inject_vv_resblock(diffusion_model: UNetModel):
    input = _get_resnet_modules(diffusion_model.input_blocks)
    middle = _get_resnet_modules(diffusion_model.middle_block)
    output = _get_resnet_modules(diffusion_model.output_blocks)

    for idx, resnet in enumerate(input):
        resnet.__class__ = ResBlock
        resnet.init_module('input', idx)

    for idx, resnet in enumerate(middle):
        resnet.__class__ = ResBlock
        resnet.init_module('middle', idx)

    for idx, resnet in enumerate(output):
        resnet.__class__ = ResBlock
        resnet.init_module('output', idx)