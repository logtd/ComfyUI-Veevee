from collections import defaultdict
from typing import List
import torch

from comfy.ldm.modules.attention import SpatialTransformer
from comfy.ldm.modules.diffusionmodules.openaimodel import TimestepBlock, UNetModel, Upsample, apply_control
from comfy.ldm.modules.diffusionmodules.util import timestep_embedding


def forward_timestep_embed(ts, x, emb, context=None, transformer_options={}, output_shape=None):
    for layer in ts:
        if isinstance(layer, TimestepBlock):
            x = layer(x, emb, transformer_options)
        elif isinstance(layer, SpatialTransformer):
            x = layer(x, context, transformer_options)
            if "transformer_index" in transformer_options:
                transformer_options["transformer_index"] += 1
        elif isinstance(layer, Upsample):
            x = layer(x, output_shape=output_shape)
        elif 'raunet' in str(layer.__class__):
            x = layer(x, output_shape=output_shape, transformer_options=transformer_options)
        elif 'Temporal' in str(layer.__class__):
            temporal_config = transformer_options.get('TEMPORAL_CONFIG', None)
            step_percent = transformer_options.get('STEP_PERCENT', 999)
            if temporal_config is None or (temporal_config.start_percent <= step_percent <= temporal_config.end_percent):
                x = layer(x, context)
        else:
            x = layer(x)
    return x


class VVUNetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        return self._forward_sample(x, timesteps, context, y, control, transformer_options=transformer_options)
    
    def _forward_sample(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_patches = transformer_options.get("patches", {})

        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for id, module in enumerate(self.input_blocks):
            transformer_options["block"] = ("input", id)
            h = forward_timestep_embed(module, h, emb, context, transformer_options)
            if control is not None:
                h = apply_control(h, control, 'input')
            if "input_block_patch" in transformer_patches:
                patch = transformer_patches["input_block_patch"]
                for p in patch:
                    h = p(h, transformer_options)

            hs.append(h)
            if "input_block_patch_after_skip" in transformer_patches:
                patch = transformer_patches["input_block_patch_after_skip"]
                for p in patch:
                    h = p(h, transformer_options)

        transformer_options["block"] = ("middle", 0)
        h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options)
        h = apply_control(h, control, 'middle')

        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            if control is not None:
                hsp = apply_control(hsp, control, 'output')

            if "output_block_patch" in transformer_patches:
                patch = transformer_patches["output_block_patch"]
                for p in patch:
                    h, hsp = p(h, hsp, transformer_options)

            h = torch.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                output_shape = hs[-1].shape
            else:
                output_shape = None
            h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape)
        
        h = h.type(x.dtype)
        return self.out(h)


def inject_unet(diffusion_model):
    diffusion_model.__class__ = VVUNetModel
