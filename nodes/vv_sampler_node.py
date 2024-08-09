import comfy.samplers
from comfy.samplers import KSAMPLER

from ..utils.sampler_utils import get_sampler_fn, create_sampler



class VVSamplerSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "sampler_name": (comfy.samplers.SAMPLER_NAMES, ),
            "attn_injection_steps": ("INT", {"default": 3, "min": 0, "max": 999, "step": 1}),
            "res_injection_steps": ("INT", {"default": 3, "min": 0, "max": 999, "step": 1}),
            "overlap": ("INT", {"default": 5, "min": 0, "max": 999, "step": 1}),
        }, "optional": {
            "flow_config": ("FLOW_CONFIG",),
            "inj_config": ("INJ_CONFIG",),
            "sca_config": ('SCA_CONFIG',),
            "pivot_config": ('PIVOT_CONFIG',),
            "rave_config": ('RAVE_CONFIG',),
            "temporal_config": ('TEMPORAL_CONFIG',),
            "sampler": ("SAMPLER",),
        }}
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"

    CATEGORY = "vv/sampling"

    def build(self, 
              sampler_name, 
              attn_injection_steps, 
              res_injection_steps, 
              overlap, 
              flow_config=None, 
              inj_config=None,
              sca_config=None,
              pivot_config=None,
              rave_config=None,
              temporal_config=None,
              sampler=None):

        sampler_fn = get_sampler_fn(sampler_name)
        sampler_fn = create_sampler(sampler_fn, attn_injection_steps, res_injection_steps, overlap,
                                    flow_config, inj_config,pivot_config, rave_config, sca_config,
                                    temporal_config)
        
        if sampler is None:
            sampler = KSAMPLER(sampler_fn)
        else:
            sampler.sampler_function = sampler_fn

        return (sampler, )