import comfy.samplers
from comfy.samplers import KSAMPLER

from ..utils.sampler_utils import get_sampler_fn, create_unsampler


class VVUnsamplerSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "sampler_name": (comfy.samplers.SAMPLER_NAMES, ),
            "overlap": ("INT", {"default": 5, "min": 0, "max": 999, "step": 1}),
        }, "optional": {
            "flow_config": ("FLOW_CONFIG",),
            "inj_config": ("INJ_CONFIG",),
            "sampler": ("SAMPLER",)
        }}
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"

    CATEGORY = "vv/sampling"

    def build(self, 
              sampler_name, 
              overlap, 
              flow_config=None, 
              inj_config=None,
              sampler=None):

        sampler_fn = get_sampler_fn(sampler_name)
        sampler_fn = create_unsampler(sampler_fn, overlap, flow_config, inj_config)
        
        if sampler is None:
            sampler = KSAMPLER(sampler_fn)
        else:
            sampler.sampler_function = sampler_fn

        return (sampler, )