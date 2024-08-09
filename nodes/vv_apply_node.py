from ..modules.vv_attention import inject_vv_atn
from ..modules.vv_block import inject_vv_block
from ..modules.vv_resnet import inject_vv_resblock
from ..modules.vv_unet import inject_unet


class ApplyVVModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "vv"

    def apply(self, model):
        inject_vv_atn(model.model.diffusion_model)
        inject_vv_block(model.model.diffusion_model)
        inject_vv_resblock(model.model.diffusion_model)
        inject_unet(model.model.diffusion_model)
        return (model, )
