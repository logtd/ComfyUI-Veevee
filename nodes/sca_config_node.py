from ..vv_defaults import MAP_TYPES


class SCAConfigNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "direction": (['PREVIOUS', 'NEXT', 'BOTH'],),
            "targets": (MAP_TYPES,),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}
    RETURN_TYPES = ("SCA_CONFIG",)
    FUNCTION = "build"

    CATEGORY = "vv/configs"

    def build(self, direction, targets, start_percent, end_percent):

        return ({ 'direction': direction, 'targets': targets, 'start_percent': start_percent, 'end_percent': end_percent},)