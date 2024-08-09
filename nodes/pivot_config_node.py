from ..vv_defaults import MAP_TYPES

class PivotConfigNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "batch_size": ("INT", {"default": 3, "min": 2, "max": 9, "step": 1}),
            "targets": (MAP_TYPES,),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}
    RETURN_TYPES = ("PIVOT_CONFIG",)
    FUNCTION = "build"

    CATEGORY = "vv/configs"

    def build(self, batch_size, targets, seed, start_percent, end_percent):
        return ({ 'batch_size': batch_size, 'targets': targets, 'seed': seed, 'start_percent': start_percent, 'end_percent': end_percent},)
