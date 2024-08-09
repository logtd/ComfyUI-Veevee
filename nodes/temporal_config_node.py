
class TemporalConfigNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}
    RETURN_TYPES = ("TEMPORAL_CONFIG",)
    FUNCTION = "build"

    CATEGORY = "vv/configs"

    def build(self, start_percent, end_percent):

        return ({ 'start_percent': start_percent, 'end_percent': end_percent},)