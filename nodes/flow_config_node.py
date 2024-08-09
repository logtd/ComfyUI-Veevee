
class FlowConfigNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "flow": ("FLOW",),
            "targets": (['full', 'inner', 'outer', 'none'],),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}
    RETURN_TYPES = ("FLOW_CONFIG",)
    FUNCTION = "build"

    CATEGORY = "vv/configs"

    def build(self, flow, targets, start_percent, end_percent):
        return ({ 'targets': targets, 'flow': flow, 'start_percent': start_percent, 'end_percent': end_percent},)