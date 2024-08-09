SD1_ATTN_DEFAULTS = set([1,2,3,4,5,6])
SD1_RES_DEFAULTS = set([3,4,6])


class ConfigFlowAttnInjSD1Node:
    @classmethod
    def INPUT_TYPES(s):
        base = {"required": { 
        }}

        for i in range(9):
            base['required'][f'output_{i}'] = ("BOOLEAN", { "default": i in SD1_ATTN_DEFAULTS })
        
        return base
    RETURN_TYPES = ("ATTN_INJ_MAP",)
    FUNCTION = "apply"

    CATEGORY = "vv"

    def apply(self, **kwargs):

        attention_map = set()
        for key, value in kwargs.items():
            if value:
                block, idx = key.split('_')
                attention_map.add((block, int(idx)))
            
        return (attention_map, )


# 0-35
class ConfigFlowAttnInjSDXLNode:
    @classmethod
    def INPUT_TYPES(s):
        base = {"required": { 
            "attns": ("STRING", {"multiline": True }),
        }}
        return base
    RETURN_TYPES = ("ATTN_INJ_MAP",)
    FUNCTION = "apply"

    CATEGORY = "flow"

    def apply(self, attns):

        attention_map = set()
        for idx in attns.split(','):
            idx = int(idx.strip())
            attention_map.add(('output', int(idx)))
            
        return (attention_map, )


class ConfigFlowResInjSD1Node:
    @classmethod
    def INPUT_TYPES(s):
        base = {"required": { 
        }}

        for i in range(9):
            base['required'][f'output_{i}'] = ("BOOLEAN", { "default": i in SD1_RES_DEFAULTS })
        
        return base
    RETURN_TYPES = ("RES_INJ_MAP",)
    FUNCTION = "apply"

    CATEGORY = "flow"

    def apply(self, **kwargs):

        resnet_map = set()
        for key, value in kwargs.items():
            if value:
                block, idx = key.split('_')
                resnet_map.add((block, int(idx)))
            
        return (resnet_map, )
    

class ConfigFlowResInjSDXLNode:
    @classmethod
    def INPUT_TYPES(s):
        base = {"required": { 
        }}

        for i in range(9):
            base['required'][f'output_{i}'] = ("BOOLEAN", { "default": i in SD1_RES_DEFAULTS })
        
        return base
    RETURN_TYPES = ("RES_INJ_MAP",)
    FUNCTION = "apply"

    CATEGORY = "flow"

    def apply(self, **kwargs):

        resnet_map = set()
        for key, value in kwargs.items():
            if value:
                block, idx = key.split('_')
                resnet_map.add((block, int(idx)))
            
        return (resnet_map, )