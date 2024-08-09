

class ConfigFlowAttnFlowSD1Node:
    @classmethod
    def INPUT_TYPES(s):
        base = {"required": { 
        }}
        for i in range(8):
            base['required'][f'input_{i}'] = ("BOOLEAN", { "default": i < 2 })

        base['required'][f'middle_0'] = ("BOOLEAN", { "default": False })

        for i in range(9):
            base['required'][f'output_{i}'] = ("BOOLEAN", { "default": i > 6 })
        
        return base
    RETURN_TYPES = ("ATTN_MAP",)
    FUNCTION = "apply"

    CATEGORY = "vv"

    def apply(self, **kwargs):

        attention_map = set()
        for key, value in kwargs.items():
            if value:
                block, idx = key.split('_')
                attention_map.add((block, int(idx)))
            
        return (attention_map, )


SDXL_DEFAULT_ALL_ATTNS = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35"

class ConfigFlowAttnFlowSDXLNode:
    @classmethod
    def INPUT_TYPES(s):
        base = {"required": { 
            "input_attns": ("STRING", {"multiline": True, "default": SDXL_DEFAULT_ALL_ATTNS }),
            "output_attns": ("STRING", {"multiline": True, "default": SDXL_DEFAULT_ALL_ATTNS }),
        }}
        return base
    RETURN_TYPES = ("ATTN_MAP",)
    FUNCTION = "apply"

    CATEGORY = "flow"

    def apply(self, input_attns, output_attns):

        attention_map = set()
        if input_attns != '' and input_attns is not None:
            for idx in input_attns.split(','):
                idx = idx.strip()
                if idx is '':
                    continue
                attention_map.add(('input', int(idx)))
        if input_attns != '' and input_attns is not None:
            for idx in output_attns.split(','):
                idx = idx.strip()
                if idx is '':
                    continue
                attention_map.add(('output', int(idx)))
            
        return (attention_map, )
    

class ConfigFlowAttnCrossFrameSDXLNode:
    @classmethod
    def INPUT_TYPES(s):
        base = {"required": { 
            "input_attns": ("STRING", {"multiline": True, "default": SDXL_DEFAULT_ALL_ATTNS }),
            "output_attns": ("STRING", {"multiline": True, "default": SDXL_DEFAULT_ALL_ATTNS }),
        }}
        return base
    RETURN_TYPES = ("ATTN_MAP",)
    FUNCTION = "apply"

    CATEGORY = "flow"

    def apply(self, input_attns, output_attns):

        attention_map = set()
        if input_attns != '' and input_attns is not None:
            for idx in input_attns.split(','):
                idx = idx.strip()
                if idx == '' or idx is None:
                    continue
                attention_map.add(('input', int(idx)))
        if output_attns != '' and output_attns is not None:
            for idx in output_attns.split(','):
                idx = idx.strip()
                if idx == '' or idx is None:
                    continue
                attention_map.add(('output', int(idx)))
            
        return (attention_map, )