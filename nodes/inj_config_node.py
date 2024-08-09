
def find_closest_index(lst, target):
    return min(range(len(lst)), key=lambda i: abs(lst[i] - target))


class InjectionConfigNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "unsampler_sigmas": ("SIGMAS",),
            "sampler_sigmas": ("SIGMAS",),
            "save_attn_steps": ("INT", {"default": 3, "min": 0, "max": 999, "step": 1}),
            "save_res_steps": ("INT", {"default": 3, "min": 0, "max": 999, "step": 1}),
        }}
    RETURN_TYPES = ("INJ_CONFIG",)
    FUNCTION = "create"

    CATEGORY = "vv/configs"

    def create(self, unsampler_sigmas, sampler_sigmas, save_attn_steps, save_res_steps):
        attn_injections = {}
        attn_sigmas = []
        for i in range(save_attn_steps):
            sampler_sigma = sampler_sigmas[i].item()
            unsampler_idx = find_closest_index(unsampler_sigmas, sampler_sigma)
            unsampler_sigma = unsampler_sigmas[unsampler_idx].item()
            attn_injections[unsampler_sigma] = {}
            attn_sigmas.append(unsampler_sigma)

        res_injections = {}
        res_sigmas = []
        for i in range(save_res_steps):
            sampler_sigma = sampler_sigmas[i].item()
            unsampler_idx = find_closest_index(unsampler_sigmas, sampler_sigma)
            unsampler_sigma = unsampler_sigmas[unsampler_idx].item()
            res_injections[unsampler_sigma] = {}
            res_sigmas.append(unsampler_sigma)

        config = {
            'attn_injections': attn_injections,
            'res_injections': res_injections,
            'attn_save_steps': save_attn_steps,
            'res_save_steps': save_res_steps,
            'attn_sigmas': attn_sigmas,
            'res_sigmas': res_sigmas
        }
        return (config, )
