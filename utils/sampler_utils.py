import torch
import numpy as np

import comfy.k_diffusion.sampling as k_diffusion_sampling

from ..constants import ModelOptionKey, InjectionType
from ..configs.create_configs import create_configs


def find_closest_index(array, value):
    array = np.array(array)
    index = (np.abs(array - value)).argmin()
    return index


def prepare_flows(flow_list, sub_idxs, x):
    prepared_flows = []
    for flow in flow_list:
        prepared_flow = {}
        if len(sub_idxs):
            prepared_flow['forward_trajectory'] = flow['forward_trajectory'][sub_idxs].to(x.device)
            prepared_flow['backward_trajectory'] = flow['backward_trajectory'][sub_idxs].to(x.device)
            prepared_flow['attn_masks'] = flow['attn_masks'][:,:,sub_idxs,:][:,:,:,sub_idxs].to(x.device)
        else:
            prepared_flow['forward_trajectory'] = flow['forward_trajectory'].to(x.device)
            prepared_flow['backward_trajectory'] = flow['backward_trajectory'].to(x.device)
            prepared_flow['attn_masks'] = flow['attn_masks'].to(x.device)
        prepared_flows.append(prepared_flow)
    return prepared_flows


def get_model_function_wrapper():
    def model_function_wrapper(apply_model_func, apply_params):
        timestep = apply_params['timestep']
        ad_params = apply_params['c']['transformer_options'].get('ad_params', {})
        sub_idxs = ad_params.get('sub_idxs', [])
        if sub_idxs is None:
            sub_idxs = []
        if not sub_idxs:
            sub_idx = 0
        else:
            sub_idx = sub_idxs[0]
        prepared = apply_params['c']['transformer_options']['FLOW']
        flow_config = apply_params['c']['transformer_options'].get('FLOW_CONFIG', None)
        
        prepared_idx = prepared.get('idx', -1)
        prepared_len = prepared.get('batch_len', 0)
        prepared_dir = prepared.get('direction', None)

        flow_direction = flow_config.flow['direction'] if flow_config is not None else None
        if (sub_idx != prepared_idx or len(sub_idxs) != prepared_len or flow_direction != prepared_dir) and flow_config:
            x = apply_params['input']
            prepared = { 'idx': sub_idx, 'batch_len': len(sub_idxs), 'forward_flows': [], 'backward_flows': [], 'direction': prepared_dir }
            flow_list = flow_config.flow['forward_flows']
            prepared['forward_flows'] = prepare_flows(flow_list, sub_idxs, x)
            flow_list = flow_config.flow['backward_flows']
            prepared['backward_flows'] = prepare_flows(flow_list, sub_idxs, x)
            apply_params['c']['transformer_options']['FLOW'] = prepared

        model_out = apply_model_func(apply_params['input'], timestep, **apply_params['c'])

        return model_out
    
    return model_function_wrapper


def get_stepper(model, model_options):
    timestep_bank = { 'step': -1 }
    def sample_step(x, sigma, **extra_args):
        timestep_bank['step'] += 1
        step = timestep_bank['step']
        model_options['transformer_options']['SIGMA'] = sigma[0].item()
        model_options['transformer_options']['STEP'] = step
        model_options['transformer_options']['STEP_PERCENT'] = step / model_options['transformer_options']['TOTAL_STEPS']

        output =  model(x, sigma, **extra_args)

        return output
    
    return sample_step


def create_sampler(sample_fn, attn_injection_steps, res_injection_steps, overlap, 
                   flow_config, inj_config, pivot_config, rave_config, sca_config, temporal_config):
    @torch.no_grad()
    def sample(model, latents, sigmas, extra_args=None, callback=None, disable=None, **extra_options):
        configs = create_configs(model, 
                                 inj_config, 
                                 flow_config,
                                 pivot_config,
                                 rave_config,
                                 sca_config,
                                 temporal_config)
        
        model_options = extra_args.get('model_options', {})
        transformer_options = model_options.get('transformer_options', {})
        model_options = {
            **model_options,
            'model_function_wrapper': get_model_function_wrapper(),
            'transformer_options': {
                **transformer_options,
                ModelOptionKey.INJECTION_TYPE: InjectionType.SAMPLING,
                'OVERLAP': overlap,
                'ATTN_INJECTION_STEPS': attn_injection_steps,
                'RES_INJECTION_STEPS': res_injection_steps,
                'TOTAL_STEPS': len(sigmas),
                'FLOW': {},
                **configs,
            }
        }
        extra_args = {**extra_args, 'model_options': model_options}

        sampler_stepper = get_stepper(model, model_options)

        output = sample_fn(sampler_stepper, latents, sigmas, extra_args=extra_args, callback=callback, disable=disable, **extra_options)

        if 'FLOW' in model_options['transformer_options']:
            del model_options['transformer_options']['FLOW']

        return output
    
    return sample


def create_unsampler(sample_fn, overlap, flow_config_dict, inj_config_dict):
    @torch.no_grad()
    def sample(model, latents, sigmas, extra_args=None, callback=None, disable=None, **extra_options):
        configs = create_configs(model, inj_config_dict, flow_config_dict)
        model_options = extra_args.get('model_options', {})
        transformer_options = model_options.get('transformer_options', {})
        model_options = {
            **model_options,
            'model_function_wrapper': get_model_function_wrapper(),
            'transformer_options': {
                **transformer_options,
                ModelOptionKey.INJECTION_TYPE: InjectionType.UNSAMPLING,
                'OVERLAP': overlap,
                'TOTAL_STEPS': len(sigmas),
                'FLOW': {},
                **configs,
            }
        }
        extra_args = {**extra_args, 'model_options': model_options}
        inj_config = configs['INJ_CONFIG']
        unsampler_stepper = get_stepper(model, model_options)
        if inj_config is not None:
            inj_config.attn_injections.clear()
            for sigma_key in inj_config.attn_sigmas:
                inj_config.attn_injections[sigma_key] = {}
                for block_key in inj_config.attn_map:
                    inj_config.attn_injections[sigma_key][block_key] = []

            inj_config.res_injections.clear()
            for sigma_key in inj_config.res_sigmas:
                inj_config.res_injections[sigma_key] = {}
                for block_key in inj_config.res_map:
                    inj_config.res_injections[sigma_key][block_key] = []

        output = sample_fn(unsampler_stepper, latents, sigmas, extra_args=extra_args, callback=callback, disable=disable, **extra_options)

        if inj_config is not None:
            for sigma_key in inj_config.attn_injections.keys():
                for block_key in inj_config.attn_injections[sigma_key].keys():
                    inj_config.attn_injections[sigma_key][block_key] = torch.cat(inj_config.attn_injections[sigma_key][block_key])

            for sigma_key in inj_config.res_injections.keys():
                for block_key in inj_config.res_injections[sigma_key].keys():
                    inj_config.res_injections[sigma_key][block_key] = torch.cat(inj_config.res_injections[sigma_key][block_key])

        if 'FLOW' in model_options['transformer_options']:
            del model_options['transformer_options']['FLOW']
        return output
    
    return sample


def get_sampler_fn(sampler_name):
    if sampler_name == "dpm_fast":
        def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):
            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            total_steps = len(sigmas) - 1
            return k_diffusion_sampling.sample_dpm_fast(model, noise, sigma_min, sigmas[0], total_steps, extra_args=extra_args, callback=callback, disable=disable)
        sampler_function = dpm_fast_function
    elif sampler_name == "dpm_adaptive":
        def dpm_adaptive_function(model, noise, sigmas, extra_args, callback, disable):
            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return k_diffusion_sampling.sample_dpm_adaptive(model, noise, sigma_min, sigmas[0], extra_args=extra_args, callback=callback, disable=disable)
        sampler_function = dpm_adaptive_function
    else:
        sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))
    return sampler_function
