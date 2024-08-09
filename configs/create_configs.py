from ..configs.rave_config import RaveAttentionConfig
from ..configs.pivot_config import PivotAttentionConfig
from ..configs.sca_config import SparseCasualAttentionConfig
from ..configs.flow_config import FlowAttentionConfig
from ..configs.inj_config import InjectionConfig
from ..configs.temporal_config import TemporalAttentionConfig
from ..vv_defaults import SD1_MAPS, SDXL_MAPS, SD1_ATTN_INJ_DEFAULTS, SD1_RES_INJ_DEFAULTS, SD1_FLOW_MAP, SDXL_ATTN_INJ_DEFAULTS, SDXL_RES_INJ_DEFAULTS, SDXL_FLOW_MAP


def create_configs(
        model,
        inj_config=None,
        flow_config=None,
        pivot_config=None,
        rave_config=None,
        sca_config=None,
        temporal_config=None,
):
    model_type = 'SD1.5' #TODO
    if model_type == 'SD1.5':
        if inj_config is not None:
            inj_config = InjectionConfig(
                SD1_ATTN_INJ_DEFAULTS,
                SD1_RES_INJ_DEFAULTS,
                inj_config['attn_injections'],
                inj_config['res_injections'],
                inj_config['attn_save_steps'],
                inj_config['res_save_steps'],
                inj_config['attn_sigmas'],
                inj_config['res_sigmas']
            )
        if flow_config is not None:
            flow_config = FlowAttentionConfig(
                SD1_MAPS[flow_config['targets']],
                flow_config['flow'],
                flow_config['start_percent'],
                flow_config['end_percent']
            )
        if rave_config is not None:
            rave_config = RaveAttentionConfig(
                SD1_MAPS[rave_config['targets']], 
                rave_config['grid_size'], 
                rave_config['seed'],
                rave_config['start_percent'],
                rave_config['end_percent'])
        if pivot_config is not None:
            pivot_config = PivotAttentionConfig(
                SD1_MAPS[pivot_config['targets']], 
                pivot_config['batch_size'], 
                pivot_config['seed'],
                pivot_config['start_percent'],
                pivot_config['end_percent'])
        if sca_config is not None:
            sca_config = SparseCasualAttentionConfig(
                SD1_MAPS[sca_config['targets']], 
                sca_config['direction'],
                sca_config['start_percent'],
                sca_config['end_percent'])
        if temporal_config is not None:
            temporal_config = TemporalAttentionConfig(
                temporal_config['start_percent'],
                temporal_config['end_percent'],
            )
    else:
        if inj_config is not None:
            inj_config = InjectionConfig(
                SDXL_ATTN_INJ_DEFAULTS,
                SDXL_RES_INJ_DEFAULTS,
                inj_config['attn_injections'],
                inj_config['res_injections'],
                inj_config['attn_save_steps'],
                inj_config['res_save_steps'],
                inj_config['attn_sigmas'],
                inj_config['res_sigmas']
            )
        if flow_config is not None:
            flow_config = FlowAttentionConfig(
                SDXL_MAPS[flow_config['targets']],
                flow_config['flow'],
                flow_config['start_percent'],
                flow_config['end_percent']
            )
        if rave_config is not None:
            rave_config = RaveAttentionConfig(
                SDXL_MAPS[rave_config['targets']], 
                rave_config['grid_size'], 
                rave_config['seed'],
                rave_config['start_percent'],
                rave_config['end_percent'])
        if pivot_config is not None:
            pivot_config = PivotAttentionConfig(
                SDXL_MAPS[pivot_config['targets']], 
                pivot_config['batch_size'], 
                pivot_config['seed'],
                pivot_config['start_percent'],
                pivot_config['end_percent'])
        if sca_config is not None:
            sca_config = SparseCasualAttentionConfig(
                SDXL_MAPS[sca_config['targets']], 
                sca_config['direction'],
                sca_config['start_percent'],
                sca_config['end_percent'])
        if temporal_config is not None:
            temporal_config = TemporalAttentionConfig(
                temporal_config['start_percent'],
                temporal_config['end_percent'],
            )
    return {
        'INJ_CONFIG': inj_config,
        'FLOW_CONFIG': flow_config,
        'RAVE_CONFIG': rave_config,
        'PIVOT_CONFIG': pivot_config,
        'SCA_CONFIG': sca_config,
        'TEMPORAL_CONFIG': temporal_config
    }