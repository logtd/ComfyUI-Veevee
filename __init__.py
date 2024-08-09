from .nodes.vv_apply_node import ApplyVVModel
from .nodes.get_flow_node import FlowGetFlowNode
from .nodes.vv_unsampler_node import VVUnsamplerSamplerNode
from .nodes.vv_sampler_node import VVSamplerSamplerNode
from .nodes.inj_config_node import InjectionConfigNode
from .nodes.pivot_config_node import PivotConfigNode
from .nodes.rave_config_node import RaveConfigNode
from .nodes.sca_config_node import SCAConfigNode
from .nodes.flow_config_node import FlowConfigNode
from .nodes.temporal_config_node import TemporalConfigNode


NODE_CLASS_MAPPINGS = {
    "ApplyVVModel": ApplyVVModel,
    "FlowGetFlow": FlowGetFlowNode,
    "VVUnsamplerSampler": VVUnsamplerSamplerNode,
    "VVSamplerSampler": VVSamplerSamplerNode,
    "InjectionConfig": InjectionConfigNode,
    "FlowConfig": FlowConfigNode,
    "PivotConfig": PivotConfigNode,
    "RaveConfig": RaveConfigNode,
    "SCAConfig": SCAConfigNode,
    "TemporalConfig": TemporalConfigNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyVVModel": "VV] Apply Model",
    "FlowGetFlow": "VV] Get Flow",
    "VVUnsamplerSampler": "VV] Unsampler",
    "VVSamplerSampler": "VV] Sampler",
    "InjectionConfig": "VV] Injection Config",
    "FlowConfig": "VV] Flow Attn Config",
    "PivotConfig": "VV] Pivot Attn Config",
    "RaveConfig": "VV] Rave Attn Config",
    "SCAConfig": "VV] Sparse Casual Attn Config",
    "TemporalConfig": "VV] Temporal Attn Config",
}
