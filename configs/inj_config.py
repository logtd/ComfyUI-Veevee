from dataclasses import dataclass
from typing import Dict, List, Union

import torch


@dataclass
class InjectionConfig:
    attn_map: set[tuple[str, int]]
    res_map: set[tuple[str, int]]
    attn_injections: Dict[float, Dict[tuple[str, int], Union[torch.Tensor, List[torch.Tensor]]]]
    res_injections: Dict[float, Dict[tuple[str, int], Union[torch.Tensor, List[torch.Tensor]]]]
    attn_save_steps: int
    res_save_steps: int
    attn_sigmas: List[float]
    res_sigmas: List[float]
