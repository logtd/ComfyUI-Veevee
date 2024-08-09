from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class FlowAttentionConfig:
    targets: set[tuple[str, int]]
    flow: Dict[Any, Any]
    start_percent: float
    end_percent: float
