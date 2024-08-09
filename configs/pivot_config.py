from dataclasses import dataclass


@dataclass
class PivotAttentionConfig:
    targets: set[tuple[str, int]]
    batch_size: int
    seed: int
    start_percent: float
    end_percent: float