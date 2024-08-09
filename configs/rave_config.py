


from dataclasses import dataclass


@dataclass
class RaveAttentionConfig:
    targets: set[tuple[str, int]]
    grid_size: int
    seed: int
    start_percent: float
    end_percent: float

