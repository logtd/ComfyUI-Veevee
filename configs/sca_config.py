


from dataclasses import dataclass


class SCADirection:
    PREVIOUS = 'PREVIOUS'
    NEXT = 'NEXT'
    BOTH = 'BOTH'
    

@dataclass
class SparseCasualAttentionConfig:
    targets: set[tuple[str, int]]
    direction: SCADirection
    start_percent: float
    end_percent: float

