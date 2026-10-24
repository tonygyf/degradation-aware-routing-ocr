from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    image_id: str
    region_id: str
    polygon: tuple[tuple[float, float], ...]
    text: str
    confidence: float
    estimated_cost_units: float = 1.0
