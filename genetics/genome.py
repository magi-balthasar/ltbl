import numpy as np
from dataclasses import dataclass, field
from typing import List
import uuid


GENOME_SCHEMA = [
    # (name, default, min, max)
    ('sensor_range',          2.0,  0.5,  5.0),
    ('max_speed',             0.5,  0.05, 2.0),
    ('actuator_sensitivity',  1.0,  0.1,  5.0),
    ('energy_decay',          0.01, 0.001, 0.1),
    ('energy_consume_rate',   0.05, 0.005, 0.5),
    ('replication_threshold', 5.0,  1.0,  20.0),
    ('replication_cost',      3.0,  0.5,  10.0),
    ('toxin_resistance',      0.1,  0.0,   1.0),
    ('membrane_permeability', 0.5,  0.1,   1.0),
]

SENSOR_COUNT = 8
SCALAR_COUNT = len(GENOME_SCHEMA)
GENOME_DIM = SENSOR_COUNT + SCALAR_COUNT


def _default_sensors() -> np.ndarray:
    return np.random.randn(SENSOR_COUNT) * 0.1


@dataclass
class Genome:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    mutation_count: int = 0

    sensor_weights: np.ndarray = field(default_factory=_default_sensors)

    sensor_range: float = 2.0
    max_speed: float = 0.5
    actuator_sensitivity: float = 1.0
    energy_decay: float = 0.01
    energy_consume_rate: float = 0.05
    replication_threshold: float = 5.0
    replication_cost: float = 3.0
    toxin_resistance: float = 0.1
    membrane_permeability: float = 0.5

    def to_vector(self) -> np.ndarray:
        scalars = [getattr(self, name) for name, *_ in GENOME_SCHEMA]
        return np.concatenate([self.sensor_weights, scalars])

    @classmethod
    def from_vector(cls, vec: np.ndarray, parent_ids: List[str] = None,
                    generation: int = 0) -> 'Genome':
        g = cls.__new__(cls)
        g.id = str(uuid.uuid4())[:8]
        g.parent_ids = parent_ids or []
        g.generation = generation
        g.mutation_count = 0
        g.sensor_weights = vec[:SENSOR_COUNT].copy()
        for i, (name, default, lo, hi) in enumerate(GENOME_SCHEMA):
            val = float(np.clip(vec[SENSOR_COUNT + i], lo, hi))
            setattr(g, name, val)
        return g

    def copy(self) -> 'Genome':
        return Genome.from_vector(self.to_vector(), parent_ids=[self.id],
                                  generation=self.generation + 1)
