import numpy as np
from dataclasses import dataclass, field
from typing import List
import uuid

# ── Phase 2 nerve-net constants ───────────────────────────────────────────────
NERVE_N     = 3   # number of recurrent neurons (fixed: physics we declare)
NERVE_W_IN  = NERVE_N * NERVE_N   # (N, N): 3 abstract signals → N neurons
NERVE_W_REC = NERVE_N * NERVE_N   # (N, N): recurrent connections (CPG possible)
NERVE_W_OUT = 2   * NERVE_N       # (2, N): neurons → vx, vy
NERVE_EXTRA = 1                   # neural_weight blend scalar
NERVE_TOTAL = NERVE_W_IN + NERVE_W_REC + NERVE_W_OUT + NERVE_EXTRA  # 25


GENOME_SCHEMA = [
    # (name, default, min, max)
    # ── Phase 1-A: membrane scalars ──────────────────────────────────────────
    ('sensor_range',          2.0,  0.5,  5.0),
    ('max_speed',             0.5,  0.05, 2.0),
    ('actuator_sensitivity',  1.0,  0.1,  5.0),
    ('energy_decay',          0.01, 0.001, 0.1),
    ('energy_consume_rate',   0.05, 0.005, 0.5),
    ('replication_threshold', 5.0,  1.0,  20.0),
    ('replication_cost',      3.0,  0.5,  10.0),
    ('toxin_resistance',      0.1,  0.0,   1.0),
    ('membrane_permeability', 0.5,  0.1,   1.0),
    # ── Phase 1-B: chemotaxis scalars (ignored by membrane agents) ───────────
    ('run_duration',          8.0,  1.0,  30.0),   # ticks to keep running
    ('tumble_bias',           0.15, 0.01,  0.6),   # base tumble rate (no history)
    ('gradient_weight',       0.5,  0.0,   1.0),   # gradient vs internal urgency
    # ── Phase 1-C: phototaxis scalars (ignored by 1-A/1-B agents) ───────────
    ('light_sensitivity',     0.3,  0.0,   1.0),   # photosynthesis efficiency
    ('uv_resistance',         0.5,  0.0,   1.0),   # protection from excess light
    ('prediction_depth',      3.0,  1.0,  10.0),   # ticks ahead to predict
    # ── Phase 1-D: quorum sensing scalars (ignored by earlier agents) ────────
    ('signal_production',     0.3,  0.0,   1.0),   # autoinducer deposit rate
    ('signal_sensitivity',    0.5,  0.0,   1.0),   # gain on sensed quorum signal
    ('quorum_threshold',      0.4,  0.05,  1.0),   # signal level to enter collective mode
    ('collective_bias',       0.5,  0.0,   1.0),   # blend: 0=individual, 1=collective
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
    # Phase 1-B chemotaxis
    run_duration: float = 8.0
    tumble_bias: float = 0.15
    gradient_weight: float = 0.5
    # Phase 1-C phototaxis
    light_sensitivity: float = 0.3
    uv_resistance: float = 0.5
    prediction_depth: float = 3.0
    # Phase 1-D quorum sensing
    signal_production: float = 0.3
    signal_sensitivity: float = 0.5
    quorum_threshold: float = 0.4
    collective_bias: float = 0.5
    # Phase 2: distributed nerve net (N=3 recurrent neurons)
    # Matrices stored as arrays; serialised as flat vector after GENOME_SCHEMA scalars.
    # Design: we declare "N neurons with recurrent connections exist."
    # We do NOT declare which connectivity pattern emerges — evolution decides.
    nerve_w_in:    np.ndarray = field(default_factory=lambda: np.zeros((NERVE_N, NERVE_N)))
    nerve_w_rec:   np.ndarray = field(default_factory=lambda: np.zeros((NERVE_N, NERVE_N)))
    nerve_w_out:   np.ndarray = field(default_factory=lambda: np.zeros((2, NERVE_N)))
    neural_weight: float = 0.1   # blend: 0=pure Phase-1 behaviour, 1=pure neural

    def to_vector(self) -> np.ndarray:
        scalars = [getattr(self, name) for name, *_ in GENOME_SCHEMA]
        return np.concatenate([
            self.sensor_weights,        # 8
            scalars,                    # 19  (SCALAR_COUNT)
            self.nerve_w_in.ravel(),    # 9
            self.nerve_w_rec.ravel(),   # 9
            self.nerve_w_out.ravel(),   # 6
            [self.neural_weight],       # 1
        ])  # total: GENOME_DIM + NERVE_TOTAL = 27 + 25 = 52

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
            idx = SENSOR_COUNT + i
            val = float(np.clip(vec[idx], lo, hi)) if idx < len(vec) else default
            setattr(g, name, val)
        # ── Phase 2 nerve-net matrices (backward-compat: zeros if vec too short) ──
        offset = GENOME_DIM  # = SENSOR_COUNT + SCALAR_COUNT
        if len(vec) >= offset + NERVE_TOTAL:
            g.nerve_w_in  = vec[offset:offset + NERVE_W_IN].reshape(NERVE_N, NERVE_N).copy()
            offset += NERVE_W_IN
            g.nerve_w_rec = vec[offset:offset + NERVE_W_REC].reshape(NERVE_N, NERVE_N).copy()
            offset += NERVE_W_REC
            g.nerve_w_out = vec[offset:offset + NERVE_W_OUT].reshape(2, NERVE_N).copy()
            offset += NERVE_W_OUT
            g.neural_weight = float(np.clip(vec[offset], 0.0, 1.0))
        else:
            g.nerve_w_in    = np.zeros((NERVE_N, NERVE_N))
            g.nerve_w_rec   = np.zeros((NERVE_N, NERVE_N))
            g.nerve_w_out   = np.zeros((2, NERVE_N))
            g.neural_weight = 0.1
        return g

    def copy(self) -> 'Genome':
        return Genome.from_vector(self.to_vector(), parent_ids=[self.id],
                                  generation=self.generation + 1)
