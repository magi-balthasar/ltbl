import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import uuid
from genetics.genome import Genome, SENSOR_COUNT


@dataclass
class InternalState:
    energy: float = 2.0
    internal_nutrient: float = 0.5
    internal_toxin: float = 0.0
    age: int = 0
    energy_history: List[float] = field(default_factory=list)
    max_history: int = 16

    def push_history(self):
        self.energy_history.append(self.energy)
        if len(self.energy_history) > self.max_history:
            self.energy_history.pop(0)

    def energy_gradient(self) -> float:
        if len(self.energy_history) < 2:
            return 0.0
        return self.energy_history[-1] - self.energy_history[-2]

    def energy_trend(self) -> float:
        """Slope over recent history via linear fit."""
        h = self.energy_history
        if len(h) < 3:
            return 0.0
        x = np.arange(len(h), dtype=float)
        return float(np.polyfit(x, h, 1)[0])


class CellAgent:
    """Phase 1-A: minimal cell with membrane, homeostasis, and sensor-actuator loop."""

    def __init__(self, x: float, y: float, genome: Genome, mode: str = 'asexual'):
        self.id = str(uuid.uuid4())[:8]
        self.x = x
        self.y = y
        self.genome = genome
        self.mode = mode
        self.state = InternalState()
        self.alive = True

        # Lamarckian experience: accumulated gradient of (reward signal × sensor input)
        self.experience_accum = np.zeros(len(genome.to_vector()))
        self.experience_steps = 0

        # Consciousness C1: log whether each action was internally driven
        self.behavior_log: List[Tuple[bool, Tuple[float, float]]] = []
        self.max_behavior_log = 64

    # ── Sensing ──────────────────────────────────────────────────────────────

    def sense(self, sea) -> Tuple[np.ndarray, dict]:
        radius = max(1, int(self.genome.sensor_range))
        raw = sea.sample(self.x, self.y, radius=radius)

        n_nutrient = raw['nutrient']
        n_toxin = raw['toxin']

        # Subsample or pad to SENSOR_COUNT
        nutrient_signal = self._resize_signal(n_nutrient, SENSOR_COUNT)
        toxin_signal = self._resize_signal(n_toxin, SENSOR_COUNT)

        combined = nutrient_signal - toxin_signal * (1.0 - self.genome.toxin_resistance)
        weighted = combined * self.genome.sensor_weights
        return weighted, raw

    @staticmethod
    def _resize_signal(arr: np.ndarray, target: int) -> np.ndarray:
        if len(arr) >= target:
            idx = np.linspace(0, len(arr) - 1, target, dtype=int)
            return arr[idx]
        return np.pad(arr, (0, target - len(arr)))

    # ── Decision ─────────────────────────────────────────────────────────────

    def decide(self, sensor_output: np.ndarray) -> Tuple[float, float]:
        n = len(sensor_output)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

        dx = float(np.sum(sensor_output * np.cos(angles)))
        dy = float(np.sum(sensor_output * np.sin(angles)))

        # Homeostatic urgency: energy below 30% of replication threshold
        low_energy = self.state.energy < self.genome.replication_threshold * 0.3
        high_toxin = self.state.internal_toxin > 0.3
        internally_driven = low_energy or high_toxin

        urgency = 1.2 if low_energy else (0.8 if high_toxin else 0.5)

        norm = np.sqrt(dx**2 + dy**2) + 1e-8
        speed = self.genome.max_speed * urgency * self.genome.actuator_sensitivity
        vx = dx / norm * speed
        vy = dy / norm * speed

        if len(self.behavior_log) >= self.max_behavior_log:
            self.behavior_log.pop(0)
        self.behavior_log.append((internally_driven, (vx, vy)))

        return vx, vy

    # ── Action ───────────────────────────────────────────────────────────────

    def act(self, vx: float, vy: float, sea, world_size: Tuple[int, int]):
        W, H = world_size
        self.x = (self.x + vx) % W
        self.y = (self.y + vy) % H

        consumed = sea.consume(self.x, self.y, self.genome.energy_consume_rate)
        self.state.energy += consumed * self.genome.membrane_permeability

        env = sea.sample(self.x, self.y, radius=0)
        local_toxin = float(env['toxin'].mean())
        self.state.internal_toxin += local_toxin * (1.0 - self.genome.toxin_resistance) * 0.01
        self.state.internal_toxin *= 0.98

        move_cost = (abs(vx) + abs(vy)) * 0.005
        self.state.energy -= self.genome.energy_decay + move_cost
        self.state.age += 1
        self.state.push_history()

        # Accumulate Lamarckian experience signal
        reward_signal = consumed - self.genome.energy_decay
        sensor_flat = np.concatenate([env['nutrient'], env['toxin']])
        exp_dim = len(self.experience_accum)
        sensor_trimmed = self._resize_signal(sensor_flat, exp_dim)
        self.experience_accum += reward_signal * sensor_trimmed
        self.experience_steps += 1

        if self.state.energy <= 0:
            self.alive = False

    # ── Replication ──────────────────────────────────────────────────────────

    def can_replicate(self) -> bool:
        return self.state.energy >= self.genome.replication_threshold

    def pay_replication_cost(self):
        self.state.energy -= self.genome.replication_cost

    @property
    def avg_experience(self) -> np.ndarray:
        if self.experience_steps == 0:
            return self.experience_accum
        return self.experience_accum / self.experience_steps

    # ── Main step ────────────────────────────────────────────────────────────

    def step(self, sea, world_size: Tuple[int, int]):
        if not self.alive:
            return
        sensor_out, _ = self.sense(sea)
        vx, vy = self.decide(sensor_out)
        self.act(vx, vy, sea, world_size)
