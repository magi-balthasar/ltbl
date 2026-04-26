import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class SeaConfig:
    width: int = 200
    height: int = 200
    dt: float = 0.1
    nutrient_diffusion: float = 0.05
    toxin_diffusion: float = 0.03
    nutrient_decay: float = 0.001
    toxin_decay: float = 0.002
    vent_count: int = 3
    tidal_period: float = 100.0
    vent_pulse_strength: float = 2.0


class PrimordialSea:
    """2D chemical gradient world with hydrothermal vents, diffusion, and tidal effects."""

    def __init__(self, config: SeaConfig = None):
        self.config = config or SeaConfig()
        self.W = self.config.width
        self.H = self.config.height
        self.t = 0.0

        self.nutrient = np.zeros((self.H, self.W))
        self.toxin = np.zeros((self.H, self.W))
        self.temperature = np.ones((self.H, self.W)) * 0.3

        self.vents: List[Tuple[int, int, float]] = self._init_vents()
        self._init_toxin_patches()

    def _init_vents(self) -> List[Tuple[int, int, float]]:
        vents = []
        margin = 20
        for _ in range(self.config.vent_count):
            x = np.random.randint(margin, self.W - margin)
            y = np.random.randint(margin, self.H - margin)
            strength = np.random.uniform(0.5, 1.5)
            vents.append((x, y, strength))
        return vents

    def _init_toxin_patches(self):
        for _ in range(5):
            cx = np.random.randint(0, self.W)
            cy = np.random.randint(0, self.H)
            ys, xs = np.mgrid[-15:16, -15:16]
            d2 = xs**2 + ys**2
            patch = np.exp(-d2 / 30.0) * 0.3
            y0 = (cy - 15) % self.H
            x0 = (cx - 15) % self.W
            for dy in range(31):
                for dx in range(31):
                    ny = (cy - 15 + dy) % self.H
                    nx = (cx - 15 + dx) % self.W
                    self.toxin[ny, nx] += patch[dy, dx]

    def _diffuse(self, grid: np.ndarray, rate: float) -> np.ndarray:
        laplacian = (
            np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
            np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) -
            4 * grid
        )
        return grid + rate * laplacian

    def _tidal_factor(self) -> float:
        return 0.5 + 0.5 * np.sin(2 * np.pi * self.t / self.config.tidal_period)

    def step(self):
        self.nutrient = self._diffuse(self.nutrient, self.config.nutrient_diffusion)
        self.toxin = self._diffuse(self.toxin, self.config.toxin_diffusion)

        self.nutrient *= (1 - self.config.nutrient_decay)
        self.toxin *= (1 - self.config.toxin_decay)

        tidal = self._tidal_factor()
        for vx, vy, strength in self.vents:
            pulse = 0.5 + 0.5 * np.sin(self.t * 0.3 + strength)
            emission = strength * tidal * pulse * self.config.vent_pulse_strength
            self.nutrient[vy, vx] += emission
            self.temperature[vy, vx] = 0.3 + strength * 0.5

        np.clip(self.nutrient, 0, 10, out=self.nutrient)
        np.clip(self.toxin, 0, 5, out=self.toxin)
        self.t += self.config.dt

    def sample(self, x: float, y: float, radius: int = 1) -> Dict[str, np.ndarray]:
        ix, iy = int(x) % self.W, int(y) % self.H
        nutrient_vals, toxin_vals = [], []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = (ix + dx) % self.W, (iy + dy) % self.H
                nutrient_vals.append(self.nutrient[ny, nx])
                toxin_vals.append(self.toxin[ny, nx])
        return {
            'nutrient': np.array(nutrient_vals),
            'toxin': np.array(toxin_vals),
            'temp': float(self.temperature[iy, ix]),
        }

    def consume(self, x: float, y: float, amount: float) -> float:
        ix, iy = int(x) % self.W, int(y) % self.H
        available = self.nutrient[iy, ix]
        consumed = min(available, amount)
        self.nutrient[iy, ix] -= consumed
        return consumed

    def state_snapshot(self) -> Dict[str, np.ndarray]:
        return {
            'nutrient': self.nutrient.copy(),
            'toxin': self.toxin.copy(),
            't': self.t,
        }
