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
    # Phase 1-C: light
    light_orbit_period: float = 200.0   # ticks per full orbit
    light_orbit_radius: float = 0.35    # fraction of world size
    light_sigma: float = 0.15          # Gaussian spread (fraction of world size)
    light_max_intensity: float = 1.0


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
        self.light = np.zeros((self.H, self.W))   # Phase 1-C: orbiting light
        self.light_tick = 0

        # Pre-compute meshgrid for vectorised light calculation
        ys, xs = np.mgrid[0:self.H, 0:self.W]
        self._ys = ys.astype(float)
        self._xs = xs.astype(float)

        self.vents: List[Tuple[int, int, float]] = self._init_vents()
        self._init_toxin_patches()
        self._update_light()   # initialise light

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

    def _update_light(self):
        """Orbiting light source: predictable sinusoidal path around world centre."""
        cfg = self.config
        angle = 2 * np.pi * self.light_tick / cfg.light_orbit_period
        # Day/night intensity: max at peak, fades to 0.1 at nadir
        intensity = cfg.light_max_intensity * (0.55 + 0.45 * np.sin(angle))
        cx = self.W / 2 + self.W * cfg.light_orbit_radius * np.cos(angle)
        cy = self.H / 2 + self.H * cfg.light_orbit_radius * np.sin(angle)
        sigma_x = self.W * cfg.light_sigma
        sigma_y = self.H * cfg.light_sigma
        d2 = ((self._xs - cx) / sigma_x) ** 2 + ((self._ys - cy) / sigma_y) ** 2
        self.light = intensity * np.exp(-0.5 * d2)
        np.clip(self.light, 0.0, cfg.light_max_intensity, out=self.light)

    def _diffuse(self, grid: np.ndarray, rate: float) -> np.ndarray:
        laplacian = (
            np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
            np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) -
            4 * grid
        )
        return grid + rate * laplacian

    def _tidal_factor(self) -> float:
        return 0.5 + 0.5 * np.sin(2 * np.pi * self.t / self.config.tidal_period)

    def step(self, pressure: float = 0.0):
        """
        pressure: 0.0 = pristine, 1.0 = extreme stress.
        Higher pressure → weaker vent output, stronger toxin, erratic pulsation.
        The environment changes independently of agents — it is not a reward signal.
        """
        self.nutrient = self._diffuse(self.nutrient, self.config.nutrient_diffusion)
        self.toxin = self._diffuse(self.toxin, self.config.toxin_diffusion)

        # Pressure increases natural decay (resource depletion over geological time)
        nutrient_decay = self.config.nutrient_decay * (1 + pressure * 4)
        toxin_decay    = self.config.toxin_decay    * (1 - pressure * 0.5)  # toxin lingers longer
        self.nutrient *= (1 - nutrient_decay)
        self.toxin    *= (1 - max(0.0001, toxin_decay))

        # Vent output weakens under pressure (hydrothermal activity declining)
        vent_modifier = max(0.05, 1.0 - pressure * 0.85)
        # Pulsation becomes more erratic under pressure
        erratic = 1.0 + pressure * 2.0

        tidal = self._tidal_factor()
        for vx, vy, strength in self.vents:
            pulse = 0.5 + 0.5 * np.sin(self.t * 0.3 * erratic + strength)
            emission = strength * tidal * pulse * self.config.vent_pulse_strength * vent_modifier
            self.nutrient[vy, vx] += emission
            self.temperature[vy, vx] = 0.3 + strength * 0.5 * vent_modifier

        # New toxin seeps in as pressure rises (volcanic / UV analogue)
        if pressure > 0.2:
            seep_intensity = (pressure - 0.2) * 0.05
            seep_x = int(self.t * 7.3) % self.W
            seep_y = int(self.t * 3.7) % self.H
            self.toxin[seep_y, seep_x] += seep_intensity

        np.clip(self.nutrient, 0, 10, out=self.nutrient)
        np.clip(self.toxin, 0, 5, out=self.toxin)
        self.light_tick += 1
        self._update_light()
        self.t += self.config.dt

    def sample(self, x: float, y: float, radius: int = 1) -> Dict[str, np.ndarray]:
        ix, iy = int(x) % self.W, int(y) % self.H
        nutrient_vals, toxin_vals = [], []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = (ix + dx) % self.W, (iy + dy) % self.H
                nutrient_vals.append(self.nutrient[ny, nx])
                toxin_vals.append(self.toxin[ny, nx])
        light_vals = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = (ix + dx) % self.W, (iy + dy) % self.H
                light_vals.append(self.light[ny, nx])
        return {
            'nutrient': np.array(nutrient_vals),
            'toxin':    np.array(toxin_vals),
            'light':    np.array(light_vals),
            'temp':     float(self.temperature[iy, ix]),
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
