"""
Environmental pressure schedule.

Nature does not declare phase transitions.
Pressure builds. Agents adapt or die.
If something new emerges, we observe it.

Calibration rule:
  survival < 10%  → ease  (extinction pressure — too fast)
  survival > 85% and no clustering signal → intensify (stasis — too slow)
  clustering signal rising → hold (something is happening, let it play out)
"""

from dataclasses import dataclass, field
from typing import Deque
from collections import deque


@dataclass
class PressureSchedule:
    level: float = 0.0           # 0 = pristine ocean, 1 = extreme stress
    base_step: float = 0.0005    # how fast pressure rises each simulation step
    min_level: float = 0.0
    max_level: float = 0.95
    _survival_history: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    _cluster_history: Deque[float] = field(default_factory=lambda: deque(maxlen=30))

    def tick(self):
        """Pressure rises autonomously every simulation step."""
        self.level = min(self.max_level, self.level + self.base_step)

    def record(self, survival_rate: float, cluster_signal: float):
        self._survival_history.append(survival_rate)
        self._cluster_history.append(cluster_signal)

    def calibrate(self):
        """
        God adjusts pressure rate based on observed outcomes.
        Called periodically (not every step) by the intervener.
        """
        if len(self._survival_history) < 5:
            return

        avg_survival = sum(self._survival_history) / len(self._survival_history)
        avg_cluster  = sum(self._cluster_history)  / len(self._cluster_history)

        if avg_survival < 0.10:
            # Mass extinction risk — ease pressure, slow the rise
            self.level = max(self.min_level, self.level - 0.05)
            self.base_step = max(0.0001, self.base_step * 0.7)
        elif avg_survival > 0.85 and avg_cluster < 0.1:
            # Agents are thriving with no new behavior — ramp pressure faster
            self.base_step = min(0.002, self.base_step * 1.3)
        elif avg_cluster > 0.3:
            # Clustering signal detected — hold pressure, let it stabilize
            self.base_step = max(0.0001, self.base_step * 0.5)

    @property
    def summary(self) -> str:
        avg_s = (sum(self._survival_history) / len(self._survival_history)
                 if self._survival_history else 0.0)
        return f'pressure={self.level:.3f}  base_step={self.base_step:.5f}  avg_survival={avg_s:.2f}'
