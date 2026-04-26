from typing import List, Dict, Deque
from collections import deque
import numpy as np


class EmergenceDetector:
    """Tracks consciousness level history and fires signals on threshold crossings."""

    def __init__(self, window: int = 20, min_population: int = 5, warmup_steps: int = 30):
        self.window = window
        self.min_population = min_population
        # Ignore early spikes: agents start energy-depleted, which caused false positives
        self.warmup_steps = warmup_steps
        self._history: Deque[Dict] = deque(maxlen=window)
        # Rising-edge state: only fire once per crossing, not every step above threshold
        self._c1_above: bool = False
        self._prev_c_level: int = 0

    def record(self, step: int, island_id: int, metrics: Dict[str, float],
               c_level: int, population: int):
        self._history.append({
            'step': step, 'island_id': island_id, 'c_level': c_level,
            'population': population, **metrics,
        })

    def detect(self) -> List[Dict]:
        """Return emergence events, ignoring warmup. Fires each event only once per crossing."""
        if not self._history:
            return []
        curr = self._history[-1]
        if curr['step'] < self.warmup_steps:
            return []

        events = []
        if curr['population'] >= self.min_population:
            # LEVEL_UP: rising edge on consciousness level
            if curr['c_level'] > self._prev_c_level:
                events.append({
                    'type': 'LEVEL_UP',
                    'island_id': curr['island_id'],
                    'step': curr['step'],
                    'from': self._prev_c_level,
                    'to': curr['c_level'],
                })
            self._prev_c_level = curr['c_level']

            # C1_THRESHOLD: fire ONCE on the first rising edge above 0.5
            c1_now_above = curr['C1'] > 0.5
            if c1_now_above and not self._c1_above:
                events.append({
                    'type': 'C1_THRESHOLD',
                    'island_id': curr['island_id'],
                    'step': curr['step'],
                    'C1': curr['C1'],
                })
            self._c1_above = c1_now_above

        return events
