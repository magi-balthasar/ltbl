from typing import List, Dict, Deque
from collections import deque
import numpy as np


class EmergenceDetector:
    """Tracks consciousness level history and fires signals on threshold crossings."""

    def __init__(self, window: int = 20, min_population: int = 5):
        self.window = window
        self.min_population = min_population
        self._history: Deque[Dict] = deque(maxlen=window)

    def record(self, step: int, island_id: int, metrics: Dict[str, float],
               c_level: int, population: int):
        self._history.append({
            'step': step, 'island_id': island_id, 'c_level': c_level,
            'population': population, **metrics,
        })

    def detect(self) -> List[Dict]:
        """Return emergence events from recent history."""
        if len(self._history) < 2:
            return []
        events = []
        prev = self._history[-2]
        curr = self._history[-1]
        if curr['population'] >= self.min_population:
            if curr['c_level'] > prev['c_level']:
                events.append({
                    'type': 'LEVEL_UP',
                    'island_id': curr['island_id'],
                    'step': curr['step'],
                    'from': prev['c_level'],
                    'to': curr['c_level'],
                })
            if curr['C1'] > 0.5 and prev['C1'] <= 0.5:
                events.append({
                    'type': 'C1_THRESHOLD',
                    'island_id': curr['island_id'],
                    'step': curr['step'],
                    'C1': curr['C1'],
                })
        return events
