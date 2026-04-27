"""
Phase 1-B: Chemotaxis — Run / Tumble

대장균(E. coli) 화학주성 모델.

핵심 원리:
  - 현재 화학 농도를 직전 기록과 비교한다 (시간적 비교 = C2 창발)
  - 농도가 개선 중이면 계속 달린다 (Run)
  - 농도가 악화되거나 런 시간이 초과되면 무작위 방향으로 전환한다 (Tumble)

run_duration, tumble_bias, gradient_weight 는 유전된다.
어떤 파라미터 조합이 생존에 유리한지는 환경이 선택한다.

behavior_log 포맷 (6-tuple, CellAgent의 4-tuple 확장):
  (internally_driven, (vx, vy), energy_trend, action_mag, gradient_trend, tumbled)
  [0]                 [1]        [2]           [3]         [4]             [5]
ConsciousnessMonitor는 len(entry) >= 6 로 chemotaxis 에이전트를 구별한다.
"""

import numpy as np
from collections import deque
from typing import Tuple

from agents.membrane import CellAgent
from genetics.genome import Genome


class ChemotaxisAgent(CellAgent):
    """E. coli 화학주성: Run/Tumble + 시간적 그라디언트 비교."""

    def __init__(self, x: float, y: float, genome: Genome, mode: str = 'asexual'):
        super().__init__(x, y, genome, mode)
        self.concentration_history: deque = deque(maxlen=16)
        self.direction: float = np.random.uniform(0, 2 * np.pi)
        self.run_ticks: int = 0
        self.is_running: bool = True

    # ── Override: step needs raw concentration ────────────────────────────────

    def step(self, sea, world_size: Tuple[int, int]):
        if not self.alive:
            return
        sensor_out, raw = self.sense(sea)
        local_conc = (float(raw['nutrient'].mean())
                      - float(raw['toxin'].mean()) * self.genome.toxin_resistance)
        vx, vy = self.decide(sensor_out, local_conc=local_conc)
        self.act(vx, vy, sea, world_size)

    # ── Override: chemotaxis decision ─────────────────────────────────────────

    def decide(self, sensor_output: np.ndarray,
               local_conc: float = 0.0) -> Tuple[float, float]:
        self.concentration_history.append(local_conc)

        tumbled = False
        gradient_trend = 0.0

        hist = list(self.concentration_history)
        if len(hist) >= 4:
            recent = np.mean(hist[-2:])
            older  = np.mean(hist[-4:-2])
            gradient_trend = float(recent - older)
            improving = gradient_trend > 0

            run_limit = max(1, int(self.genome.run_duration))
            if self.is_running:
                if (not improving) or (self.run_ticks >= run_limit):
                    self.direction = np.random.uniform(0, 2 * np.pi)
                    self.is_running = False
                    self.run_ticks = 0
                    tumbled = True
                else:
                    self.run_ticks += 1
            else:
                # One tick after tumble → start running
                self.is_running = True
                self.run_ticks = 0
        else:
            # Insufficient history: biased random walk
            if np.random.random() < self.genome.tumble_bias:
                self.direction = np.random.uniform(0, 2 * np.pi)
                tumbled = True

        # Speed modulated by energy trend (inherited from 1-A)
        trend = self.state.energy_trend()
        has_temporal = len(self.state.energy_history) >= 3
        internally_driven = has_temporal and trend < -0.008

        if has_temporal and trend < 0:
            urgency = 1.0 + min(2.0, abs(trend) * 15.0)
        elif has_temporal and trend > 0.02:
            urgency = 0.4
        else:
            urgency = 0.7
        if self.state.internal_toxin > 0.3:
            urgency = max(urgency, 1.0)

        speed = self.genome.max_speed * urgency * self.genome.actuator_sensitivity
        vx = float(np.cos(self.direction) * speed)
        vy = float(np.sin(self.direction) * speed)
        action_mag = float(np.sqrt(vx ** 2 + vy ** 2))

        if len(self.behavior_log) >= self.max_behavior_log:
            self.behavior_log.pop(0)
        self.behavior_log.append(
            (internally_driven, (vx, vy), trend, action_mag, gradient_trend, tumbled)
        )

        return vx, vy
