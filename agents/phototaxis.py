"""
Phase 1-C: Phototaxis — 주광성 (시아노박테리아)

프로젝트명 "빛이 있으라"와 직접 연결되는 단계.

핵심 원리:
  - 빛(궤도 운동 광원)을 에너지로 수확 (광합성)
  - 빛의 시간적 패턴을 기억하고 미래 위치를 예측 (C3 창발)
  - 과거 기록으로 미래를 시뮬레이션 — 예측의 원형

Phase 1-C = Phase 1-B(Run/Tumble) + Phase 1-A(항상성) + 광합성 + 예측

prediction_depth 유전자:
  - 몇 틱 앞을 예측할지 결정
  - 너무 짧으면 반응적 행동에 그침
  - 너무 길면 부정확한 예측으로 에너지 낭비
  - 최적값은 환경의 빛 주기와 맞물려 진화로 발견됨

behavior_log 포맷 (8-tuple, 6-tuple 확장):
  (internally_driven, (vx,vy), energy_trend, action_mag,
   gradient_trend, tumbled,
   light_trend,      [6] 빛 추세 (선형 기울기)
   actual_light)     [7] 현재 실제 빛 강도

C3은 predicted_light[t] = actual_light[t] + light_trend[t] * prediction_depth
와 actual_light[t + prediction_depth]의 상관계수로 계산.
ConsciousnessMonitor가 len(entry) >= 8로 감지.
"""

import numpy as np
from collections import deque
from typing import Tuple

from agents.chemotaxis import ChemotaxisAgent
from genetics.genome import Genome


class PhototaxisAgent(ChemotaxisAgent):
    """시아노박테리아 모델: 광합성 + 빛 예측 (C3 창발)."""

    def __init__(self, x: float, y: float, genome: Genome, mode: str = 'asexual'):
        super().__init__(x, y, genome, mode)
        self.light_history: deque = deque(maxlen=16)

    # ── Override: step needs light from environment ───────────────────────────

    def step(self, sea, world_size: Tuple[int, int]):
        if not self.alive:
            return
        sensor_out, raw = self.sense(sea)
        local_conc  = (float(raw['nutrient'].mean())
                       - float(raw['toxin'].mean()) * self.genome.toxin_resistance)
        local_light = float(raw['light'].mean()) if 'light' in raw else 0.0
        vx, vy = self.decide(sensor_out, local_conc=local_conc, local_light=local_light)
        self.act(vx, vy, sea, world_size)
        self._photosynthesize(local_light)

    # ── Override: phototaxis decision ────────────────────────────────────────

    def decide(self, sensor_output: np.ndarray,
               local_conc: float = 0.0,
               local_light: float = 0.0) -> Tuple[float, float]:
        """
        Run/Tumble inherited from ChemotaxisAgent.
        Additionally: bias direction toward predicted future light peak.
        """
        self.light_history.append(local_light)

        # ── Light trend & prediction ──────────────────────────────────────────
        light_hist = list(self.light_history)
        light_trend = 0.0
        if len(light_hist) >= 4:
            x = np.arange(len(light_hist), dtype=float)
            light_trend = float(np.polyfit(x, light_hist, 1)[0])

        # ── Run/Tumble (from ChemotaxisAgent logic) ───────────────────────────
        self.concentration_history.append(local_conc)
        tumbled = False
        gradient_trend = 0.0
        chem_hist = list(self.concentration_history)
        if len(chem_hist) >= 4:
            recent = np.mean(chem_hist[-2:])
            older  = np.mean(chem_hist[-4:-2])
            gradient_trend = float(recent - older)
            improving = gradient_trend > 0

            run_limit = max(1, int(self.genome.run_duration))
            if self.is_running:
                if (not improving) or (self.run_ticks >= run_limit):
                    self.direction = np.random.uniform(0, 2 * np.pi)
                    self.is_running = False
                    self.run_ticks  = 0
                    tumbled = True
                else:
                    self.run_ticks += 1
            else:
                self.is_running = True
                self.run_ticks  = 0
        else:
            if np.random.random() < self.genome.tumble_bias:
                self.direction = np.random.uniform(0, 2 * np.pi)
                tumbled = True

        # ── Predictive light-seeking: bias direction toward future light ──────
        # If light is improving in current direction → keep running
        # If light is declining AND we just tumbled → pick direction biased
        #   toward where light_trend predicts peak will be
        pred_depth = max(1, int(self.genome.prediction_depth))
        predicted_light = local_light + light_trend * pred_depth

        # When tumbling: bias new direction by gradient_weight × light_trend
        # (agent "hopes" more light is ahead)
        if tumbled and len(light_hist) >= 4 and light_trend < 0:
            # Light declining here → try a light-seeking direction offset
            # Use light_trend magnitude to add bias to the new random direction
            bias_angle = np.random.normal(0, np.pi * (1.0 - self.genome.gradient_weight))
            self.direction = (self.direction + np.pi + bias_angle) % (2 * np.pi)

        # ── Speed (inherited energy-trend urgency) ────────────────────────────
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
        # 8-tuple: extends 6-tuple with light_trend + actual_light
        self.behavior_log.append((
            internally_driven, (vx, vy), trend, action_mag,
            gradient_trend, tumbled,
            light_trend, local_light,
        ))

        return vx, vy

    # ── Photosynthesis ────────────────────────────────────────────────────────

    def _photosynthesize(self, light_level: float):
        """Light → energy gain. Excess light → UV toxin cost."""
        gain = light_level * self.genome.light_sensitivity * 0.15
        self.state.energy += gain

        # UV damage: very bright light harms poorly-shielded agents
        uv_threshold = 0.65
        if light_level > uv_threshold:
            uv_damage = ((light_level - uv_threshold)
                         * (1.0 - self.genome.uv_resistance) * 0.03)
            self.state.internal_toxin += uv_damage
