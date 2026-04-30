"""
Phase 1-D: Quorum Sensing — 쿼럼 센싱 (집단 신호)

생물학적 모델: 시아노박테리아 → 그람음성균 쿼럼 센싱 (AHL 자가유도물질)
  - 개체가 화학 신호(autoinducer)를 분비
  - 국소 신호 농도가 임계값 초과 → 집단 모드 전환
  - 개체 행동 ≠ 설계. 집단 행동은 신호 장과의 상호작용에서 창발

핵심 원칙:
  - 신호 장(agent_signal grid)은 세계 물리의 일부 — 보상이 아님
  - 에이전트는 신호를 분비하고 감지한다. 협력은 설계하지 않는다
  - 집단 편향이 생존을 돕는지는 진화가 결정한다

behavior_log 포맷 (10-tuple, 8-tuple 확장):
  (...8 inherited fields from PhototaxisAgent...,
   local_signal,   [8] 현재 위치 국소 신호 농도
   in_quorum)      [9] 쿼럼 상태 (0.0 or 1.0)

C_QS = corr(local_signal[t], quorum_state_change[t])
       신호 강도와 집단 모드 전환이 실제로 연결되어 있는지 측정.
ConsciousnessMonitor가 len(entry) >= 10으로 감지.
"""

import numpy as np
from typing import Tuple

from agents.phototaxis import PhototaxisAgent
from genetics.genome import Genome


class QuorumSensingAgent(PhototaxisAgent):
    """그람음성균 모델: 자가유도물질 분비 + 쿼럼 기반 집단 행동 창발."""

    def __init__(self, x: float, y: float, genome: Genome, mode: str = 'asexual'):
        super().__init__(x, y, genome, mode)
        self.in_quorum: bool = False

    # ── Override: step deposits signal before sensing ─────────────────────────

    def step(self, sea, world_size: Tuple[int, int]):
        if not self.alive:
            return

        # Deposit autoinducer proportional to signal_production gene
        sea.deposit_signal(self.x, self.y,
                           self.genome.signal_production * 0.1)

        sensor_out, raw = self.sense(sea)
        local_conc   = (float(raw['nutrient'].mean())
                        - float(raw['toxin'].mean()) * self.genome.toxin_resistance)
        local_light  = float(raw['light'].mean()) if 'light' in raw else 0.0
        signal_vals  = raw.get('agent_signal', np.zeros(9))
        local_signal = float(signal_vals.mean())

        vx, vy = self.decide(sensor_out,
                             local_conc=local_conc,
                             local_light=local_light,
                             local_signal=local_signal,
                             signal_window=signal_vals)
        self.act(vx, vy, sea, world_size)
        self._photosynthesize(local_light)

    # ── Override: quorum-blended decision ────────────────────────────────────

    def decide(self,
               sensor_output: np.ndarray,
               local_conc: float = 0.0,
               local_light: float = 0.0,
               local_signal: float = 0.0,
               signal_window: np.ndarray = None) -> Tuple[float, float]:
        """
        Inherits Run/Tumble + light prediction from PhototaxisAgent.
        Blends result with collective gradient when quorum is detected.
        """
        # 1. Get base direction from phototaxis parent (appends 8-tuple to log)
        vx, vy = super().decide(sensor_output,
                                local_conc=local_conc,
                                local_light=local_light)

        # 2. Quorum detection
        self.in_quorum = local_signal > self.genome.quorum_threshold

        # 3. Collective direction: gradient of agent_signal field
        if self.in_quorum and signal_window is not None and len(signal_window) >= 9:
            side = int(round(len(signal_window) ** 0.5))
            w = np.array(signal_window, dtype=float).reshape(side, side)
            # Gradient: direction of increasing signal density
            grad_x = float(w[:, -1].mean() - w[:, 0].mean())
            grad_y = float(w[-1, :].mean() - w[0, :].mean())
            mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
            if mag > 1e-8:
                grad_x /= mag
                grad_y /= mag
                # Blend individual with collective
                speed = np.sqrt(vx ** 2 + vy ** 2) + 1e-8
                cbias = np.clip(
                    self.genome.collective_bias * self.genome.signal_sensitivity,
                    0.0, 1.0
                )
                vx = (1.0 - cbias) * vx + cbias * grad_x * speed
                vy = (1.0 - cbias) * vy + cbias * grad_y * speed

        # 4. Extend last behavior_log entry from 8-tuple to 10-tuple
        if self.behavior_log:
            last = self.behavior_log[-1]
            self.behavior_log[-1] = (*last[:8], local_signal, float(self.in_quorum))

        return vx, vy
