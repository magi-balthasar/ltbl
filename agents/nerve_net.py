"""
Phase 2: Distributed Nerve Net (분산 신경망 / 히드라)

생물학적 모델: 히드라 (Hydra vulgaris)
  - 뉴런 ~1,000개, 뇌 없음
  - 분산 신경망(nerve net): 중앙 제어 없음, 국소 처리
  - 자극이 신경망을 통해 파급 (wave propagation)

철학적 전제:
  "진화는 창조가 아니라 물리법칙이 허용한 해공간을 탐색하는 과정이다."

  설계한 것 (물리법칙 역할):
    - N=3 재귀 뉴런의 존재
    - tanh 활성화 (억제/흥분 모두 가능)
    - 3개 추상 신호 입력 (내수용성·외수용성·사회적)
    - 재귀 연결 가능성 (W_rec 행렬)

  설계하지 않은 것:
    - 어떤 연결 패턴이 선택될지
    - CPG 링 오실레이터 vs 승자독식 vs 통합기
    - 어떤 섬이 어떤 회로로 수렴할지

  관찰 대상:
    - hidden_history 자기상관 → 음수=진동(CPG) / 양수=통합기
    - 섬 간 회로 수렴 여부 → 수렴 진화의 시뮬레이션

3개 추상 입력 (behavior_log에서 직접 추출):
  [0] energy_trend   (log[2])  내수용성 — 에너지 증감 추세
  [1] gradient_trend (log[4])  외수용성 — 화학/빛 구배 추세
  [2] social_signal  (log[8])  사회적   — 쿼럼 신호 강도

→ 생물학에서 고등 뇌 영역이 하위에서 추상화된 신호를 받는 구조와 동일.
   감각 원신호가 아니라 이미 처리된 의미 있는 신호가 신경망에 입력됨.

behavior_log: 12-tuple (10-tuple 확장)
  [10] hidden_norm         = ‖hidden‖ — 신경 활성화 수준
  [11] neural_contribution = ‖neural_out − base_out‖ — 신경층 기여도

C_NET (consciousness/level_monitor.py):
  lag-1 자기상관의 절댓값 평균
  → 0 = 무작위, 1 = 완전 일관 (진동이든 안정이든)
  sign 따로 기록 → 음수=CPG, 양수=통합기
"""

import numpy as np
from collections import deque
from typing import Tuple

from agents.quorum import QuorumSensingAgent
from genetics.genome import Genome, NERVE_N


class NerveNetAgent(QuorumSensingAgent):
    """히드라 수준의 분산 신경망 에이전트."""

    def __init__(self, x: float, y: float, genome: Genome, mode: str = 'asexual'):
        super().__init__(x, y, genome, mode)
        # 재귀 숨겨진 상태 (시간에 걸쳐 유지 → 시간적 기억의 원형)
        self.hidden: np.ndarray = np.zeros(NERVE_N)
        # C_NET 측정용 활성화 norm 이력
        self.hidden_history: deque = deque(maxlen=32)

    # ── Override: step (QuorumSensingAgent.step 재사용, decide만 교체) ────────

    def step(self, sea, world_size: Tuple[int, int]):
        if not self.alive:
            return

        # 쿼럼 신호 분비 상속 (Phase 1-D)
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

    # ── Override: decide — 신경망 레이어 추가 ────────────────────────────────

    def decide(self,
               sensor_output: np.ndarray,
               local_conc: float = 0.0,
               local_light: float = 0.0,
               local_signal: float = 0.0,
               signal_window: np.ndarray = None) -> Tuple[float, float]:
        """
        1. 부모(QuorumSensingAgent) decide → 10-tuple log 생성, 기저 vx/vy 획득
        2. behavior_log에서 3개 추상 신호 추출 (이미 처리된 의미 있는 신호)
        3. RNN 한 스텝: hidden = tanh(W_rec @ hidden + W_in @ net_input)
        4. 신경 출력: neural_out = W_out @ hidden
        5. 혼합: final = (1 - α) * base + α * neural
        6. behavior_log를 12-tuple로 확장
        """
        # 1. 기저 행동 (모든 Phase 1 능력 상속)
        vx, vy = super().decide(sensor_output,
                                local_conc=local_conc,
                                local_light=local_light,
                                local_signal=local_signal,
                                signal_window=signal_window)

        # 2. 3개 추상 신호 추출 (behavior_log의 10-tuple에서)
        if self.behavior_log:
            last = self.behavior_log[-1]
            energy_trend   = float(last[2]) if len(last) > 2 else 0.0
            gradient_trend = float(last[4]) if len(last) > 4 else 0.0
            social_sig     = float(last[8]) if len(last) > 8 else 0.0
        else:
            energy_trend = gradient_trend = social_sig = 0.0

        net_input = np.array([energy_trend, gradient_trend, social_sig],
                             dtype=np.float64)

        # 3. RNN step
        # W_rec @ hidden: 재귀 연결 (이 행렬이 CPG 가능성을 열어줌)
        # W_in  @ input:  추상 신호 → 뉴런
        # tanh: 억제(-1)와 흥분(+1) 모두 허용 → 억제 회로 창발 가능
        pre_act = (self.genome.nerve_w_rec @ self.hidden
                   + self.genome.nerve_w_in  @ net_input)
        self.hidden = np.tanh(pre_act)
        hidden_norm = float(np.linalg.norm(self.hidden))
        self.hidden_history.append(hidden_norm)

        # 4. 신경 출력 → 이동 벡터
        neural_out = self.genome.nerve_w_out @ self.hidden  # shape (2,)
        neural_vx  = float(neural_out[0])
        neural_vy  = float(neural_out[1])

        # 5. 혼합 (neural_weight 유전자: 0.1 기본값으로 초기엔 기저 행동 우세)
        alpha     = float(np.clip(self.genome.neural_weight, 0.0, 1.0))
        final_vx  = (1.0 - alpha) * vx + alpha * neural_vx
        final_vy  = (1.0 - alpha) * vy + alpha * neural_vy

        # 6. 10-tuple → 12-tuple 확장
        neural_contribution = float(np.sqrt(
            (neural_vx - vx) ** 2 + (neural_vy - vy) ** 2
        ))
        if self.behavior_log:
            base = self.behavior_log[-1]
            self.behavior_log[-1] = (*base[:10], hidden_norm, neural_contribution)

        return final_vx, final_vy
