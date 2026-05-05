"""
Phase 4: Vertebrate Transition (척추동물 전이 / 어류)

생물학적 모델: 원시 어류 (기악류 등가물)
  - 측선(lateral line): 물 압력파 감지 → 주변 에이전트 이동 감지
  - 척수 계층화: N=3 척수 CPG + N=4 뇌간(hindbrain) 통합 레이어
  - 포식자 압력: 즉각적 통합 판단 필요 → 계층적 신경 통합 선택압
  - 군집(schooling): 측선 + 포식압 → 집단 행동 자발 창발 가능

철학적 전제:
  "어류가 선충보다 더 잘 학습할 수 있었던 이유는 새 알고리즘이 생겨서가 아니다.
   새로운 물리적 신체 구조가 생겨서, 그것을 통합하는 신경계가 필요해졌기 때문이다."

설계한 것 (물리법칙):
  - 측선 감각 기관의 존재 (흐름장에서 읽기)
  - 척수(N=3) + 뇌간(N=4) 계층 구조의 존재
  - 뉴런이 함께 발화하면 연결이 변할 수 있다는 가능성 (Oja's rule)
  - 이동 시 흐름장에 속도 기록 (물리적 측선 신호원)

설계하지 않은 것:
  - 측선 정보로 무엇을 할지 (진화 결정)
  - 어떤 뇌간 연결 패턴이 선택될지
  - 학습률 η의 최적값 (진화 결정)
  - 군집이 형성될지, 포식자 회피에 쓰일지

behavior_log: 16-tuple (14-tuple 확장)
  [14] hind_norm         = ‖hind_hidden‖ — 뇌간 활성화 수준
  [15] lateral_coherence = ‖agent_flow‖  — 측선 신호 강도

C5 (consciousness/level_monitor.py):
  집단 반응 일관성: 측선 신호 변화 → 집단 이동 방향 변화 상관관계
  → C5 > 0.3 = 군집이 외부 자극에 집단적으로 반응함

C_PLASTIC:
  mean(|Δplastic_delta_w|) per step → 학습이 실제 일어나는가?

C_BALDWIN:
  corr(parent.plastic_delta_w, child.hind_w_rec - baseline)
  → 부모 학습이 자식 게놈에 반영되는가?
"""

import numpy as np
from collections import deque
from typing import Tuple

from agents.nematode import NematodeAgent
from genetics.genome import Genome, NERVE_N, FISH_HIND_N


class FishAgent(NematodeAgent):
    """
    어류 수준의 측선 + 척수/뇌간 계층 + Hebbian 가소성 에이전트.

    핵심 물리:
      흐름장(flow field)에 이동 벡터 기록 → 이웃 에이전트가 측선으로 감지
      뇌간(N=4)이 척수 CPG(N=3) 위에서 측선 신호를 통합
      Oja's rule: 함께 발화하면 연결 강화 (보상 없음, 순수 co-activation)
    """

    def __init__(self, x: float, y: float, genome: Genome, mode: str = 'asexual'):
        super().__init__(x, y, genome, mode)
        # 뇌간 숨겨진 상태 (N=4, 척수 hidden N=3과 분리)
        self.hind_hidden: np.ndarray = np.zeros(FISH_HIND_N)
        # Hebbian 소성 가중치 변화 (수명 동안 누적)
        # 초기화: 0 (게놈의 hind_w_rec가 진화, plastic_delta_w는 경험)
        self.plastic_delta_w: np.ndarray = np.zeros((FISH_HIND_N, FISH_HIND_N))
        # 마지막 sense() 결과 캐시 (측선 신호 접근용)
        self._last_raw: dict = {}
        # 측선 신호 이력 (C5 집단 반응 측정용)
        self.lateral_flow_history: deque = deque(maxlen=32)
        # 뇌간 활성화 이력 (C_PLASTIC 측정용)
        self.hind_norm_history: deque = deque(maxlen=32)

    # ── Override: sense — raw 캐시 ────────────────────────────────────────────

    def sense(self, sea):
        sensor_out, raw = super().sense(sea)
        self._last_raw = raw
        return sensor_out, raw

    # ── Override: step — 이동 후 흐름장 기록 (측선 신호원) ──────────────────────

    def step(self, sea, world_size: Tuple[int, int]):
        if not self.alive:
            return
        prev_x, prev_y = self.x, self.y
        # 부모 step: 신호 분비 → sense → decide(뇌간 포함) → act → _update_body
        super().step(sea, world_size)
        # 이동 벡터를 흐름장에 기록 — 이웃의 측선이 감지하는 물리 신호
        dx = self.x - prev_x
        dy = self.y - prev_y
        sea.deposit_flow(self.x, self.y, dx, dy)

    # ── Override: decide — 뇌간 RNN + Oja Hebbian ────────────────────────────

    def decide(self,
               sensor_output: np.ndarray,
               local_conc: float = 0.0,
               local_light: float = 0.0,
               local_signal: float = 0.0,
               signal_window: np.ndarray = None) -> Tuple[float, float]:
        """
        1. 척수 레이어 (NematodeAgent.decide()) → 14-tuple log, 기저 vx/vy
        2. 측선 신호 읽기 (_last_raw['agent_flow'])
        3. 뇌간 RNN step: hind_w_rec @ hind_hidden + hind_w_cross @ spinal + input
        4. Oja's rule (보상 없는 순수 co-activation Hebbian)
        5. 뇌간 출력으로 속도 조율
        6. 14-tuple → 16-tuple 확장
        """
        # 1. 척수 레이어 (Phase 1~3 전체 상속)
        vx, vy = super().decide(sensor_output,
                                local_conc=local_conc,
                                local_light=local_light,
                                local_signal=local_signal,
                                signal_window=signal_window)

        # 2. 측선 신호 (흐름장에서 읽은 주변 이동 벡터 평균)
        agent_flow = self._last_raw.get('agent_flow', np.zeros(2))
        lateral_sens = float(np.clip(self.genome.lateral_sensitivity, 0.0, 1.0))
        lat_x = float(agent_flow[0]) * lateral_sens
        lat_y = float(agent_flow[1]) * lateral_sens
        lat_mag = float(np.linalg.norm(agent_flow)) * lateral_sens

        # 3. 뇌간 입력: [에너지 추세, 측선_x, 측선_y, 측선_강도]
        #    에너지 추세: behavior_log에서 추출 (Phase 2 동일 구조)
        if self.behavior_log:
            last = self.behavior_log[-1]
            energy_trend = float(last[2]) if len(last) > 2 else 0.0
        else:
            energy_trend = 0.0

        hind_input = np.array([energy_trend, lat_x, lat_y, lat_mag], dtype=np.float64)

        # 4. 뇌간 RNN step
        #    척수(spinal, N=3) → 뇌간 교차 연결 (hind_w_cross 4×3)
        #    설계: "척수 CPG 신호가 뇌간에 입력된다" (물리)
        #    어떤 신호를 통합할지는 진화가 결정
        spinal_hidden = self.hidden  # N=3 from NerveNetAgent
        pre_hind = (self.genome.hind_w_rec @ self.hind_hidden
                    + self.genome.hind_w_cross @ spinal_hidden
                    + hind_input)
        self.hind_hidden = np.tanh(pre_hind)

        # 5. Oja's rule — 순수 Hebbian (보상 함수 없음)
        #    Δw_ij = η × h_i × h_j − η × h_i² × w_ij
        #    → 동시 활성화 시 연결 강화, 자기 연결 감쇠 (안정화)
        #    η = plastic_learning_rate (진화가 최적값 찾음: 0이면 학습 없음)
        eta = float(np.clip(self.genome.plastic_learning_rate, 0.0, 0.05))
        if eta > 1e-7:
            h = self.hind_hidden
            # 유효 가중치 = 게놈 기저 + 생애 학습 누적
            w_eff = self.genome.hind_w_rec + self.plastic_delta_w
            oja_delta = eta * (np.outer(h, h) - np.diag(h ** 2) @ w_eff)
            self.plastic_delta_w = np.clip(
                self.plastic_delta_w + oja_delta, -2.0, 2.0
            )

        # 6. 뇌간 출력 → 척수 출력에 더함 (조율, 억제/강화 모두 가능)
        hind_out = self.genome.hind_w_out @ self.hind_hidden  # shape (2,)
        final_vx = vx + float(hind_out[0])
        final_vy = vy + float(hind_out[1])

        # 7. 14-tuple → 16-tuple 확장
        hind_norm = float(np.linalg.norm(self.hind_hidden))
        lat_coherence = float(np.linalg.norm(agent_flow))  # 비스케일 강도
        self.hind_norm_history.append(hind_norm)
        self.lateral_flow_history.append(np.array([agent_flow[0], agent_flow[1]]))

        if self.behavior_log:
            base = self.behavior_log[-1]
            self.behavior_log[-1] = (*base[:14], hind_norm, lat_coherence)

        return final_vx, final_vy

    # ── Lamarckian epigenetic: plastic_delta_w → 자식 게놈으로 약한 전달 ─────

    @property
    def avg_experience(self) -> np.ndarray:
        """
        Baldwin 효과 기반 약한 후성유전 전달.
        plastic_delta_w (4×4)를 평탄화해서 경험 벡터로 반환.
        replication.lamarckian()이 이를 hind_w_rec에 약하게 반영.
        """
        return self.plastic_delta_w.ravel()
