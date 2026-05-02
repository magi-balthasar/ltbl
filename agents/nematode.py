"""
Phase 3: Nematode (선충 / C. elegans 수준)

생물학적 모델: Caenorhabditis elegans
  - 뉴런 302개, 완전한 커넥톰 알려짐
  - 분절 없는 선형 몸체 (7 segments 시뮬레이션)
  - 좌우 근육 교번 수축으로 정현파 파형 운동 (undulatory locomotion)
  - CPG 없으면 S자 파형 불가 → 에너지 비효율 → 선택 압력

철학적 전제 (사용자 노트):
  "물리법칙이라는 제약 안에서 생존이라는 문제의 최적해는 수렴할 수밖에 없다."

  Phase 2: 모든 섬이 통합기(→) 회로로 수렴 — 리듬이 필요 없었기 때문.
  Phase 3: 분절 몸체가 생기면 W_rec에서 억제 링 연결이 자발적으로 선택될 것이다.
  이것이 Phase 2의 핵심 예측. Phase 3가 이 예측을 검증한다.

설계한 것 (물리법칙):
  - N=7 분절 몸체와 관절 각도의 존재
  - 파형 운동이 에너지 효율적이라는 물리법칙
    (CPG 리듬 없이 굽히면 낭비 에너지 → 선택 압력)
  - 고유감각(proprioception): 몸 상태 → 신경 피드백 (자기 모델 원형 가능성)
  - 3-layer 몸 설계 원칙: 감각 → 통합 → 운동 (발생학적 3배엽 등가물)

설계하지 않은 것:
  - CPG 회로가 창발하는지
  - wave_amplitude, body_stiffness 최적값
  - 어느 분절 패턴이 선택될지

behavior_log: 14-tuple (12-tuple 확장)
  [12] curvature             = mean |joint_angles| — 몸 굽힘 정도
  [13] locomotion_efficiency = 이동거리 / 에너지소모 — 운동 효율 (이전 틱 기준)

C4 (consciousness/level_monitor.py):
  corr(hidden_history, curvature_history)
  신경 활성화 norm이 자기 몸 상태와 상관되는가?
  → 양의 상관 = 신경망이 자기 몸을 "알고 있다" = 자기 모델 원형

C_CPG (consciousness/level_monitor.py):
  hidden[0]의 lag-1 자기상관 (부호 있는)
  → 음수 = CPG 진동 attractor 창발 (rhythm_h_history 사용)
"""

import numpy as np
from collections import deque
from typing import Tuple

from agents.nerve_net import NerveNetAgent
from genetics.genome import Genome, NERVE_N, NEMA_N_SEGS


class NematodeAgent(NerveNetAgent):
    """
    C. elegans 수준의 분절 몸체 + CPG 운동 에이전트.

    핵심 물리:
      hidden[0]이 진동(CPG)하면 → 분절에 위상차 사인파 → S자 파형 → 효율적 이동
      hidden[0]이 상수(통합기)면 → 직선 이동 + 비효율적 굽힘 에너지 낭비
    """

    N_SEGS  = NEMA_N_SEGS   # 7: 머리 + 6 몸체 분절
    SEG_LEN = 0.3            # 분절 간 거리 (물리 상수, 진화 대상 아님)

    # 분절당 시간 지연 (틱 단위).
    # 분절 i는 i * SEG_DELAY 틱 전의 rhythm 신호를 받음.
    # → CPG 진동 시: 각 분절이 다른 위상 → 진행파 (traveling wave)
    # → 통합기 상수 시: 모든 분절이 동일 각도 → 진행파 없음 (분산=0)
    SEG_DELAY = 2

    def __init__(self, x: float, y: float, genome: Genome, mode: str = 'asexual'):
        super().__init__(x, y, genome, mode)
        # 분절 관절 각도 (N_SEGS - 1개)
        self.seg_angles: np.ndarray = np.zeros(self.N_SEGS - 1, dtype=float)
        # 분절 위치 [[x0,y0], ..., [x6,y6]]
        self.seg_positions: np.ndarray = np.tile([x, y], (self.N_SEGS, 1)).astype(float)
        # 고유감각 이력 (C4 자기 모델 측정용)
        self.curvature_history: deque = deque(maxlen=32)
        # hidden[0] 이력 (C_CPG 탐지용 — ||hidden|| 노름이 아닌 부호 있는 성분)
        self.rhythm_h_history: deque = deque(maxlen=32)
        # 리듬 이력: 분절별 시간 지연 진행파 생성에 사용
        self._rhythm_history: deque = deque(maxlen=self.N_SEGS * self.SEG_DELAY + 4)
        # 이전 틱 기준 파형 효율 (decide()에서 속도 스케일링에 사용)
        self._wave_efficiency: float = 0.0
        # 이전 틱 이동 효율 (decide() 호출 시점에 이동 전이라 직접 계산 불가)
        # → _update_body()에서 계산 후 다음 틱 behavior_log에 기록
        self._prev_loco_eff: float = 0.0
        # 이전 틱 위치/에너지 (_update_body에서 계산 후 저장)
        self._prev_pos: np.ndarray = np.array([x, y], dtype=float)
        self._prev_energy: float = 5.0   # 초기 에너지 기본값

    # ── Override: step — 몸체 업데이트를 이동 후에 추가 ──────────────────────

    def step(self, sea, world_size: Tuple[int, int]):
        if not self.alive:
            return
        # 이동 전 상태 기록 (_update_body에서 locomotion_efficiency 계산용)
        self._prev_pos[:] = [self.x, self.y]
        self._prev_energy = self.state.energy

        # 이전 틱의 분절 각도로 파형 효율 계산 → decide()에서 속도 스케일링에 사용
        amp = float(np.clip(self.genome.wave_amplitude, 0.0, 1.0))
        max_var = (amp ** 2) * 0.5 + 1e-8
        self._wave_efficiency = min(1.0, float(np.var(self.seg_angles)) / max_var)

        # 부모 step: 신호 분비 → sense → decide(신경망 포함) → act(머리 이동)
        super().step(sea, world_size)

        # 몸체 운동학 업데이트 (머리 이동 후 — locomotion_efficiency 계산 가능)
        self._update_body()

    # ── Override: decide — 고유감각 + behavior_log 14-tuple 확장 ─────────────

    def decide(self,
               sensor_output: np.ndarray,
               local_conc: float = 0.0,
               local_light: float = 0.0,
               local_signal: float = 0.0,
               signal_window: np.ndarray = None) -> Tuple[float, float]:
        """
        NerveNetAgent.decide() 확장:
          - 파형 효율로 속도 스케일링 (CPG 선택압)
          - 고유감각(curvature)을 hidden state에 피드백 (자기 모델 원형)
          - behavior_log를 14-tuple로 확장
        """
        # 1. 부모 decide() → 12-tuple log 생성, 기저 vx/vy 획득
        vx, vy = super().decide(sensor_output,
                                local_conc=local_conc,
                                local_light=local_light,
                                local_signal=local_signal,
                                signal_window=signal_window)

        # 1-b. 파형 속도 스케일링: 이동 속도가 wave_efficiency에 비례
        #
        #  [CPG 선택 압력 물리법칙]
        #  wave_amplitude → 0으로 진화하면 분절 굽힘이 없어지고
        #  에너지 비용은 0이지만 이동도 불가능해짐 → CPG가 유일한 고속 이동 수단
        #
        #  speed_factor: 0.3 (파형 없음) ~ 1.0 (완전 파형)
        #  CPG 없이는 최대 속도의 30%만 달성 가능 → 먹이 경쟁에서 불리
        speed_factor = 0.3 + 0.7 * self._wave_efficiency
        vx *= speed_factor
        vy *= speed_factor

        # 2. 고유감각: 현재 몸 곡률 → hidden state 피드백
        #    생물학: 근방추(muscle spindle)가 관절 각도를 CNS로 피드백
        #    → 자기 몸 상태를 신경망이 반영 → C4 자기 모델 원형
        #    proprioception_weight 유전자가 피드백 강도 결정
        curvature = float(np.mean(np.abs(self.seg_angles)))
        prop_w = float(np.clip(self.genome.proprioception_weight, 0.0, 1.0))
        if prop_w > 1e-4 and len(self.hidden) == NERVE_N:
            self.hidden = np.tanh(self.hidden + prop_w * curvature * 0.5)

        # 3. 12-tuple → 14-tuple 확장
        #    loco_eff: 이전 틱에서 계산된 값 사용 (현재 틱은 이동 전이라 불가)
        if self.behavior_log:
            base = self.behavior_log[-1]
            self.behavior_log[-1] = (*base[:12], curvature, self._prev_loco_eff)

        return vx, vy

    # ── 몸체 운동학 ──────────────────────────────────────────────────────────

    def _update_body(self):
        """
        진행파(traveling wave) 기반 운동학 + CPG 선택 압력.

        핵심 물리법칙 설계:
          분절 i는 i*SEG_DELAY 틱 전의 rhythm 신호를 받음 (시간 지연)
          → CPG (진동) : 각 분절이 서로 다른 위상 → 분산 높음 → 진행파 O → 고속 이동
          → 통합기 (상수): 모든 분절이 동일 각도  → 분산=0  → 진행파 X → 저속 이동

        wave_efficiency = normalized variance of seg_angles
          → CPG 진동이 없으면 wave_efficiency ≈ 0
          → decide()에서 speed *= (0.3 + 0.7 * wave_efficiency) 적용
          → CPG 회로에 대한 실질적 선택 압력
        """
        rhythm = float(self.hidden[0]) if len(self.hidden) > 0 else 0.0
        amp    = float(np.clip(self.genome.wave_amplitude, 0.0, 1.0))
        stiff  = float(np.clip(self.genome.body_stiffness, 0.0, 0.95))

        # ① 리듬 이력에 현재 신호 기록
        self._rhythm_history.append(rhythm)
        self.rhythm_h_history.append(rhythm)   # C_CPG 측정용 (부호 있는 hidden[0])
        hist = list(self._rhythm_history)

        # ② 진행파: 분절 i = i*SEG_DELAY 틱 전 rhythm
        target_angles = np.zeros(self.N_SEGS - 1)
        for i in range(self.N_SEGS - 1):
            delay = i * self.SEG_DELAY
            if delay < len(hist):
                target_angles[i] = amp * hist[-(delay + 1)]

        # ③ 관절 각도 업데이트 (stiffness = 관성)
        self.seg_angles = stiff * self.seg_angles + (1.0 - stiff) * target_angles

        # ④ 체인 운동학: 머리에서 꼬리로 분절 위치 계산
        dx = self.x - self._prev_pos[0]
        dy = self.y - self._prev_pos[1]
        head_dir = float(np.arctan2(dy, dx)) if (dx*dx + dy*dy) > 1e-8 else 0.0

        self.seg_positions[0] = [self.x, self.y]
        cum_angle = head_dir
        for i, delta in enumerate(self.seg_angles):
            cum_angle += delta
            prev = self.seg_positions[i]
            self.seg_positions[i + 1] = (
                prev + self.SEG_LEN * np.array([np.cos(cum_angle), np.sin(cum_angle)])
            )

        # ⑤ 에너지 비용: 진행파 없이 굽히면 낭비 에너지
        curvature = float(np.mean(np.abs(self.seg_angles)))
        seg_var   = float(np.var(self.seg_angles))
        max_var   = (amp ** 2) * 0.5 + 1e-8
        wave_efficiency = min(1.0, seg_var / max_var)
        wasted_bending  = curvature * (1.0 - wave_efficiency)
        energy_cost     = 0.002 * (curvature + wasted_bending)
        self.state.energy = max(0.0, self.state.energy - energy_cost)

        self.curvature_history.append(curvature)

        # ⑥ locomotion_efficiency: 이동 후 계산 가능 → 다음 틱 behavior_log에 기록
        dist = float(np.linalg.norm([self.x - self._prev_pos[0],
                                     self.y - self._prev_pos[1]]))
        energy_used = max(self._prev_energy - self.state.energy, 1e-6)
        self._prev_loco_eff = dist / energy_used
