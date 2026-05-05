from dataclasses import dataclass, field
from typing import List


@dataclass
class IslandConfig:
    island_id: int
    world_width: int = 200
    world_height: int = 200
    initial_agents: int = 80
    max_agents: int = 800
    replication_mode: str = 'asexual'   # asexual / sexual / lamarckian
    mutation_rate: float = 0.01
    mutation_strength: float = 0.1
    migration_interval: int = 20        # steps between migration events
    ticks_per_step: int = 50
    vent_count: int = 3
    agent_type: str = 'membrane'  # 'membrane'|'chemotaxis'|'phototaxis'|'quorum'|'nerve_net'|'nematode'|'fish'
    mate_selection: str = 'random'  # 'random' | 'mhc' (MHC 등가물: 유전적 다양성 선호)
    predator_count: int = 0            # Phase 4: 포식자 수 (fish 섬에서만 사용)


def default_experiment() -> List[IslandConfig]:
    """
    8-island setup (Phase 4: Vertebrate Transition / 어류):
      1 nematode (Phase 3 기준선) + 7 fish (Phase 4 주실험)

    Phase 4 설계:
      - 측선(lateral line): 흐름장에서 주변 에이전트 이동 감지
      - 척수(N=3) + 뇌간(N=4) 계층 구조
      - 포식자 압력: 섬당 N=3 고정, 지능 없음 (조건만 선언)
      - 순수 Hebbian 가소성 (Oja's rule, 보상 함수 없음)
      - Baldwin 효과: plastic_delta_w → 자식 hind_w_rec 약한 전달

    핵심 관찰 질문:
      ① C5 > 0.3이 되는가? (군집 schooling 창발)
      ② C_PLASTIC > 0이 되는가? (Hebbian 학습 실제 발생)
      ③ C_BALDWIN 상승하는가? (볼드윈 효과: 학습→게놈 반영)
      ④ 포식자 있는 섬 vs 없는 섬: 군집 창발 속도 차이?
    """
    return [
        # ── Phase 3 기준선: 몸체 CPG — C_CPG 음수 유지 확인 ──────────────────
        IslandConfig(island_id=0, agent_type='nematode', replication_mode='sexual',
                     mutation_rate=0.01),
        # ── Phase 4: Fish ─────────────────────────────────────────────────────
        # 포식자 없음: 측선만으로 군집이 창발하는가?
        IslandConfig(island_id=1, agent_type='fish', replication_mode='asexual',
                     mutation_rate=0.01, predator_count=0),
        # 포식자 3마리: 핵심 선택압 — 군집 창발 가속?
        IslandConfig(island_id=2, agent_type='fish', replication_mode='sexual',
                     mutation_rate=0.01, predator_count=3),
        # 포식자 + 라마르크: Baldwin 효과 탐지 (plastic_delta_w → 게놈)
        IslandConfig(island_id=3, agent_type='fish', replication_mode='lamarckian',
                     mutation_rate=0.01, predator_count=3),
        # 포식자 + 넓은 탐색: η 유전자 빠른 수렴
        IslandConfig(island_id=4, agent_type='fish', replication_mode='asexual',
                     mutation_rate=0.02, predator_count=3),
        # MHC 배우자 선택 + 포식자: 유전적 다양성이 군집 탐색 가속?
        IslandConfig(island_id=5, agent_type='fish', replication_mode='sexual',
                     mutation_rate=0.02, mate_selection='mhc', predator_count=3),
        # 고포식 압력: 포식자 6마리 → 강한 선택압
        IslandConfig(island_id=6, agent_type='fish', replication_mode='asexual',
                     mutation_rate=0.02, predator_count=6),
        # 소규모 고압 환경: 고압 스트레스 + 포식자
        IslandConfig(island_id=7, agent_type='fish', replication_mode='asexual',
                     mutation_rate=0.02, world_width=80, world_height=80,
                     initial_agents=30, max_agents=200, vent_count=1,
                     predator_count=3),
    ]
