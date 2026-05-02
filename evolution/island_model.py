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
    agent_type: str = 'membrane'  # 'membrane'|'chemotaxis'|'phototaxis'|'quorum'|'nerve_net'|'nematode'
    mate_selection: str = 'random'  # 'random' | 'mhc' (MHC 등가물: 유전적 다양성 선호)


def default_experiment() -> List[IslandConfig]:
    """
    8-island setup (Phase 3: Nematode + Genetics Redesign):
      1 nerve_net (Phase 2 기준선) + 7 nematode (Phase 3 주실험)

    Phase 3 유전 시스템 재설계 반영:
      - 감수분열 등가물 (Crossing Over) — sexual 섬 모두
      - 진화하는 돌연변이율 (mutation_rate_gene)
      - 바이스만 장벽 (lamarckian 대폭 감소)
      - MHC 등가물 배우자 선택 (island_id=5)

    핵심 관찰 질문:
      ① C_NET_sign이 음수로 전환되는가? (CPG attractor 창발 — Phase 2 예측 검증)
      ② C4 > 0이 되는가? (자기 몸 모델 창발)
      ③ MHC 선택이 CPG 회로 탐색을 가속하는가? (유전적 다양성 효과)
      ④ mutation_rate_gene이 수렴하는 값은? (최적 돌연변이율)
    """
    return [
        # ── Phase 2 기준선: 몸 없는 신경망 — C_NET_sign > 0 (통합기) 유지 예상 ─
        IslandConfig(island_id=0, agent_type='nerve_net', replication_mode='sexual',
                     mutation_rate=0.02),
        # ── Phase 3: Nematode ─────────────────────────────────────────────────
        IslandConfig(island_id=1, agent_type='nematode', replication_mode='asexual',
                     mutation_rate=0.01),   # 기본 실험: mutation_rate_gene 수렴값 관찰
        IslandConfig(island_id=2, agent_type='nematode', replication_mode='sexual',
                     mutation_rate=0.01),   # 감수분열 교차: CPG 탐색 가속?
        IslandConfig(island_id=3, agent_type='nematode', replication_mode='lamarckian',
                     mutation_rate=0.01),   # 바이스만 장벽 적용: 경험 간접 전달
        IslandConfig(island_id=4, agent_type='nematode', replication_mode='asexual',
                     mutation_rate=0.02),   # 넓은 탐색 공간
        # MHC 등가물 배우자 선택: 유전적 다양성 선호가 CPG 발견을 가속하는가?
        IslandConfig(island_id=5, agent_type='nematode', replication_mode='sexual',
                     mutation_rate=0.02, mate_selection='mhc'),
        IslandConfig(island_id=6, agent_type='nematode', replication_mode='asexual',
                     mutation_rate=0.04),   # 고돌연변이: 초기 CPG 창발 속도?
        IslandConfig(island_id=7, agent_type='nematode', replication_mode='asexual',
                     mutation_rate=0.02, world_width=80, world_height=80,
                     initial_agents=30, max_agents=200, vent_count=1),  # 고압 스트레스
    ]
