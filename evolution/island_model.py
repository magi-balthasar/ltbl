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
    agent_type: str = 'membrane'  # 'membrane'|'chemotaxis'|'phototaxis'|'quorum'|'nerve_net'


def default_experiment() -> List[IslandConfig]:
    """
    8-island setup (Phase 2):
      1 phototaxis (1-C baseline) + 1 quorum (1-D baseline) + 6 nerve_net (Phase 2)

    핵심 관찰 질문:
      - Island 2–7이 독립적으로 비슷한 회로에 수렴하는가? (수렴 진화 재현)
      - 어떤 섬이 CPG (음의 C_NET_sign)로 수렴하는가?
      - sexual vs asexual의 C_NET 차이? (재조합이 회로 탐색 가속?)
    """
    return [
        # ── Phase 1-C: Phototaxis — 하한 기준선 (신경망 없음) ─────────────────
        IslandConfig(island_id=0, agent_type='phototaxis', replication_mode='asexual',
                     mutation_rate=0.01),
        # ── Phase 1-D: Quorum — 중간 기준선 (신경망 없음) ─────────────────────
        IslandConfig(island_id=1, agent_type='quorum', replication_mode='asexual',
                     mutation_rate=0.01),
        # ── Phase 2: Nerve Net — 주실험 (6 variants) ──────────────────────────
        # 각 섬은 독립적으로 탐색 → 어떤 회로가 이 물리계의 attractor인지 관찰
        IslandConfig(island_id=2, agent_type='nerve_net', replication_mode='asexual',
                     mutation_rate=0.01),
        IslandConfig(island_id=3, agent_type='nerve_net', replication_mode='sexual',
                     mutation_rate=0.01),
        IslandConfig(island_id=4, agent_type='nerve_net', replication_mode='lamarckian',
                     mutation_rate=0.01),
        IslandConfig(island_id=5, agent_type='nerve_net', replication_mode='asexual',
                     mutation_rate=0.02),   # 높은 돌연변이 → 더 넓은 회로 탐색
        IslandConfig(island_id=6, agent_type='nerve_net', replication_mode='sexual',
                     mutation_rate=0.02),   # 재조합 + 넓은 탐색
        IslandConfig(island_id=7, agent_type='nerve_net', replication_mode='asexual',
                     mutation_rate=0.04, world_width=80, world_height=80,
                     initial_agents=30, max_agents=200, vent_count=1),  # 고압 스트레스
    ]
