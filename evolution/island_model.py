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
    agent_type: str = 'membrane'        # 'membrane' (1-A) | 'chemotaxis' (1-B)


def default_experiment() -> List[IslandConfig]:
    """
    8-island setup: 4 membrane (Phase 1-A) + 4 chemotaxis (Phase 1-B).
    Migration is enabled between all islands — chemotaxis genes are present
    but silent in membrane agents (backward-compatible genome).
    """
    return [
        # ── Phase 1-A: Membrane (homeostasis + energy-trend) ─────────────────
        IslandConfig(island_id=0, agent_type='membrane',   replication_mode='asexual',
                     mutation_rate=0.01),
        IslandConfig(island_id=1, agent_type='membrane',   replication_mode='sexual',
                     mutation_rate=0.01),
        IslandConfig(island_id=2, agent_type='membrane',   replication_mode='lamarckian',
                     mutation_rate=0.01),
        IslandConfig(island_id=3, agent_type='membrane',   replication_mode='asexual',
                     mutation_rate=0.05, world_width=80, world_height=80,
                     initial_agents=30, max_agents=200, vent_count=1),
        # ── Phase 1-B: Chemotaxis (Run/Tumble + gradient temporal awareness) ─
        IslandConfig(island_id=4, agent_type='chemotaxis', replication_mode='asexual',
                     mutation_rate=0.01),
        IslandConfig(island_id=5, agent_type='chemotaxis', replication_mode='sexual',
                     mutation_rate=0.01),
        IslandConfig(island_id=6, agent_type='chemotaxis', replication_mode='lamarckian',
                     mutation_rate=0.01),
        IslandConfig(island_id=7, agent_type='chemotaxis', replication_mode='asexual',
                     mutation_rate=0.05, world_width=80, world_height=80,
                     initial_agents=30, max_agents=200, vent_count=1),
    ]
