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
    agent_type: str = 'membrane'  # 'membrane'|'chemotaxis'|'phototaxis'


def default_experiment() -> List[IslandConfig]:
    """
    8-island setup (Phase 1-C):
      2 membrane (1-A) + 2 chemotaxis (1-B) + 4 phototaxis (1-C)
    Migration between all islands — higher-phase genes are silent in
    lower-phase agents (backward-compatible genome).
    """
    return [
        # ── Phase 1-A: Membrane — control baseline ────────────────────────────
        IslandConfig(island_id=0, agent_type='membrane',   replication_mode='asexual',
                     mutation_rate=0.01),
        IslandConfig(island_id=1, agent_type='membrane',   replication_mode='lamarckian',
                     mutation_rate=0.01),
        # ── Phase 1-B: Chemotaxis — intermediate ─────────────────────────────
        IslandConfig(island_id=2, agent_type='chemotaxis', replication_mode='asexual',
                     mutation_rate=0.01),
        IslandConfig(island_id=3, agent_type='chemotaxis', replication_mode='sexual',
                     mutation_rate=0.01),
        # ── Phase 1-C: Phototaxis — main experiment ───────────────────────────
        IslandConfig(island_id=4, agent_type='phototaxis', replication_mode='asexual',
                     mutation_rate=0.01),
        IslandConfig(island_id=5, agent_type='phototaxis', replication_mode='sexual',
                     mutation_rate=0.01),
        IslandConfig(island_id=6, agent_type='phototaxis', replication_mode='lamarckian',
                     mutation_rate=0.01),
        IslandConfig(island_id=7, agent_type='phototaxis', replication_mode='asexual',
                     mutation_rate=0.04, world_width=80, world_height=80,
                     initial_agents=30, max_agents=200, vent_count=1),
    ]
