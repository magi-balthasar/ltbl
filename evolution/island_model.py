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


def default_experiment() -> List[IslandConfig]:
    """8-island setup: 3 replication modes × mutation rates + high-stress control."""
    return [
        # Asexual — low / medium / high mutation
        IslandConfig(island_id=0, replication_mode='asexual',    mutation_rate=0.005),
        IslandConfig(island_id=1, replication_mode='asexual',    mutation_rate=0.02),
        IslandConfig(island_id=2, replication_mode='asexual',    mutation_rate=0.08),
        # Sexual
        IslandConfig(island_id=3, replication_mode='sexual',     mutation_rate=0.01),
        IslandConfig(island_id=4, replication_mode='sexual',     mutation_rate=0.04),
        # Lamarckian
        IslandConfig(island_id=5, replication_mode='lamarckian', mutation_rate=0.01),
        IslandConfig(island_id=6, replication_mode='lamarckian', mutation_rate=0.04),
        # High-stress: small world, few vents
        IslandConfig(island_id=7, replication_mode='asexual',    mutation_rate=0.03,
                     world_width=80, world_height=80, initial_agents=30,
                     max_agents=200, vent_count=1),
    ]
