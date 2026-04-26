from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LineageRecord:
    agent_id: str
    genome_id: str
    parent_genome_ids: List[str]
    generation: int
    birth_tick: int
    replication_mode: str
    death_tick: Optional[int] = None
    consciousness_level: int = 0
    lifespan_energy_sum: float = 0.0


class LineageTracker:
    def __init__(self):
        self.records: Dict[str, LineageRecord] = {}

    def register(self, agent, tick: int):
        self.records[agent.id] = LineageRecord(
            agent_id=agent.id,
            genome_id=agent.genome.id,
            parent_genome_ids=list(agent.genome.parent_ids),
            generation=agent.genome.generation,
            birth_tick=tick,
            replication_mode=agent.mode,
        )

    def record_death(self, agent, tick: int, c_level: int):
        rec = self.records.get(agent.id)
        if rec:
            rec.death_tick = tick
            rec.consciousness_level = c_level
            rec.lifespan_energy_sum = sum(agent.state.energy_history)

    def generation_stats(self) -> Dict[int, Dict]:
        groups: Dict[int, List[LineageRecord]] = {}
        for r in self.records.values():
            groups.setdefault(r.generation, []).append(r)

        stats = {}
        for gen, recs in groups.items():
            lifespans = [r.death_tick - r.birth_tick for r in recs if r.death_tick]
            stats[gen] = {
                'count': len(recs),
                'avg_lifespan': sum(lifespans) / len(lifespans) if lifespans else None,
                'avg_c_level': sum(r.consciousness_level for r in recs) / len(recs),
                'survived': sum(1 for r in recs if r.death_tick is None),
            }
        return stats

    def max_generation(self) -> int:
        if not self.records:
            return 0
        return max(r.generation for r in self.records.values())
