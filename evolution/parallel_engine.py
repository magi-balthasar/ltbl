import sys, os
import numpy as np
import ray
from typing import List, Dict, Any, Tuple

from evolution.island_model import IslandConfig


@ray.remote
class IslandActor:
    """Ray remote actor: one isolated population in its own process."""

    def __init__(self, config: IslandConfig):
        # Ensure project root is importable inside the Ray worker
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)

        from world.primordial_sea import PrimordialSea, SeaConfig
        from agents.membrane import CellAgent
        from genetics.genome import Genome
        from genetics.replication import ReplicationEngine
        from genetics.lineage_tracker import LineageTracker
        from consciousness.level_monitor import ConsciousnessMonitor

        self.cfg = config
        self.sea = PrimordialSea(SeaConfig(
            width=config.world_width, height=config.world_height,
            vent_count=config.vent_count,
        ))
        self.replicator = ReplicationEngine(
            base_mutation_rate=config.mutation_rate,
            mutation_strength=config.mutation_strength,
        )
        self.tracker = LineageTracker()
        self.monitor = ConsciousnessMonitor()
        self.tick = 0
        self.agents: List[CellAgent] = []

        for _ in range(config.initial_agents):
            g = Genome()
            x = np.random.uniform(0, config.world_width)
            y = np.random.uniform(0, config.world_height)
            a = CellAgent(x, y, g, mode=config.replication_mode)
            self.agents.append(a)
            self.tracker.register(a, self.tick)

    # ── Simulation step ──────────────────────────────────────────────────────

    def step(self) -> Dict[str, Any]:
        from agents.membrane import CellAgent
        from genetics.genome import Genome

        W, H = self.cfg.world_width, self.cfg.world_height

        for _ in range(self.cfg.ticks_per_step):
            self.sea.step()
            next_gen: List[CellAgent] = []

            for agent in self.agents:
                agent.step(self.sea, (W, H))

                if not agent.alive:
                    metrics = self.monitor.measure([agent])
                    c_level = self.monitor.consciousness_level(metrics)
                    self.tracker.record_death(agent, self.tick, c_level)
                    continue

                if (agent.can_replicate()
                        and len(next_gen) + len(self.agents) < self.cfg.max_agents):
                    agent.pay_replication_cost()
                    child_genome = self._replicate(agent)
                    child = CellAgent(
                        (agent.x + np.random.uniform(-1, 1)) % W,
                        (agent.y + np.random.uniform(-1, 1)) % H,
                        child_genome,
                        mode=self.cfg.replication_mode,
                    )
                    next_gen.append(child)
                    self.tracker.register(child, self.tick)

                next_gen.append(agent)

            self.agents = next_gen
            self.tick += 1

        metrics = self.monitor.measure(self.agents)
        c_level = self.monitor.consciousness_level(metrics)

        return {
            'island_id':            self.cfg.island_id,
            'replication_mode':     self.cfg.replication_mode,
            'mutation_rate':        self.cfg.mutation_rate,
            'tick':                 self.tick,
            'population':           len(self.agents),
            'consciousness_metrics': metrics,
            'consciousness_level':  c_level,
            'avg_generation':       float(np.mean([a.genome.generation for a in self.agents])) if self.agents else 0.0,
            'max_generation':       self.tracker.max_generation(),
            'avg_energy':           float(np.mean([a.state.energy for a in self.agents])) if self.agents else 0.0,
        }

    def _replicate(self, agent):
        mode = self.cfg.replication_mode
        if mode == 'asexual':
            return self.replicator.asexual(agent.genome)
        if mode == 'sexual' and len(self.agents) > 1:
            partner = self.agents[np.random.randint(len(self.agents))]
            return self.replicator.sexual(agent.genome, partner.genome)
        if mode == 'lamarckian':
            return self.replicator.lamarckian(agent.genome, agent.avg_experience)
        return self.replicator.asexual(agent.genome)

    # ── Migration ────────────────────────────────────────────────────────────

    def get_emigrants(self, n: int) -> List[Tuple]:
        if len(self.agents) <= n:
            return []
        chosen = np.random.choice(self.agents, min(n, len(self.agents)), replace=False)
        return [(a.genome.to_vector().tolist(), a.genome.generation, list(a.genome.parent_ids))
                for a in chosen]

    def receive_immigrants(self, data: List[Tuple]):
        from agents.membrane import CellAgent
        from genetics.genome import Genome
        for vec_list, gen, parent_ids in data:
            vec = np.array(vec_list)
            genome = Genome.from_vector(vec, parent_ids=parent_ids, generation=gen)
            x = np.random.uniform(0, self.cfg.world_width)
            y = np.random.uniform(0, self.cfg.world_height)
            a = CellAgent(x, y, genome, mode=self.cfg.replication_mode)
            self.agents.append(a)
            self.tracker.register(a, self.tick)


# ── Orchestrator ─────────────────────────────────────────────────────────────

class ParallelExperimentEngine:
    """Manages N islands running in parallel via Ray."""

    def __init__(self, configs: List[IslandConfig], working_dir: str = '.'):
        if not ray.is_initialized():
            ray.init(
                runtime_env={"working_dir": working_dir},
                ignore_reinit_error=True,
                log_to_driver=False,
            )
        self.islands = [IslandActor.remote(cfg) for cfg in configs]
        self.configs = configs
        self.step_count = 0

    def run_step(self) -> List[Dict[str, Any]]:
        futures = [island.step.remote() for island in self.islands]
        results = ray.get(futures)

        # Periodic migration between islands
        if self.step_count > 0 and self.step_count % self.configs[0].migration_interval == 0:
            self._migrate(n=3)

        self.step_count += 1
        return results

    def _migrate(self, n: int):
        emigrant_futures = [isl.get_emigrants.remote(n) for isl in self.islands]
        all_emigrants = ray.get(emigrant_futures)

        for i, emigrants in enumerate(all_emigrants):
            if not emigrants:
                continue
            targets = [j for j in range(len(self.islands)) if j != i]
            target = int(np.random.choice(targets))
            self.islands[target].receive_immigrants.remote(emigrants)

    def shutdown(self):
        ray.shutdown()
