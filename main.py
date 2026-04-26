#!/usr/bin/env python3
"""
LTBL — Let There Be Light
Phase 1-A: Primordial Sea + Cell Membrane

"조건을 선언한다. 결과는 창발한다."
"""

import sys
import os
import argparse

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evolution.island_model import default_experiment, IslandConfig
from evolution.parallel_engine import ParallelExperimentEngine
from god.observer import GodObserver
from god.intervener import GodIntervener


def parse_args():
    p = argparse.ArgumentParser(description='LTBL Phase 1 simulation')
    p.add_argument('--steps',      type=int, default=200,   help='Number of macro steps')
    p.add_argument('--report',     type=int, default=10,    help='Print report every N steps')
    p.add_argument('--intervene',  type=int, default=50,    help='God intervention every N steps')
    p.add_argument('--db',         type=str, default='logs/ltbl.db')
    p.add_argument('--single',     action='store_true',     help='Single island (no Ray) for debugging')
    return p.parse_args()


def run_single_island(args):
    """Debug mode: one island, no Ray."""
    from world.primordial_sea import PrimordialSea, SeaConfig
    from agents.membrane import CellAgent
    from genetics.genome import Genome
    from genetics.replication import ReplicationEngine
    from consciousness.level_monitor import ConsciousnessMonitor

    cfg = IslandConfig(island_id=0, replication_mode='asexual', mutation_rate=0.02,
                       world_width=100, world_height=100, initial_agents=30,
                       max_agents=200, ticks_per_step=20)

    sea = PrimordialSea(SeaConfig(width=cfg.world_width, height=cfg.world_height))
    replicator = ReplicationEngine(base_mutation_rate=cfg.mutation_rate)
    monitor = ConsciousnessMonitor()
    agents = [CellAgent(
        float(cfg.world_width * (i % 10) / 10 + 5),
        float(cfg.world_height * (i // 10) / 10 + 5),
        Genome(), mode='asexual'
    ) for i in range(cfg.initial_agents)]

    print("=== LTBL [single-island debug] ===\n")
    for step in range(args.steps):
        sea.step()
        next_gen = []
        for a in agents:
            a.step(sea, (cfg.world_width, cfg.world_height))
            if not a.alive:
                continue
            if a.can_replicate() and len(next_gen) < cfg.max_agents:
                a.pay_replication_cost()
                child = CellAgent(a.x, a.y, replicator.asexual(a.genome))
                next_gen.append(child)
            next_gen.append(a)
        agents = next_gen

        if step % args.report == 0:
            metrics = monitor.measure(agents)
            c = monitor.consciousness_level(metrics)
            avg_e = sum(a.state.energy for a in agents) / max(len(agents), 1)
            avg_g = sum(a.genome.generation for a in agents) / max(len(agents), 1)
            print(f"step={step:4d}  pop={len(agents):4d}  gen={avg_g:5.1f}  "
                  f"E={avg_e:5.2f}  C={c}  C1={metrics['C1']:.2f}  C2={metrics['C2']:.2f}")


def run_parallel(args):
    configs = default_experiment()
    engine = ParallelExperimentEngine(configs, working_dir=os.path.dirname(os.path.abspath(__file__)))
    observer = GodObserver(db_path=args.db)
    intervener = GodIntervener(engine.islands)

    print("=== LTBL: Let There Be Light ===")
    print(f"Phase 1-A | {len(configs)} islands | {args.steps} steps\n")

    try:
        for step in range(args.steps):
            results = engine.run_step()
            observer.observe(step, results)

            if step % args.report == 0:
                print(observer.report(step, results))
                events = observer.detect_emergence(results)
                for ev in events:
                    print(f"  *** EMERGENCE [{ev['type']}] island={ev['island_id']} ***")
                print()

            if step > 0 and step % args.intervene == 0:
                intervener.redistribute_resources(results)
                intervener.cross_pollinate(results)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        observer.close()
        engine.shutdown()
        print("Simulation complete. Data saved to", args.db)


def main():
    args = parse_args()
    if args.single:
        run_single_island(args)
    else:
        run_parallel(args)


if __name__ == '__main__':
    main()
