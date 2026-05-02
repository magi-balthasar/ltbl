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
from god.evolutionary_map import EvolutionaryMap
from genetics.genome_archive import GenomeArchive
from genetics.genome import Genome


def parse_args():
    p = argparse.ArgumentParser(description='LTBL Phase 1 simulation')
    p.add_argument('--steps',      type=int, default=200,   help='Number of macro steps')
    p.add_argument('--report',     type=int, default=10,    help='Print report every N steps')
    p.add_argument('--intervene',  type=int, default=50,    help='God intervention every N steps')
    p.add_argument('--db',         type=str, default='logs/ltbl.db')
    p.add_argument('--single',     action='store_true',     help='Single island (no Ray) for debugging')
    return p.parse_args()


def run_single_island(args):
    """Debug mode: one nematode island, no Ray (Phase 3)."""
    from world.primordial_sea import PrimordialSea, SeaConfig
    from agents.nematode import NematodeAgent
    from genetics.genome import Genome
    from genetics.replication import ReplicationEngine
    from consciousness.level_monitor import ConsciousnessMonitor

    cfg = IslandConfig(island_id=0, agent_type='nematode',
                       replication_mode='sexual', mutation_rate=0.02,
                       world_width=100, world_height=100, initial_agents=30,
                       max_agents=200, ticks_per_step=20)

    sea = PrimordialSea(SeaConfig(width=cfg.world_width, height=cfg.world_height))
    replicator = ReplicationEngine(base_mutation_rate=cfg.mutation_rate)
    monitor = ConsciousnessMonitor()
    import numpy as np
    agents = [NematodeAgent(
        float(cfg.world_width * (i % 10) / 10 + 5),
        float(cfg.world_height * (i // 10) / 10 + 5),
        Genome(), mode=cfg.replication_mode
    ) for i in range(cfg.initial_agents)]

    from world.pressure_schedule import PressureSchedule
    pressure = PressureSchedule()

    print("=== LTBL [single-island debug] ===\n")
    for step in range(args.steps):
        pressure.tick()
        sea.step(pressure.level)
        next_gen = []
        for a in agents:
            a.step(sea, (cfg.world_width, cfg.world_height))
            if not a.alive:
                continue
            if a.can_replicate() and len(next_gen) < cfg.max_agents:
                a.pay_replication_cost()
                if cfg.replication_mode == 'sexual' and len(agents) > 1:
                    partner = agents[np.random.randint(len(agents))]
                    child_genome = replicator.sexual(a.genome, partner.genome)
                else:
                    child_genome = replicator.asexual(a.genome)
                child = NematodeAgent(a.x, a.y, child_genome, mode=cfg.replication_mode)
                next_gen.append(child)
            next_gen.append(a)
        agents = next_gen

        if step % args.report == 0:
            metrics = monitor.measure(agents)
            c = monitor.consciousness_level(metrics)
            avg_e = sum(a.state.energy for a in agents) / max(len(agents), 1)
            avg_g = sum(a.genome.generation for a in agents) / max(len(agents), 1)
            avg_mut = sum(a.genome.mutation_rate_gene for a in agents) / max(len(agents), 1)
            c_cpg = metrics.get('C_CPG', 0.0)
            c4    = metrics.get('C4', 0.0)
            print(f"step={step:4d}  pop={len(agents):4d}  gen={avg_g:5.1f}  "
                  f"E={avg_e:5.2f}  P={pressure.level:.3f}  C={c}  "
                  f"C_CPG={c_cpg:+.3f}  C4={c4:.3f}  mut_gene={avg_mut:.4f}")


def run_parallel(args):
    from world.pressure_schedule import PressureSchedule

    configs = default_experiment()
    engine    = ParallelExperimentEngine(configs, working_dir=os.path.dirname(os.path.abspath(__file__)))
    observer  = GodObserver(db_path=args.db)
    evo_map   = EvolutionaryMap(db_path=args.db)
    archive   = GenomeArchive(capacity=300, min_distance=0.02)
    pressure  = PressureSchedule()
    intervener = GodIntervener(engine.islands, pressure)

    print("=== LTBL: Let There Be Light ===")
    print(f"8 islands | {args.steps} steps | pressure-driven evolution\n")

    try:
        for step in range(args.steps):
            pressure.tick()
            results = engine.run_step(pressure.level)
            observer.observe(step, results)

            # ── 2차: Noah's Ark + 3차: Evolutionary Map ─────────────────────
            all_died = []
            for r in results:
                for rec in r.get('died_records', []):
                    all_died.append(rec)
                    # Noah's Ark: consider every dead genome for structural diversity
                    g = Genome.from_vector(
                        __import__('numpy').array(rec['genome_snapshot']),
                        parent_ids=rec['parent_ids'],
                        generation=rec['generation'],
                    )
                    archive.consider(g, rec)
            if all_died:
                evo_map.record_lineage(all_died)

            # ── Report ───────────────────────────────────────────────────────
            if step % args.report == 0:
                print(observer.report(step, results, pressure.level))
                print(f"  [Archive] {archive.stats}")
                for ev in observer.detect_emergence(results):
                    print(f"  *** [{ev['type']}] island={ev['island_id']} ***")
                for sig in observer.phase_transition_signals(results):
                    print(f"  >>> {sig}")
                print()

            # ── Intervention ─────────────────────────────────────────────────
            if step > 0 and step % args.intervene == 0:
                intervener.calibrate_pressure(results)
                intervener.rescue_dying_islands(results)
                intervener.cross_pollinate_top(results)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # Print evolutionary map summary before exit
        print()
        print(evo_map.phase_transition_hints())
        observer.close()
        evo_map.close()
        engine.shutdown()
        print("\nSimulation complete. Data saved to", args.db)


def main():
    args = parse_args()
    if args.single:
        run_single_island(args)
    else:
        run_parallel(args)


if __name__ == '__main__':
    main()
