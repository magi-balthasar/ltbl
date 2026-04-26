"""
God intervener: calibrates environmental pressure.

Does NOT:
  - Set phase transition conditions
  - Declare when a phase has been reached
  - Reward or punish agents directly

Does:
  - Increase pressure when agents are too comfortable (stasis)
  - Ease pressure when extinction is imminent (too fast)
  - Hold pressure when something new is emerging (let it play out)
  - Cross-pollinate high-consciousness islands (accelerate diversity)
"""

from typing import List, Dict, Any
from world.pressure_schedule import PressureSchedule


class GodIntervener:
    def __init__(self, islands: list, pressure: PressureSchedule):
        self.islands = islands
        self.pressure = pressure

    # ── Pressure calibration ─────────────────────────────────────────────────

    def calibrate_pressure(self, results: List[Dict[str, Any]]):
        """
        Observe survival and clustering, then adjust pressure rate.
        Called every N steps — not every step.
        """
        if not results:
            return

        survivals = [r.get('survival_rate', 1.0) for r in results]
        clusters  = [r.get('cluster_signal', 0.0) for r in results]
        avg_surv  = sum(survivals) / len(survivals)
        avg_clust = sum(clusters)  / len(clusters)

        self.pressure.record(avg_surv, avg_clust)
        self.pressure.calibrate()

    # ── Population rescue ─────────────────────────────────────────────────────

    def rescue_dying_islands(self, results: List[Dict[str, Any]]):
        """Send migrants from thriving islands to those near extinction."""
        import ray
        dying    = [r for r in results if r['population'] < 10]
        thriving = sorted(results, key=lambda r: r['population'], reverse=True)
        if not dying or not thriving:
            return
        src_id = thriving[0]['island_id']
        for d in dying:
            emigrants = ray.get(self.islands[src_id].get_emigrants.remote(10))
            if emigrants:
                self.islands[d['island_id']].receive_immigrants.remote(emigrants)

    # ── Cross-pollination ─────────────────────────────────────────────────────

    def cross_pollinate_top(self, results: List[Dict[str, Any]]):
        """Mix genomes between the two highest-C islands to seed diversity."""
        import ray
        sorted_r = sorted(results, key=lambda r: r['consciousness_level'], reverse=True)
        if len(sorted_r) < 2:
            return
        a_id = sorted_r[0]['island_id']
        b_id = sorted_r[1]['island_id']
        em_a = ray.get(self.islands[a_id].get_emigrants.remote(5))
        em_b = ray.get(self.islands[b_id].get_emigrants.remote(5))
        if em_a:
            self.islands[b_id].receive_immigrants.remote(em_a)
        if em_b:
            self.islands[a_id].receive_immigrants.remote(em_b)
