from typing import List, Dict, Any


class GodIntervener:
    """Divine interventions: resource redistribution, cross-pollination."""

    def __init__(self, islands: list):
        self.islands = islands  # List[IslandActor] (Ray handles)

    def redistribute_resources(self, results: List[Dict[str, Any]]):
        """Islands with dying populations get a migration boost from thriving ones."""
        import ray
        dying = [r for r in results if r['population'] < 10]
        thriving = [r for r in results if r['population'] > 100]
        if not dying or not thriving:
            return
        for d in dying:
            src_id = thriving[0]['island_id']
            dst_id = d['island_id']
            emigrants = ray.get(self.islands[src_id].get_emigrants.remote(10))
            if emigrants:
                self.islands[dst_id].receive_immigrants.remote(emigrants)

    def cross_pollinate(self, results: List[Dict[str, Any]]):
        """Force migration between the two highest-consciousness islands."""
        import ray
        sorted_r = sorted(results, key=lambda r: r['consciousness_level'], reverse=True)
        if len(sorted_r) < 2:
            return
        a_id = sorted_r[0]['island_id']
        b_id = sorted_r[1]['island_id']
        emigrants_a = ray.get(self.islands[a_id].get_emigrants.remote(5))
        emigrants_b = ray.get(self.islands[b_id].get_emigrants.remote(5))
        if emigrants_a:
            self.islands[b_id].receive_immigrants.remote(emigrants_a)
        if emigrants_b:
            self.islands[a_id].receive_immigrants.remote(emigrants_b)
