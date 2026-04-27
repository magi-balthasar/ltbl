import numpy as np
from typing import List, Dict


class ConsciousnessMonitor:
    """Measures C0-C6 consciousness metrics over a population of agents."""

    def measure(self, agents: list) -> Dict[str, float]:
        if not agents:
            return {f'C{i}': 0.0 for i in range(7)}
        return {
            'C0':  self._c0_internal_complexity(agents),
            'C1':  self._c1_internal_driven_ratio(agents),
            'C2':  self._c2_temporal_depth(agents),
            'C2b': self._c2b_gradient_temporal(agents),  # Phase 1-B: gradient temporal
            'C3':  0.0,  # prediction accuracy — Phase 1-C
            'C4':  0.0,  # self-model — Phase 1-C+
            'C5':  0.0,  # other-model — Phase 1-D+
            'C6':  0.0,  # recursive self-reference — Phase 7
        }

    def consciousness_level(self, metrics: Dict[str, float]) -> int:
        c0, c1, c2 = metrics['C0'], metrics['C1'], metrics['C2']
        c2b = metrics.get('C2b', 0.0)
        # C2: energy-trend steers action (1-A temporal loop)
        # C2b: gradient-trend steers tumble decision (1-B temporal loop)
        if (c1 > 0.5 and c2 > 0.2) or c2b > 0.3:
            return 2
        if c1 > 0.2 or c0 > 2.0 or c2b > 0.1:
            return 1
        return 0

    # ── C0: internal state complexity ────────────────────────────────────────

    def _c0_internal_complexity(self, agents: list) -> float:
        scores = []
        for a in agents:
            s = a.state
            active = sum([
                s.energy > 0.01,
                s.internal_nutrient > 0.01,
                s.internal_toxin > 0.01,
                len(s.energy_history) > 2,
                abs(s.energy_gradient()) > 0.001,
            ])
            genome_dim = len(a.genome.to_vector())
            scores.append(active * genome_dim / 25.0)
        return float(np.mean(scores))

    # ── C1: fraction of trend-driven behavior ───────────────────────────────
    # "internally driven" = agent responded to energy TREND (temporal signal),
    # not to instantaneous low energy. Separates temporal awareness from mere crisis.

    def _c1_internal_driven_ratio(self, agents: list) -> float:
        ratios = []
        for a in agents:
            if not a.behavior_log:
                continue
            driven = sum(1 for entry in a.behavior_log if entry[0])
            ratios.append(driven / len(a.behavior_log))
        return float(np.mean(ratios)) if ratios else 0.0

    # ── C2: correlation between energy trend and action intensity ────────────
    # High C2 = agent consistently moves harder when energy is declining.
    # Measures whether temporal comparison (trend) actually steers behavior.

    def _c2_temporal_depth(self, agents: list) -> float:
        corrs = []
        for a in agents:
            log = a.behavior_log
            if len(log) < 8:
                continue
            trends = np.array([entry[2] for entry in log])
            mags   = np.array([entry[3] for entry in log])
            if np.std(trends) < 1e-8 or np.std(mags) < 1e-8:
                continue
            corr = float(np.corrcoef(trends, mags)[0, 1])
            corrs.append(max(0.0, -corr))
        return float(np.mean(corrs)) if corrs else 0.0

    # ── C2b: gradient temporal depth (Phase 1-B chemotaxis) ─────────────────
    # High C2b = worsening chemical gradient reliably triggers tumble.
    # Detects entries with len >= 6 (chemotaxis 6-tuple vs membrane 4-tuple).

    def _c2b_gradient_temporal(self, agents: list) -> float:
        corrs = []
        for a in agents:
            log = [e for e in a.behavior_log if len(e) >= 6]
            if len(log) < 8:
                continue
            grad_trends = np.array([float(e[4]) for e in log])
            tumbled     = np.array([float(e[5]) for e in log])
            if np.std(grad_trends) < 1e-8 or np.std(tumbled) < 1e-8:
                continue
            corr = float(np.corrcoef(grad_trends, tumbled)[0, 1])
            # Negative gradient → tumble: negative corr → good temporal response.
            corrs.append(max(0.0, -corr))
        return float(np.mean(corrs)) if corrs else 0.0
