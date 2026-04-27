"""
Evolutionary map — survival pattern recording.

Which structures survived in which environments?
This map is the hint carried into the next phase.
Not answers. Possibility space.

Schema:
  lineages: one row per dead agent
    - genome snapshot (compressed)
    - pressure_born / pressure_died
    - lifespan, generation, c_level
    - replication_mode, island_id

  patterns: aggregated view
    - For each pressure bucket: which genome cluster dominated?
    - Evolvability ranking across lineages

God reads this map before designing Phase 2 conditions.
"""

import sqlite3
import json
import numpy as np
from typing import List, Dict, Any, Optional


class EvolutionaryMap:
    def __init__(self, db_path: str = 'logs/ltbl.db'):
        self.conn = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS lineages (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                genome_id        TEXT,
                parent_ids       TEXT,
                generation       INTEGER,
                replication_mode TEXT,
                island_id        INTEGER,
                pressure_born    REAL,
                pressure_died    REAL,
                lifespan         INTEGER,
                c_level          INTEGER,
                genome_snapshot  TEXT       -- JSON-encoded float list
            )
        ''')
        self.conn.commit()

    # ── Recording ────────────────────────────────────────────────────────────

    def record_lineage(self, entries: List[Dict[str, Any]]):
        """
        entries: list of dicts with keys matching lineages schema.
        Each dict represents one dead agent's complete record.
        """
        rows = []
        for e in entries:
            rows.append((
                e.get('genome_id', ''),
                json.dumps(e.get('parent_ids', [])),
                e.get('generation', 0),
                e.get('replication_mode', ''),
                e.get('island_id', -1),
                e.get('pressure_born', 0.0),
                e.get('pressure_died', 0.0),
                e.get('lifespan', 0),
                e.get('c_level', 0),
                json.dumps(e.get('genome_snapshot', [])),
            ))
        self.conn.executemany(
            'INSERT INTO lineages VALUES (NULL,?,?,?,?,?,?,?,?,?,?)', rows
        )
        self.conn.commit()

    # ── Analysis ─────────────────────────────────────────────────────────────

    def survival_by_pressure(self, n_buckets: int = 10) -> List[Dict]:
        """Average lifespan and C level per pressure bucket."""
        rows = self.conn.execute('''
            SELECT pressure_died, lifespan, c_level, replication_mode
            FROM lineages WHERE lifespan > 0
        ''').fetchall()
        if not rows:
            return []

        pressures = np.array([r[0] for r in rows])
        lifespans = np.array([r[1] for r in rows])
        c_levels  = np.array([r[2] for r in rows])

        edges = np.linspace(pressures.min(), pressures.max(), n_buckets + 1)
        result = []
        for i in range(n_buckets):
            mask = (pressures >= edges[i]) & (pressures < edges[i + 1])
            if not mask.any():
                continue
            result.append({
                'pressure_range': (float(edges[i]), float(edges[i + 1])),
                'count':          int(mask.sum()),
                'avg_lifespan':   float(lifespans[mask].mean()),
                'avg_c_level':    float(c_levels[mask].mean()),
            })
        return result

    def evolvability_ranking(self, top_n: int = 20) -> List[Dict]:
        """Top lineages ranked by evolvability proxy."""
        rows = self.conn.execute('''
            SELECT genome_id, generation, replication_mode,
                   pressure_born, pressure_died, lifespan, c_level
            FROM lineages
            ORDER BY (1 + c_level) * lifespan * MAX(pressure_died - pressure_born, 0.001) DESC
            LIMIT ?
        ''', (top_n,)).fetchall()
        return [
            {
                'genome_id': r[0], 'generation': r[1], 'mode': r[2],
                'pressure_range': round(r[4] - r[3], 3),
                'lifespan': r[5], 'c_level': r[6],
            }
            for r in rows
        ]

    def phase_transition_hints(self) -> str:
        """
        Summarise what the evolutionary map suggests for next-phase design.
        These are observations, not prescriptions.
        """
        lines = ['=== Evolutionary Map — Phase Transition Hints ===']

        ranking = self.evolvability_ranking(10)
        if ranking:
            top_modes = {}
            for r in ranking:
                top_modes[r['mode']] = top_modes.get(r['mode'], 0) + 1
            lines.append(f'Top evolvable replication modes: {top_modes}')
            top = ranking[0]
            lines.append(
                f'Most evolvable lineage: {top["genome_id"]} '
                f'gen={top["generation"]} mode={top["mode"]} '
                f'pressure_range={top["pressure_range"]:.3f} C={top["c_level"]}'
            )

        surv = self.survival_by_pressure(5)
        if surv:
            best = max(surv, key=lambda x: x['avg_lifespan'])
            lines.append(
                f'Best survival pressure zone: {best["pressure_range"][0]:.2f}–'
                f'{best["pressure_range"][1]:.2f}  avg_lifespan={best["avg_lifespan"]:.0f}'
            )

        total = self.conn.execute('SELECT COUNT(*) FROM lineages').fetchone()[0]
        lines.append(f'Total lineages recorded: {total}')
        lines.append('→ Carry: diverse genome pool + pressure-survival curve')
        lines.append('→ Do NOT carry: the fittest individual at final pressure')
        return '\n'.join(lines)

    def close(self):
        self.conn.close()
