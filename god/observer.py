import sqlite3
from datetime import datetime
from typing import List, Dict, Any

from consciousness.emergence_signals import EmergenceDetector


class GodObserver:
    """Scans all islands, records observations, detects emergence, reports."""

    def __init__(self, db_path: str = 'logs/ltbl.db'):
        import os
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.detectors: Dict[int, EmergenceDetector] = {}
        self._init_db()

    def _init_db(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS observations (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT,
                step         INTEGER,
                island_id    INTEGER,
                replication_mode TEXT,
                mutation_rate    REAL,
                population   INTEGER,
                c_level      INTEGER,
                c0 REAL, c1 REAL, c2 REAL,
                avg_generation REAL,
                max_generation INTEGER,
                avg_energy     REAL
            )
        ''')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS emergence_events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                step      INTEGER,
                island_id INTEGER,
                event_type TEXT,
                detail     TEXT
            )
        ''')
        self.conn.commit()

    # ── Observation ──────────────────────────────────────────────────────────

    def observe(self, step: int, results: List[Dict[str, Any]]):
        ts = datetime.now().isoformat()
        rows = []
        for r in results:
            m = r['consciousness_metrics']
            rows.append((
                ts, step, r['island_id'], r['replication_mode'], r['mutation_rate'],
                r['population'], r['consciousness_level'],
                m['C0'], m['C1'], m['C2'],
                r['avg_generation'], r['max_generation'], r['avg_energy'],
            ))
            # Update detector
            iid = r['island_id']
            if iid not in self.detectors:
                self.detectors[iid] = EmergenceDetector()
            self.detectors[iid].record(step, iid, m, r['consciousness_level'], r['population'])
            for event in self.detectors[iid].detect():
                self.conn.execute(
                    'INSERT INTO emergence_events VALUES (NULL,?,?,?,?,?)',
                    (ts, step, iid, event['type'], str(event)),
                )

        self.conn.executemany(
            'INSERT INTO observations VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?)', rows
        )
        self.conn.commit()

    # ── Reporting ────────────────────────────────────────────────────────────

    def report(self, step: int, results: List[Dict[str, Any]]) -> str:
        lines = [f"┌── Step {step:>5d} ──────────────────────────────────────────────┐"]
        for r in results:
            m = r['consciousness_metrics']
            lines.append(
                f"│ [{r['island_id']}] {r['replication_mode']:12s} μ={r['mutation_rate']:.3f} "
                f"pop={r['population']:4d} gen={r['avg_generation']:5.1f}/{r['max_generation']:3d} "
                f"E={r['avg_energy']:5.2f}  "
                f"C={r['consciousness_level']} (C0={m['C0']:.2f} C1={m['C1']:.2f} C2={m['C2']:.2f})"
            )
        lines.append("└───────────────────────────────────────────────────────────────┘")
        return "\n".join(lines)

    def detect_emergence(self, results: List[Dict]) -> List[Dict]:
        events = []
        for r in results:
            det = self.detectors.get(r['island_id'])
            if det:
                events.extend(det.detect())
        return events

    def close(self):
        self.conn.close()
