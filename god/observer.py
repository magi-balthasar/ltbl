import sqlite3
from datetime import datetime
from typing import List, Dict, Any

from consciousness.emergence_signals import EmergenceDetector


class GodObserver:
    """
    Scans all islands, records observations, detects emergence signals.

    God does not declare phase transitions.
    God observes, records, and reports what is happening.
    Transition is detected when patterns in the data justify the label.
    """

    def __init__(self, db_path: str = 'logs/ltbl.db'):
        import os
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.detectors: Dict[int, EmergenceDetector] = {}
        self._init_db()

    def _init_db(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS observations (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT,
                step             INTEGER,
                island_id        INTEGER,
                replication_mode TEXT,
                mutation_rate    REAL,
                population       INTEGER,
                c_level          INTEGER,
                c0 REAL, c1 REAL, c2 REAL, c2b REAL,
                avg_generation   REAL,
                max_generation   INTEGER,
                avg_energy       REAL,
                survival_rate    REAL,
                cluster_signal   REAL,
                pressure         REAL
            )
        ''')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS emergence_events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT,
                step       INTEGER,
                island_id  INTEGER,
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
                ts, step,
                r['island_id'], r['replication_mode'], r['mutation_rate'],
                r['population'], r['consciousness_level'],
                m['C0'], m['C1'], m['C2'], m.get('C2b', 0.0),
                r['avg_generation'], r['max_generation'], r['avg_energy'],
                r.get('survival_rate', 1.0),
                r.get('cluster_signal', 0.0),
                r.get('pressure', 0.0),
            ))
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
            'INSERT INTO observations VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', rows
        )
        self.conn.commit()

    # ── Reporting ────────────────────────────────────────────────────────────

    def report(self, step: int, results: List[Dict[str, Any]], pressure: float = 0.0) -> str:
        lines = [
            f"┌── Step {step:>5d}  pressure={pressure:.3f} ─────────────────────────────┐"
        ]
        for r in results:
            m = r['consciousness_metrics']
            cluster  = r.get('cluster_signal', 0.0)
            surv     = r.get('survival_rate', 1.0)
            atype    = r.get('agent_type', 'membrane')
            c2b_str  = f" C2b={m.get('C2b', 0.0):.2f}" if atype == 'chemotaxis' else ''
            lines.append(
                f"│ [{r['island_id']}] {atype[0].upper()}{r['replication_mode']:11s} μ={r['mutation_rate']:.3f} "
                f"pop={r['population']:4d} gen={r['avg_generation']:4.1f}/{r['max_generation']:3d} "
                f"E={r['avg_energy']:4.2f} surv={surv:.2f} clust={cluster:.2f}  "
                f"C={r['consciousness_level']} C1={m['C1']:.2f} C2={m['C2']:.2f}{c2b_str}"
            )
        lines.append("└───────────────────────────────────────────────────────────────────┘")
        return "\n".join(lines)

    def detect_emergence(self, results: List[Dict]) -> List[Dict]:
        events = []
        for r in results:
            det = self.detectors.get(r['island_id'])
            if det:
                events.extend(det.detect())
        return events

    def phase_transition_signals(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Observe patterns that may indicate a phase transition is occurring.
        Does not declare anything — returns human-readable observations.
        """
        signals = []
        populations = [r['population'] for r in results]
        survivals   = [r.get('survival_rate', 1.0) for r in results]
        clusters    = [r.get('cluster_signal', 0.0) for r in results]

        avg_surv    = sum(survivals) / len(survivals) if survivals else 1.0
        avg_cluster = sum(clusters)  / len(clusters)  if clusters  else 0.0
        extinct     = sum(1 for p in populations if p == 0)

        if avg_surv < 0.3:
            signals.append(f'[압력 과다] 평균 생존율 {avg_surv:.2f} — 환경 완화 권장')
        if avg_cluster > 0.25:
            signals.append(f'[클러스터링] 집합 신호 {avg_cluster:.2f} — Phase 2 전구체 관찰 중')
        if extinct > len(results) // 2:
            signals.append(f'[대멸종] {extinct}/{len(results)} 섬 소멸 — 압력 즉시 완화 필요')

        return signals

    def close(self):
        self.conn.close()
