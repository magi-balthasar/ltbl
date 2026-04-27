"""
Noah's Ark — structural diversity preservation.

God does not select for ability.
God selects for structural novelty.
A genome earns its place by being *different*, not by being *better*.

Admission rule:
  - New genome must be at least min_distance away from every existing member
  - Distance is L2 norm in normalised genome space

When archive is full:
  - New genome replaces the LEAST diverse existing member
    (the one most similar to its nearest neighbour — it is least irreplaceable)

Phase transition handoff:
  - archive.top_evolvable(n) returns the n most evolvable genomes
  - These seed the next phase — not as answers, but as possibility space
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from genetics.genome import Genome


@dataclass
class ArchiveEntry:
    vector: np.ndarray          # normalised genome vector
    raw_vector: np.ndarray      # original for reconstruction
    genome_id: str
    generation: int
    pressure_born: float
    pressure_died: float
    lifespan: int               # ticks alive
    c_level: int
    replication_mode: str
    evolvability: float = 0.0   # computed on insertion

    def compute_evolvability(self):
        """
        Evolvability proxy: adapted across how wide a pressure range,
        for how long, reaching what consciousness level.
        Wide range + long life + high C = high evolvability.
        """
        pressure_range = max(0.001, self.pressure_died - self.pressure_born)
        self.evolvability = (
            (1 + self.c_level) * self.lifespan * pressure_range
        )


class GenomeArchive:
    """Maintains a structurally diverse genome pool across all islands."""

    def __init__(self, capacity: int = 300, min_distance: float = 1.0):
        self.capacity = capacity
        self.min_distance = min_distance
        self.entries: List[ArchiveEntry] = []
        self._admitted = 0
        self._rejected = 0

    # ── Public interface ─────────────────────────────────────────────────────

    def consider(self, genome: Genome, metadata: Dict) -> bool:
        """
        Admit genome if it is structurally novel.
        Returns True if admitted.
        """
        vec = self._normalise(genome.to_vector())
        entry = ArchiveEntry(
            vector=vec,
            raw_vector=genome.to_vector(),
            genome_id=genome.id,
            generation=genome.generation,
            pressure_born=metadata.get('pressure_born', 0.0),
            pressure_died=metadata.get('pressure_died', 0.0),
            lifespan=metadata.get('lifespan', 0),
            c_level=metadata.get('c_level', 0),
            replication_mode=metadata.get('replication_mode', 'asexual'),
        )
        entry.compute_evolvability()

        if not self.entries:
            self.entries.append(entry)
            self._admitted += 1
            return True

        min_dist = self._min_distance_to_archive(vec)
        if min_dist < self.min_distance:
            self._rejected += 1
            return False

        if len(self.entries) >= self.capacity:
            self._replace_least_diverse(entry)
        else:
            self.entries.append(entry)

        self._admitted += 1
        return True

    def top_evolvable(self, n: int) -> List[np.ndarray]:
        """
        Return raw genome vectors of the n most evolvable entries.
        Used to seed Phase 2 — possibility space, not answers.
        """
        ranked = sorted(self.entries, key=lambda e: e.evolvability, reverse=True)
        return [e.raw_vector for e in ranked[:n]]

    def diversity_score(self) -> float:
        """Mean pairwise distance in the archive — higher = more diverse."""
        if len(self.entries) < 2:
            return 0.0
        vecs = np.array([e.vector for e in self.entries])
        total, count = 0.0, 0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                total += float(np.linalg.norm(vecs[i] - vecs[j]))
                count += 1
        return total / max(count, 1)

    @property
    def size(self) -> int:
        return len(self.entries)

    @property
    def stats(self) -> str:
        evols = [e.evolvability for e in self.entries]
        return (
            f"size={self.size}/{self.capacity}  "
            f"admitted={self._admitted}  rejected={self._rejected}  "
            f"diversity={self.diversity_score():.2f}  "
            f"avg_evolvability={np.mean(evols):.1f}" if evols else "empty"
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-8)

    def _min_distance_to_archive(self, vec: np.ndarray) -> float:
        return min(float(np.linalg.norm(vec - e.vector)) for e in self.entries)

    def _replace_least_diverse(self, new_entry: ArchiveEntry):
        """Replace the archive member most similar to its neighbours."""
        min_nn = []
        for i, e in enumerate(self.entries):
            nn_dist = min(
                (float(np.linalg.norm(e.vector - self.entries[j].vector))
                 for j in range(len(self.entries)) if j != i),
                default=float('inf')
            )
            min_nn.append(nn_dist)
        least_diverse_idx = int(np.argmin(min_nn))
        self.entries[least_diverse_idx] = new_entry
