import numpy as np
from .genome import Genome, GENOME_DIM
from typing import List


class ReplicationEngine:
    """Asexual, sexual, and Lamarckian replication strategies."""

    def __init__(self, base_mutation_rate: float = 0.01, mutation_strength: float = 0.1):
        self.base_mutation_rate = base_mutation_rate
        self.mutation_strength = mutation_strength

    def asexual(self, genome: Genome) -> Genome:
        vec = genome.to_vector()
        child_vec = self._mutate(vec)
        child = Genome.from_vector(child_vec, parent_ids=[genome.id],
                                   generation=genome.generation + 1)
        child.mutation_count = genome.mutation_count + int(np.sum(np.abs(child_vec - vec) > 1e-10))
        return child

    def sexual(self, parent_a: Genome, parent_b: Genome) -> Genome:
        vec_a = parent_a.to_vector()
        vec_b = parent_b.to_vector()
        # Use full vector length (supports genome expansion across phases)
        n = max(len(vec_a), len(vec_b))
        if len(vec_a) < n:
            vec_a = np.pad(vec_a, (0, n - len(vec_a)))
        if len(vec_b) < n:
            vec_b = np.pad(vec_b, (0, n - len(vec_b)))
        mask = np.random.rand(n) < 0.5
        child_vec = self._mutate(np.where(mask, vec_a, vec_b))
        return Genome.from_vector(
            child_vec,
            parent_ids=[parent_a.id, parent_b.id],
            generation=max(parent_a.generation, parent_b.generation) + 1,
        )

    def lamarckian(self, genome: Genome, experience: np.ndarray, lr: float = 0.02) -> Genome:
        """Compress lifetime experience gradient into genome."""
        vec = genome.to_vector()
        n = len(vec)
        exp_clipped = experience[:n]
        if len(exp_clipped) < n:
            exp_clipped = np.pad(exp_clipped, (0, n - len(exp_clipped)))
        # Normalize experience signal
        norm = np.linalg.norm(exp_clipped) + 1e-8
        child_vec = self._mutate(vec + lr * exp_clipped / norm,
                                 rate=self.base_mutation_rate * 0.5)
        return Genome.from_vector(child_vec, parent_ids=[genome.id],
                                  generation=genome.generation + 1)

    def _mutate(self, vec: np.ndarray, rate: float = None) -> np.ndarray:
        rate = rate if rate is not None else self.base_mutation_rate
        mask = np.random.rand(len(vec)) < rate
        noise = np.random.randn(len(vec)) * self.mutation_strength
        return vec + mask * noise
