"""
Phase 3: 유전 시스템 재설계

사용자 철학 노트에서:
  "현재 설계의 한계: 단순 가중치 전달 → 라마르크식에 가까움 → 실제 진화와 거리"

핵심 변경:
  1. 감수분열 등가물 (Meiosis) — crossing over
     - 게놈 벡터를 N=4 염색체 구역으로 분할
     - 각 염색체 내 무작위 교차점 (crossing over)
     - "어머니 쪽 절반 + 아버지 쪽 절반" 대신 "4개 염색체 각각 교차"
     → 부모 어느 쪽과도 다른 새로운 조합 창출

  2. 진화하는 돌연변이율 (Metamutation)
     - mutation_rate_gene (genome 필드)에서 실제 돌연변이율 결정
     - 너무 낮으면 적응 불가, 너무 높으면 정보 붕괴 → 중간값이 선택됨
     - "돌연변이율 자체가 선택압 대상"

  3. 바이스만 장벽 (Weismann Barrier)
     - 체세포(경험, lamarckian) → 생식세포(genome) 직접 전달 금지
     - 경험은 epigenetic_state로만 영향 (간접, 감쇠)
     - 자식: parent_genome + ε × experience (lr: 0.005, 대폭 감소)
     - 잡음 추가: 생식세포로 가는 경로는 불완전함

  4. MHC 등가물 배우자 선택 (유전적 다양성 선호)
     - 아무 에이전트나 교배 X
     - 유전적으로 가장 다른 파트너를 샘플에서 선호
     - "면역 다양성 확보" — 자식의 유전자 탐색 공간 확장
"""

import numpy as np
from .genome import Genome, GENOME_DIM
from typing import List

# 감수분열 염색체 분할 수
N_CHROMOSOMES = 4


class ReplicationEngine:
    """Phase 3 재설계: Meiosis + Metamutation + Weismann Barrier + MHC Selection."""

    def __init__(self, base_mutation_rate: float = 0.01, mutation_strength: float = 0.1):
        self.base_mutation_rate = base_mutation_rate
        self.mutation_strength  = mutation_strength

    # ── 1. 무성생식 (돌연변이율 유전자 사용) ────────────────────────────────────

    def asexual(self, genome: Genome) -> Genome:
        vec  = genome.to_vector()
        rate = self._effective_rate(genome)
        child_vec = self._mutate(vec, rate=rate)
        child = Genome.from_vector(child_vec, parent_ids=[genome.id],
                                   generation=genome.generation + 1)
        child.mutation_count = genome.mutation_count + int(
            np.sum(np.abs(child_vec - vec) > 1e-10)
        )
        return child

    # ── 2. 유성생식 — 감수분열 + 교차 (Meiosis with Crossing Over) ──────────────

    def sexual(self, parent_a: Genome, parent_b: Genome) -> Genome:
        """
        감수분열 등가물:
          1. 게놈을 N_CHROMOSOMES 구역으로 분할
          2. 각 염색체 내 단일 교차점 결정 (70% 확률)
          3. 교차점 기준으로 두 부모의 절반 합치기
          → 부모 어느 쪽과도 다른 새로운 조합

        이것이 기존 random 50% mask와 다른 이유:
          - 기존: 각 유전자가 독립적으로 A/B 부모에서 선택 → 연관된 유전자들이 분리됨
          - 감수분열: 염색체 단위로 유지, 연관 유전자가 함께 이동
          → 협력하는 유전자 조합이 보존될 가능성 높음
        """
        vec_a = parent_a.to_vector()
        vec_b = parent_b.to_vector()
        n = max(len(vec_a), len(vec_b))
        if len(vec_a) < n:
            vec_a = np.pad(vec_a, (0, n - len(vec_a)))
        if len(vec_b) < n:
            vec_b = np.pad(vec_b, (0, n - len(vec_b)))

        child_vec = self._meiosis_crossover(vec_a, vec_b, n)

        # 돌연변이율: 두 부모의 mutation_rate_gene 평균
        rate = (self._effective_rate(parent_a) + self._effective_rate(parent_b)) / 2.0
        child_vec = self._mutate(child_vec, rate=rate)

        return Genome.from_vector(
            child_vec,
            parent_ids=[parent_a.id, parent_b.id],
            generation=max(parent_a.generation, parent_b.generation) + 1,
        )

    def _meiosis_crossover(self, vec_a: np.ndarray, vec_b: np.ndarray,
                            n: int) -> np.ndarray:
        """4개 염색체 구역으로 분할 후 각 구역 내 교차."""
        child_vec = np.empty(n, dtype=float)
        seg_size  = n // N_CHROMOSOMES

        for c in range(N_CHROMOSOMES):
            start = c * seg_size
            end   = (c + 1) * seg_size if c < N_CHROMOSOMES - 1 else n

            if np.random.rand() < 0.7 and end > start + 1:
                # 교차 발생: 무작위 교차점
                xo = np.random.randint(start, end)
                if np.random.rand() < 0.5:
                    child_vec[start:xo]  = vec_a[start:xo]
                    child_vec[xo:end]    = vec_b[xo:end]
                else:
                    child_vec[start:xo]  = vec_b[start:xo]
                    child_vec[xo:end]    = vec_a[xo:end]
            else:
                # 교차 없음: 한 부모에서 전체 염색체
                if np.random.rand() < 0.5:
                    child_vec[start:end] = vec_a[start:end]
                else:
                    child_vec[start:end] = vec_b[start:end]

        return child_vec

    # ── 3. 라마르크식 — 바이스만 장벽 적용 ──────────────────────────────────────

    def lamarckian(self, genome: Genome, experience: np.ndarray,
                   lr: float = 0.005) -> Genome:
        """
        바이스만 장벽 (Weismann Barrier):
          "에이전트의 경험(체세포 등가물)은 직접 전달 안 됨.
           발달 프로그램(생식세포 등가물)만 전달."

          [변경]
          - lr: 0.02 → 0.005 (간접 영향만 허용)
          - 노이즈 추가: 생식세포 경로의 불완전성 모델링
            exp_noisy = experience + N(0, 0.5) * std(experience)
          - 결과: 경험이 진화를 약하게 유도할 뿐, 직접 각인 X

        생물학적 근거:
          후성유전학(epigenetics): DNA 서열은 바뀌지 않지만
          DNA가 읽히는 방식이 일부 자식에게 전달됨.
          → 완전한 장벽(바이스만) + 미약한 투과(후성유전)의 중간
        """
        vec = genome.to_vector()
        n   = len(vec)
        exp_clipped = experience[:n]
        if len(exp_clipped) < n:
            exp_clipped = np.pad(exp_clipped, (0, n - len(exp_clipped)))

        # 바이스만 장벽: 경험에 잡음 추가 (생식세포 경로의 불완전성)
        exp_std   = np.std(exp_clipped) + 1e-8
        exp_noisy = exp_clipped + np.random.randn(n) * exp_std * 0.5

        norm      = np.linalg.norm(exp_noisy) + 1e-8
        rate      = self._effective_rate(genome) * 0.5   # 반감: 보수적 탐색
        child_vec = self._mutate(vec + lr * exp_noisy / norm, rate=rate)

        return Genome.from_vector(child_vec, parent_ids=[genome.id],
                                  generation=genome.generation + 1)

    # ── 4. MHC 등가물 배우자 선택 (유전적 다양성 선호) ─────────────────────────

    def select_mate_mhc(self, focal_genome: Genome,
                         candidates: List, sample_k: int = 10):
        """
        MHC 등가물: 유전적으로 가장 다른 파트너 선택.

        생물학적 근거:
          - 인간 MHC 유전자: 무의식적으로 다른 면역 유전자 보유자 선호
          - 자식의 면역 다양성 확보 → 병원균 저항성 향상

        LTBL:
          - 전체 게놈 벡터의 유클리드 거리로 유전적 다양성 측정
          - sample_k개 중에서 가장 거리가 먼 파트너 선택
          - 완전 탐색 대신 샘플링 (생물학적으로도 무작위 만남)
        """
        if len(candidates) <= 1:
            return candidates[0]

        vec_focal = focal_genome.to_vector()
        sample    = candidates if len(candidates) <= sample_k else \
                    list(np.random.choice(candidates, sample_k, replace=False))

        distances = []
        for c in sample:
            vec_c = c.genome.to_vector()
            n = min(len(vec_focal), len(vec_c))
            distances.append(float(np.linalg.norm(vec_focal[:n] - vec_c[:n])))

        return sample[int(np.argmax(distances))]

    # ── 내부 유틸 ────────────────────────────────────────────────────────────

    def _effective_rate(self, genome: Genome) -> float:
        """mutation_rate_gene이 있으면 그 값, 없으면 base_mutation_rate."""
        gene_rate = getattr(genome, 'mutation_rate_gene', None)
        if gene_rate is not None:
            return float(np.clip(gene_rate, 0.001, 0.1))
        return self.base_mutation_rate

    def _mutate(self, vec: np.ndarray, rate: float = None) -> np.ndarray:
        rate = rate if rate is not None else self.base_mutation_rate
        mask  = np.random.rand(len(vec)) < rate
        noise = np.random.randn(len(vec)) * self.mutation_strength
        return vec + mask * noise
