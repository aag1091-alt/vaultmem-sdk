"""
VaultMem three-tier retrieval (§6.2).

Search hierarchy:
    Tier 1 — AFFINITY:   standing patterns; score = α·significance + (1−α)·cosine
    Tier 2 — COMPOSITE:  type-homogeneous aggregations; cosine similarity
    Tier 3 — ATOM:       leaf atoms; cosine similarity

All embedding vectors are assumed unit-length (normalised at write time by
LocalEmbedder).  Cosine similarity of unit vectors reduces to a dot product,
which numpy computes as a single batched matrix-vector multiply.

An atom returned from a higher tier is excluded from lower tiers, so the
same atom never appears twice in the result list.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .models import Granularity, MemoryObject, MemoryType


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Single retrieval result."""

    atom: MemoryObject
    score: float   # Blended (AFFINITY) or cosine score ∈ [-1, 1]
    tier: str      # "AFFINITY" | "COMPOSITE" | "ATOM"


# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------

def _batch_cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Dot product of a (D,) query against an (N, D) matrix of unit vectors.
    Returns an (N,) array of cosine similarities.
    """
    return matrix @ query


def _affinity_scores(
    atoms: list[MemoryObject],
    query: np.ndarray,
    alpha: float,
) -> list[tuple[MemoryObject, float]]:
    """
    Blended score = α·significance + (1−α)·cosine for AFFINITY atoms.
    Atoms without a significance value are treated as significance=0.
    """
    if not atoms:
        return []
    embs = np.array([a.embedding for a in atoms], dtype=np.float32)
    cosines = _batch_cosine(query, embs).tolist()
    out = []
    for atom, cos in zip(atoms, cosines):
        sig = atom.significance if atom.significance is not None else 0.0
        score = alpha * sig + (1.0 - alpha) * float(cos)
        out.append((atom, score))
    return out


def _cosine_scores(
    atoms: list[MemoryObject],
    query: np.ndarray,
) -> list[tuple[MemoryObject, float]]:
    """Pure cosine scores for COMPOSITE / ATOM tiers."""
    if not atoms:
        return []
    embs = np.array([a.embedding for a in atoms], dtype=np.float32)
    cosines = _batch_cosine(query, embs).tolist()
    return [(atom, float(cos)) for atom, cos in zip(atoms, cosines)]


def _topk(
    scored: list[tuple[MemoryObject, float]],
    k: int,
    min_score: float = -1.0,
) -> list[tuple[MemoryObject, float]]:
    """Filter by min_score, sort descending, take top k."""
    filtered = [(a, s) for a, s in scored if s >= min_score]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:k]


# ---------------------------------------------------------------------------
# Public search function
# ---------------------------------------------------------------------------

def search(
    query_embedding: list[float],
    atoms: dict[str, MemoryObject],
    *,
    top_k: int = 10,
    memory_type: Optional[MemoryType] = None,
    alpha: float = 0.5,
    affinity_min_score: float = -1.0,
    composite_min_score: float = -1.0,
    atom_min_score: float = -1.0,
) -> list[SearchResult]:
    """
    Three-tier retrieval over an atom dict.

    Args:
        query_embedding:  384-dim normalised float list (or numpy array).
        atoms:            Full atoms dict (churned atoms filtered internally).
        top_k:            Maximum results returned overall.
        memory_type:      Optional MemoryType filter.
        alpha:            AFFINITY tier blend weight [0, 1].
                          0 → pure cosine; 1 → pure significance.
        *_min_score:      Per-tier score floor; -1.0 means no cutoff.

    Returns:
        Up to ``top_k`` SearchResult objects, sorted by score descending.
        Each atom appears at most once (highest-priority tier wins).
    """
    query = np.array(query_embedding, dtype=np.float32)

    # Active atoms that have been embedded, with optional type filter
    active: list[MemoryObject] = [
        a for a in atoms.values()
        if not a.is_churned
        and a.embedding is not None
        and (memory_type is None or a.type == memory_type)
    ]

    seen: set[str] = set()
    results: list[SearchResult] = []

    # ── Tier 1: AFFINITY ─────────────────────────────────────────────────────
    tier1 = [a for a in active if a.granularity == Granularity.AFFINITY]
    for atom, score in _topk(
        _affinity_scores(tier1, query, alpha), top_k, affinity_min_score
    ):
        results.append(SearchResult(atom=atom, score=score, tier="AFFINITY"))
        seen.add(atom.id)

    # ── Tier 2: COMPOSITE ────────────────────────────────────────────────────
    tier2 = [
        a for a in active
        if a.granularity == Granularity.COMPOSITE and a.id not in seen
    ]
    for atom, score in _topk(
        _cosine_scores(tier2, query), top_k, composite_min_score
    ):
        results.append(SearchResult(atom=atom, score=score, tier="COMPOSITE"))
        seen.add(atom.id)

    # ── Tier 3: ATOM ─────────────────────────────────────────────────────────
    tier3 = [
        a for a in active
        if a.granularity == Granularity.ATOM and a.id not in seen
    ]
    for atom, score in _topk(
        _cosine_scores(tier3, query), top_k, atom_min_score
    ):
        results.append(SearchResult(atom=atom, score=score, tier="ATOM"))
        seen.add(atom.id)

    # Global re-rank and cap
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
