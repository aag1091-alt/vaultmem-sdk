"""
VaultMem vector index — ANN layer for sub-linear semantic search.

VectorIndex        — abstract interface.
HNSWVectorIndex    — hnswlib HNSW graph implementation.

Requires: pip install "vaultmem[ann]"

Why this layer exists
---------------------
The SearchIndex (SQLite/Postgres) stores *encrypted* embeddings.  Every search
would otherwise need to decrypt all N embeddings, load them into RAM, and do a
brute-force cosine scan — O(N) per query.

HNSWVectorIndex keeps plaintext embeddings in an HNSW graph in session RAM.
After a one-time load on session open the graph enables O(log N) ANN search
without touching the blob store or index DB at all.

The index file on disk is encrypted with the MEK, so the platform operator
cannot read the embeddings even if they have filesystem access.

Search modes
------------
* Large candidate set (> EXACT_THRESHOLD): hnswlib knn_query — O(log N).
* Small candidate set (≤ EXACT_THRESHOLD, e.g. after a memory_type pre-filter):
  exact cosine via hnswlib.get_items() — simple and dependency-free.

Persistence
-----------
save() serialises the HNSW index + id↔int mapping, then encrypts the payload
with AES-256-GCM (MEK, AAD = b"vector_index_v1") and writes it atomically.
load() reverses this.  The file is typically stored as
    {vault_dir}/vector_index.hnsw.enc
"""
from __future__ import annotations

import json
import os
import struct
import tempfile
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np


# Candidate sets at or below this size use exact cosine via get_items()
# rather than the HNSW filter function (simpler, no version dependency).
EXACT_THRESHOLD = 1000

# Fixed AAD for the whole-index encryption blob (not per-atom).
_INDEX_AAD = b"vector_index_v1"


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class VectorIndex(ABC):
    """Abstract ANN index for fast semantic search."""

    @abstractmethod
    def add(self, atom_id: str, embedding: list[float]) -> None:
        """Add or update an atom's embedding in the index."""

    @abstractmethod
    def remove(self, atom_id: str) -> None:
        """Mark an atom deleted. No-op if not present."""

    @abstractmethod
    def search(
        self,
        query: list[float],
        k: int,
        filter_ids: Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        """
        Return up to k (atom_id, cosine_similarity) pairs sorted descending.

        filter_ids: when set, only atoms in this set may appear in results.
        """

    @abstractmethod
    def save(self, path: "str | Path", mek: bytes) -> None:
        """Encrypt and persist the index to *path*."""

    @abstractmethod
    def load(self, path: "str | Path", mek: bytes) -> None:
        """Decrypt and restore the index from *path* (no-op if file absent)."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of active (non-deleted) vectors."""


# ---------------------------------------------------------------------------
# hnswlib implementation
# ---------------------------------------------------------------------------

class HNSWVectorIndex(VectorIndex):
    """
    hnswlib HNSW cosine-space vector index.

    Parameters
    ----------
    dim : int
        Embedding dimensionality (e.g. 384 for all-MiniLM-L6-v2).
    ef_construction : int
        hnswlib ef_construction hyperparameter (build quality vs speed).
    M : int
        hnswlib M hyperparameter (graph connectivity).
    """

    def __init__(
        self,
        dim: int,
        *,
        ef_construction: int = 200,
        M: int = 16,
    ) -> None:
        try:
            import hnswlib
            self._hnswlib = hnswlib
        except ImportError:
            raise ImportError(
                'HNSWVectorIndex requires hnswlib: pip install "vaultmem[ann]"'
            )
        self._dim             = dim
        self._ef_construction = ef_construction
        self._M               = M
        self._lock            = threading.Lock()

        # hnswlib uses integer labels; we manage the str↔int mapping ourselves.
        self._id_to_int: dict[str, int] = {}
        self._int_to_id: dict[int, str] = {}
        self._next_int:   int           = 0
        self._active_count: int         = 0

        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(
            max_elements=1024, ef_construction=ef_construction, M=M
        )
        self._index.set_ef(50)

    # ── Writes ────────────────────────────────────────────────────────────

    def add(self, atom_id: str, embedding: list[float]) -> None:
        """Add or replace an atom's embedding."""
        with self._lock:
            vec = np.array(embedding, dtype=np.float32).reshape(1, -1)

            if atom_id in self._id_to_int:
                # hnswlib has no in-place update; mark old label deleted, re-add.
                old_label = self._id_to_int[atom_id]
                try:
                    self._index.mark_deleted(old_label)
                except Exception:
                    pass
                self._active_count -= 1

            label = self._next_int
            self._next_int += 1
            self._id_to_int[atom_id] = label
            self._int_to_id[label]   = atom_id

            self._ensure_capacity(1)
            self._index.add_items(vec, [label])
            self._active_count += 1

    def remove(self, atom_id: str) -> None:
        """Mark an atom deleted (soft-delete — HNSW graph structure preserved)."""
        with self._lock:
            label = self._id_to_int.get(atom_id)
            if label is None:
                return
            try:
                self._index.mark_deleted(label)
                self._active_count -= 1
            except Exception:
                pass

    # ── Reads ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: list[float],
        k: int,
        filter_ids: Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        with self._lock:
            if self._active_count == 0:
                return []

            if filter_ids is not None and len(filter_ids) <= EXACT_THRESHOLD:
                return self._exact_search(query, k, filter_ids)

            actual_k = min(k, self._active_count)
            q = np.array(query, dtype=np.float32).reshape(1, -1)
            labels, distances = self._index.knn_query(q, k=actual_k)

            results: list[tuple[str, float]] = []
            for label, dist in zip(labels[0], distances[0]):
                atom_id = self._int_to_id.get(int(label))
                if atom_id is None:
                    continue
                if filter_ids is not None and atom_id not in filter_ids:
                    continue
                results.append((atom_id, float(1.0 - dist)))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

    def _exact_search(
        self,
        query: list[float],
        k: int,
        filter_ids: set[str],
    ) -> list[tuple[str, float]]:
        """Exact cosine for small candidate sets using hnswlib.get_items()."""
        labels = [self._id_to_int[aid] for aid in filter_ids if aid in self._id_to_int]
        if not labels:
            return []

        vecs = self._index.get_items(labels)  # shape (n, dim)
        q    = np.array(query, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-10)

        results: list[tuple[str, float]] = []
        for label, vec in zip(labels, vecs):
            v      = np.array(vec, dtype=np.float32)
            v_norm = v / (np.linalg.norm(v) + 1e-10)
            cosine = float(np.dot(q_norm, v_norm))
            results.append((self._int_to_id[label], cosine))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: "str | Path", mek: bytes) -> None:
        """Encrypt and persist the index to *path* (atomic write on POSIX)."""
        from .crypto import encrypt_atom, pack_encrypted_block

        path = Path(path)

        with self._lock:
            # 1. Serialise HNSW graph via tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".hnsw") as tmp:
                tmp_hnsw = tmp.name
            try:
                self._index.save_index(tmp_hnsw)
                hnsw_bytes = Path(tmp_hnsw).read_bytes()
            finally:
                os.unlink(tmp_hnsw)

            # 2. Serialise id↔int mapping + metadata
            mapping = {
                "id_to_int":    self._id_to_int,
                "int_to_id":    {str(k): v for k, v in self._int_to_id.items()},
                "next_int":     self._next_int,
                "active_count": self._active_count,
                "dim":          self._dim,
                "ef_construction": self._ef_construction,
                "M":            self._M,
            }
            map_bytes = json.dumps(mapping, separators=(",", ":")).encode("utf-8")

        # 3. Pack: [4B hnsw_len][hnsw_bytes][4B map_len][map_bytes]
        payload = (
            struct.pack(">I", len(hnsw_bytes)) + hnsw_bytes
            + struct.pack(">I", len(map_bytes)) + map_bytes
        )

        # 4. Encrypt with MEK; fixed AAD — not per-atom
        iv, ct, tag = encrypt_atom(mek, payload, _INDEX_AAD)
        blob = pack_encrypted_block(iv, ct, tag)

        # 5. Atomic write
        tmp_out = Path(str(path) + ".tmp")
        tmp_out.write_bytes(blob)
        tmp_out.rename(path)

    def load(self, path: "str | Path", mek: bytes) -> None:
        """Decrypt and restore from *path*. No-op if file does not exist."""
        from .crypto import decrypt_atom, unpack_encrypted_block

        path = Path(path)
        if not path.exists():
            return

        raw = path.read_bytes()
        iv, ct, tag, _ = unpack_encrypted_block(raw, 0)
        payload = decrypt_atom(mek, iv, ct, tag, _INDEX_AAD)

        # Unpack
        hnsw_len   = struct.unpack(">I", payload[:4])[0]
        hnsw_bytes = payload[4:4 + hnsw_len]
        off        = 4 + hnsw_len
        map_len    = struct.unpack(">I", payload[off:off + 4])[0]
        map_bytes  = payload[off + 4:off + 4 + map_len]

        mapping = json.loads(map_bytes.decode("utf-8"))

        with self._lock:
            self._dim             = mapping["dim"]
            self._ef_construction = mapping.get("ef_construction", self._ef_construction)
            self._M               = mapping.get("M", self._M)
            self._id_to_int       = mapping["id_to_int"]
            self._int_to_id       = {int(k): v for k, v in mapping["int_to_id"].items()}
            self._next_int        = mapping["next_int"]
            self._active_count    = mapping["active_count"]

            # Restore HNSW graph
            with tempfile.NamedTemporaryFile(delete=False, suffix=".hnsw") as tmp:
                tmp.write(hnsw_bytes)
                tmp_path = tmp.name
            try:
                self._index = self._hnswlib.Index(space="cosine", dim=self._dim)
                self._index.load_index(
                    tmp_path, max_elements=max(self._next_int + 128, 1024)
                )
                self._index.set_ef(50)
            finally:
                os.unlink(tmp_path)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _ensure_capacity(self, needed: int) -> None:
        current = self._index.get_max_elements()
        if self._next_int + needed > current:
            new_cap = max(current * 2, self._next_int + needed + 128)
            self._index.resize_index(new_cap)

    def __len__(self) -> int:
        return self._active_count
