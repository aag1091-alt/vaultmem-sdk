"""
VaultMem session — state machine and vault lifecycle.

States:
    CLOSED        No MEK in memory; vault is on-disk only.
    OPEN          MEK in RAM; MemoryState loaded; writes buffered.
    CHECKPOINTING Atomic write in progress (transient).
    CLOSING       Flush + MEK zero in progress (transient).

Checkpoint protocol (crash-safe on POSIX):
    1. Encrypt only new atoms → append to current.atoms → fsync
    2. Rebuild encrypted index in RAM
    3. Write header+index → current.vmem.tmp → fsync → rename(current.vmem)
    4. Mark state clean

    Step 1 is O(new atoms); existing atom blocks are never rewritten (atoms are
    immutable once written). Step 3 rewrites only the small index file.

Session locking (session.lock):
    An exclusive flock on session.lock prevents concurrent opens of the
    same vault directory. The file contains the locking process PID.
    On non-POSIX platforms (Windows) locking is a no-op.

MEK lifecycle:
    Stored as a bytearray so individual bytes can be zeroed on close.
    Python does not guarantee that zeroing a bytearray scrubs the
    underlying memory (GC/copy-on-write), but it is a best-effort measure
    consistent with standard practice for sensitive key material in Python.
"""
from __future__ import annotations

import base64
import enum
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Optional

from .classifier import classify
from .crypto import (
    ARGON2_MEMORY_COST,
    derive_kek,
    encrypt_atom,
    pack_encrypted_block,
    unpack_encrypted_block,
    decrypt_atom,
    unwrap_mek,
)
from .embedder import Embedder, LocalEmbedder
from .exceptions import SessionStateError, VaultLockedError
from .index import IndexQuery, IndexRecord, SearchIndex
from .models import DataClass, Granularity, MemoryObject, MemoryType, estimate_tokens
from .retrieval import SearchResult
from .retrieval import search as _search
from .storage import BlobStore
from .vector_index import VectorIndex
from .vault import (
    FILE_TYPE_WORKING,
    MemoryState,
    append_atoms_to_file,
    create_vault,
    read_working_copy,
    write_index_file,
)

try:
    import fcntl as _fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False  # Windows — locking is a no-op


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class SessionState(enum.Enum):
    CLOSED        = "CLOSED"
    OPEN          = "OPEN"
    CHECKPOINTING = "CHECKPOINTING"
    CLOSING       = "CLOSING"


# ---------------------------------------------------------------------------
# VaultSession
# ---------------------------------------------------------------------------

class VaultSession:
    """
    Active session over a VaultMem vault directory.

    Typical usage::

        # Open existing vault
        with VaultSession.open("./vault", passphrase="s3cr3t") as s:
            s.add("I met Alice yesterday at the conference")
            results = s.search("Alice conference")
            for r in results:
                print(r.score, r.atom.content)

        # Create a new vault
        with VaultSession.create("./vault", passphrase="s3cr3t", owner="alice") as s:
            s.add("My first memory")
    """

    # ── Private constructor ────────────────────────────────────────────────

    def __init__(
        self,
        vault_dir: Path,
        memory_state: Optional[MemoryState],
        mek: bytearray,
        data_class: DataClass,
        lock_fd: Optional[int],
        embedder: Embedder,
        session_id: str,
        *,
        blob_store: Optional[BlobStore] = None,
        search_index: Optional[SearchIndex] = None,
        vector_index: Optional[VectorIndex] = None,
        owner: str = "",
    ) -> None:
        if (blob_store is None) != (search_index is None):
            raise ValueError("blob_store and search_index must both be provided or both omitted")

        self._vault_dir  = vault_dir
        self._mem        = memory_state
        self._mek        = mek         # bytearray — zeroed on close
        self._data_class = data_class
        self._lock_fd    = lock_fd
        self._embedder   = embedder
        self._session_id = session_id
        self._state      = SessionState.OPEN
        self._rlock      = threading.RLock()

        # Backend-mode state (blob_store + search_index provided)
        self._use_backends: bool = blob_store is not None
        self._blob_store   = blob_store
        self._search_index = search_index
        self._vector_index = vector_index  # optional ANN layer
        self._owner        = owner if self._use_backends else (memory_state.owner if memory_state else "")
        self._pending: dict[str, MemoryObject] = {}   # buffered, not yet flushed
        self._dirty: bool = False

    # ── Factory: open existing vault ───────────────────────────────────────

    @classmethod
    def open(
        cls,
        vault_dir: "str | Path",
        passphrase: str,
        embedder: Optional[Embedder] = None,
        *,
        blob_store: Optional[BlobStore] = None,
        search_index: Optional[SearchIndex] = None,
        vector_index: Optional[VectorIndex] = None,
    ) -> "VaultSession":
        """
        Open an existing vault.

        Raises:
            FileNotFoundError:    vault_dir or current.vmem not found.
            VaultLockedError:     Another process holds the session lock.
            WrongPassphraseError: Bad passphrase (GCM tag mismatch).
            VaultTamperedError:   File contents failed authentication.
        """
        vault_dir = Path(vault_dir)
        lock_fd = cls._acquire_lock(vault_dir)
        try:
            meta       = cls._read_meta(vault_dir)
            slot       = next(
                s for s in meta["credential_slots"]
                if s["slot_id"] == "primary"
            )
            kdf_salt    = base64.b64decode(slot["kdf_salt"])
            wrapped_iv  = base64.b64decode(slot["wrapped_mek_iv"])
            wrapped_ct  = base64.b64decode(slot["wrapped_mek_ct"])
            wrapped_tag = base64.b64decode(slot["wrapped_mek_tag"])
            memory_cost = meta.get("memory_cost", ARGON2_MEMORY_COST)
            data_class  = DataClass(meta.get("data_class", "GENERAL"))

            # Derive KEK → unwrap MEK → zero KEK immediately
            kek = derive_kek(passphrase, kdf_salt, memory_cost)
            try:
                mek_bytes = unwrap_mek(kek, wrapped_iv, wrapped_ct, wrapped_tag)
            finally:
                kek_ba = bytearray(kek)
                for i in range(len(kek_ba)):
                    kek_ba[i] = 0

            # Load vault from disk (split working copy format)
            vmem_path  = vault_dir / "current.vmem"
            atoms_path = vault_dir / "current.atoms"

            mek_ba = bytearray(mek_bytes)
            del mek_bytes

            if blob_store is not None or search_index is not None:
                # Backend mode: don't load all atoms into RAM
                session = cls(
                    vault_dir=vault_dir,
                    memory_state=None,
                    mek=mek_ba,
                    data_class=data_class,
                    lock_fd=lock_fd,
                    embedder=embedder or LocalEmbedder(),
                    session_id=str(uuid.uuid4()),
                    blob_store=blob_store,
                    search_index=search_index,
                    vector_index=vector_index,
                    owner=meta.get("owner", ""),
                )
                if vector_index is not None:
                    vi_path = vault_dir / "vector_index.hnsw.enc"
                    vector_index.load(vi_path, bytes(mek_ba))
                return session

            state = read_working_copy(vmem_path, atoms_path, bytes(mek_ba))

            return cls(
                vault_dir=vault_dir,
                memory_state=state,
                mek=mek_ba,
                data_class=data_class,
                lock_fd=lock_fd,
                embedder=embedder or LocalEmbedder(),
                session_id=str(uuid.uuid4()),
            )
        except Exception:
            cls._release_lock(lock_fd)
            raise

    # ── Factory: create new vault ──────────────────────────────────────────

    @classmethod
    def create(
        cls,
        vault_dir: "str | Path",
        passphrase: str,
        owner: str,
        data_class: str = "GENERAL",
        embedder: Optional[Embedder] = None,
        *,
        blob_store: Optional[BlobStore] = None,
        search_index: Optional[SearchIndex] = None,
        vector_index: Optional[VectorIndex] = None,
    ) -> "VaultSession":
        """
        Initialise a new vault and open it.

        Raises:
            FileExistsError: A vault already exists at vault_dir.
        """
        vault_dir = Path(vault_dir)
        if (vault_dir / "meta.json").exists():
            raise FileExistsError(f"Vault already exists at {vault_dir}")

        state, mek_bytes = create_vault(vault_dir, passphrase, owner, data_class)

        lock_fd = cls._acquire_lock(vault_dir)
        try:
            mek_ba = bytearray(mek_bytes)
            del mek_bytes
            if blob_store is not None or search_index is not None:
                return cls(
                    vault_dir=vault_dir,
                    memory_state=None,
                    mek=mek_ba,
                    data_class=DataClass(data_class),
                    lock_fd=lock_fd,
                    embedder=embedder or LocalEmbedder(),
                    session_id=str(uuid.uuid4()),
                    blob_store=blob_store,
                    search_index=search_index,
                    vector_index=vector_index,
                    owner=owner,
                )
            return cls(
                vault_dir=vault_dir,
                memory_state=state,
                mek=mek_ba,
                data_class=DataClass(data_class),
                lock_fd=lock_fd,
                embedder=embedder or LocalEmbedder(),
                session_id=str(uuid.uuid4()),
            )
        except Exception:
            cls._release_lock(lock_fd)
            raise

    # ── Locking helpers ────────────────────────────────────────────────────

    @staticmethod
    def _acquire_lock(vault_dir: Path) -> Optional[int]:
        if not _HAS_FCNTL:
            return None
        lock_path = vault_dir / "session.lock"
        fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise VaultLockedError(
                f"Vault is locked by another process: {vault_dir}"
            )
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, str(os.getpid()).encode())
        return fd

    @staticmethod
    def _release_lock(lock_fd: Optional[int]) -> None:
        if lock_fd is None or not _HAS_FCNTL:
            return
        try:
            _fcntl.flock(lock_fd, _fcntl.LOCK_UN)
            os.close(lock_fd)
        except OSError:
            pass

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _read_meta(vault_dir: Path) -> dict:
        with open(vault_dir / "meta.json") as fh:
            return json.load(fh)

    def _require_open(self) -> None:
        if self._state != SessionState.OPEN:
            raise SessionStateError(
                f"Operation requires OPEN session (current: {self._state.value})"
            )

    def _checkpoint(self) -> None:
        """
        Two-step append-only checkpoint (crash-safe on POSIX).

        Step 1 — append new atoms to current.atoms and fsync. Existing atom
        blocks are never modified; atoms are immutable once written (§4.1).
        Write cost is O(new atoms), not O(total atoms).

        Step 2 — rebuild the encrypted index and atomically replace
        current.vmem via tmp-file rename. If the process crashes between
        steps 1 and 2, the orphaned blocks in current.atoms are detected and
        truncated on the next open (read_working_copy handles this).
        """
        prev_state = self._state
        self._state = SessionState.CHECKPOINTING
        try:
            atoms_path = self._vault_dir / "current.atoms"
            vmem_path  = self._vault_dir / "current.vmem"
            mek_bytes  = bytes(self._mek)

            # Step 1: append only atoms not yet on disk
            new_ids = self._mem.new_atom_ids()
            if new_ids:
                new_offsets = append_atoms_to_file(
                    atoms_path,
                    mek_bytes,
                    self._mem.atoms,
                    new_ids,
                    start_offset=self._mem._atoms_file_size,
                )
                self._mem._atom_offsets.update(new_offsets)
                self._mem._atoms_file_size += sum(
                    sz for _, sz in new_offsets.values()
                )

            # Step 2: rewrite header + index atomically
            write_index_file(vmem_path, mek_bytes, self._mem)
            self._mem.mark_clean()
        finally:
            self._state = prev_state

    def _zero_mek(self) -> None:
        """Best-effort MEK erasure."""
        for i in range(len(self._mek)):
            self._mek[i] = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        *,
        confidence: float = 1.0,
        owner: str = "",
        auto_embed: bool = True,
    ) -> MemoryObject:
        """
        Add a plaintext memory atom to the in-memory state.

        Memory type is classified automatically by the deterministic 4-feature
        classifier (§5.1). An embedding is generated with the session embedder
        (all-MiniLM-L6-v2 by default) unless auto_embed=False.

        The atom is not persisted until flush() or close() is called.

        Returns:
            The new MemoryObject (id is assigned immediately).
        """
        with self._rlock:
            self._require_open()
            mem_type = classify(content)
            size_tok = estimate_tokens(content)
            effective_owner = owner or (self._owner if self._use_backends else self._mem.owner)
            atom = MemoryObject(
                id="",                          # UUID assigned in __post_init__
                type=mem_type,
                granularity=Granularity.ATOM,
                content=content,
                size_tokens=size_tok,
                session_id=self._session_id,
                confidence=confidence,
                owner=effective_owner,
                data_class=self._data_class,
            )
            if auto_embed:
                atom.embedding = self._embedder.embed(content)

            if self._use_backends:
                self._pending[atom.id] = atom
                self._dirty = True
            else:
                self._mem.atoms[atom.id] = atom
                self._mem.mark_dirty()
            return atom

    def search(
        self,
        query: "str | list[float]",
        *,
        top_k: int = 10,
        memory_type: Optional[MemoryType] = None,
        alpha: float = 0.5,
    ) -> list[SearchResult]:
        """
        Three-tier semantic search (AFFINITY → COMPOSITE → ATOM).

        Args:
            query:       Plain text (embedded by the session embedder) or a
                         pre-computed 384-dim unit vector.
            top_k:       Maximum number of results.
            memory_type: Optional MemoryType filter.
            alpha:       AFFINITY blend weight (0 = pure cosine, 1 = pure
                         significance). Default 0.5.

        Returns:
            SearchResult list sorted by score descending.
        """
        with self._rlock:
            self._require_open()
            if isinstance(query, str):
                q_emb = self._embedder.embed(query)
            else:
                q_emb = list(query)
            if self._use_backends:
                return self._backend_search(q_emb, top_k=top_k, memory_type=memory_type, alpha=alpha)
            return _search(
                query_embedding=q_emb,
                atoms=self._mem.active_atoms,
                top_k=top_k,
                memory_type=memory_type,
                alpha=alpha,
            )

    def flush(self) -> None:
        """Persist any pending changes to disk without closing the session."""
        with self._rlock:
            self._require_open()
            if self._use_backends:
                if self._dirty:
                    self._flush_pending()
            elif self._mem.is_dirty:
                self._checkpoint()

    def close(self) -> None:
        """
        Flush pending writes, zero the MEK, and release the session lock.
        Safe to call multiple times (idempotent after first call).
        """
        with self._rlock:
            if self._state not in (SessionState.OPEN, SessionState.CHECKPOINTING):
                return
            self._state = SessionState.CLOSING
            try:
                if self._use_backends:
                    if self._dirty:
                        self._flush_pending()
                    self._search_index.close()
                elif self._mem.is_dirty:
                    self._checkpoint()
            finally:
                self._zero_mek()
                self._release_lock(self._lock_fd)
                self._state = SessionState.CLOSED

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def owner(self) -> str:
        """Vault owner string."""
        return self._owner if self._use_backends else self._mem.owner

    @property
    def vault_dir(self) -> Path:
        return self._vault_dir

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def atom_count(self) -> int:
        """Number of active (non-churned) atoms."""
        with self._rlock:
            if self._use_backends:
                return self._search_index.count() + len(self._pending)
            return len(self._mem.active_atoms)

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "VaultSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        if self._use_backends:
            n = self._search_index.count() if self._state == SessionState.OPEN else "?"
            return (
                f"VaultSession(owner={self._owner!r}, "
                f"state={self._state.value}, "
                f"atoms={n}, backend=True)"
            )
        return (
            f"VaultSession(owner={self.owner!r}, "
            f"state={self._state.value}, "
            f"atoms={len(self._mem.atoms)})"
        )

    # ── Backend-mode private methods ────────────────────────────────────────

    def _flush_pending(self) -> None:
        """Encrypt and persist all pending atoms to blob store + search index."""
        import json
        import uuid as _uuid
        import zlib

        mek_bytes = bytes(self._mek)
        prev_state = self._state
        self._state = SessionState.CHECKPOINTING
        try:
            for atom_id, atom in list(self._pending.items()):
                atom_id_bytes = _uuid.UUID(atom_id).bytes

                # 1. Encrypt full atom content → blob store
                atom_json = json.dumps(atom.to_dict(), ensure_ascii=False).encode("utf-8")
                atom_compressed = zlib.compress(atom_json)
                iv, ct, tag = encrypt_atom(mek_bytes, atom_compressed, atom_id_bytes)
                blob = pack_encrypted_block(iv, ct, tag)
                self._blob_store.put(atom_id, blob)

                # 2. Encrypt embedding → pack into bytes for search index
                if atom.embedding is not None:
                    emb_bytes = json.dumps(atom.embedding).encode("utf-8")
                    emb_iv, emb_ct, emb_tag = encrypt_atom(mek_bytes, emb_bytes, atom_id_bytes)
                    enc_emb = pack_encrypted_block(emb_iv, emb_ct, emb_tag)
                else:
                    enc_emb = pack_encrypted_block(
                        *encrypt_atom(mek_bytes, b"null", atom_id_bytes)
                    )

                # 3. Upsert index record
                self._search_index.upsert(IndexRecord(
                    atom_id=atom_id,
                    tier=atom.granularity.value,
                    memory_type=atom.type.value,
                    data_class=atom.data_class.value,
                    significance=atom.significance,
                    created_at=atom.created_at,
                    session_id=atom.session_id,
                    owner=atom.owner,
                    is_churned=atom.is_churned,
                    enc_embedding=enc_emb,
                ))

                # 4. Add plaintext embedding to ANN index (if present)
                if self._vector_index is not None and atom.embedding is not None:
                    self._vector_index.add(atom_id, atom.embedding)

            # Persist ANN index if used
            if self._vector_index is not None:
                vi_path = self._vault_dir / "vector_index.hnsw.enc"
                self._vector_index.save(vi_path, mek_bytes)

            self._pending.clear()
            self._dirty = False
        finally:
            self._state = prev_state

    def _backend_search(
        self,
        q_emb: list,
        *,
        top_k: int,
        memory_type: Optional[MemoryType],
        alpha: float,
    ) -> list[SearchResult]:
        """Three-tier search using the blob store + search index backends."""
        if self._vector_index is not None:
            return self._ann_search(q_emb, top_k=top_k, memory_type=memory_type, alpha=alpha)
        return self._exact_backend_search(q_emb, top_k=top_k, memory_type=memory_type, alpha=alpha)

    def _ann_search(
        self,
        q_emb: list,
        *,
        top_k: int,
        memory_type: Optional[MemoryType],
        alpha: float,
    ) -> list[SearchResult]:
        """ANN search path: O(log N) via HNSW, then fetch only top-k atoms."""
        import json
        import uuid as _uuid
        import zlib

        mek_bytes = bytes(self._mek)

        # Step 1: optional SQLite pre-filter to a candidate set
        filter_ids: Optional[set[str]] = None
        if memory_type is not None:
            iq = IndexQuery(is_churned=False, memory_type=memory_type.value)
            filter_ids = {r.atom_id for r in self._search_index.query(iq)}

        # Step 2: ANN search — returns (atom_id, cosine) sorted descending
        ann_hits: dict[str, float] = dict(
            self._vector_index.search(q_emb, k=top_k * 3, filter_ids=filter_ids)
        )

        # Step 3: exact cosine for unflushed pending atoms
        pending_scores = self._pending_cosine(q_emb, memory_type=memory_type)
        ann_hits.update(pending_scores)

        if not ann_hits:
            return []

        # Step 4: fetch metadata for only the top candidates
        top_ids = sorted(ann_hits, key=lambda x: -ann_hits[x])[: top_k * 2]
        non_pending = [aid for aid in top_ids if aid not in self._pending]
        meta_map = {r.atom_id: r for r in self._search_index.fetch_many(non_pending)}

        # Step 5: significance blending for AFFINITY atoms
        scored: list[tuple[str, float, str]] = []
        for atom_id in top_ids:
            cosine = ann_hits[atom_id]
            if atom_id in self._pending:
                atom = self._pending[atom_id]
                tier = atom.granularity.value
                sig  = atom.significance
            else:
                meta = meta_map.get(atom_id)
                if meta is None:
                    continue
                tier = meta.tier
                sig  = meta.significance

            if tier == Granularity.AFFINITY.value and sig is not None:
                score = alpha * sig + (1.0 - alpha) * cosine
            else:
                score = cosine
            scored.append((atom_id, score, tier))

        scored.sort(key=lambda x: -x[1])
        scored = scored[:top_k]

        # Step 6: fetch full atom content from blob store for the top-k only
        full_results: list[SearchResult] = []
        for atom_id, score, tier in scored:
            if atom_id in self._pending:
                full_results.append(
                    SearchResult(atom=self._pending[atom_id], score=score, tier=tier)
                )
                continue
            try:
                raw = self._blob_store.get(atom_id)
                iv, ct, tag, _ = unpack_encrypted_block(raw, 0)
                compressed = decrypt_atom(mek_bytes, iv, ct, tag, _uuid.UUID(atom_id).bytes)
                full_atom = MemoryObject.from_dict(json.loads(zlib.decompress(compressed)))
                full_results.append(SearchResult(atom=full_atom, score=score, tier=tier))
            except Exception:
                continue

        return full_results

    def _exact_backend_search(
        self,
        q_emb: list,
        *,
        top_k: int,
        memory_type: Optional[MemoryType],
        alpha: float,
    ) -> list[SearchResult]:
        """Legacy O(N) exact search: decrypts all embeddings from SQLite into RAM."""
        import json
        import uuid as _uuid

        mek_bytes = bytes(self._mek)

        # Step 1: pull candidate records from the search index
        iq = IndexQuery(
            is_churned=False,
            memory_type=memory_type.value if memory_type is not None else None,
        )
        records = self._search_index.query(iq)

        # Step 2: decrypt embeddings, build lightweight MemoryObject stubs for ranking
        stub_atoms: dict[str, MemoryObject] = {}
        for rec in records:
            try:
                iv, ct, tag, _ = unpack_encrypted_block(rec.enc_embedding, 0)
                emb_bytes = decrypt_atom(mek_bytes, iv, ct, tag, _uuid.UUID(rec.atom_id).bytes)
                embedding = json.loads(emb_bytes.decode("utf-8"))
                if embedding is None:
                    continue
            except Exception:
                continue  # corrupted or missing embedding — skip gracefully

            stub = MemoryObject(
                id=rec.atom_id,
                type=MemoryType(rec.memory_type),
                granularity=Granularity(rec.tier),
                content="",          # not needed for ranking
                size_tokens=0,
                embedding=embedding,
                session_id=rec.session_id,
                created_at=rec.created_at,
                significance=rec.significance,
                is_churned=rec.is_churned,
                owner=rec.owner,
                data_class=DataClass(rec.data_class),
            )
            stub_atoms[rec.atom_id] = stub

        # Step 3: merge pending (in-session, not yet flushed) atoms
        for atom in self._pending.values():
            if not atom.is_churned and atom.embedding is not None:
                if memory_type is None or atom.type == memory_type:
                    stub_atoms[atom.id] = atom

        # Step 4: run three-tier retrieval over stubs
        ranked = _search(
            query_embedding=q_emb,
            atoms=stub_atoms,
            top_k=top_k,
            memory_type=memory_type,
            alpha=alpha,
        )

        # Step 5: fetch full atom content from blob store for the top results
        import zlib
        full_results: list[SearchResult] = []
        for sr in ranked:
            if sr.atom.id in self._pending:
                full_results.append(sr)
                continue
            try:
                raw = self._blob_store.get(sr.atom.id)
                iv, ct, tag, _ = unpack_encrypted_block(raw, 0)
                compressed = decrypt_atom(
                    mek_bytes, iv, ct, tag, _uuid.UUID(sr.atom.id).bytes
                )
                full_atom = MemoryObject.from_dict(
                    json.loads(zlib.decompress(compressed))
                )
                full_results.append(SearchResult(atom=full_atom, score=sr.score, tier=sr.tier))
            except Exception:
                continue  # missing or tampered blob — skip

        return full_results

    def _pending_cosine(
        self,
        q_emb: list,
        *,
        memory_type: Optional[MemoryType],
    ) -> dict[str, float]:
        """Compute exact cosine similarity for unflushed pending atoms."""
        if not self._pending:
            return {}
        import numpy as np
        q = np.array(q_emb, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        scores: dict[str, float] = {}
        for atom_id, atom in self._pending.items():
            if atom.is_churned or atom.embedding is None:
                continue
            if memory_type is not None and atom.type != memory_type:
                continue
            v      = np.array(atom.embedding, dtype=np.float32)
            v_norm = v / (np.linalg.norm(v) + 1e-10)
            scores[atom_id] = float(np.dot(q_norm, v_norm))
        return scores
