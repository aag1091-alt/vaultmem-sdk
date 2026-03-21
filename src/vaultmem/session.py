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
    unwrap_mek,
)
from .embedder import Embedder, LocalEmbedder
from .exceptions import SessionStateError, VaultLockedError
from .models import DataClass, Granularity, MemoryObject, MemoryType, estimate_tokens
from .retrieval import SearchResult
from .retrieval import search as _search
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
        memory_state: MemoryState,
        mek: bytearray,
        data_class: DataClass,
        lock_fd: Optional[int],
        embedder: Embedder,
        session_id: str,
    ) -> None:
        self._vault_dir  = vault_dir
        self._mem        = memory_state
        self._mek        = mek         # bytearray — zeroed on close
        self._data_class = data_class
        self._lock_fd    = lock_fd
        self._embedder   = embedder
        self._session_id = session_id
        self._state      = SessionState.OPEN
        self._rlock      = threading.RLock()

    # ── Factory: open existing vault ───────────────────────────────────────

    @classmethod
    def open(
        cls,
        vault_dir: "str | Path",
        passphrase: str,
        embedder: Optional[Embedder] = None,
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
            mem_type  = classify(content)
            size_tok  = estimate_tokens(content)
            atom = MemoryObject(
                id="",                          # UUID assigned in __post_init__
                type=mem_type,
                granularity=Granularity.ATOM,
                content=content,
                size_tokens=size_tok,
                session_id=self._session_id,
                confidence=confidence,
                owner=owner or self._mem.owner,
                data_class=self._data_class,
            )
            if auto_embed:
                atom.embedding = self._embedder.embed(content)

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
            if self._mem.is_dirty:
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
                if self._mem.is_dirty:
                    self._checkpoint()
            finally:
                self._zero_mek()
                self._release_lock(self._lock_fd)
                self._state = SessionState.CLOSED

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def owner(self) -> str:
        """Vault owner string."""
        return self._mem.owner

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
        """Number of active (non-churned) atoms currently loaded."""
        with self._rlock:
            return len(self._mem.active_atoms)

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "VaultSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"VaultSession(owner={self.owner!r}, "
            f"state={self._state.value}, "
            f"atoms={len(self._mem.atoms)})"
        )
