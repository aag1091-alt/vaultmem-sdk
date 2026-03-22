"""
VaultMem search index layer.

SearchIndex       — abstract interface for the pre-filter search index.
SQLiteSearchIndex — SQLite-backed implementation (WAL mode).
PostgresSearchIndex — PostgreSQL-backed implementation. Requires: pip install "vaultmem[postgres]"

Privacy tradeoff (explicitly accepted by callers):
    Plaintext in the index: atom_id, tier, memory_type, data_class,
        significance, created_at, session_id, owner, is_churned.
    Encrypted in the index: enc_embedding (AES-256-GCM, keyed with MEK).

The index reveals structural information (what kind of memories exist and
when they were created) but never content. It is a performance layer —
it can always be rebuilt from the blob store given the MEK.

The pre-filter query narrows the candidate set before embeddings are
decrypted and cosine search runs in session RAM.
"""
from __future__ import annotations

import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class IndexRecord:
    """One row in the search index."""
    atom_id:       str
    tier:          str            # Granularity value: ATOM | COMPOSITE | AFFINITY
    memory_type:   str            # MemoryType value: EPISODIC | SEMANTIC | PERSONA | PROCEDURAL
    data_class:    str            # DataClass value: GENERAL | MEDICAL | ARCHIVAL
    significance:  Optional[float]  # None for ATOM/COMPOSITE; precomputed for AFFINITY
    created_at:    int            # Unix timestamp — plaintext (accepted tradeoff)
    session_id:    str
    owner:         str            # plaintext (accepted tradeoff)
    is_churned:    bool
    enc_embedding: bytes          # pack_encrypted_block output; encrypted with MEK


@dataclass
class IndexQuery:
    """Filter criteria for SearchIndex.query()."""
    tier:             Optional[str]   = None   # filter by Granularity value
    memory_type:      Optional[str]   = None   # filter by MemoryType value
    min_significance: Optional[float] = None   # lower bound on significance
    is_churned:       Optional[bool]  = False  # False = active atoms only; None = all
    owner:            Optional[str]   = None   # None = no filter


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class SearchIndex(ABC):
    """Abstract pre-filter search index."""

    @abstractmethod
    def upsert(self, record: IndexRecord) -> None:
        """Insert or replace a record."""

    @abstractmethod
    def query(self, q: IndexQuery) -> list[IndexRecord]:
        """Return IndexRecord list matching all provided filters."""

    @abstractmethod
    def remove(self, atom_id: str) -> None:
        """Remove a record. No-op if atom_id not found."""

    @abstractmethod
    def count(self) -> int:
        """Return total number of records (including churned)."""

    @abstractmethod
    def fetch_many(self, atom_ids: list[str]) -> list[IndexRecord]:
        """Return IndexRecords for the given atom_ids (any order, missing IDs skipped)."""

    @abstractmethod
    def close(self) -> None:
        """Release resources. Idempotent."""

    def migrate_to(self, other: "SearchIndex") -> None:
        """
        Copy all records (including churned) to *other*.

        Use migrate_vault() to move both the blob store and search index
        together in a single call.
        """
        all_records = self.query(IndexQuery(is_churned=None))
        for record in all_records:
            other.upsert(record)


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------

class SQLiteSearchIndex(SearchIndex):
    """
    SQLite-backed search index.

    Schema::

        atoms(atom_id PK, tier, memory_type, data_class, significance,
              created_at, session_id, owner, is_churned, enc_embedding)

    Indexes on tier, significance, is_churned, memory_type for fast filtering.
    WAL journal mode so reads never block writes.
    """

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS atoms (
            atom_id       TEXT    PRIMARY KEY,
            tier          TEXT    NOT NULL,
            memory_type   TEXT    NOT NULL,
            data_class    TEXT    NOT NULL,
            significance  REAL,
            created_at    INTEGER NOT NULL,
            session_id    TEXT    NOT NULL DEFAULT '',
            owner         TEXT    NOT NULL DEFAULT '',
            is_churned    INTEGER NOT NULL DEFAULT 0,
            enc_embedding BLOB    NOT NULL
        );
    """

    _CREATE_INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_tier        ON atoms(tier);",
        "CREATE INDEX IF NOT EXISTS idx_significance ON atoms(significance);",
        "CREATE INDEX IF NOT EXISTS idx_is_churned  ON atoms(is_churned);",
        "CREATE INDEX IF NOT EXISTS idx_memory_type ON atoms(memory_type);",
    ]

    def __init__(self, db_path: "str | Path") -> None:
        self._db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = sqlite3.connect(
            str(self._db_path), check_same_thread=False
        )
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(self._CREATE_TABLE)
        for idx_sql in self._CREATE_INDEXES:
            self._conn.execute(idx_sql)
        self._conn.commit()

    # ── Writes ────────────────────────────────────────────────────────────

    def upsert(self, record: IndexRecord) -> None:
        self._require_open()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO atoms
                    (atom_id, tier, memory_type, data_class, significance,
                     created_at, session_id, owner, is_churned, enc_embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.atom_id,
                    record.tier,
                    record.memory_type,
                    record.data_class,
                    record.significance,
                    record.created_at,
                    record.session_id,
                    record.owner,
                    int(record.is_churned),
                    record.enc_embedding,
                ),
            )
            self._conn.commit()

    def remove(self, atom_id: str) -> None:
        self._require_open()
        with self._lock:
            self._conn.execute("DELETE FROM atoms WHERE atom_id = ?", (atom_id,))
            self._conn.commit()

    # ── Reads ─────────────────────────────────────────────────────────────

    def query(self, q: IndexQuery) -> list[IndexRecord]:
        self._require_open()
        clauses: list[str] = []
        params: list = []

        if q.tier is not None:
            clauses.append("tier = ?")
            params.append(q.tier)
        if q.memory_type is not None:
            clauses.append("memory_type = ?")
            params.append(q.memory_type)
        if q.min_significance is not None:
            clauses.append("significance >= ?")
            params.append(q.min_significance)
        if q.is_churned is not None:
            clauses.append("is_churned = ?")
            params.append(int(q.is_churned))
        if q.owner is not None:
            clauses.append("owner = ?")
            params.append(q.owner)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            f"SELECT atom_id, tier, memory_type, data_class, significance, "
            f"created_at, session_id, owner, is_churned, enc_embedding "
            f"FROM atoms {where}"
        )

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        return [
            IndexRecord(
                atom_id=r[0],
                tier=r[1],
                memory_type=r[2],
                data_class=r[3],
                significance=r[4],
                created_at=r[5],
                session_id=r[6],
                owner=r[7],
                is_churned=bool(r[8]),
                enc_embedding=r[9],
            )
            for r in rows
        ]

    def count(self) -> int:
        self._require_open()
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]

    def fetch_many(self, atom_ids: list[str]) -> list[IndexRecord]:
        self._require_open()
        if not atom_ids:
            return []
        placeholders = ",".join("?" * len(atom_ids))
        sql = (
            f"SELECT atom_id, tier, memory_type, data_class, significance, "
            f"created_at, session_id, owner, is_churned, enc_embedding "
            f"FROM atoms WHERE atom_id IN ({placeholders})"
        )
        with self._lock:
            rows = self._conn.execute(sql, atom_ids).fetchall()
        return [
            IndexRecord(
                atom_id=r[0], tier=r[1], memory_type=r[2], data_class=r[3],
                significance=r[4], created_at=r[5], session_id=r[6],
                owner=r[7], is_churned=bool(r[8]), enc_embedding=r[9],
            )
            for r in rows
        ]

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ── Internal ──────────────────────────────────────────────────────────

    def _require_open(self) -> None:
        if self._conn is None:
            raise RuntimeError("SQLiteSearchIndex is closed")


# ---------------------------------------------------------------------------
# PostgreSQL implementation
# ---------------------------------------------------------------------------

import re

_VALID_IDENTIFIER = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]{0,62}$')


class PostgresSearchIndex(SearchIndex):
    """
    PostgreSQL-backed search index.

    Requires: ``pip install "vaultmem[postgres]"``

    The table is created automatically on first open. Pass *table* to use a
    custom table name (alphanumeric + underscores, max 63 chars).

    Example::

        idx = PostgresSearchIndex("postgresql://user:pass@host/dbname")

    The *dsn* is passed directly to psycopg2.connect() — any libpq-compatible
    connection string or keyword arguments are accepted.
    """

    def __init__(self, dsn: str, *, table: str = "vaultmem_atoms") -> None:
        if not _VALID_IDENTIFIER.match(table):
            raise ValueError(
                f"Invalid table name {table!r}. Use alphanumeric characters and underscores only."
            )
        try:
            import psycopg2
            import psycopg2.extras
            self._psycopg2 = psycopg2
        except ImportError:
            raise ImportError(
                'PostgresSearchIndex requires psycopg2: pip install "vaultmem[postgres]"'
            )
        self._dsn = dsn
        self._table = table
        self._conn = psycopg2.connect(dsn)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        t = self._table
        with self._lock, self._conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {t} (
                    atom_id       TEXT    PRIMARY KEY,
                    tier          TEXT    NOT NULL,
                    memory_type   TEXT    NOT NULL,
                    data_class    TEXT    NOT NULL,
                    significance  REAL,
                    created_at    BIGINT  NOT NULL,
                    session_id    TEXT    NOT NULL DEFAULT '',
                    owner         TEXT    NOT NULL DEFAULT '',
                    is_churned    BOOLEAN NOT NULL DEFAULT FALSE,
                    enc_embedding BYTEA   NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_{t}_tier        ON {t}(tier);
                CREATE INDEX IF NOT EXISTS idx_{t}_significance ON {t}(significance);
                CREATE INDEX IF NOT EXISTS idx_{t}_is_churned  ON {t}(is_churned);
                CREATE INDEX IF NOT EXISTS idx_{t}_memory_type ON {t}(memory_type);
            """)
            self._conn.commit()

    # ── Writes ────────────────────────────────────────────────────────────

    def upsert(self, record: IndexRecord) -> None:
        self._require_open()
        t = self._table
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {t}
                    (atom_id, tier, memory_type, data_class, significance,
                     created_at, session_id, owner, is_churned, enc_embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (atom_id) DO UPDATE SET
                    tier          = EXCLUDED.tier,
                    memory_type   = EXCLUDED.memory_type,
                    data_class    = EXCLUDED.data_class,
                    significance  = EXCLUDED.significance,
                    created_at    = EXCLUDED.created_at,
                    session_id    = EXCLUDED.session_id,
                    owner         = EXCLUDED.owner,
                    is_churned    = EXCLUDED.is_churned,
                    enc_embedding = EXCLUDED.enc_embedding
                """,
                (
                    record.atom_id,
                    record.tier,
                    record.memory_type,
                    record.data_class,
                    record.significance,
                    record.created_at,
                    record.session_id,
                    record.owner,
                    record.is_churned,
                    self._psycopg2.Binary(record.enc_embedding),
                ),
            )
            self._conn.commit()

    def remove(self, atom_id: str) -> None:
        self._require_open()
        with self._lock, self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table} WHERE atom_id = %s", (atom_id,))
            self._conn.commit()

    # ── Reads ─────────────────────────────────────────────────────────────

    def query(self, q: IndexQuery) -> list[IndexRecord]:
        self._require_open()
        clauses: list[str] = []
        params: list = []

        if q.tier is not None:
            clauses.append("tier = %s"); params.append(q.tier)
        if q.memory_type is not None:
            clauses.append("memory_type = %s"); params.append(q.memory_type)
        if q.min_significance is not None:
            clauses.append("significance >= %s"); params.append(q.min_significance)
        if q.is_churned is not None:
            clauses.append("is_churned = %s"); params.append(q.is_churned)
        if q.owner is not None:
            clauses.append("owner = %s"); params.append(q.owner)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            f"SELECT atom_id, tier, memory_type, data_class, significance, "
            f"created_at, session_id, owner, is_churned, enc_embedding "
            f"FROM {self._table} {where}"
        )

        with self._lock, self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            IndexRecord(
                atom_id=r[0],
                tier=r[1],
                memory_type=r[2],
                data_class=r[3],
                significance=r[4],
                created_at=r[5],
                session_id=r[6],
                owner=r[7],
                is_churned=bool(r[8]),
                enc_embedding=bytes(r[9]),
            )
            for r in rows
        ]

    def count(self) -> int:
        self._require_open()
        with self._lock, self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            return cur.fetchone()[0]

    def fetch_many(self, atom_ids: list[str]) -> list[IndexRecord]:
        self._require_open()
        if not atom_ids:
            return []
        sql = (
            f"SELECT atom_id, tier, memory_type, data_class, significance, "
            f"created_at, session_id, owner, is_churned, enc_embedding "
            f"FROM {self._table} WHERE atom_id = ANY(%s)"
        )
        with self._lock, self._conn.cursor() as cur:
            cur.execute(sql, (atom_ids,))
            rows = cur.fetchall()
        return [
            IndexRecord(
                atom_id=r[0], tier=r[1], memory_type=r[2], data_class=r[3],
                significance=r[4], created_at=r[5], session_id=r[6],
                owner=r[7], is_churned=bool(r[8]), enc_embedding=bytes(r[9]),
            )
            for r in rows
        ]

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ── Internal ──────────────────────────────────────────────────────────

    def _require_open(self) -> None:
        if self._conn is None:
            raise RuntimeError("PostgresSearchIndex is closed")
