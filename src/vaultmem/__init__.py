"""
VaultMem — Zero-knowledge encrypted memory library for personal AI agents.

A zero-knowledge encrypted memory layer that any platform developer can embed
to give their users persistent, portable, private AI memory. The platform
operator cryptographically cannot read user data.

Implements the Personal Storage Layer (Layer 1) of the Memory-as-Asset
framework (Pan, Huang & Yang, arXiv:2603.14212).

Quick start::

    from vaultmem import VaultSession

    # Create a new vault
    with VaultSession.create("./my_vault", passphrase="secret", owner="alice") as s:
        s.add("I met Bob at the AI conference yesterday")
        s.add("I prefer concise answers")

    # Re-open and search
    with VaultSession.open("./my_vault", passphrase="secret") as s:
        results = s.search("Bob conference")
        for r in results:
            print(f"[{r.tier}] {r.score:.3f}  {r.atom.content}")

GitHub: https://github.com/aag1091-alt/vaultmem
"""

__version__ = "0.3.0"
__author__ = "Avinash Gosavi"

# ── Core session ──────────────────────────────────────────────────────────────
from .session import VaultSession, SessionState

# ── Retrieval result ──────────────────────────────────────────────────────────
from .retrieval import SearchResult

# ── Data model ───────────────────────────────────────────────────────────────
from .models import (
    MemoryObject,
    MemoryType,
    Granularity,
    DataClass,
    ChurnReason,
    PolicyObject,
    compute_significance,
)

# ── Embedders ─────────────────────────────────────────────────────────────────
from .embedder import Embedder, LocalEmbedder, NullEmbedder, OllamaEmbedder

# ── Storage backends ──────────────────────────────────────────────────────────
from .storage import BlobStore, FileBlobStore, S3BlobStore, migrate_vault
from .index import SearchIndex, SQLiteSearchIndex, PostgresSearchIndex, IndexRecord, IndexQuery
from .vector_index import VectorIndex, HNSWVectorIndex

# ── Media ingestion + temporal search ────────────────────────────────────────
from .media import (
    MediaExtractionResult,
    MediaExtractor,
    MediaIngester,
    ImageExtractor,
    AudioExtractor,
    DocumentExtractor,
    VideoExtractor,
    TimeQueryParser,
    QueryNormalizer,
    RegexQueryNormalizer,
)

# ── Sanitizer ────────────────────────────────────────────────────────────────
from .sanitize import Sanitizer

# ── Classifier ───────────────────────────────────────────────────────────────
from .classifier import classify, classify_with_features

# ── Exceptions ───────────────────────────────────────────────────────────────
from .exceptions import (
    VaultMemError,
    WrongPassphraseError,
    VaultTamperedError,
    VaultLockedError,
    VaultAlreadyOpenError,
    SessionStateError,
    MemorySchemaError,
    RotationRequiredError,
)

__all__ = [
    # Session
    "VaultSession",
    "SessionState",
    # Retrieval
    "SearchResult",
    # Models
    "MemoryObject",
    "MemoryType",
    "Granularity",
    "DataClass",
    "ChurnReason",
    "PolicyObject",
    "compute_significance",
    # Embedders
    "Embedder",
    "LocalEmbedder",
    "NullEmbedder",
    "OllamaEmbedder",
    # Storage backends
    "BlobStore",
    "FileBlobStore",
    "S3BlobStore",
    "migrate_vault",
    "SearchIndex",
    "SQLiteSearchIndex",
    "PostgresSearchIndex",
    "IndexRecord",
    "IndexQuery",
    # Vector index
    "VectorIndex",
    "HNSWVectorIndex",
    # Media
    "MediaExtractionResult",
    "MediaExtractor",
    "MediaIngester",
    "ImageExtractor",
    "AudioExtractor",
    "DocumentExtractor",
    "VideoExtractor",
    "TimeQueryParser",
    "QueryNormalizer",
    "RegexQueryNormalizer",
    # Sanitizer
    "Sanitizer",
    # Classifier
    "classify",
    "classify_with_features",
    # Exceptions
    "VaultMemError",
    "WrongPassphraseError",
    "VaultTamperedError",
    "VaultLockedError",
    "VaultAlreadyOpenError",
    "SessionStateError",
    "MemorySchemaError",
    "RotationRequiredError",
]
