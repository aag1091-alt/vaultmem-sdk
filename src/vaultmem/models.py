"""
VaultMem memory object model.

Implements the MAS 5-tuple m = (c, e, o, π, v):
  c → content, embedding
  e → type, created_at, session_id, confidence
  o → owner
  π → permissions, data_class
  v → is_churned, supersedes, schema_version
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MemoryType(str, Enum):
    EPISODIC   = "EPISODIC"    # Time-anchored experience or event
    SEMANTIC   = "SEMANTIC"    # Timeless declarative fact
    PERSONA    = "PERSONA"     # Stable preference, trait, or identity
    PROCEDURAL = "PROCEDURAL"  # Workflow or how-to


class Granularity(str, Enum):
    ATOM      = "ATOM"       # Leaf unit — single extracted statement, immutable
    COMPOSITE = "COMPOSITE"  # Type-homogeneous aggregation of ATOMs
    AFFINITY  = "AFFINITY"   # Derived recurring-pattern atom


class DataClass(str, Enum):
    GENERAL  = "GENERAL"   # Default — standard Argon2id params
    MEDICAL  = "MEDICAL"   # Doubled memory_cost, mandatory timeout, mlock
    ARCHIVAL = "ARCHIVAL"  # Slowest significance decay, no auto-prune


class ChurnReason(str, Enum):
    AGE        = "AGE"        # Automatically aged out
    IMPORTANCE = "IMPORTANCE" # Displaced by higher-significance atoms
    SUPERSEDED = "SUPERSEDED" # Replaced by a correction atom
    ABSORBED   = "ABSORBED"   # Rolled into a composite or affinity
    USER       = "USER"       # Explicit user removal


# ---------------------------------------------------------------------------
# Size limits (tokens) — enforced at write time, spec §1.2
# ---------------------------------------------------------------------------

SIZE_LIMITS: dict[tuple[MemoryType, Granularity], tuple[int, int]] = {
    (MemoryType.EPISODIC,   Granularity.ATOM):      (20,  300),
    (MemoryType.EPISODIC,   Granularity.COMPOSITE): (100, 1000),
    (MemoryType.EPISODIC,   Granularity.AFFINITY):  (10,  80),
    (MemoryType.SEMANTIC,   Granularity.ATOM):      (5,   150),
    (MemoryType.SEMANTIC,   Granularity.COMPOSITE): (50,  800),
    (MemoryType.SEMANTIC,   Granularity.AFFINITY):  (10,  80),
    (MemoryType.PERSONA,    Granularity.ATOM):      (5,   100),
    (MemoryType.PERSONA,    Granularity.COMPOSITE): (50,  600),
    (MemoryType.PERSONA,    Granularity.AFFINITY):  (5,   60),
    (MemoryType.PROCEDURAL, Granularity.ATOM):      (20,  500),
    (MemoryType.PROCEDURAL, Granularity.COMPOSITE): (100, 1200),
    (MemoryType.PROCEDURAL, Granularity.AFFINITY):  (10,  80),
}


def estimate_tokens(text: str) -> int:
    """Approximate token count (chars / 4, GPT-style). Used for size enforcement."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Policy object (MAS π field)
# ---------------------------------------------------------------------------

@dataclass
class PolicyObject:
    caregiver_read: bool = True     # Can a caregiver credential slot read this atom?
    export_allowed: bool = True     # Can this atom be included in a portable export?
    retention_days: Optional[int] = None  # None = no expiry

    def to_dict(self) -> dict:
        return {
            "caregiver_read": self.caregiver_read,
            "export_allowed": self.export_allowed,
            "retention_days": self.retention_days,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyObject":
        return cls(
            caregiver_read=d.get("caregiver_read", True),
            export_allowed=d.get("export_allowed", True),
            retention_days=d.get("retention_days"),
        )

    @classmethod
    def default(cls) -> "PolicyObject":
        return cls()


# ---------------------------------------------------------------------------
# MemoryObject — the core atom schema
# ---------------------------------------------------------------------------

@dataclass
class MemoryObject:
    # ── Identity ──────────────────────────────────────────────────────────
    id: str                          # UUID4 string, globally unique, immutable

    # ── Classification (MAS e field) ──────────────────────────────────────
    type: MemoryType
    granularity: Granularity

    # ── Content ───────────────────────────────────────────────────────────
    content: str                     # Memory text (atom), summary (composite), pattern label (affinity)
    size_tokens: int                 # Computed at write time, enforced against SIZE_LIMITS
    embedding: Optional[list[float]] = None  # 384-dim float32 vector; None until embedded

    # ── Structural links ──────────────────────────────────────────────────
    composite_of: list[str] = field(default_factory=list)  # COMPOSITE only: constituent atom IDs
    affinity_of: list[str] = field(default_factory=list)   # AFFINITY only: source atom IDs
    supersedes: list[str] = field(default_factory=list)    # Atoms this correction replaces

    # ── Provenance (MAS e field) ───────────────────────────────────────────
    session_id: str = ""             # Session UUID that produced this memory
    created_at: int = 0             # Unix timestamp, immutable after write

    # ── Affinity-specific fields ──────────────────────────────────────────
    frequency_count: Optional[int] = None      # Number of source atoms (AFFINITY only)
    first_observed_at: Optional[int] = None    # Earliest source atom timestamp (AFFINITY only)
    last_observed_at: Optional[int] = None     # Most recent source atom timestamp (AFFINITY only)
    significance: Optional[float] = None       # Precomputed importance score ∈ (0, 1] (AFFINITY only)

    # ── Quality ───────────────────────────────────────────────────────────
    confidence: float = 1.0         # Extraction confidence: 0.5–1.0

    # ── Lifecycle (MAS v field) ────────────────────────────────────────────
    is_churned: bool = False
    churn_reason: Optional[str] = None  # ChurnReason value as string

    # ── Ownership & policy (MAS o, π fields) ──────────────────────────────
    owner: str = ""
    permissions: PolicyObject = field(default_factory=PolicyObject.default)
    data_class: DataClass = DataClass.GENERAL

    # ── Schema version ────────────────────────────────────────────────────
    schema_version: int = 1

    # ── Language (always per-atom, auto-detected) ─────────────────────────
    language: str = "en"

    # ── Media / multi-modal fields (v2, all optional) ─────────────────────
    content_type: str = "text/plain"          # MIME type; "text/plain" for text atoms
    captured_at: Optional[int] = None         # Real-world event timestamp (≠ created_at)
    media_blob_id: Optional[str] = None       # UUID → encrypted media file in vault/media/
    media_source_path: Optional[str] = None   # Original filename (stored encrypted inside atom)
    location: Optional[dict] = None           # {lat, lon, place} from EXIF or manual

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = int(time.time())
        # Auto-bump schema_version when any media field is set
        if (
            self.content_type != "text/plain"
            or self.captured_at is not None
            or self.media_blob_id is not None
            or self.media_source_path is not None
            or self.location is not None
        ) and self.schema_version < 2:
            self.schema_version = 2

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "granularity": self.granularity.value,
            "content": self.content,
            "size_tokens": self.size_tokens,
            "embedding": self.embedding,
            "composite_of": self.composite_of,
            "affinity_of": self.affinity_of,
            "supersedes": self.supersedes,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "frequency_count": self.frequency_count,
            "first_observed_at": self.first_observed_at,
            "last_observed_at": self.last_observed_at,
            "significance": self.significance,
            "confidence": self.confidence,
            "is_churned": self.is_churned,
            "churn_reason": self.churn_reason,
            "owner": self.owner,
            "permissions": self.permissions.to_dict(),
            "data_class": self.data_class.value,
            "schema_version": self.schema_version,
            "language": self.language,
            "content_type": self.content_type,
            "captured_at": self.captured_at,
            "media_blob_id": self.media_blob_id,
            "media_source_path": self.media_source_path,
            "location": self.location,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryObject":
        return cls(
            id=d["id"],
            type=MemoryType(d["type"]),
            granularity=Granularity(d["granularity"]),
            content=d["content"],
            size_tokens=d["size_tokens"],
            embedding=d.get("embedding"),
            composite_of=d.get("composite_of", []),
            affinity_of=d.get("affinity_of", []),
            supersedes=d.get("supersedes", []),
            session_id=d.get("session_id", ""),
            created_at=d.get("created_at", 0),
            frequency_count=d.get("frequency_count"),
            first_observed_at=d.get("first_observed_at"),
            last_observed_at=d.get("last_observed_at"),
            significance=d.get("significance"),
            confidence=d.get("confidence", 1.0),
            is_churned=d.get("is_churned", False),
            churn_reason=d.get("churn_reason"),
            owner=d.get("owner", ""),
            permissions=PolicyObject.from_dict(d.get("permissions", {})),
            data_class=DataClass(d.get("data_class", "GENERAL")),
            schema_version=d.get("schema_version", 1),
            language=d.get("language", "en"),
            content_type=d.get("content_type", "text/plain"),
            captured_at=d.get("captured_at"),
            media_blob_id=d.get("media_blob_id"),
            media_source_path=d.get("media_source_path"),
            location=d.get("location"),
        )


# ---------------------------------------------------------------------------
# Significance formula (§4.1.1)
# ---------------------------------------------------------------------------

# Decay rates per data class
_LAMBDA: dict[DataClass, float] = {
    DataClass.GENERAL:  0.005,
    DataClass.MEDICAL:  0.002,
    DataClass.ARCHIVAL: 0.0007,
}
_KAPPA = 0.3  # Frequency growth rate, common to all types


def compute_significance(
    freq_count: int,
    last_observed_at: int,
    data_class: DataClass = DataClass.GENERAL,
) -> float:
    """
    significance = (1 − e^(−κ × freq_count)) × e^(−λ × days_since_last)

    κ = 0.3 (frequency saturation)
    λ = data-class-specific decay rate
    Result ∈ (0, 1].
    """
    import math
    days = max(0.0, (time.time() - last_observed_at) / 86400)
    lam = _LAMBDA[data_class]
    freq_factor = 1.0 - math.exp(-_KAPPA * freq_count)
    decay_factor = math.exp(-lam * days)
    return max(1e-6, freq_factor * decay_factor)
