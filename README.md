# VaultMem

**Zero-knowledge encrypted memory for personal AI agents.**

[![PyPI](https://img.shields.io/pypi/v/vaultmem)](https://pypi.org/project/vaultmem/)
[![Python](https://img.shields.io/pypi/pyversions/vaultmem)](https://pypi.org/project/vaultmem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

VaultMem is an embeddable Python library that gives AI agents persistent, encrypted, portable memory — where the **platform operator cryptographically cannot read user data**.

It implements Layer 1 (Personal Storage Layer) of the [Memory-as-Asset framework](https://arxiv.org/abs/2603.14212) (arXiv:2603.14212, March 2026).

**Preprint:** https://doi.org/10.5281/zenodo.19154079 · **Live demo:** https://vaultmem-demo.streamlit.app

---

## The Problem

Every existing AI memory library (mem0, Zep, LangMem, Letta) stores user memories in plaintext on the platform's servers. The platform owns your memory.

VaultMem flips this: the user holds the encryption key. The platform stores opaque ciphertext it cannot read. The guarantee is mathematical, not a privacy policy.

| | mem0 / Zep / Letta | Self-hosted mem0 | VaultMem |
|---|---|---|---|
| Operator can read memories | Yes | Yes (you = operator) | **No** |
| User holds key | No | No | **Yes** |
| Embeddable library | Yes | Yes | **Yes** |
| Portable vault format | No | No | **Yes (.vmem)** |

---

## Install

```bash
pip install vaultmem                  # core: AES-256-GCM, Argon2id, NumPy
pip install "vaultmem[local]"         # + sentence-transformers (LocalEmbedder)
pip install "vaultmem[ann]"           # + hnswlib (HNSWVectorIndex, O(log N) search)
pip install "vaultmem[media]"         # + Pillow, Whisper, PyMuPDF, ffmpeg-python
pip install "vaultmem[s3]"            # + boto3 (S3BlobStore)
pip install "vaultmem[postgres]"      # + psycopg2-binary (PostgresSearchIndex)
```

Core has no ML dependencies — just `cryptography`, `argon2-cffi`, `numpy`.

---

## Quick Start

```python
from vaultmem import VaultSession, LocalEmbedder

embedder = LocalEmbedder()  # all-MiniLM-L6-v2, fully local, no network calls

# Create a new encrypted vault
with VaultSession.create("./my_vault", "s3cr3t", owner="alice",
                          embedder=embedder) as s:
    s.add("I met Bob at the AI conference yesterday")
    s.add("I prefer concise bullet-point answers over long paragraphs")
    s.add("My go-to language for data work is Python")
    s.add("I drink two coffees every morning before coding")
    s.flush()

# Re-open and search — memories persist across sessions
with VaultSession.open("./my_vault", "s3cr3t", embedder=embedder) as s:
    results = s.search("Bob conference", top_k=3)
    for r in results:
        print(f"[{r.tier:<9}] {r.score:.3f}  {r.atom.content}")
```

```
[ATOM     ] 0.847  I met Bob at the AI conference yesterday
```

Wrong passphrase is cryptographically rejected — no bypass:

```python
from vaultmem import WrongPassphraseError

try:
    VaultSession.open("./my_vault", "wrong")
except WrongPassphraseError:
    print("GCM authentication tag mismatch — cannot open vault")
```

---

## Memory Model

### Types

Each atom is classified into one of four memory types (auto-detected from content):

| Type | What it captures | Example |
|------|-----------------|---------|
| `EPISODIC` | Time-anchored events | "I met Bob at the conference yesterday" |
| `SEMANTIC` | Timeless facts and knowledge | "AES-256-GCM provides authenticated encryption" |
| `PERSONA` | Stable preferences and traits | "I prefer dark mode in all my editors" |
| `PROCEDURAL` | Workflows and how-tos | "To deploy: run tests → PR → squash merge → watch logs" |

```python
atom = s.add("I prefer dark mode in all my editors")
print(atom.type)  # MemoryType.PERSONA

# Override classification
from vaultmem import MemoryType
atom = s.add("AES stands for Advanced Encryption Standard", memory_type=MemoryType.SEMANTIC)
```

### Granularity Tiers

| Tier | Description |
|------|-------------|
| `ATOM` | A single leaf memory — immutable once written |
| `COMPOSITE` | A type-homogeneous aggregation of related atoms |
| `AFFINITY` | A recurring-pattern summary derived from repeated atoms; carries a significance score that time-decays when the pattern stops occurring |

### Data Classes

Data class controls KDF parameters and significance decay rate:

| Class | Use case | Argon2id memory | Decay λ (half-life) |
|-------|----------|-----------------|---------------------|
| `GENERAL` (default) | Standard memories | 64 MiB | 0.005 (~5 months) |
| `MEDICAL` | Health data | 128 MiB + mlock | 0.002 (~12 months) |
| `ARCHIVAL` | Long-term records | 64 MiB | 0.0007 (~3 years) |

```python
VaultSession.create("./vault", passphrase, owner="alice", data_class="MEDICAL")
```

---

## Three-Tier Retrieval

`s.search()` runs all three tiers automatically in one call:

```
Tier 1 — AFFINITY:   score = α · significance + (1−α) · cosine
Tier 2 — COMPOSITE:  score = cosine
Tier 3 — ATOM:       score = cosine
```

AFFINITY significance formula:
```
σ = (1 − e^{−κ × freq_count}) × e^{−λ × days_since_last}
  κ = 0.3 (frequency growth rate)
  λ = data-class-specific decay (GENERAL: 0.005)
```

```python
results = s.search("coffee habits", top_k=5)
for r in results:
    print(f"[{r.tier:<9}] score={r.score:.3f}  type={r.atom.type.value}")
    print(f"  {r.atom.content}")
```

A habit mentioned five times in the last week scores higher than a one-off
mention three months ago. The blend weight `alpha` is configurable:

```python
# alpha=0 → pure cosine (ignores significance)
# alpha=1 → pure significance (ignores embedding similarity)
results = s.search("coffee", alpha=0.7)
```

**Benchmark (18-atom corpus):** three-tier retrieval achieves MRR = 1.00 on
pattern queries vs MRR = 0.50 for flat cosine, with zero regression on
specific/factual queries.

---

## Embedders

VaultMem accepts any object with `.embed(text) -> list[float]`. Three are included:

| Class | Install | Privacy | Notes |
|-------|---------|---------|-------|
| `LocalEmbedder` | `vaultmem[local]` | Fully local | all-MiniLM-L6-v2, 384-dim, ~80ms/sentence on CPU |
| `OllamaEmbedder` | Running Ollama | Sends text to Ollama host | Use on localhost or Tailscale |
| `NullEmbedder` | Nothing | N/A | Zero vectors — for testing only |

```python
from vaultmem import LocalEmbedder, OllamaEmbedder, NullEmbedder

# Local — no network, ~2 GB PyTorch download on first use
embedder = LocalEmbedder()

# Ollama — GPU-accelerated, localhost or Tailscale VPN
embedder = OllamaEmbedder("http://localhost:11434", model="all-minilm")

# Custom — duck-typed, no inheritance required
class MyEmbedder:
    def embed(self, text: str) -> list[float]:
        ...  # call your embedding API

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return 384
```

---

## Query Normalizer

For bag-of-words or hash-projection embedders, question framing dilutes the
semantic signal. `"What do I know about Sarah Chen?"` has equal token weight on
"what", "do", "i", "know", "about" as on "Sarah", "Chen".

VaultMem exposes a `QueryNormalizer` Protocol so you can preprocess queries
before they reach the embedder.

### Built-in (regex)

```python
from vaultmem import VaultSession, RegexQueryNormalizer

with VaultSession.open("./vault", passphrase,
                        query_normalizer=RegexQueryNormalizer()) as s:
    # "What do I know about Sarah Chen?" → embedded as "Sarah Chen"
    # "How does Flash Attention work?"   → embedded as "Flash Attention work"
    # "Am I vegetarian?"                 → embedded as "vegetarian"
    results = s.search("What do I know about Sarah Chen?", normalize_query=True)
```

Leave `normalize_query=False` (the default) when using `LocalEmbedder` —
sentence-transformers handle question framing natively via attention.

### Custom (LLM-backed)

Implement `normalize(self, text: str) -> str` — no import or base class needed:

```python
class GroqQueryNormalizer:
    """Runs on your own Groq key — query text never leaves your process."""

    def __init__(self, api_key: str) -> None:
        from groq import Groq
        self._client = Groq(api_key=api_key)

    def normalize(self, text: str) -> str:
        resp = self._client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=30,
            messages=[
                {"role": "system",
                 "content": "Extract key search terms. Return only keywords."},
                {"role": "user", "content": text},
            ],
        )
        return resp.choices[0].message.content.strip() or text


with VaultSession.open("./vault", passphrase,
                        query_normalizer=GroqQueryNormalizer("gsk_...")) as s:
    results = s.search("What do I know about Sarah?", normalize_query=True)
```

---

## Temporal Search

Every atom carries two timestamps:
- `created_at` — when it was written to the vault (immutable, set automatically)
- `captured_at` — the real-world event time (e.g. EXIF date on a photo, or set manually)

### Browse by time window

```python
from datetime import datetime, timezone

def ts(year, month=1, day=1):
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())

with VaultSession.open("./vault", passphrase, embedder=embedder) as s:
    # All memories from 2019
    memories_2019 = s.search_by_time(ts(2019), ts(2020))

    # Q3 2021
    q3 = s.search_by_time(ts(2021, 7), ts(2021, 10))

    # Atoms written to the vault in the last 24 hours (by created_at)
    import time
    recent = s.diff(int(time.time()) - 86400, int(time.time()))
```

### Set real-world event time

```python
with VaultSession.open("./vault", passphrase, embedder=embedder) as s:
    atom = s.add("Visited Tokyo for the first time — cherry blossom season.")
    atom.captured_at = ts(2019, 4, 8)   # backdate to the real event
    s.flush()
```

### Natural-language time phrases in queries

```python
from vaultmem import TimeQueryParser

from_ts, to_ts, remainder = TimeQueryParser.parse("what happened summer 2019")
# from_ts  → 2019-06-01 UTC
# to_ts    → 2019-09-01 UTC
# remainder → "what happened"

# Supported phrases:
# "summer 2019", "March 2020", "in 2019", "last year",
# "this month", "last month", "yesterday", "last 30 days"
```

### Semantic search with automatic time filtering

```python
with VaultSession.open("./vault", passphrase, embedder=embedder) as s:
    # Strips "in 2019" → timestamp range, embeds "what did I do"
    results = s.search("what did I do in 2019", top_k=10, parse_time=True)
```

---

## Multi-Modal Memory

Ingest images, audio, PDFs, and video as encrypted memory atoms.

```bash
pip install "vaultmem[media]"
```

```python
from vaultmem import VaultSession, LocalEmbedder

with VaultSession.open("./vault", passphrase, embedder=LocalEmbedder()) as s:

    # Image — extracts EXIF date, GPS, optional OCR
    atom = s.add_media("holiday_photo.jpg", passphrase)
    print(atom.content)         # "Photo taken 2023-07-14 in Split, Croatia"
    print(atom.captured_at)     # EXIF timestamp as Unix int
    print(atom.location)        # {"lat": 43.5, "lon": 16.4, "place": "Split"}

    # Audio — Whisper transcription
    atom = s.add_media("voice_note.mp3", passphrase)

    # PDF — PyMuPDF text extraction
    atom = s.add_media("contract.pdf", passphrase)

    # Batch
    atoms = s.add_media_batch(["img1.jpg", "img2.jpg", "notes.pdf"], passphrase)

    # Retrieve raw bytes (decrypted on demand)
    raw = s.get_media(atom.id, passphrase)
```

Media bytes are stored encrypted at `vault/media/{uuid}.enc`. The atom content
is the extracted text; the raw file is fetched separately only when needed.

### Custom extractor

```python
from vaultmem import MediaExtractor, MediaExtractionResult

class HeicExtractor:
    MIME_TYPES = {"image/heic", "image/heif"}

    def extract(self, path, mime_type):
        # your extraction logic
        return MediaExtractionResult(
            content_type=mime_type,
            transcript="HEIC photo from iPhone",
            captured_at=...,
        )

from vaultmem import MediaIngester
ingester = MediaIngester(extra_extractors=[HeicExtractor()])
```

---

## Pluggable Storage Backends

By default VaultMem stores everything locally. Swap in production backends
without changing any other code.

```bash
pip install "vaultmem[s3]"        # S3BlobStore
pip install "vaultmem[postgres]"  # PostgresSearchIndex
pip install "vaultmem[ann]"       # HNSWVectorIndex
```

```python
from vaultmem import (
    VaultSession, LocalEmbedder,
    FileBlobStore, S3BlobStore,
    SQLiteSearchIndex, PostgresSearchIndex,
    HNSWVectorIndex,
)

# Local (default, no extra dependencies)
with VaultSession.open(
    "./vault", passphrase,
    embedder=LocalEmbedder(),
    blob_store=FileBlobStore("./vault/atoms"),
    search_index=SQLiteSearchIndex("./vault/index.db"),
) as s:
    ...

# Production: S3 + Postgres + HNSW ANN
with VaultSession.open(
    "./vault", passphrase,
    embedder=LocalEmbedder(),
    blob_store=S3BlobStore(bucket="my-vault-atoms"),
    search_index=PostgresSearchIndex("postgresql://user:pass@host/db"),
    vector_index=HNSWVectorIndex(),  # O(log N) ANN search
) as s:
    ...
```

### HNSW vector index

At small scale (< 1,000 atoms) VaultMem uses exact cosine search. For large
vaults, `HNSWVectorIndex` drops search latency from ~148ms to < 1ms at 10,000
atoms while preserving the same top-k accuracy.

```python
from vaultmem import HNSWVectorIndex

vi = HNSWVectorIndex(ef_construction=200, M=16)

with VaultSession.open("./vault", passphrase,
                        blob_store=blob_store,
                        search_index=search_index,
                        vector_index=vi) as s:
    results = s.search("coffee", top_k=10)  # O(log N) via HNSW graph
```

The HNSW graph is serialized to `vault/vector_index.hnsw.enc` (AES-256-GCM)
and reloaded on session open.

### Custom backends

Implement the ABCs to plug in any storage layer:

```python
from vaultmem import BlobStore, SearchIndex

class RedisBlobStore(BlobStore):
    def put(self, atom_id: str, ciphertext: bytes) -> None: ...
    def get(self, atom_id: str) -> bytes: ...
    def delete(self, atom_id: str) -> None: ...
    def exists(self, atom_id: str) -> bool: ...

class ElasticsearchIndex(SearchIndex):
    def upsert(self, record) -> None: ...
    def query(self, iq) -> list: ...
    def fetch_many(self, ids) -> list: ...
    def count(self) -> int: ...
    def close(self) -> None: ...
```

### Migrate between backends

```python
from vaultmem import migrate_vault

# Copy encrypted bytes opaquely — no decryption required
migrate_vault(
    src_blob_store=FileBlobStore("./old"),
    dst_blob_store=S3BlobStore(bucket="new-vault"),
    passphrase=passphrase,
)
```

---

## Vault Format

```
my_vault/
├── meta.json              # KDF params, credential slots, owner (plaintext)
├── current.vmem           # encrypted index (AES-256-GCM, self-describing header)
├── current.atoms          # append-only encrypted atom blocks
├── session.lock           # OS-level exclusive lock (flock / LockFileEx)
├── snapshots/             # named checkpoints (optional)
│   └── v1.vmem
└── media/                 # encrypted raw media files (optional)
    └── {uuid}.enc
```

The `.vmem` header is plaintext (48 bytes): magic `VMEM`, version, algorithm
IDs, offsets, atom count, file UUID. Everything else is ciphertext. The format
is language-agnostic — any implementation can read a vault given the passphrase.

---

## Session Lifecycle

```python
# Context manager — MEK is zeroed automatically on exit
with VaultSession.open("./vault", passphrase, embedder=embedder) as s:
    s.add("new memory")
    s.flush()                # checkpoint to disk mid-session (optional)
    results = s.search("query")
    print(s.atom_count)
    print(s.owner)
# MEK zeroed here — vault unlocked

# Manual management
s = VaultSession.open("./vault", passphrase, embedder=embedder)
try:
    s.add("memory")
    s.flush()
finally:
    s.close()   # always call — zeros MEK and releases file lock
```

Only one session can hold a vault open at a time (`flock(LOCK_EX)` on POSIX,
`LockFileEx` on Windows). Attempting to open a locked vault raises
`VaultLockedError`.

### Snapshots

```python
with VaultSession.open("./vault", passphrase, embedder=embedder) as s:
    s.take_snapshot("before-migration")
    # ... do risky operations ...
    s.restore_snapshot("before-migration")
    print(s.list_snapshots())   # ["before-migration"]
```

### Export and erasure

```python
with VaultSession.open("./vault", passphrase, embedder=embedder) as s:
    # GDPR-compliant full erasure — overwrites MEK, makes vault unreadable
    s.erase()
```

---

## Security Properties

| Property | How it's enforced |
|----------|------------------|
| Platform cannot read memories | AES-256-GCM; MEK never written to disk |
| Tampering is detected | GCM authentication tag per atom — single flipped bit causes decryption failure |
| Wrong passphrase is rejected | MEK unwrap GCM auth failure → `WrongPassphraseError` |
| Brute-force is expensive | Argon2id: 64 MiB / 3 iterations / 4 threads per attempt |
| Single writer | `flock(LOCK_EX)` on POSIX, `LockFileEx` on Windows |
| MEK zeroed on close | `bytearray` byte-by-byte zero fill |
| Ciphertext transplant blocked | Atom UUID bound as AAD — swapping encrypted blobs is detected |
| Atomic checkpoint | `fsync` + `os.rename()` — crash-safe |
| Passphrase rotation | O(1) — re-wrap MEK only, zero atom re-encryption |

---

## Full API Reference

### `VaultSession`

```python
# Factory methods
VaultSession.create(
    vault_dir,           # str | Path
    passphrase,          # str
    owner,               # str
    data_class="GENERAL",
    embedder=None,       # Embedder — defaults to LocalEmbedder()
    *,
    blob_store=None,     # BlobStore
    search_index=None,   # SearchIndex
    vector_index=None,   # VectorIndex
    query_normalizer=None,  # QueryNormalizer
) -> VaultSession

VaultSession.open(
    vault_dir,
    passphrase,
    embedder=None,
    *,
    blob_store=None,
    search_index=None,
    vector_index=None,
    query_normalizer=None,
) -> VaultSession

# Memory operations
session.add(text, *, memory_type=None) -> MemoryObject
session.add_media(path, passphrase, *, override_captured_at=None) -> MemoryObject
session.add_media_batch(paths, passphrase) -> list[MemoryObject]
session.get_media(atom_id, passphrase) -> bytes

# Search
session.search(
    query,               # str | list[float] (pre-computed embedding)
    *,
    top_k=10,
    memory_type=None,    # filter to one MemoryType
    alpha=0.5,           # AFFINITY blend weight
    parse_time=False,    # strip time phrases, pre-filter by captured_at
    normalize_query=False,  # strip question preamble before embedding
) -> list[SearchResult]

session.search_by_time(from_ts, to_ts, *, top_k=None) -> list[MemoryObject]
session.diff(from_ts, to_ts) -> list[MemoryObject]

# Lifecycle
session.flush() -> None          # checkpoint to disk without closing
session.close() -> None          # flush + zero MEK + release lock

# Vault management
session.take_snapshot(name) -> None
session.restore_snapshot(name) -> None
session.list_snapshots() -> list[str]
session.erase() -> None          # GDPR full erasure

# Properties
session.atom_count -> int
session.owner -> str
```

### `SearchResult`

```python
@dataclass
class SearchResult:
    atom: MemoryObject
    score: float    # blended AFFINITY score or cosine ∈ [-1, 1]
    tier: str       # "AFFINITY" | "COMPOSITE" | "ATOM"
```

### `MemoryObject` (key fields)

```python
@dataclass
class MemoryObject:
    id: str                  # UUID4, globally unique
    type: MemoryType         # EPISODIC | SEMANTIC | PERSONA | PROCEDURAL
    granularity: Granularity # ATOM | COMPOSITE | AFFINITY
    content: str             # memory text
    embedding: list[float]   # 384-dim vector

    # Provenance
    created_at: int          # Unix timestamp — when written to vault
    captured_at: int | None  # Unix timestamp — real-world event time
    session_id: str          # session that produced this atom

    # Affinity-specific
    frequency_count: int | None   # how many source atoms
    significance: float | None    # σ = (1−e^{−κf}) × e^{−λd}

    # Media
    content_type: str        # MIME type, default "text/plain"
    media_blob_id: str | None
    location: dict | None    # {"lat": ..., "lon": ..., "place": ...}

    # Policy
    owner: str
    data_class: DataClass    # GENERAL | MEDICAL | ARCHIVAL
    confidence: float        # 0.5–1.0
    is_churned: bool
```

### Extension Protocols and ABCs

| Name | Type | Purpose |
|------|------|---------|
| `Embedder` | Protocol | `embed(text) -> list[float]` |
| `QueryNormalizer` | Protocol | `normalize(text) -> str` |
| `MediaExtractor` | Protocol | `extract(path, mime) -> MediaExtractionResult` |
| `BlobStore` | ABC | encrypted atom storage (`put`, `get`, `delete`, `exists`) |
| `SearchIndex` | ABC | pre-filter metadata index (`upsert`, `query`, `fetch_many`) |
| `VectorIndex` | ABC | ANN graph (`add`, `search`, `save`, `load`) |

### Built-in implementations

| Protocol/ABC | Built-in | Extra |
|---|---|---|
| `Embedder` | `LocalEmbedder`, `OllamaEmbedder`, `NullEmbedder` | — |
| `QueryNormalizer` | `RegexQueryNormalizer` | — |
| `MediaExtractor` | `ImageExtractor`, `AudioExtractor`, `DocumentExtractor`, `VideoExtractor` | `[media]` |
| `BlobStore` | `FileBlobStore` | `S3BlobStore` (`[s3]`) |
| `SearchIndex` | `SQLiteSearchIndex` | `PostgresSearchIndex` (`[postgres]`) |
| `VectorIndex` | — | `HNSWVectorIndex` (`[ann]`) |

### Exceptions

```python
from vaultmem import (
    WrongPassphraseError,   # bad passphrase (GCM tag mismatch)
    VaultTamperedError,     # atom authentication failed
    VaultLockedError,       # another process holds the session lock
    VaultAlreadyOpenError,  # this process already has the vault open
    SessionStateError,      # operation called on wrong state (e.g. closed session)
    MemorySchemaError,      # atom content violates size limits
    RotationRequiredError,  # vault KDF params are below current minimums
)
```

---

## Running Tests

```bash
pip install "vaultmem[dev]"
pytest tests/test_unit.py -v           # 39 unit tests, Python 3.10–3.13
python tests/bench_affinity.py         # AFFINITY tier uplift vs flat cosine
python tests/bench_section7.py         # crypto + search latency table (§7 of paper)
python tests/bench_recall.py           # Recall@10 / MRR on LoCoMo-10 benchmark
```

---

## Examples

| Script | What it shows |
|--------|--------------|
| `examples/demo_01_quickstart.py` | Create vault, add memories, search, wrong passphrase |
| `examples/demo_02_temporal.py` | `captured_at`, `search_by_time`, `diff`, `TimeQueryParser`, `parse_time` |
| `examples/demo_03_media.py` | Image/audio/PDF ingestion, EXIF, GPS, Whisper |
| `examples/demo_backends.py` | `FileBlobStore` + `SQLiteSearchIndex` + `HNSWVectorIndex` |

---

## Background

VaultMem is the reference implementation of the Personal Storage Layer described in:

> **Memory as Asset: Towards User-Owned Persistent AI Memory**
> Gosavi — arXiv:2603.14212, Zenodo DOI: 10.5281/zenodo.19154079, March 2026

The paper defines a three-layer architecture for user-owned AI memory. VaultMem
implements Layer 1. Layers 2 (collaborative memory groups) and 3 (decentralized
memory exchange) are future work.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Contributing

Issues and pull requests welcome. The library is intentionally focused (~4,500
lines across 13 modules) — the code is the specification.
