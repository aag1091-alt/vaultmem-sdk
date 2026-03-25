# VaultMem SDK Architecture — v0.2.0

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        PLATFORM DEVELOPER                                ║
║   (embeds VaultMem into their app — never sees plaintext user data)      ║
╚════════════════════════════╤═════════════════════════════════════════════╝
                             │  Python API
                             ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                         VaultSession                          session.py ║
║                                                                          ║
║  .create(vault_dir, passphrase, owner)   .open(vault_dir, passphrase)   ║
║  .add(text)            .add_media(path)  .add_media_batch(paths)         ║
║  .search(query)        .search(parse_time=True)                          ║
║  .search_by_time(from_ts, to_ts)         .diff(from_ts, to_ts)           ║
║  .get_media(media_id)  .flush()          .close()                        ║
║                                                                          ║
║  State: CLOSED → OPEN → CHECKPOINTING → CLOSING → CLOSED                ║
║  MEK lives only here (bytearray, zeroed on close)                        ║
╚══════╤═══════════╤══════════════╤════════════════╤════════════════╤══════╝
       │           │              │                │                │
       ▼           ▼              ▼                ▼                ▼
  ┌─────────┐ ┌─────────┐  ┌──────────┐   ┌────────────┐   ┌───────────┐
  │Classifi-│ │Embedder │  │  Media   │   │TimeQuery   │   │ Retrieval │
  │er       │ │         │  │Ingester  │   │Parser      │   │           │
  │         │ │Local    │  │          │   │            │   │3-tier     │
  │4-feature│ │Embedder │  │Image     │   │"summer     │   │search:    │
  │rule-    │ │(all-    │  │Extractor │   │ 2019"      │   │AFFINITY   │
  │based:   │ │MiniLM)  │  │Audio     │   │→ unix ts   │   │COMPOSITE  │
  │EPISODIC │ │Ollama   │  │Extractor │   │+ remainder │   │ATOM       │
  │SEMANTIC │ │Embedder │  │Document  │   │            │   │           │
  │PERSONA  │ │Null     │  │Extractor │   │            │   │cosine +   │
  │PROCEDU- │ │Embedder │  │Video     │   │            │   │signific.  │
  │RAL      │ │         │  │Extractor │   │            │   │weighted   │
  └────┬────┘ └────┬────┘  └────┬─────┘   └────────────┘   └───────────┘
       │           │            │
       └───────────┴────────────┘
                   │  builds
                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                        MemoryObject                           models.py  ║
║                                                                          ║
║  id           type (EPISODIC│SEMANTIC│PERSONA│PROCEDURAL)                ║
║  granularity  (ATOM│COMPOSITE│AFFINITY)                                  ║
║  content      size_tokens    embedding (384-dim float32)                 ║
║  ── media fields (v0.2, all optional) ─────────────────────────────────  ║
║  content_type   captured_at  (real-world event ts, ≠ created_at)         ║
║  media_blob_id  media_source_path   location {lat, lon, place}           ║
║  ── provenance ────────────────────────────────────────────────────────  ║
║  owner   session_id   created_at   confidence   data_class   permissions ║
║  is_churned   schema_version (auto=2 when any media field set)   language║
╚══════════════════════════════════════════════════════════════════════════╝
                   │ encrypted by
                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                        Crypto Layer                           crypto.py  ║
║                                                                          ║
║  passphrase                                                              ║
║      │                                                                   ║
║      ▼  Argon2id(passphrase, salt, 64MiB, 3×, 4‖)                       ║
║     KEK  ──────────────────────────── zeroed immediately after MEK unwrap║
║      │ unwrap                                                            ║
║      ▼                                                                   ║
║     MEK  (32 random bytes, generated once at vault creation)             ║
║      │    lives only in VaultSession RAM as bytearray                    ║
║      │    zeroed on session.close()                                      ║
║      │                                                                   ║
║      ├── AES-256-GCM(MEK, atom_json,  AAD=uuid_bytes) → atom ciphertext ║
║      ├── AES-256-GCM(MEK, embedding,  AAD=uuid_bytes) → enc_embedding   ║
║      ├── AES-256-GCM(MEK, index_json, AAD=header)     → index block     ║
║      └── AES-256-GCM(MEK, raw_media,  AAD=uuid_bytes) → media blob      ║
║                                                                          ║
║  Each ciphertext: [12B IV] [4B ct_len] [N bytes CT] [16B GCM tag]       ║
╚═══════════╤══════════════════════════════════╤═══════════════════════════╝
            │                                  │
            ▼                                  ▼
  ┌──────────────────────┐       ┌─────────────────────────────────────┐
  │   File mode          │       │          Backend mode               │
  │  (default)           │       │    (scalable, pass blob_store=)     │
  │                      │       │                                     │
  │  vault.py            │       │  storage.py       index.py          │
  │                      │       │                                     │
  │  current.vmem        │       │  BlobStore   ←→  SearchIndex        │
  │  ├─ 48B header       │       │  │               │                  │
  │  └─ encrypted index  │       │  FileBlobStore   SQLiteSearchIndex  │
  │                      │       │  S3BlobStore     PostgresSearchIndex│
  │  current.atoms       │       │                                     │
  │  └─ append-only      │       │  Optional:                          │
  │     encrypted blobs  │       │  vector_index.py                    │
  │                      │       │  HNSWVectorIndex (ANN, O(log N))    │
  │  Crash-safe:         │       │                                     │
  │  1. append+fsync     │       │                                     │
  │  2. index tmp+rename │       │                                     │
  └──────────┬───────────┘       └────────────────────────────────────-┘
             │
             ▼
  vault_dir/media/
  {uuid}.enc  ← encrypted raw file (JPEG, PDF, MP3, MP4 …)
               AES-256-GCM, AAD = media UUID bytes
               retrieved via session.get_media(media_id)
```

---

## Reading it top-to-bottom

### 1. Platform Developer
The only consumer of the SDK. Calls `VaultSession` on behalf of their user. Never has access to plaintext content — only the user's passphrase unlocks the MEK.

### 2. VaultSession (`session.py`)
Single public entry point. Manages the state machine. Holds the MEK in RAM as a `bytearray` and zeros it on `close()`. Orchestrates all subsystems.

### 3. Five subsystems

| Module | Role |
|---|---|
| `classifier.py` | 4-feature rule classifier → `EPISODIC / SEMANTIC / PERSONA / PROCEDURAL`. No LLM required. |
| `embedder.py` | Pluggable: `LocalEmbedder` (sentence-transformers, 384-dim), `OllamaEmbedder`, `NullEmbedder`. |
| `media.py` | `MediaIngester` dispatches to extractors. `TimeQueryParser` — pure Python, zero deps. |
| `retrieval.py` | Three-tier cosine search: AFFINITY → COMPOSITE → ATOM. Blends cosine with significance score. |
| `crypto.py` | Argon2id KDF + AES-256-GCM. MEK wrap/unwrap. Per-atom AAD binds ciphertext to identity. |

### 4. MemoryObject (`models.py`)
Single schema for every memory regardless of type or modality. A photo atom and a text atom are the same struct — photos just have `media_blob_id` pointing to their encrypted file and `captured_at` holding the EXIF date. `schema_version` auto-bumps to 2 when any media field is set; old vaults read fine with `d.get(field, default)`.

### 5. Crypto layer (`crypto.py`)
```
passphrase
  │
  ▼  Argon2id (64 MiB, 3 iterations, 4 parallelism)
 KEK  — zeroed immediately after MEK unwrap
  │
  ▼  AES-256-GCM decrypt
 MEK  — lives in VaultSession RAM only, zeroed on close()
  │
  ├── encrypts every atom        (AAD = atom UUID bytes)
  ├── encrypts every embedding   (same AAD)
  ├── encrypts the vault index   (AAD = file header)
  └── encrypts every media file  (AAD = media UUID bytes)
```
Key insight: MEK is generated once at vault creation. Changing the passphrase only re-wraps the MEK — zero atom re-encryption needed.

### 6. Storage

**File mode (default — simple vaults):**
- Atoms held in `MemoryState` (RAM dict) during the session
- `flush()` → append new atom blobs to `current.atoms` → fsync → rewrite encrypted index in `current.vmem` atomically (tmp + rename)
- O(new atoms) per checkpoint; existing atom blocks never touched (immutable)

**Backend mode (scalable vaults):**
- Pass `blob_store=` and `search_index=` to `VaultSession`
- `FileBlobStore` or `S3BlobStore` for atom blobs
- `SQLiteSearchIndex` or `PostgresSearchIndex` for metadata + encrypted embeddings
- Optional `HNSWVectorIndex` for O(log N) ANN search

**Media (both modes):**
- Raw files encrypted to `vault_dir/media/{uuid}.enc`
- Atoms carry `media_blob_id` pointer
- Retrieved and decrypted on demand via `session.get_media(id)`

### 7. Temporal search
- `captured_at` = when the real-world event happened (EXIF date, audio metadata, manual)
- `created_at` = when the atom was written to the vault
- `search_by_time(from_ts, to_ts)` — browse by event time (the "vault" experience)
- `diff(from_ts, to_ts)` — what was written to the vault in a time window (sync/audit)
- `search(query, parse_time=True)` — `TimeQueryParser` strips "summer 2019" from the query, pre-filters candidates, then runs semantic search on the remainder

---

## File map

```
src/vaultmem/
├── __init__.py      Public API surface — all exports
├── session.py       VaultSession state machine
├── models.py        MemoryObject dataclass + enums + significance formula
├── crypto.py        Argon2id KDF, AES-256-GCM, MEK wrapping
├── vault.py         .vmem/.atoms format, crash-safe checkpoint protocol
├── classifier.py    4-feature deterministic memory type classifier
├── embedder.py      Embedder protocol + LocalEmbedder / OllamaEmbedder / NullEmbedder
├── retrieval.py     Three-tier cosine search + significance weighting
├── media.py         MediaIngester, extractors (Image/Audio/Document/Video), TimeQueryParser
├── storage.py       BlobStore ABC + FileBlobStore + S3BlobStore
├── index.py         SearchIndex ABC + SQLiteSearchIndex + PostgresSearchIndex
├── vector_index.py  VectorIndex ABC + HNSWVectorIndex (ANN)
└── exceptions.py    All custom exception types

examples/
├── demo_01_quickstart.py   Create vault, add text, search, wrong-passphrase rejection
├── demo_02_temporal.py     captured_at, search_by_time, diff, TimeQueryParser, parse_time
├── demo_03_media.py        ImageExtractor EXIF, add_media, get_media, add_media_batch
└── demo_backends.py        FileBlobStore + SQLiteSearchIndex scalable mode
```

---

## What a memory looks like on disk

```
vault_dir/
├── meta.json          ← credential material (kdf_salt, wrapped MEK) — plaintext
├── session.lock       ← exclusive flock, contains PID
├── current.vmem       ← 48B header + encrypted index (atom offsets + owner)
├── current.atoms      ← append-only encrypted atom blobs
└── media/
    ├── {uuid}.enc     ← encrypted JPEG / MP3 / PDF / MP4 …
    └── {uuid}.enc
```

Every byte in `current.atoms` and `media/*.enc` is AES-256-GCM ciphertext. The platform developer cannot read any of it without the user's passphrase.
