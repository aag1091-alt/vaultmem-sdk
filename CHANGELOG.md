# Changelog

All notable changes to VaultMem are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.2.1] — March 2026

### Added

**Pluggable query normalizer**
- `QueryNormalizer` Protocol — implement `normalize(self, text: str) -> str` to preprocess search queries before they reach the embedder; injected at session creation, applied automatically when `search(..., normalize_query=True)` is called
- `RegexQueryNormalizer` — built-in implementation; strips common English question preamble (`"What do I know about X?"` → `"X"`, `"How does X work?"` → `"X work"`, etc.) using regex; most valuable for bag-of-words / hash-projection embedders that assign equal weight to every token
- `VaultSession.search(..., normalize_query=False)` — new keyword argument; when `True`, runs the active normalizer before embedding; defaults to `False` so sentence-transformer users are unaffected
- `VaultSession.open()` and `VaultSession.create()` gain `query_normalizer=` keyword argument; stores the normalizer on the session for all subsequent searches

**LLM normalizer example** (not shipped, implement as shown in docs):
```python
class GroqQueryNormalizer:
    def __init__(self, api_key: str):
        from groq import Groq
        self._client = Groq(api_key=api_key)

    def normalize(self, text: str) -> str:
        resp = self._client.chat.completions.create(
            model="llama-3.3-70b-versatile", max_tokens=30,
            messages=[
                {"role": "system", "content": "Extract key search terms. Return only keywords."},
                {"role": "user", "content": text},
            ],
        )
        return resp.choices[0].message.content.strip() or text
```

### Exports
- `QueryNormalizer`, `RegexQueryNormalizer` added to top-level `vaultmem` package

---

## [0.2.0] — March 2026

### Added

**Multi-modal memory ingestion**
- `VaultSession.add_media(path, passphrase)` — ingest an image, audio, document, or video file as an encrypted memory atom; raw bytes stored at `vault/media/{uuid}.enc`
- `VaultSession.add_media_batch(paths, passphrase)` — batch variant
- `VaultSession.get_media(atom_id, passphrase)` — decrypt and return raw media bytes on demand
- `media.py` — new module containing:
  - `MediaExtractor` Protocol — implement to add custom file-type support
  - `ImageExtractor` — Pillow + piexif; extracts EXIF date, GPS coordinates, optional OCR (pytesseract)
  - `AudioExtractor` — Whisper transcription
  - `DocumentExtractor` — PyMuPDF text extraction
  - `VideoExtractor` — ffmpeg key-frame metadata and duration
  - `MediaIngester` — dispatcher that routes by MIME type to registered `MediaExtractor` implementations; all imports are lazy so the core library remains dependency-free
  - `TimeQueryParser` — pure-Python natural-language time phrase parser; converts phrases like `"summer 2019"` or `"last Tuesday"` to UNIX timestamp ranges with no dependencies

**Temporal search**
- `VaultSession.search_by_time(start, end)` — browse atoms by `captured_at` timestamp range
- `VaultSession.diff(since, until)` — enumerate atoms written (by `created_at`) in a given range
- `VaultSession.search(..., parse_time=True)` — auto-extracts a time phrase from the query string, pre-filters candidate atoms by timestamp, then runs embedding comparison on the narrowed set

**MemoryObject schema v2**
- Five new optional media fields: `content_type`, `captured_at`, `media_blob_id`, `media_source_path`, `location`
- Fully backward-compatible; `schema_version` auto-bumps to `2` only when a media field is set; v1 vaults read without migration

**Optional extras**
- `[local]` — `sentence-transformers>=3.0` for `LocalEmbedder`
- `[media]` — Pillow, piexif, PyMuPDF, openai-whisper, ffmpeg-python for `MediaIngester`
- `[s3]` — `boto3>=1.26` for `S3BlobStore`
- `[postgres]` — `psycopg2-binary>=2.9` for `PostgresSearchIndex`
- `[ann]` — `hnswlib>=0.8` for `HNSWVectorIndex`

**Documentation**
- `ARCHITECTURE.md` — full component diagram and module breakdown for v0.2.0
- Three working examples: `examples/quickstart.py`, `examples/temporal_search.py`, `examples/media_ingestion.py`

### Changed
- `MemoryObject` dataclass gains 5 optional fields (all default `None`); no breaking change to existing vaults or code

---

## [0.1.3] — March 2026

### Added

**Pluggable storage backends**
- `BlobStore` ABC — defines the encrypted-atom storage contract
  - `FileBlobStore` (default) — one encrypted file per atom in a local `atoms/` directory; packed format `[12B IV][4B len][ciphertext][16B tag]`
  - `S3BlobStore` — any S3-compatible object store via `boto3`
- `migrate_vault(src, dst, passphrase)` — moves a complete vault between any two backends by copying encrypted bytes opaquely; no decryption required

**Pluggable search index**
- `SearchIndex` ABC — pre-filter metadata index contract (`IndexQuery` / `IndexRecord`)
  - `SQLiteSearchIndex` (default, WAL mode) — local SQLite with encrypted embeddings column
  - `PostgresSearchIndex` — drop-in Postgres backend
- `fetch_many(ids)` — targeted metadata retrieval by atom ID list

**HNSW ANN vector index**
- `VectorIndex` ABC — ANN search contract over session-RAM embeddings
  - `HNSWVectorIndex` — hnswlib HNSW graph; O(log N) `knn_query` for candidate sets > 1,000 atoms; exact-cosine fallback for ≤ 1,000 atoms; graph serialized encrypted to `vector_index.hnsw.enc`; loaded once at session open
- Without a `VectorIndex`, the legacy O(N) exact-cosine path is fully preserved

**Session API**
- `VaultSession.create()` and `.open()` gain `blob_store=`, `search_index=`, and `vector_index=` keyword arguments; all default to the existing local implementations

**README**
- Zenodo preprint DOI: `https://doi.org/10.5281/zenodo.19154079`

---

## [0.1.2] — March 2026

### Fixed
- Corrected GitHub repository URLs in `pyproject.toml` to point to `aag1091-alt/vaultmem-sdk`
- Updated README with correct install instructions and repository links

---

## [0.1.1] — March 2026 — Initial Release

**Zero-knowledge encrypted memory SDK for personal AI agents.**
Implements Layer 1 of the Memory-as-Asset (MAS) framework ([arXiv:2603.14212](https://arxiv.org/abs/2603.14212)).

### Core features

**Cryptographic layer** (`crypto.py`, 281 lines)
- AES-256-GCM encryption with a fresh 12-byte IV per atom
- Argon2id key derivation (RFC 9106): `time_cost=3`, `memory_cost=65536` (64 MiB), `parallelism=4`
- Session-scoped MEK: generated once at vault creation, stored only as KEK-wrapped ciphertext in `meta.json`, zeroed byte-by-byte on session close
- AAD binding of atom UUID prevents ciphertext transplantation
- Passphrase rotation cost: O(1) — one MEK re-wrap, zero atom re-encryption

**Vault format** (`vault.py`, 605 lines)
- `.vmem` binary format: 48-byte plaintext header (`VMEM` magic, version, algo, KDF, file-type, offsets, atom count, file UUID) + encrypted index block + encrypted atom blocks
- `write_vmem()` / `read_vmem()` for working copies
- Named snapshot support: `take_snapshot(name)` / `restore_snapshot(name)` / `list_snapshots()`
- Vault export and GDPR-compliant full erasure
- Crash-safe atomic checkpoint: `fsync` + `os.rename()`

**Session state machine** (`session.py`, 458 lines in v0.1.x)
- `VaultSession.create(path, passphrase)` and `.open(path, passphrase)` as context managers
- States: `CREATED → LOCKED → OPEN → CLOSING → CLOSED` (+ `CORRUPTED`)
- `.add(text)` — auto-classifies type, embeds, writes encrypted atom
- `.search(query, top_k)` — three-tier cosine retrieval
- `.flush()` — checkpoint without closing; `.close()` — checkpoint + MEK zero + unlock
- Session locking via `flock(LOCK_EX)` — enforces single writer
- Idle timeout with configurable grace period; MEDICAL vaults enforce minimum 900 s timeout

**Memory object model** (`models.py`, 251 lines in v0.1.x)
- `MemoryObject` dataclass implementing MAS 5-tuple `m = (c, e, o, π, v)`
- Types: `EPISODIC`, `SEMANTIC`, `PERSONA`, `PROCEDURAL`
- Granularities: `ATOM`, `COMPOSITE`, `AFFINITY`
- Data classes: `GENERAL`, `MEDICAL` (2× memory cost), `ARCHIVAL` (slow decay)
- AFFINITY significance: `σ = (1 − e^{−κf}) · e^{−λd}`; decay rates per data class

**Three-tier retrieval** (`retrieval.py`, 171 lines)
- AFFINITY → COMPOSITE → ATOM search order
- Blended AFFINITY score: `0.5σ + 0.5cos`; degrades to flat cosine on ATOM-only vaults
- `SearchResult` dataclass with rank, score, atom reference

**Classifier** (`classifier.py`, 185 lines)
- Deterministic 4-feature linguistic classifier (temporal anchoring, dispositional framing, sequential structure, self-reference)
- No LLM call required for type assignment

**Embedders** (`embedder.py`, 169 lines)
- `LocalEmbedder` — sentence-transformers `all-MiniLM-L6-v2` (384-dim); no external calls
- `OllamaEmbedder` — HTTP to a local Ollama instance via `urllib` only; no opaque dependencies
- `NullEmbedder` — all-zero vectors for testing

**Exceptions** (`exceptions.py`, 33 lines)
- `WrongPassphraseError`, `VaultTamperedError`, `VaultLockedError`, `VaultAlreadyOpenError`, `SessionStateError`, `MemorySchemaError`, `RotationRequiredError`

**Test suite**
- 39 unit tests covering all components; all pass on Python 3.11–3.13

**Dependencies**
- Core: `cryptography>=42.0`, `argon2-cffi>=23.1`, `numpy>=1.24`
- Optional: `sentence-transformers>=3.0` for `LocalEmbedder`
