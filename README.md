# VaultMem

**Zero-knowledge encrypted memory for personal AI agents.**

[![PyPI](https://img.shields.io/pypi/v/vaultmem)](https://pypi.org/project/vaultmem/)
[![Python](https://img.shields.io/pypi/pyversions/vaultmem)](https://pypi.org/project/vaultmem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

VaultMem is an embeddable Python library that gives AI agents persistent, encrypted, portable memory — where the **platform operator cryptographically cannot read user data**.

It implements Layer 1 (Personal Storage Layer) of the [Memory-as-Asset framework](https://arxiv.org/abs/2603.14212) (Pan, Huang & Yang, 2026).

---

## The Problem

Every existing AI memory library (mem0, Zep, LangMem, Letta) stores user memories in plaintext on the platform's servers. The platform owns your memory.

VaultMem flips this: the user holds the encryption key. The platform stores opaque ciphertext it cannot read. The guarantee is mathematical, not a privacy policy.

---

## Install

```bash
pip install vaultmem                  # core (AES-256-GCM, Argon2id, NumPy)
pip install "vaultmem[local]"         # + sentence-transformers for local embeddings
```

Core dependencies only: `cryptography`, `argon2-cffi`, `numpy`. No PyTorch required unless you use `[local]`.

---

## Quick Start

```python
from vaultmem import VaultSession
from vaultmem import OllamaEmbedder   # or LocalEmbedder, or bring your own

embedder = OllamaEmbedder("http://localhost:11434")  # any Ollama instance

# Create a new encrypted vault
with VaultSession.create("./my_vault", passphrase="s3cr3t", owner="alice",
                          embedder=embedder) as s:
    s.add("I met Bob at the AI conference yesterday")
    s.add("I prefer concise bullet-point answers over long paragraphs")
    s.add("My go-to language for data work is Python")

# Re-open and search across sessions
with VaultSession.open("./my_vault", passphrase="s3cr3t",
                        embedder=embedder) as s:
    results = s.search("Bob conference")
    for r in results:
        print(f"[{r.tier}] {r.score:.3f}  {r.atom.content}")
```

Output:
```
[ATOM] 0.847  I met Bob at the AI conference yesterday
```

---

## How It Works

### Cryptographic Layer

| Component | Algorithm | Detail |
|-----------|-----------|--------|
| Key derivation | Argon2id (RFC 9106) | 64 MiB, 3 iterations, 4 threads |
| Encryption | AES-256-GCM | Per-atom nonce, authentication tag |
| Key architecture | KEK wraps MEK | MEK lives only in session RAM; zeroed on close |
| On-disk format | `.vmem` binary | Self-describing, tamper-evident, portable |

The session key (MEK) is derived from the user's passphrase and **never written to disk**. It is zeroed from memory when the session closes. A platform developer who integrates VaultMem receives opaque ciphertext from their users — they can store it, back it up, serve it back — but cannot read it.

### Memory Model

Each memory is stored as a `MemoryObject` with four types:

| Type | What it stores |
|------|---------------|
| `EPISODIC` | Events: "I met Bob at the conference" |
| `SEMANTIC` | Facts: "Paris is the capital of France" |
| `PERSONA` | Preferences and habits: "I prefer concise answers" |
| `PROCEDURAL` | How-to knowledge: "To deploy, run `make prod`" |

Memories are organized across three granularity tiers:

- **ATOM** — a single leaf memory
- **COMPOSITE** — a type-homogeneous aggregation of related atoms
- **AFFINITY** — a recurring pattern summary (e.g., "user drinks coffee every morning"), with a significance score that decays when the pattern stops occurring

### Three-Tier Retrieval

```
AFFINITY   score = α · significance + (1−α) · cosine
COMPOSITE  score = cosine
ATOM       score = cosine
```

Pattern queries (habits, preferences) surface AFFINITY summaries above individual episodic memories. Factual queries fall through to ATOM-level cosine retrieval. Both cases use the same `s.search()` call — the tier structure is automatic.

**Benchmark:** On a corpus of 18 atoms (3 topic clusters), three-tier retrieval achieves MRR = 1.00 on pattern queries vs MRR = 0.50 for flat cosine, with zero regression on specific/factual queries.

---

## Embedders

VaultMem ships three embedders and accepts any object implementing `.embed(text) -> list[float]`:

| Class | Requires | Privacy |
|-------|----------|---------|
| `LocalEmbedder` | `pip install "vaultmem[local]"` (PyTorch) | Fully local, no network |
| `OllamaEmbedder` | Running Ollama instance | Sends text to Ollama host |
| `NullEmbedder` | Nothing | For testing only (zero vectors) |

```python
from vaultmem import LocalEmbedder, OllamaEmbedder

# Local (no network, ~80ms/sentence on CPU)
embedder = LocalEmbedder()

# Ollama (GPU-accelerated, Tailscale or localhost only)
embedder = OllamaEmbedder("http://localhost:11434", model="all-minilm")

# Custom
class MyEmbedder:
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimension(self) -> int: return 384
```

---

## Vault Format

Vaults are stored as two files:

```
my_vault/
├── current.vmem      # encrypted index (AES-256-GCM, self-describing header)
└── current.atoms     # append-only encrypted atom blocks
```

The `.vmem` format is language-agnostic and self-describing — any conforming implementation can read a vault given the correct passphrase. Checkpoints are crash-safe via `fsync` + `os.rename()`.

---

## Session Lifecycle

```python
# Sessions are state machines: CLOSED → OPEN → CHECKPOINTING → CLOSING

with VaultSession.open("./vault", passphrase="...", embedder=embedder) as s:
    s.add("new memory")            # writes to in-memory buffer
    s.checkpoint()                 # optional: flush to disk mid-session
    results = s.search("query")    # three-tier retrieval
# MEK zeroed automatically on context manager exit
```

Only one session can hold a vault open at a time (exclusive OS file lock). Attempting to open a locked vault raises `VaultAlreadyOpenError`.

---

## Security Properties

| Property | How it's enforced |
|----------|------------------|
| Platform cannot read memories | AES-256-GCM; MEK never leaves session RAM |
| Tampering is detected | GCM authentication tag per atom block |
| Wrong passphrase is rejected | MEK unwrap authentication failure → `WrongPassphraseError` |
| Single writer | `flock(LOCK_EX)` on POSIX, `LockFileEx` on Windows |
| MEK zeroed on close | `bytearray` byte-by-byte zero fill |
| Atomic checkpoint | `fsync` + `os.rename()` |
| Portable vault | `.vmem` magic header + version field |

---

## Running Tests

```bash
pip install "vaultmem[dev]"
pytest tests/test_unit.py           # 39 unit tests
python tests/bench_affinity.py      # AFFINITY tier uplift benchmark
python tests/bench_section7.py      # crypto + search latency
```

---

## API Reference

### `VaultSession`

```python
VaultSession.create(path, *, passphrase, owner, embedder=None) -> VaultSession
VaultSession.open(path, *, passphrase, embedder=None) -> VaultSession

session.add(text, *, memory_type=None) -> MemoryObject
session.search(query, *, top_k=10, memory_type=None) -> list[SearchResult]
session.checkpoint() -> None
session.close() -> None
```

### `SearchResult`

```python
@dataclass
class SearchResult:
    atom: MemoryObject
    score: float       # blended score (AFFINITY) or cosine (COMPOSITE/ATOM)
    tier: str          # "AFFINITY" | "COMPOSITE" | "ATOM"
```

---

## Background

VaultMem is the reference implementation of the Personal Storage Layer described in:

> **Memory as Asset: Towards User-Owned Persistent AI Memory**
> Pan, Huang & Yang — arXiv:2603.14212, March 2026

The paper defines a three-layer architecture for user-owned AI memory. VaultMem implements Layer 1. Layers 2 (collaborative memory groups) and 3 (decentralized memory exchange) are future work.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Contributing

Issues and pull requests welcome. The library is intentionally small (~2,100 lines excluding tests) — the code is the specification.
