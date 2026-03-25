"""
VaultMem backend demo — FileBlobStore + SQLiteSearchIndex + HNSWVectorIndex

Run from the vaultmem repo root:
    .venv/bin/python examples/demo_backends.py

Uses the local vaultmem-sdk (installed via: pip install -e ../vaultmem-sdk)

What this script shows:
  1. Create a vault with all three backends
  2. Add a mix of typed memories and flush to disk
  3. Inspect the blob files on disk (encrypted)
  4. Inspect the SQLite index (plaintext metadata, no content)
  5. Inspect the encrypted HNSW vector index file
  6. Search using the ANN path and show ranked results
  7. Reopen the vault to verify persistence (HNSW index reloaded from disk)
"""

import os
import shutil
import sqlite3
import tracemalloc
from pathlib import Path

from vaultmem import (
    FileBlobStore,
    SQLiteSearchIndex,
    HNSWVectorIndex,
    VaultSession,
    NullEmbedder,
    LocalEmbedder,
    OllamaEmbedder,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VAULT_DIR   = Path("/tmp/vaultmem_demo")
PASSPHRASE  = "demo-passphrase-123"
OWNER       = "alice"

# Embedder priority:
#   1. OllamaEmbedder — all-minilm via local Ollama (no GPU needed)
#   2. LocalEmbedder  — all-MiniLM-L6-v2 via sentence-transformers (needs torch)
#   3. NullEmbedder   — zero vectors (no cosine search, but everything else works)
EMBED_DIM = 384

try:
    embedder = OllamaEmbedder()   # default: all-minilm, localhost:11434
    embedder.embed("warmup")       # fail fast if Ollama is unreachable
    EMBEDDER_NAME = "OllamaEmbedder (all-minilm via Ollama)"
except Exception:
    try:
        embedder = LocalEmbedder()
        embedder.embed("warmup")
        EMBEDDER_NAME = "LocalEmbedder (all-MiniLM-L6-v2)"
    except Exception:
        embedder = NullEmbedder()
        EMBEDDER_NAME = "NullEmbedder  (start Ollama or install torch for real embeddings)"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP  = "─" * 60
SEP2 = "═" * 60

def header(title: str) -> None:
    print(f"\n{SEP2}\n  {title}\n{SEP2}")

def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------

header("VaultMem Backend Demo")
print(f"  Embedder : {EMBEDDER_NAME}")
print(f"  Vault    : {VAULT_DIR}")

# Start fresh each run
if VAULT_DIR.exists():
    shutil.rmtree(VAULT_DIR)

blob    = FileBlobStore(VAULT_DIR)
index   = SQLiteSearchIndex(VAULT_DIR / "search_index.db")
vectors = HNSWVectorIndex(dim=EMBED_DIM)


# ---------------------------------------------------------------------------
# 2. Create vault and add memories
# ---------------------------------------------------------------------------

section("Creating vault and adding memories")

MEMORIES = [
    "I met Sarah at the NeurIPS conference last week",
    "I prefer dark mode in all my editors",
    "My go-to language for data work is Python",
    "I drink two cups of coffee every morning before coding",
    "I drink coffee with oat milk, no sugar",
    "The team decided to use Postgres for the new backend",
    "Sarah works on privacy-preserving ML at Stanford",
    "I go for a 30-minute run every morning",
    "I go for a run every morning before breakfast",
    "Always write tests before merging a feature branch",
]

with VaultSession.create(
    VAULT_DIR, PASSPHRASE, OWNER,
    embedder=embedder,
    blob_store=blob,
    search_index=index,
    vector_index=vectors,
) as s:
    for mem in MEMORIES:
        atom = s.add(mem)
        print(f"  [{atom.type.value:10s}] {mem}")
    s.flush()
    print(f"\n  Flushed {s.atom_count} atoms to disk")
    print(f"  HNSW index  : {len(vectors)} vectors in RAM")


# ---------------------------------------------------------------------------
# 3. Inspect blob files on disk
# ---------------------------------------------------------------------------

section("Blob files on disk  (atoms/ directory)")

atoms_dir = VAULT_DIR / "atoms"
enc_files = sorted(atoms_dir.glob("*.enc"))

print(f"  Files: {len(enc_files)}")
print()
for f in enc_files:
    size = f.stat().st_size
    raw  = f.read_bytes()
    # Header layout: [12B IV] [4B ct_len] [ciphertext] [16B tag]
    ct_len = int.from_bytes(raw[12:16], "big")
    print(f"  {f.name}")
    print(f"    size      : {size} bytes")
    print(f"    IV        : {raw[:12].hex()}")
    print(f"    ct_len    : {ct_len} bytes")
    print(f"    tag       : {raw[-16:].hex()}")
    print(f"    content?  : [encrypted — unreadable without MEK]")
    print()


# ---------------------------------------------------------------------------
# 4. Inspect SQLite index
# ---------------------------------------------------------------------------

section("SQLite search index  (search_index.db)")

db_path = VAULT_DIR / "search_index.db"
print(f"  DB size: {db_path.stat().st_size} bytes\n")

conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT atom_id, tier, memory_type, data_class, significance, "
    "created_at, owner, is_churned, length(enc_embedding) as emb_len "
    "FROM atoms ORDER BY created_at"
).fetchall()

print(f"  {'atom_id'[:8]:<8}  {'tier':<10} {'memory_type':<12} {'significance':<13} {'emb_len':>7}  content?")
print(f"  {'-'*8}  {'-'*10} {'-'*12} {'-'*13} {'-'*7}  {'-'*30}")
for r in rows:
    short_id = r["atom_id"][:8]
    sig = f"{r['significance']:.4f}" if r["significance"] is not None else "  —  "
    print(
        f"  {short_id}  {r['tier']:<10} {r['memory_type']:<12} {sig:<13} "
        f"{r['emb_len']:>7}B  [encrypted]"
    )

conn.close()


# ---------------------------------------------------------------------------
# 5. Inspect the encrypted HNSW vector index
# ---------------------------------------------------------------------------

section("HNSW vector index  (vector_index.hnsw.enc)")

vi_path = VAULT_DIR / "vector_index.hnsw.enc"
vi_raw  = vi_path.read_bytes()
vi_ct_len = int.from_bytes(vi_raw[12:16], "big")
print(f"  File     : {vi_path.name}")
print(f"  Size     : {vi_path.stat().st_size} bytes")
print(f"  IV       : {vi_raw[:12].hex()}")
print(f"  ct_len   : {vi_ct_len} bytes")
print(f"  tag      : {vi_raw[-16:].hex()}")
print(f"  content? : [encrypted — HNSW graph + id mapping, unreadable without MEK]")
print()
print("  What's inside (after decryption with MEK):")
print("    - HNSW graph structure  (edges, layers, entry point)")
print("    - atom_id ↔ int label mapping")
print("    - 384-dim float32 vectors (plaintext — only safe in session RAM)")


# ---------------------------------------------------------------------------
# 6. Search  (ANN path via HNSW)
# ---------------------------------------------------------------------------

section("Search  (vault reopened — HNSW index loaded from disk)")

blob2    = FileBlobStore(VAULT_DIR)
index2   = SQLiteSearchIndex(VAULT_DIR / "search_index.db")
vectors2 = HNSWVectorIndex(dim=EMBED_DIM)   # empty; load() fills it from disk

with VaultSession.open(
    VAULT_DIR, PASSPHRASE,
    embedder=embedder,
    blob_store=blob2,
    search_index=index2,
    vector_index=vectors2,
) as s:
    print(f"  Vault reopened — {s.atom_count} atoms in index")
    print(f"  HNSW index   — {len(vectors2)} vectors loaded from disk\n")
    print(f"  Search path  : {'ANN (HNSW O(log N))' if len(vectors2) > 0 else 'exact (NullEmbedder — no vectors)'}\n")

    for query in ["morning coffee habits", "Sarah", "running exercise"]:
        results = s.search(query, top_k=3)
        print(f"  Query: \"{query}\"")
        if not results:
            print("    (no results — NullEmbedder has no vectors)")
        for r in results:
            print(f"    [{r.tier:<10}] score={r.score:.3f}  {r.atom.content}")
        print()


# ---------------------------------------------------------------------------
# 7. ANN vs exact — what actually gets loaded per search
# ---------------------------------------------------------------------------
#
# ANN path (VectorIndex present):
#   - HNSW graph loaded ONCE on session open, stays in RAM
#   - Each search: graph traversal only → fetch_many(top_k) rows from SQLite
#   - Embeddings decrypted from DB per search: 0
#
# Exact path (no VectorIndex):
#   - Each search: query() fetches ALL N rows + decrypts ALL N enc_embeddings
#   - Everything lands in RAM as Python objects, then gets GC'd
# ---------------------------------------------------------------------------

section("ANN vs exact — what gets loaded per search")

QUERY = "morning coffee habits"
TOP_K = 3

# Wrap SQLiteSearchIndex to count rows touched per method call
class CountingIndex(SQLiteSearchIndex):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.query_rows    = 0
        self.fetchmany_rows = 0
    def query(self, q):
        rows = super().query(q)
        self.query_rows += len(rows)
        return rows
    def fetch_many(self, ids):
        rows = super().fetch_many(ids)
        self.fetchmany_rows += len(rows)
        return rows

# — ANN path —
ann_idx  = CountingIndex(VAULT_DIR / "search_index.db")
ann_vec  = HNSWVectorIndex(dim=EMBED_DIM)
with VaultSession.open(
    VAULT_DIR, PASSPHRASE, embedder=embedder,
    blob_store=FileBlobStore(VAULT_DIR),
    search_index=ann_idx,
    vector_index=ann_vec,
) as s:
    tracemalloc.start()
    s.search(QUERY, top_k=TOP_K)
    _, ann_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

# — Exact path (no vector_index) —
exact_idx = CountingIndex(VAULT_DIR / "search_index.db")
with VaultSession.open(
    VAULT_DIR, PASSPHRASE, embedder=embedder,
    blob_store=FileBlobStore(VAULT_DIR),
    search_index=exact_idx,
) as s:
    tracemalloc.start()
    s.search(QUERY, top_k=TOP_K)
    _, exact_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

n        = exact_idx.query_rows                           # all atoms read by exact path = N
emb_size = rows[0]["emb_len"]                             # enc_embedding bytes per atom (from section 4)

print(f"  Query: \"{QUERY}\"  (top_k={TOP_K}, vault size N={n})\n")
print(f"  {'Metric':<38} {'ANN (HNSW)':>12}  {'Exact (no HNSW)':>15}")
print(f"  {'-'*38} {'-'*12}  {'-'*15}")
print(f"  {'SQLite rows read (query all N)':38} {'0':>12}  {exact_idx.query_rows:>15}")
print(f"  {'SQLite rows read (fetch_many top-k)':38} {ann_idx.fetchmany_rows:>12}  {'0':>15}")
print(f"  {'Enc embeddings decrypted from DB':38} {0:>12}  {exact_idx.query_rows:>15}")
print(f"  {'Peak RAM allocated (search call)':38} {ann_peak//1024:>10} KB  {exact_peak//1024:>13} KB")
print()
print(f"  Notes:")
print(f"    ANN fetch_many reads top_k×2={TOP_K*2} rows (headroom for significance blending),")
print(f"    not top_k={TOP_K} — still O(1) vs O(N).")
print(f"    ANN decrypts 0 embeddings from SQLite; the HNSW graph already holds them in RAM.")
print()
print(f"  As N grows:")
print(f"    Exact : O(N) — every search decrypts all N × {emb_size} B enc_embeddings into RAM")
print(f"    ANN   : O(log N) — HNSW traversal only; fetch_many loads {TOP_K*2} rows regardless of N")

# ---------------------------------------------------------------------------
# 8. Vault directory layout
# ---------------------------------------------------------------------------

section("Vault directory layout")

for p in sorted(VAULT_DIR.rglob("*")):
    rel  = p.relative_to(VAULT_DIR)
    size = f"{p.stat().st_size:>8} bytes" if p.is_file() else ""
    print(f"  {str(rel):<45} {size}")

header("Done")
