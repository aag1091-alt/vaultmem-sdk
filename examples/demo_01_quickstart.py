"""
VaultMem SDK — Example 01: Quickstart

Shows the minimal integration pattern a platform developer would use:
  1. Create a vault for a user
  2. Add text memories on their behalf
  3. Search across those memories
  4. Re-open and verify persistence

The vault is encrypted end-to-end. The platform developer never sees
plaintext content — only the user's passphrase unlocks the MEK.

Run:
    PYTHONPATH=src .venv/bin/python examples/demo_01_quickstart.py
"""

import shutil
from pathlib import Path

from vaultmem import VaultSession, NullEmbedder

# ── Config ────────────────────────────────────────────────────────────────────
VAULT_DIR  = Path("/tmp/vaultmem_quickstart")
PASSPHRASE = "hunter2"
OWNER      = "alice"

# NullEmbedder so the demo runs without PyTorch installed.
# Swap in LocalEmbedder() for real cosine search.
EMBEDDER = NullEmbedder()

SEP = "─" * 60

def hr(title: str = "") -> None:
    print(f"\n{SEP}")
    if title:
        print(f"  {title}")
        print(SEP)

# ── Setup ─────────────────────────────────────────────────────────────────────
if VAULT_DIR.exists():
    shutil.rmtree(VAULT_DIR)

# ── 1. Create vault and add memories ─────────────────────────────────────────
hr("1. Creating vault and adding memories")

with VaultSession.create(VAULT_DIR, PASSPHRASE, OWNER, embedder=EMBEDDER) as s:

    memories = [
        "I met Sarah at the NeurIPS conference — she works on privacy-preserving ML at Stanford.",
        "I prefer dark mode in all my editors and terminals.",
        "My go-to language for data work is Python.",
        "I drink two cups of coffee every morning before coding.",
        "The team decided to use Postgres for the new backend.",
        "Always write tests before merging a feature branch.",
    ]

    atoms = [s.add(m) for m in memories]
    s.flush()

    print(f"  Vault created at : {VAULT_DIR}")
    print(f"  Owner            : {s.owner}")
    print(f"  Atoms stored     : {s.atom_count}")
    print()
    for atom in atoms:
        print(f"  [{atom.type.value:<10}] {atom.content[:70]}")

# At this point the vault is closed and the MEK is zeroed from RAM.
# On disk: current.vmem (encrypted index) + current.atoms (encrypted blobs)

# ── 2. Inspect what's on disk (all ciphertext) ────────────────────────────────
hr("2. What's on disk — all ciphertext, unreadable without passphrase")

vmem_path  = VAULT_DIR / "current.vmem"
atoms_path = VAULT_DIR / "current.atoms"
print(f"  current.vmem  : {vmem_path.stat().st_size:,} bytes  (header + encrypted index)")
print(f"  current.atoms : {atoms_path.stat().st_size:,} bytes  (append-only encrypted atom blocks)")
print()

# Show that the raw bytes reveal nothing
vmem_raw = vmem_path.read_bytes()
print(f"  First 32 bytes of current.vmem: {vmem_raw[:32].hex()}")
print(f"  Looks like random noise — this is what the platform sees.")

# ── 3. Re-open and search ─────────────────────────────────────────────────────
hr("3. Re-opening vault and searching")

with VaultSession.open(VAULT_DIR, PASSPHRASE, embedder=EMBEDDER) as s:
    print(f"  Vault re-opened. Atoms: {s.atom_count}")
    print()

    # With NullEmbedder all cosine scores are 0, so results are arbitrary.
    # With LocalEmbedder() or OllamaEmbedder() these would rank by meaning.
    queries = ["morning coffee", "Sarah Stanford", "testing practices"]
    for q in queries:
        results = s.search(q, top_k=2)
        print(f"  Query: {q!r}")
        for r in results:
            print(f"    score={r.score:.3f}  [{r.tier:<9}]  {r.atom.content[:65]}")
        print()

# ── 4. Wrong passphrase is rejected ──────────────────────────────────────────
hr("4. Wrong passphrase is cryptographically rejected")

from vaultmem import WrongPassphraseError
try:
    VaultSession.open(VAULT_DIR, "wrong-passphrase", embedder=EMBEDDER)
    print("  ERROR: should have raised!")
except WrongPassphraseError:
    print("  WrongPassphraseError raised — GCM authentication tag mismatch.")
    print("  The platform cannot brute-force this without the user's passphrase.")

print(f"\n{SEP}\nDone.\n")
