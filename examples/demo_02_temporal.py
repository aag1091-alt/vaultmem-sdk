"""
VaultMem SDK — Example 02: Temporal Search & Timeline Browsing

Shows how to build time-aware memory on top of VaultMem:
  1. Store memories with real-world capture timestamps (captured_at)
  2. search_by_time()  — browse a time window (the "vault" experience)
  3. diff()            — what was recorded in a date range
  4. TimeQueryParser   — extract time bounds from natural language queries
  5. search(parse_time=True) — semantic search with automatic time pre-filtering

The captured_at field represents *when the event happened* in the real world,
independent of *when the atom was written* to the vault (created_at).

Run:
    PYTHONPATH=src .venv/bin/python examples/demo_02_temporal.py
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path

from vaultmem import VaultSession, NullEmbedder, TimeQueryParser

# ── Config ────────────────────────────────────────────────────────────────────
VAULT_DIR  = Path("/tmp/vaultmem_temporal")
PASSPHRASE = "hunter2"
OWNER      = "alice"
EMBEDDER   = NullEmbedder()

def ts(year, month=1, day=1) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())

def fmt(unix_ts) -> str:
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%Y-%m-%d")

SEP = "─" * 60
def hr(title: str = "") -> None:
    print(f"\n{SEP}")
    if title:
        print(f"  {title}")
        print(SEP)

# ── Setup ─────────────────────────────────────────────────────────────────────
if VAULT_DIR.exists():
    shutil.rmtree(VAULT_DIR)

# ── 1. Store memories with real-world timestamps ──────────────────────────────
hr("1. Storing memories with real-world captured_at timestamps")

DIARY = [
    # (content, year, month, day)
    ("Started my first job at a startup in San Francisco.", 2018, 7, 2),
    ("Visited Tokyo for the first time — cherry blossom season.", 2019, 4, 8),
    ("Graduated with my MSc in Computer Science.", 2019, 6, 15),
    ("Moved to Berlin. Took the leap after years of planning.", 2019, 9, 1),
    ("Presented a paper at NeurIPS. Terrifying and exhilarating.", 2019, 12, 10),
    ("Started learning Rust. Borrow checker is humbling.", 2020, 2, 5),
    ("Locked down. Started an online course on cryptography.", 2020, 3, 20),
    ("Got my scuba diving certification in Croatia.", 2021, 7, 14),
    ("Joined a new team working on privacy infrastructure.", 2022, 1, 10),
    ("Ran my first half-marathon in 2:04.", 2023, 5, 21),
    ("Launched the first public version of the side project.", 2024, 3, 18),
]

with VaultSession.create(VAULT_DIR, PASSPHRASE, OWNER, embedder=EMBEDDER) as s:
    for content, year, month, day in DIARY:
        atom = s.add(content)
        atom.captured_at = ts(year, month, day)   # backdate to real-world event

    s.flush()
    print(f"  Stored {s.atom_count} diary entries spanning 2018–2024")

# ── 2. search_by_time — browse a time window ─────────────────────────────────
hr("2. search_by_time() — browse a time window")

with VaultSession.open(VAULT_DIR, PASSPHRASE, embedder=EMBEDDER) as s:

    # Browse 2019
    results = s.search_by_time(ts(2019), ts(2020), top_k=10)
    print(f"  Memories from 2019 ({len(results)} entries):")
    for atom in results:
        print(f"    {fmt(atom.captured_at)}  {atom.content}")

    # Browse a specific quarter (Q3 2021)
    print()
    q3_2021 = s.search_by_time(ts(2021, 7), ts(2021, 10), top_k=10)
    print(f"  Memories from Q3 2021 ({len(q3_2021)} entries):")
    for atom in q3_2021:
        print(f"    {fmt(atom.captured_at)}  {atom.content}")

# ── 3. diff — what was written to the vault recently ─────────────────────────
hr("3. diff() — atoms written to the vault in a time range (by created_at)")

import time

with VaultSession.open(VAULT_DIR, PASSPHRASE, embedder=EMBEDDER) as s:
    # Add a couple of new entries for this session
    new_entry = s.add("Thinking about writing a book on software craftsmanship.")
    s.flush()

    # diff returns atoms by *write time*, not event time
    # useful for syncing, auditing, incremental processing
    now = int(time.time())
    recent = s.diff(now - 5, now + 5)
    print(f"  Atoms written in the last 5 seconds: {len(recent)}")
    for atom in recent:
        print(f"    created={fmt(atom.created_at)}  captured={fmt(atom.captured_at) if atom.captured_at else '—'}")
        print(f"    {atom.content}")

# ── 4. TimeQueryParser — extract time bounds from natural language ─────────────
hr("4. TimeQueryParser — parse time phrases from natural language")

examples = [
    "what did I do summer 2019",
    "memories from March 2020",
    "what happened in 2019",
    "last year projects",
    "this month updates",
    "yesterday",
]

print(f"  {'Query':<40}  {'From':<12}  {'To':<12}  Remainder")
print(f"  {'-'*40}  {'-'*12}  {'-'*12}  {'-'*20}")
for q in examples:
    from_ts, to_ts, remainder = TimeQueryParser.parse(q)
    f = fmt(from_ts) if from_ts else "—"
    t = fmt(to_ts)   if to_ts   else "—"
    print(f"  {q:<40}  {f:<12}  {t:<12}  {remainder!r}")

# ── 5. search with parse_time=True ────────────────────────────────────────────
hr("5. search(parse_time=True) — semantic search with automatic time filtering")

with VaultSession.open(VAULT_DIR, PASSPHRASE, embedder=EMBEDDER) as s:
    # The query "what happened in 2019" automatically becomes:
    #   - semantic query: "what happened in"  (time phrase stripped)
    #   - captured_at filter: 2019-01-01 → 2020-01-01
    query = "what did I do in 2019"
    from_ts, to_ts, remainder = TimeQueryParser.parse(query)
    print(f"  Query: {query!r}")
    print(f"  TimeQueryParser extracted: {fmt(from_ts)} → {fmt(to_ts)}, semantic remainder: {remainder!r}")

    results = s.search(query, top_k=10, parse_time=True)
    print(f"  Results ({len(results)} atoms in 2019):")
    for r in results:
        ts_str = fmt(r.atom.captured_at) if r.atom.captured_at else "—"
        print(f"    {ts_str}  {r.atom.content}")

print(f"\n{SEP}\nDone.\n")
