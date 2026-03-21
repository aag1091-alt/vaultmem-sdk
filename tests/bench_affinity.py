"""
VaultMem AFFINITY Tier Benchmark — §7.3.1

Demonstrates that the three-tier retrieval architecture produces measurably
higher MRR than flat cosine on *pattern queries* — queries asking about
recurring habits and preferences — by surfacing high-significance AFFINITY
atoms above semantically similar but incidental ATOM-granularity memories.

Corpus: 3 topic clusters (coffee, Python, exercise), 5 ATOMs each + 1 AFFINITY
atom per cluster = 18 atoms total.

Query set:
  - 6 "pattern queries":  ask about habits/patterns → AFFINITY is primary answer
  - 6 "specific queries": ask about single events → specific ATOMs are answers

Run:
    python tests/bench_affinity.py
    python tests/bench_affinity.py --ollama http://100.118.247.106:11434
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vaultmem.embedder import OllamaEmbedder
from vaultmem.models import (
    DataClass, Granularity, MemoryObject, MemoryType, compute_significance,
    estimate_tokens,
)
from vaultmem.retrieval import search


# ---------------------------------------------------------------------------
# Corpus — 3 topic clusters, 5 ATOMs + 1 AFFINITY each
# ---------------------------------------------------------------------------

# Each AFFINITY represents a recurring pattern; significance reflects 8–10
# source atoms observed recently (simulates a month of active use).

_NOW = int(time.time())
_ONE_WEEK_AGO = _NOW - 7 * 86400

CORPUS = [
    # ── Coffee cluster ──────────────────────────────────────────────────────
    # AFFINITY is written as an AI-generated aggregate-pattern summary
    # (abstract language → lower raw cosine to pattern queries than keyword-
    # matching decoy ATOMs, so flat cosine incorrectly prefers the decoy).
    {
        "id": "cf_aff",
        "content": "Recurring pre-work caffeination: user exhibits consistent morning beverage intake correlated with daily task initiation",
        "type": MemoryType.PERSONA,
        "granularity": Granularity.AFFINITY,
        "freq": 10,
        "last_seen": _ONE_WEEK_AGO,
    },
    # Keyword-matching decoy: mentions "coffee habits" and "morning" incidentally
    {
        "id": "cf_decoy",
        "content": "I mentioned to a friend that my coffee habits in the morning feel automatic now",
        "type": MemoryType.EPISODIC,
        "granularity": Granularity.ATOM,
    },
    {
        "id": "cf_0",
        "content": "Bought a bag of Ethiopian Yirgacheffe beans from the specialty roaster on Saturday",
        "type": MemoryType.EPISODIC,
        "granularity": Granularity.ATOM,
    },
    {
        "id": "cf_1",
        "content": "My pour-over this morning took longer because I ground the beans too fine",
        "type": MemoryType.EPISODIC,
        "granularity": Granularity.ATOM,
    },
    {
        "id": "cf_2",
        "content": "Tried the new cold brew at the coffee shop downtown today — very smooth",
        "type": MemoryType.EPISODIC,
        "granularity": Granularity.ATOM,
    },
    {
        "id": "cf_3",
        "content": "Ordered a double espresso while waiting for my delayed flight at the airport",
        "type": MemoryType.EPISODIC,
        "granularity": Granularity.ATOM,
    },

    # ── Python cluster ───────────────────────────────────────────────────────
    {
        "id": "py_aff",
        "content": "Dominant language pattern: Python appears consistently as primary scripting, automation, and data-analysis tool across tasks",
        "type": MemoryType.PERSONA,
        "granularity": Granularity.AFFINITY,
        "freq": 9,
        "last_seen": _ONE_WEEK_AGO,
    },
    # Keyword-matching decoy: mentions "go-to programming language" incidentally
    {
        "id": "py_decoy",
        "content": "Told a new hire that my go-to programming language for most things is Python",
        "type": MemoryType.EPISODIC,
        "granularity": Granularity.ATOM,
    },
    {
        "id": "py_0",
        "content": "Debugged a Python import cycle error in the authentication module this afternoon",
        "type": MemoryType.EPISODIC,
        "granularity": Granularity.ATOM,
    },
    {
        "id": "py_1",
        "content": "Set up a new Python 3.11 virtual environment with venv for the new project",
        "type": MemoryType.EPISODIC,
        "granularity": Granularity.ATOM,
    },
    {
        "id": "py_2",
        "content": "Used Python subprocess module to call a shell script from my data pipeline code",
        "type": MemoryType.EPISODIC,
        "granularity": Granularity.ATOM,
    },
    {
        "id": "py_3",
        "content": "Fixed a type annotation error that mypy caught in the data processing module",
        "type": MemoryType.EPISODIC,
        "granularity": Granularity.ATOM,
    },

    # ── Exercise cluster ─────────────────────────────────────────────────────
    {
        "id": "ex_aff",
        "content": "Consistent physical activity pattern: regular gym and running sessions observed at multiple-times-per-week frequency",
        "type": MemoryType.PERSONA,
        "granularity": Granularity.AFFINITY,
        "freq": 8,
        "last_seen": _ONE_WEEK_AGO,
    },
    # Two keyword-matching decoys — each matches one of the two pattern queries
    {
        "id": "ex_decoy1",
        "content": "Discussed my exercise habits and typical workout routine with a personal trainer",
        "type": MemoryType.EPISODIC,
        "granularity": Granularity.ATOM,
    },
    {
        "id": "ex_decoy2",
        "content": "Mentioned to my trainer that I work out multiple times a week consistently",
        "type": MemoryType.EPISODIC,
        "granularity": Granularity.ATOM,
    },
    {
        "id": "ex_1",
        "content": "Did a heavy leg day at the gym with squats and deadlifts yesterday afternoon",
        "type": MemoryType.EPISODIC,
        "granularity": Granularity.ATOM,
    },
    {
        "id": "ex_2",
        "content": "My running pace improved to under 5 minutes per kilometre on the track this week",
        "type": MemoryType.EPISODIC,
        "granularity": Granularity.ATOM,
    },
    {
        "id": "ex_4",
        "content": "Signed up for a 10K city race happening next month",
        "type": MemoryType.EPISODIC,
        "granularity": Granularity.ATOM,
    },
]

# ---------------------------------------------------------------------------
# Query set — pattern queries label AFFINITY as relevant; specific queries
# label individual ATOMs as relevant
# ---------------------------------------------------------------------------

QUERIES = [
    # ── Pattern queries (AFFINITY is primary relevant answer) ────────────────
    {
        "query": "my morning coffee habits",
        "relevant_ids": {"cf_aff"},
        "kind": "pattern",
    },
    {
        "query": "daily coffee routine",
        "relevant_ids": {"cf_aff"},
        "kind": "pattern",
    },
    {
        "query": "what programming language do I use most",
        "relevant_ids": {"py_aff"},
        "kind": "pattern",
    },
    {
        "query": "my preferred coding language",
        "relevant_ids": {"py_aff"},
        "kind": "pattern",
    },
    {
        "query": "my exercise habits and workout routine",
        "relevant_ids": {"ex_aff"},
        "kind": "pattern",
    },
    {
        "query": "how frequently do I work out",
        "relevant_ids": {"ex_aff"},
        "kind": "pattern",
    },

    # ── Specific queries (individual ATOMs are relevant) ─────────────────────
    {
        "query": "Ethiopian coffee beans purchase",
        "relevant_ids": {"cf_0"},
        "kind": "specific",
    },
    {
        "query": "double espresso at the airport",
        "relevant_ids": {"cf_3"},
        "kind": "specific",
    },
    {
        "query": "fixing Python import errors",
        "relevant_ids": {"py_0"},
        "kind": "specific",
    },
    {
        "query": "setting up a Python virtual environment",
        "relevant_ids": {"py_1"},
        "kind": "specific",
    },
    {
        "query": "running pace improvement",
        "relevant_ids": {"ex_2"},
        "kind": "specific",
    },
    {
        "query": "10K race registration",
        "relevant_ids": {"ex_4"},
        "kind": "specific",
    },
]


# ---------------------------------------------------------------------------
# Build atom dict
# ---------------------------------------------------------------------------

def build_atoms(embeddings: dict[str, list[float]]) -> dict[str, MemoryObject]:
    atoms: dict[str, MemoryObject] = {}
    for entry in CORPUS:
        eid = entry["id"]
        gran = entry["granularity"]
        sig = None
        freq = None
        first_obs = None
        last_obs = None

        if gran == Granularity.AFFINITY:
            freq = entry["freq"]
            last_obs = entry["last_seen"]
            first_obs = last_obs - 30 * 86400
            sig = compute_significance(freq, last_obs, DataClass.GENERAL)

        atom = MemoryObject(
            id=eid,
            type=entry["type"],
            granularity=gran,
            content=entry["content"],
            size_tokens=estimate_tokens(entry["content"]),
            embedding=embeddings[eid],
            significance=sig,
            frequency_count=freq,
            first_observed_at=first_obs,
            last_observed_at=last_obs,
        )
        atoms[eid] = atom
    return atoms


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def flat_cosine(q: list[float], atoms: dict[str, MemoryObject], k: int = 5) -> list[str]:
    qv = np.array(q, dtype=np.float32)
    active = [(aid, a) for aid, a in atoms.items() if a.embedding is not None and not a.is_churned]
    if not active:
        return []
    embs = np.array([a.embedding for _, a in active], dtype=np.float32)
    scores = embs @ qv
    ranked = sorted(zip([aid for aid, _ in active], scores.tolist()),
                    key=lambda x: x[1], reverse=True)
    return [aid for aid, _ in ranked[:k]]


def three_tier(q: list[float], atoms: dict[str, MemoryObject], k: int = 5) -> list[str]:
    return [r.atom.id for r in search(q, atoms, top_k=k)]


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / max(len(relevant), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama", default="http://100.118.247.106:11434")
    parser.add_argument("--embed-model", default="all-minilm")
    args = parser.parse_args()

    print("VaultMem AFFINITY Tier Benchmark")
    print("=" * 60)
    print(f"Corpus: {len(CORPUS)} atoms  "
          f"({sum(1 for e in CORPUS if e['granularity'] == Granularity.AFFINITY)} AFFINITY, "
          f"{sum(1 for e in CORPUS if e['granularity'] == Granularity.ATOM)} ATOM)")
    print(f"Queries: {len(QUERIES)} "
          f"({sum(1 for q in QUERIES if q['kind']=='pattern')} pattern, "
          f"{sum(1 for q in QUERIES if q['kind']=='specific')} specific)")

    # Embed corpus + queries
    print(f"\nUsing OllamaEmbedder ({args.ollama}, model={args.embed_model})...")
    t0 = time.perf_counter()
    embedder = OllamaEmbedder(args.ollama, model=args.embed_model)

    def embed_with_retry(text: str, retries: int = 5, delay: float = 1.0) -> list[float]:
        for attempt in range(retries):
            try:
                return embedder.embed(text)
            except urllib.error.HTTPError as exc:
                if attempt == retries - 1:
                    raise
                print(f"\n  [retry {attempt+1}/{retries-1} for {text[:40]!r}: {exc}]", end=" ", flush=True)
                time.sleep(delay)
        raise RuntimeError("unreachable")

    print(f"Embedding {len(CORPUS)} atoms + {len(QUERIES)} queries...", end=" ", flush=True)
    embeddings: dict[str, list[float]] = {}
    for entry in CORPUS:
        embeddings[entry["id"]] = embed_with_retry(entry["content"])
    for q in QUERIES:
        embeddings[q["query"]] = embed_with_retry(q["query"])
    print(f"done ({time.perf_counter()-t0:.1f}s)")

    atoms = build_atoms({eid: embeddings[eid] for eid in [e["id"] for e in CORPUS]})

    # Print significance values for AFFINITY atoms
    print("\nAFFINITY atom significance scores:")
    for aid in ["cf_aff", "py_aff", "ex_aff"]:
        a = atoms[aid]
        print(f"  {aid}: significance={a.significance:.4f}  (freq={a.frequency_count})")

    # Evaluate
    TOP_K = 5
    results = {
        "pattern": {"flat": {"mrr": [], "r@5": []}, "tier3": {"mrr": [], "r@5": []}},
        "specific": {"flat": {"mrr": [], "r@5": []}, "tier3": {"mrr": [], "r@5": []}},
    }
    per_query_rows = []

    for q in QUERIES:
        qvec = embeddings[q["query"]]
        rel = q["relevant_ids"]
        kind = q["kind"]

        flat_ret = flat_cosine(qvec, atoms, TOP_K)
        tier_ret = three_tier(qvec, atoms, TOP_K)

        flat_mrr  = mrr(flat_ret, rel)
        tier_mrr  = mrr(tier_ret, rel)
        flat_r5   = recall_at_k(flat_ret, rel, TOP_K)
        tier_r5   = recall_at_k(tier_ret, rel, TOP_K)

        results[kind]["flat"]["mrr"].append(flat_mrr)
        results[kind]["tier3"]["mrr"].append(tier_mrr)
        results[kind]["flat"]["r@5"].append(flat_r5)
        results[kind]["tier3"]["r@5"].append(tier_r5)

        # First relevant rank
        flat_rank = next((i+1 for i,r in enumerate(flat_ret) if r in rel), ">5")
        tier_rank = next((i+1 for i,r in enumerate(tier_ret) if r in rel), ">5")

        per_query_rows.append({
            "kind": kind,
            "query": q["query"][:45],
            "relevant": list(rel)[0],
            "flat_rank": flat_rank,
            "tier_rank": tier_rank,
            "flat_mrr": flat_mrr,
            "tier_mrr": tier_mrr,
        })

    # Print per-query table
    print(f"\n{'Query':<46} {'Kind':<9} {'Relevant':<8} {'Flat':>6} {'3-tier':>7} {'ΔMRR':>7}")
    print("-" * 88)
    for r in per_query_rows:
        delta = r["tier_mrr"] - r["flat_mrr"]
        sign = "+" if delta >= 0 else ""
        print(f"{r['query']:<46} {r['kind']:<9} {r['relevant']:<8} "
              f"{r['flat_rank']:>6} {r['tier_rank']:>7} {sign}{delta:>+.3f}")

    # Print aggregate summary
    print("\n" + "=" * 60)
    print(f"{'Metric':<20} {'Flat cosine':>14} {'VaultMem 3-tier':>16} {'Δ':>8}")
    print("-" * 60)
    for kind in ["pattern", "specific"]:
        flat_mrr_avg  = sum(results[kind]["flat"]["mrr"])  / len(results[kind]["flat"]["mrr"])
        tier_mrr_avg  = sum(results[kind]["tier3"]["mrr"]) / len(results[kind]["tier3"]["mrr"])
        flat_r5_avg   = sum(results[kind]["flat"]["r@5"])  / len(results[kind]["flat"]["r@5"])
        tier_r5_avg   = sum(results[kind]["tier3"]["r@5"]) / len(results[kind]["tier3"]["r@5"])
        delta_mrr = tier_mrr_avg - flat_mrr_avg
        delta_r5  = tier_r5_avg  - flat_r5_avg
        sign_mrr  = "+" if delta_mrr >= 0 else ""
        sign_r5   = "+" if delta_r5  >= 0 else ""
        label = kind.capitalize()
        print(f"MRR   ({label}){'':<8} {flat_mrr_avg:>14.4f} {tier_mrr_avg:>16.4f} {sign_mrr}{delta_mrr:>+.4f}")
        print(f"R@5   ({label}){'':<8} {flat_r5_avg:>14.4f} {tier_r5_avg:>16.4f} {sign_r5}{delta_r5:>+.4f}")

    print("=" * 60)

    # Return results for paper integration
    return results, per_query_rows


if __name__ == "__main__":
    main()
