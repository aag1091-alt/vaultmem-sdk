"""
VaultMem §7 Retrieval Quality Benchmark — Recall@10 / Precision@10

Pipeline:
  1. Qwen 2.5 14B  → generates 200 synthetic memory atoms + 80 labelled queries
  2. all-minilm    → embeds every atom and query (384-dim, normalised)
  3. Two retrievers compared:
       Baseline  — flat cosine over all ATOMs (no tier structure)
       VaultMem  — three-tier (AFFINITY → COMPOSITE → ATOM)
  4. Metrics: Recall@10, Precision@10, MRR per memory type and overall

Run:
    python tests/bench_recall.py --ollama http://100.118.247.106:11434
    python tests/bench_recall.py --ollama http://100.118.247.106:11434 --skip-gen  # reuse saved corpus
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from vaultmem.embedder import OllamaEmbedder
from vaultmem.models import Granularity, MemoryObject, MemoryType, estimate_tokens
from vaultmem.retrieval import search


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORPUS_PATH = Path(__file__).parent / "recall_corpus.json"
RESULTS_PATH = Path(__file__).parent / "recall_results.json"

MEMORIES_PER_TYPE = 50   # 50 × 4 types = 200 atoms total
QUERIES_PER_TYPE  = 20   # 20 × 4 types = 80 queries total
TOP_K = 10


# ---------------------------------------------------------------------------
# Ollama chat helper (for Qwen corpus generation)
# ---------------------------------------------------------------------------

def ollama_chat(base_url: str, model: str, prompt: str, timeout: int = 600) -> str:
    """Call Ollama /api/chat (streaming NDJSON) and return the full assistant text.

    Reads the response line-by-line so each token arrives immediately.
    The socket timeout applies per-read, not to total generation time.
    """
    import http.client, urllib.parse, io
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {"temperature": 0.7},
    }).encode()
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or 11434
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request(
        "POST", "/api/chat",
        body=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    parts: list[str] = []
    # readline() reads one NDJSON chunk at a time; blocks until \n arrives
    while True:
        raw_line = resp.readline()
        if not raw_line:   # EOF
            break
        line = raw_line.decode("utf-8").strip()
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = chunk.get("message", {}).get("content", "")
        if content:
            parts.append(content)
        if chunk.get("done"):
            break
    conn.close()
    return "".join(parts)


def extract_json(text: str):
    """Extract the first JSON array or object from LLM output.

    Uses json.JSONDecoder.raw_decode which handles strings correctly
    (bracket counting without string awareness is fragile).
    """
    decoder = json.JSONDecoder()
    # Try each position where a JSON value could start
    for start_char in ("[", "{"):
        pos = text.find(start_char)
        while pos != -1:
            try:
                value, _ = decoder.raw_decode(text, pos)
                return value
            except json.JSONDecodeError:
                pos = text.find(start_char, pos + 1)
    raise ValueError(f"No valid JSON found in LLM output:\n{text[:300]}")


# ---------------------------------------------------------------------------
# Step 1: Corpus generation
# ---------------------------------------------------------------------------

MEMORY_TYPE_EXAMPLES = {
    "EPISODIC": [
        "I met Sarah at the Python conference in Austin last month",
        "Yesterday I had a difficult conversation with my manager about the project timeline",
        "Last week I finished reading Atomic Habits — it changed how I think about habits",
    ],
    "SEMANTIC": [
        "Python uses indentation to define code blocks instead of braces",
        "The capital of France is Paris with a population of about 2 million in the city proper",
        "AES-256-GCM is an authenticated encryption algorithm that provides both confidentiality and integrity",
    ],
    "PERSONA": [
        "I always prefer dark mode in all my applications",
        "I tend to work best in the early morning before 9am",
        "I dislike meetings that could have been emails",
    ],
    "PROCEDURAL": [
        "To deploy the app: run tests first, then push to main, then wait for CI to pass before merging",
        "When I feel overwhelmed: first write down all tasks, then pick the smallest one and start",
        "To set up a new Python project: create a venv, install deps, write pyproject.toml",
    ],
}

MEMORY_TYPE_DESCRIPTIONS = {
    "EPISODIC": "specific personal events, meetings, or experiences anchored to a time",
    "SEMANTIC": "timeless facts, knowledge, or definitions with no personal time anchor",
    "PERSONA": "stable personal preferences, habits, traits, or identity statements",
    "PROCEDURAL": "step-by-step workflows, how-to instructions, or processes",
}


def generate_memories(base_url: str, model: str, mem_type: str, count: int) -> list[str]:
    """Ask Qwen to generate `count` memory atoms of the given type."""
    examples = "\n".join(f'  - "{e}"' for e in MEMORY_TYPE_EXAMPLES[mem_type])
    desc = MEMORY_TYPE_DESCRIPTIONS[mem_type]
    prompt = f"""Generate {count} realistic, diverse personal memory facts of type {mem_type}.
Definition: {desc}

Examples:
{examples}

Rules:
- Each memory must be a single sentence, self-contained atomic fact
- Be specific and realistic — vary topics, people, places, domains
- Do NOT repeat the examples above
- Return ONLY a JSON array of {count} strings, no explanation

Output format:
["memory 1", "memory 2", ...]"""

    print(f"  Generating {count} {mem_type} memories...", end=" ", flush=True)
    t0 = time.perf_counter()
    raw = ollama_chat(base_url, model, prompt, timeout=600)
    elapsed = time.perf_counter() - t0
    memories = extract_json(raw)
    assert isinstance(memories, list), f"Expected list, got {type(memories)}"
    memories = [str(m).strip() for m in memories if str(m).strip()]
    print(f"{len(memories)} generated in {elapsed:.1f}s")
    return memories[:count]


def generate_queries(
    base_url: str,
    model: str,
    mem_type: str,
    memories: list[dict],  # [{"id": str, "content": str}]
    count: int,
) -> list[dict]:
    """Ask Qwen to generate queries + relevance labels for a set of memories."""
    mem_list = "\n".join(
        f'  ID={m["id"]}: {m["content"]}' for m in memories
    )
    prompt = f"""You are a memory retrieval evaluator.

Below are {len(memories)} personal memory facts (type: {mem_type}):
{mem_list}

Task: Generate {count} natural questions or search queries a user might ask when looking for information in their memory system. For each query, list the IDs of memories that are genuinely relevant (directly answer or strongly relate to the query).

Rules:
- Queries should be short natural-language phrases or questions
- Each query must have at least 1 and at most 6 relevant memory IDs
- Only include memory IDs that clearly match — be strict about relevance
- Vary query topics across the full range of memories above
- Return ONLY a JSON array, no explanation

Output format:
[
  {{"query": "query text here", "relevant_ids": ["id1", "id2"]}},
  ...
]"""

    print(f"  Generating {count} queries for {mem_type}...", end=" ", flush=True)
    t0 = time.perf_counter()
    raw = ollama_chat(base_url, model, prompt, timeout=600)
    elapsed = time.perf_counter() - t0
    try:
        queries = extract_json(raw)
    except ValueError as e:
        print(f"\n  [WARN] JSON parse failed: {e}")
        queries = []
    if not isinstance(queries, list):
        print(f"\n  [WARN] Expected list, got {type(queries).__name__}: {str(queries)[:200]}")
        queries = []

    # Validate and clean
    valid = []
    valid_ids = {m["id"] for m in memories}
    for q in queries:
        if not isinstance(q, dict):
            continue
        text = str(q.get("query", "")).strip()
        rel_ids = [r for r in q.get("relevant_ids", []) if r in valid_ids]
        if text and rel_ids:
            valid.append({"query": text, "relevant_ids": rel_ids, "type": mem_type})
    print(f"{len(valid)} valid queries in {elapsed:.1f}s")
    return valid[:count]


def build_corpus(base_url: str, chat_model: str) -> dict:
    """Generate the full labelled corpus using Qwen."""
    print("\n=== Generating synthetic corpus ===")
    all_memories = []
    all_queries = []
    mem_id = 0

    for mem_type in ["EPISODIC", "SEMANTIC", "PERSONA", "PROCEDURAL"]:
        print(f"\n[{mem_type}]")
        contents = generate_memories(base_url, chat_model, mem_type, MEMORIES_PER_TYPE)
        typed_memories = []
        for content in contents:
            atom_id = f"m{mem_id:04d}"
            typed_memories.append({
                "id": atom_id,
                "content": content,
                "type": mem_type,
            })
            mem_id += 1
        all_memories.extend(typed_memories)

        queries = generate_queries(
            base_url, chat_model, mem_type, typed_memories, QUERIES_PER_TYPE
        )
        all_queries.extend(queries)

    corpus = {"memories": all_memories, "queries": all_queries}
    CORPUS_PATH.write_text(json.dumps(corpus, indent=2, ensure_ascii=False))
    print(f"\nCorpus saved → {CORPUS_PATH}")
    print(f"  {len(all_memories)} memories, {len(all_queries)} queries")
    return corpus


# ---------------------------------------------------------------------------
# Step 2: Embed corpus
# ---------------------------------------------------------------------------

def embed_corpus(corpus: dict, embedder: OllamaEmbedder) -> tuple[dict, dict]:
    """
    Returns:
        memory_embeddings: {id: list[float]}
        query_embeddings:  {query_text: list[float]}
    """
    print("\n=== Embedding corpus ===")

    memories = corpus["memories"]
    queries = corpus["queries"]

    print(f"  Embedding {len(memories)} memories...", end=" ", flush=True)
    t0 = time.perf_counter()
    mem_vecs = embedder.embed_batch([m["content"] for m in memories])
    print(f"done in {time.perf_counter() - t0:.1f}s")

    unique_queries = list({q["query"] for q in queries})
    print(f"  Embedding {len(unique_queries)} queries...", end=" ", flush=True)
    t0 = time.perf_counter()
    # Embed one-by-one with retry so a single 500 doesn't abort everything
    q_vecs: list[list[float] | None] = []
    failed = 0
    for qtext in unique_queries:
        for attempt in range(3):
            try:
                q_vecs.append(embedder.embed(qtext))
                break
            except Exception:
                if attempt == 2:
                    q_vecs.append(None)
                    failed += 1
    skipped = f", {failed} skipped" if failed else ""
    print(f"done in {time.perf_counter() - t0:.1f}s{skipped}")

    mem_embeddings = {m["id"]: v for m, v in zip(memories, mem_vecs)}
    q_embeddings = {
        q: v for q, v in zip(unique_queries, q_vecs) if v is not None
    }
    return mem_embeddings, q_embeddings


# ---------------------------------------------------------------------------
# Step 3: Build atom dicts for retrieval
# ---------------------------------------------------------------------------

def build_atom_dict(corpus: dict, mem_embeddings: dict) -> dict[str, MemoryObject]:
    """Convert corpus memories into MemoryObject ATOMs with embeddings."""
    atoms = {}
    type_map = {
        "EPISODIC":   MemoryType.EPISODIC,
        "SEMANTIC":   MemoryType.SEMANTIC,
        "PERSONA":    MemoryType.PERSONA,
        "PROCEDURAL": MemoryType.PROCEDURAL,
    }
    for m in corpus["memories"]:
        atom = MemoryObject(
            id=m["id"],
            type=type_map[m["type"]],
            granularity=Granularity.ATOM,
            content=m["content"],
            size_tokens=estimate_tokens(m["content"]),
            embedding=mem_embeddings[m["id"]],
        )
        atoms[atom.id] = atom
    return atoms


# ---------------------------------------------------------------------------
# Step 4: Flat cosine baseline
# ---------------------------------------------------------------------------

def flat_cosine_search(
    query_vec: list[float],
    atoms: dict[str, MemoryObject],
    top_k: int = 10,
) -> list[str]:
    """Simple flat cosine — no tier structure. Returns top-k atom IDs."""
    q = np.array(query_vec, dtype=np.float32)
    active = [(aid, a) for aid, a in atoms.items() if a.embedding is not None]
    embs = np.array([a.embedding for _, a in active], dtype=np.float32)
    scores = embs @ q
    ranked = sorted(zip([aid for aid, _ in active], scores.tolist()),
                    key=lambda x: x[1], reverse=True)
    return [aid for aid, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Step 5: Evaluate
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / min(len(relevant), k)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int = 10) -> float:
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / k


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def evaluate(
    corpus: dict,
    atoms: dict[str, MemoryObject],
    q_embeddings: dict,
    top_k: int = 10,
) -> dict:
    """
    Evaluate both retrieval methods on all queries.
    Returns nested dict: method → type → metric → value.
    """
    methods = {
        "flat_cosine": lambda q_vec: flat_cosine_search(q_vec, atoms, top_k),
        "vaultmem_3tier": lambda q_vec: [
            r.atom.id for r in search(q_vec, atoms, top_k=top_k)
        ],
    }

    results = {m: {t: {"recall": [], "precision": [], "mrr": []}
                   for t in ["EPISODIC", "SEMANTIC", "PERSONA", "PROCEDURAL", "ALL"]}
               for m in methods}

    for q in corpus["queries"]:
        q_text = q["query"]
        q_type = q["type"]
        relevant = set(q["relevant_ids"])
        if q_text not in q_embeddings:
            continue  # embedding failed for this query
        q_vec = q_embeddings[q_text]

        for method_name, retriever in methods.items():
            retrieved = retriever(q_vec)
            r10  = recall_at_k(retrieved, relevant, top_k)
            p10  = precision_at_k(retrieved, relevant, top_k)
            mrr  = reciprocal_rank(retrieved, relevant)

            results[method_name][q_type]["recall"].append(r10)
            results[method_name][q_type]["precision"].append(p10)
            results[method_name][q_type]["mrr"].append(mrr)

            results[method_name]["ALL"]["recall"].append(r10)
            results[method_name]["ALL"]["precision"].append(p10)
            results[method_name]["ALL"]["mrr"].append(mrr)

    # Aggregate
    summary = {}
    for method in methods:
        summary[method] = {}
        for t in ["EPISODIC", "SEMANTIC", "PERSONA", "PROCEDURAL", "ALL"]:
            vals = results[method][t]
            n = len(vals["recall"])
            summary[method][t] = {
                "n_queries": n,
                "recall@10": round(sum(vals["recall"]) / n, 4) if n else 0,
                "precision@10": round(sum(vals["precision"]) / n, 4) if n else 0,
                "mrr": round(sum(vals["mrr"]) / n, 4) if n else 0,
            }
    return summary


# ---------------------------------------------------------------------------
# Step 6: Print results table
# ---------------------------------------------------------------------------

def print_results(summary: dict) -> None:
    print("\n" + "=" * 72)
    print("VaultMem §7 Retrieval Quality Benchmark  (top-10)")
    print("=" * 72)

    col_w = 12
    types = ["EPISODIC", "SEMANTIC", "PERSONA", "PROCEDURAL", "ALL"]
    methods = list(summary.keys())
    method_labels = {"flat_cosine": "Flat cosine", "vaultmem_3tier": "VaultMem 3-tier"}

    for metric_key, metric_label in [
        ("recall@10", "Recall@10"),
        ("precision@10", "Precision@10"),
        ("mrr", "MRR"),
    ]:
        print(f"\n  {metric_label}")
        header = f"  {'Method':<18}" + "".join(f"{t:>{col_w}}" for t in types)
        print(header)
        print("  " + "-" * (18 + col_w * len(types)))
        for method in methods:
            label = method_labels.get(method, method)
            row = f"  {label:<18}"
            for t in types:
                v = summary[method][t][metric_key]
                row += f"{v:>{col_w}.4f}"
            print(row)

    print("\n  n_queries per type:")
    for t in types:
        n = summary[methods[0]][t]["n_queries"]
        print(f"    {t}: {n}")

    # Delta
    if "flat_cosine" in summary and "vaultmem_3tier" in summary:
        print("\n  VaultMem vs Flat Cosine  (Δ Recall@10)")
        for t in types:
            vm  = summary["vaultmem_3tier"][t]["recall@10"]
            fc  = summary["flat_cosine"][t]["recall@10"]
            delta = vm - fc
            sign = "+" if delta >= 0 else ""
            print(f"    {t:<12}: {sign}{delta:+.4f}")

    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VaultMem Recall@10 Benchmark")
    parser.add_argument("--ollama", default="http://100.118.247.106:11434",
                        help="Ollama base URL")
    parser.add_argument("--chat-model", default="qwen2.5:14b",
                        help="Ollama chat model for corpus generation")
    parser.add_argument("--embed-model", default="all-minilm",
                        help="Ollama embedding model")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Skip corpus generation, reuse saved recall_corpus.json")
    args = parser.parse_args()

    print(f"Ollama: {args.ollama}")
    print(f"Chat model: {args.chat_model}  |  Embed model: {args.embed_model}")

    # Step 1: Corpus
    if args.skip_gen and CORPUS_PATH.exists():
        print(f"\nLoading existing corpus from {CORPUS_PATH}")
        corpus = json.loads(CORPUS_PATH.read_text())
        print(f"  {len(corpus['memories'])} memories, {len(corpus['queries'])} queries")
    else:
        corpus = build_corpus(args.ollama, args.chat_model)

    # Step 2: Embed
    embedder = OllamaEmbedder(args.ollama, model=args.embed_model)
    mem_embeddings, q_embeddings = embed_corpus(corpus, embedder)

    # Step 3: Build atom dict
    atoms = build_atom_dict(corpus, mem_embeddings)
    print(f"\n  Atom dict: {len(atoms)} ATOMs")

    # Step 4: Evaluate
    print("\n=== Running retrieval evaluation ===")
    t0 = time.perf_counter()
    summary = evaluate(corpus, atoms, q_embeddings, top_k=TOP_K)
    print(f"  Evaluation complete in {time.perf_counter() - t0:.2f}s")

    # Step 5: Print
    print_results(summary)

    # Save
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
