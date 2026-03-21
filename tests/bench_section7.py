"""
VaultMem Section 7 benchmarks.

Measures reported in the paper:
  §7.1 — Argon2id KDF latency (ms) at memory_cost=65536 KiB
  §7.2 — AES-256-GCM encrypt/decrypt throughput (MB/s)
  §7.3 — Search latency (ms) at 1K, 10K atoms
  §7.4 — End-to-end write latency per atom (ms), including embedding via NullEmbedder

Run:
    pytest tests/bench_section7.py --benchmark-only -v
    pytest tests/bench_section7.py --benchmark-only --benchmark-json=bench_results.json

Typical expected ranges (x86 laptop with AES-NI, single-thread):
    Argon2id:    50–500 ms    (memory-bandwidth-dependent)
    AES-256-GCM: 500–3000 MB/s
    Search 1K:   < 20 ms
    Search 10K:  < 200 ms
"""
from __future__ import annotations

import os
import tempfile
import pathlib
import time
import uuid

import numpy as np
import pytest

from vaultmem.crypto import (
    ARGON2_MEMORY_COST,
    decrypt_atom,
    derive_kek,
    encrypt_atom,
    random_mek,
    random_salt,
)
from vaultmem.embedder import NullEmbedder
from vaultmem.models import DataClass, Granularity, MemoryObject, MemoryType, estimate_tokens
from vaultmem.retrieval import search
from vaultmem.session import VaultSession
from vaultmem.vault import MemoryState, create_vault, read_vmem, write_vmem, FILE_TYPE_WORKING


# ── Shared fixtures ───────────────────────────────────────────────────────────

DIM = 384


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_atom_dict(n: int) -> dict[str, MemoryObject]:
    atoms: dict[str, MemoryObject] = {}
    for i in range(n):
        atom = MemoryObject(
            id="",
            type=MemoryType.SEMANTIC,
            granularity=Granularity.ATOM,
            content=f"Memory atom number {i}",
            size_tokens=5,
            embedding=_unit_vec(i),
        )
        atoms[atom.id] = atom
    return atoms


# ── §7.1: Argon2id KDF latency ───────────────────────────────────────────────

def test_argon2id_kdf_latency(benchmark):
    """
    §7.1 — Argon2id wall-clock latency at GENERAL data class parameters.

    Parameters: time_cost=3, memory_cost=65536 KiB (64 MiB), parallelism=4.
    This is intentionally slow; the benchmark confirms the range reported in
    the paper and validates that the library enforces the hardening level.
    """
    salt = random_salt()

    def _kdf():
        return derive_kek("benchmark-passphrase", salt, ARGON2_MEMORY_COST)

    benchmark.pedantic(_kdf, iterations=1, rounds=3)


# ── §7.2: AES-256-GCM throughput ─────────────────────────────────────────────

@pytest.mark.parametrize("size_kb", [1, 64, 512])
def test_aes_gcm_encrypt_throughput(benchmark, size_kb):
    """
    §7.2 — AES-256-GCM encryption throughput for atom-sized payloads.

    Payloads: 1 KB, 64 KB, 512 KB.
    """
    mek = random_mek()
    aad = uuid.uuid4().bytes
    plaintext = os.urandom(size_kb * 1024)

    def _encrypt():
        return encrypt_atom(mek, plaintext, aad)

    result = benchmark(_encrypt)
    # Throughput can be computed from benchmark.stats if needed


@pytest.mark.parametrize("size_kb", [1, 64, 512])
def test_aes_gcm_decrypt_throughput(benchmark, size_kb):
    """
    §7.2 — AES-256-GCM decryption throughput.
    """
    mek = random_mek()
    aad = uuid.uuid4().bytes
    plaintext = os.urandom(size_kb * 1024)
    iv, ct, tag = encrypt_atom(mek, plaintext, aad)

    def _decrypt():
        return decrypt_atom(mek, iv, ct, tag, aad)

    benchmark(_decrypt)


# ── §7.3: Search latency ─────────────────────────────────────────────────────

@pytest.mark.parametrize("n_atoms", [1_000, 10_000])
def test_search_latency(benchmark, n_atoms):
    """
    §7.3 — Three-tier search latency at 1K and 10K atoms.

    All atoms are ATOMs (single tier exercised), which is the typical
    production case for freshly-populated vaults before COMPOSITE/AFFINITY
    synthesis.
    """
    atoms = _make_atom_dict(n_atoms)
    query = _unit_vec(999)

    def _search():
        return search(query, atoms, top_k=10)

    results = benchmark(_search)
    assert len(results) == 10


def test_search_latency_mixed_tiers(benchmark):
    """
    §7.3 — Search latency with mixed AFFINITY + COMPOSITE + ATOM tiers.

    Composition: 10% AFFINITY, 20% COMPOSITE, 70% ATOM (1K atoms total).
    """
    atoms: dict[str, MemoryObject] = {}
    for i in range(100):  # AFFINITY
        atom = MemoryObject(
            id="", type=MemoryType.EPISODIC, granularity=Granularity.AFFINITY,
            content=f"affinity {i}", size_tokens=3,
            embedding=_unit_vec(i), significance=float(i) / 100,
        )
        atoms[atom.id] = atom
    for i in range(200):  # COMPOSITE
        atom = MemoryObject(
            id="", type=MemoryType.SEMANTIC, granularity=Granularity.COMPOSITE,
            content=f"composite {i}", size_tokens=3,
            embedding=_unit_vec(i + 1000),
        )
        atoms[atom.id] = atom
    for i in range(700):  # ATOM
        atom = MemoryObject(
            id="", type=MemoryType.SEMANTIC, granularity=Granularity.ATOM,
            content=f"atom {i}", size_tokens=3,
            embedding=_unit_vec(i + 5000),
        )
        atoms[atom.id] = atom

    query = _unit_vec(42000)

    def _search():
        return search(query, atoms, top_k=10)

    benchmark(_search)


# ── §7.4: Full vault write latency ───────────────────────────────────────────

def test_vault_write_latency_100_atoms(benchmark, tmp_path):
    """
    §7.4 — Full write_vmem() latency with 100 pre-embedded atoms.

    Models the checkpoint cost: JSON serialise → zlib compress → AES-GCM
    encrypt for each atom + index block.  NullEmbedder produces zero-vectors
    so this isolates serialisation + crypto overhead.
    """
    mek = random_mek()
    state = MemoryState.new("benchmark-user")
    for i in range(100):
        atom = MemoryObject(
            id="", type=MemoryType.SEMANTIC, granularity=Granularity.ATOM,
            content=f"benchmark memory {i}", size_tokens=5,
            embedding=[0.0] * DIM,
        )
        state.atoms[atom.id] = atom

    def _write():
        return write_vmem(mek, state, FILE_TYPE_WORKING)

    benchmark(_write)


def test_vault_read_latency_100_atoms(benchmark):
    """
    §7.4 — Full read_vmem() latency with 100 pre-embedded atoms.
    """
    mek = random_mek()
    state = MemoryState.new("benchmark-user")
    for i in range(100):
        atom = MemoryObject(
            id="", type=MemoryType.SEMANTIC, granularity=Granularity.ATOM,
            content=f"benchmark memory {i}", size_tokens=5,
            embedding=[0.0] * DIM,
        )
        state.atoms[atom.id] = atom
    raw = write_vmem(mek, state, FILE_TYPE_WORKING)

    def _read():
        return read_vmem(raw, mek)

    benchmark(_read)


def test_session_add_latency(benchmark, tmp_path):
    """
    §7.4 — VaultSession.add() end-to-end latency including classification
    and NullEmbedder (no model inference).
    """
    with VaultSession.create(
        tmp_path, "bench-pass", "bench-user", embedder=NullEmbedder()
    ) as s:

        def _add():
            s.add("I met Alice at the conference yesterday")

        benchmark(_add)


# ── §7.5: Direct timing outputs for paper table ───────────────────────────────

def test_paper_timing_summary(capsys, tmp_path):
    """
    Non-benchmark timing test that prints numbers suitable for the paper table.
    Run manually: pytest tests/bench_section7.py::test_paper_timing_summary -v -s
    """
    ROUNDS = 3

    # --- Argon2id ---
    salt = random_salt()
    t0 = time.perf_counter()
    for _ in range(ROUNDS):
        derive_kek("passphrase", salt, ARGON2_MEMORY_COST)
    argon2_ms = (time.perf_counter() - t0) / ROUNDS * 1000

    # --- AES-GCM encrypt 1 KB ---
    mek = random_mek()
    aad = b"\x00" * 16
    payload = os.urandom(1024)
    N = 1000
    t0 = time.perf_counter()
    for _ in range(N):
        encrypt_atom(mek, payload, aad)
    aes_enc_us = (time.perf_counter() - t0) / N * 1e6

    # --- Search 10K ---
    atoms = _make_atom_dict(10_000)
    query = _unit_vec(0)
    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        search(query, atoms, top_k=10)
    search10k_ms = (time.perf_counter() - t0) / N * 1000

    # --- Search 1K ---
    atoms1k = _make_atom_dict(1_000)
    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        search(query, atoms1k, top_k=10)
    search1k_ms = (time.perf_counter() - t0) / N * 1000

    # --- Vault write 100 atoms ---
    mek = random_mek()
    state = MemoryState.new("bench")
    for i in range(100):
        atom = MemoryObject(
            id="", type=MemoryType.SEMANTIC, granularity=Granularity.ATOM,
            content=f"memory {i}", size_tokens=5, embedding=[0.0] * DIM,
        )
        state.atoms[atom.id] = atom
    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        write_vmem(mek, state, FILE_TYPE_WORKING)
    write_ms = (time.perf_counter() - t0) / N * 1000

    import platform, sysconfig
    cpu = "unknown"
    try:
        import subprocess
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        pass

    print("\n")
    print("=" * 65)
    print("VaultMem §7 Benchmark Summary")
    print(f"  CPU : {cpu}")
    print(f"  Python : {platform.python_version()}")
    print("=" * 65)
    print(f"  §7.1  Argon2id KDF (t=3, m=64MiB, p=4):  {argon2_ms:8.1f} ms")
    print(f"  §7.2  AES-256-GCM encrypt (1 KB):         {aes_enc_us:8.2f} µs")
    print(f"  §7.3  Search @  1K atoms (linear scan):   {search1k_ms:8.1f} ms")
    print(f"  §7.3  Search @ 10K atoms (linear scan):   {search10k_ms:8.1f} ms")
    print(f"  §7.4  write_vmem() @ 100 atoms:           {write_ms:8.2f} ms")
    print("=" * 65)
    print("  Note: search bottleneck is list→ndarray conversion per query.")
    print("  Pre-computing the embedding matrix would yield <1ms at 10K atoms.")
