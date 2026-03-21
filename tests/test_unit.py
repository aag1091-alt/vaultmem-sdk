"""
VaultMem unit tests.

Run with:  pytest tests/test_unit.py -v
"""
from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

from vaultmem.classifier import classify, classify_with_features
from vaultmem.crypto import (
    ARGON2_MEMORY_COST,
    derive_kek,
    encrypt_atom,
    decrypt_atom,
    encrypt_index,
    decrypt_index,
    pack_encrypted_block,
    unpack_encrypted_block,
    random_mek,
    random_salt,
    unwrap_mek,
    wrap_mek,
)
from vaultmem.embedder import NullEmbedder
from vaultmem.exceptions import VaultLockedError, VaultTamperedError, WrongPassphraseError
from vaultmem.models import (
    DataClass,
    Granularity,
    MemoryObject,
    MemoryType,
    compute_significance,
    estimate_tokens,
)
from vaultmem.retrieval import search
from vaultmem.session import VaultSession
from vaultmem.vault import (
    FILE_TYPE_WORKING,
    MemoryState,
    create_vault,
    read_vmem,
    write_vmem,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

DIM = 384


def unit_vec(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def make_atom(
    content: str = "test",
    mtype: MemoryType = MemoryType.SEMANTIC,
    granularity: Granularity = Granularity.ATOM,
    embedding: list[float] | None = None,
    seed: int = 0,
) -> MemoryObject:
    if embedding is None:
        embedding = unit_vec(seed)
    return MemoryObject(
        id="",
        type=mtype,
        granularity=granularity,
        content=content,
        size_tokens=estimate_tokens(content),
        embedding=embedding,
    )


# ── Classifier ────────────────────────────────────────────────────────────────

class TestClassifier:
    def test_episodic_temporal_marker(self):
        t, _ = classify_with_features("I met Alice yesterday at the conference")
        assert t == MemoryType.EPISODIC

    def test_episodic_past_tense_narrative(self):
        t, _ = classify_with_features("I went to the office and met the team")
        assert t == MemoryType.EPISODIC

    def test_persona_habitual(self):
        t, _ = classify_with_features("I always prefer concise answers")
        assert t == MemoryType.PERSONA

    def test_persona_preference_verb(self):
        t, _ = classify_with_features("I love working with Python")
        assert t == MemoryType.PERSONA

    def test_procedural_sequential(self):
        t, _ = classify_with_features(
            "First, open the terminal. Then run pip install."
        )
        assert t == MemoryType.PROCEDURAL

    def test_semantic_default(self):
        t, _ = classify_with_features("Python is a high-level programming language")
        assert t == MemoryType.SEMANTIC

    def test_features_dict_keys(self):
        _, features = classify_with_features("test")
        assert set(features) == {"F1", "F2", "F3", "F4"}

    def test_deterministic(self):
        text = "I usually prefer dark mode in my editors"
        assert classify(text) == classify(text)


# ── Crypto ───────────────────────────────────────────────────────────────────

class TestCrypto:
    def test_atom_roundtrip(self):
        mek = random_mek()
        plaintext = b"hello atom"
        aad = b"\x00" * 16
        iv, ct, tag = encrypt_atom(mek, plaintext, aad)
        result = decrypt_atom(mek, iv, ct, tag, aad)
        assert result == plaintext

    def test_atom_tamper_detection(self):
        mek = random_mek()
        aad = b"\x00" * 16
        iv, ct, tag = encrypt_atom(mek, b"secret", aad)
        corrupted = bytes([ct[0] ^ 0xFF]) + ct[1:]
        with pytest.raises(VaultTamperedError):
            decrypt_atom(mek, iv, corrupted, tag, aad)

    def test_index_roundtrip(self):
        mek = random_mek()
        header = b"\xAB" * 48
        plaintext = b'{"owner": "alice"}'
        iv, ct, tag = encrypt_index(mek, plaintext, header)
        result = decrypt_index(mek, iv, ct, tag, header)
        assert result == plaintext

    def test_index_header_tamper(self):
        mek = random_mek()
        header = b"\xAB" * 48
        iv, ct, tag = encrypt_index(mek, b"index", header)
        bad_header = bytes([header[0] ^ 0x01]) + header[1:]
        with pytest.raises(VaultTamperedError):
            decrypt_index(mek, iv, ct, tag, bad_header)

    def test_mek_wrap_unwrap(self):
        kek = random_mek()
        mek = random_mek()
        iv, ct, tag = wrap_mek(kek, mek)
        recovered = unwrap_mek(kek, iv, ct, tag)
        assert recovered == mek

    def test_wrong_passphrase(self):
        salt = random_salt()
        kek_good = derive_kek("correct", salt, ARGON2_MEMORY_COST)
        kek_bad  = derive_kek("wrong",   salt, ARGON2_MEMORY_COST)
        mek = random_mek()
        iv, ct, tag = wrap_mek(kek_good, mek)
        with pytest.raises(WrongPassphraseError):
            unwrap_mek(kek_bad, iv, ct, tag)

    def test_block_pack_unpack(self):
        iv  = b"\x01" * 12
        ct  = b"\x02" * 50
        tag = b"\x03" * 16
        packed = pack_encrypted_block(iv, ct, tag)
        iv2, ct2, tag2, next_off = unpack_encrypted_block(packed)
        assert iv2 == iv
        assert ct2 == ct
        assert tag2 == tag
        assert next_off == len(packed)


# ── Vault format ─────────────────────────────────────────────────────────────

class TestVaultFormat:
    def test_empty_vault_roundtrip(self):
        mek = random_mek()
        state = MemoryState.new("alice")
        raw = write_vmem(mek, state, FILE_TYPE_WORKING)
        state2 = read_vmem(raw, mek)
        assert state2.owner == "alice"
        assert len(state2.atoms) == 0

    def test_atoms_roundtrip(self):
        mek = random_mek()
        state = MemoryState.new("bob")
        for i in range(5):
            atom = make_atom(content=f"memory {i}", seed=i)
            state.atoms[atom.id] = atom
        raw = write_vmem(mek, state, FILE_TYPE_WORKING)
        state2 = read_vmem(raw, mek)
        assert len(state2.atoms) == 5
        contents = {a.content for a in state2.atoms.values()}
        assert contents == {f"memory {i}" for i in range(5)}

    def test_wrong_mek_raises(self):
        mek1 = random_mek()
        mek2 = random_mek()
        state = MemoryState.new("alice")
        atom = make_atom()
        state.atoms[atom.id] = atom
        raw = write_vmem(mek1, state, FILE_TYPE_WORKING)
        with pytest.raises(VaultTamperedError):
            read_vmem(raw, mek2)

    def test_working_copy_append_only(self, tmp_path):
        """Checkpoint should only append new atoms, not rewrite existing ones."""
        from vaultmem.vault import append_atoms_to_file, write_index_file, read_working_copy
        state, mek = create_vault(tmp_path, "pass", "alice")
        atoms_path = tmp_path / "current.atoms"
        vmem_path  = tmp_path / "current.vmem"

        # Add first batch of atoms and checkpoint
        for i in range(3):
            atom = make_atom(content=f"first batch {i}", seed=i)
            state.atoms[atom.id] = atom
        new_offsets = append_atoms_to_file(
            atoms_path, mek, state.atoms, state.new_atom_ids(), 0
        )
        state._atom_offsets.update(new_offsets)
        state._atoms_file_size = sum(sz for _, sz in new_offsets.values())
        write_index_file(vmem_path, mek, state)
        size_after_first = atoms_path.stat().st_size

        # Add a second atom and checkpoint
        atom4 = make_atom(content="second batch", seed=99)
        state.atoms[atom4.id] = atom4
        new_offsets2 = append_atoms_to_file(
            atoms_path, mek, state.atoms, state.new_atom_ids(),
            start_offset=state._atoms_file_size,
        )
        state._atom_offsets.update(new_offsets2)
        state._atoms_file_size += sum(sz for _, sz in new_offsets2.values())
        write_index_file(vmem_path, mek, state)

        # The atoms file must have grown (only one atom appended)
        size_after_second = atoms_path.stat().st_size
        assert size_after_second > size_after_first

        # The new atom's offset must start exactly where the first batch ended
        assert new_offsets2[atom4.id][0] == size_after_first

        # Round-trip: read back and verify all 4 atoms
        state2 = read_working_copy(vmem_path, atoms_path, mek)
        assert len(state2.atoms) == 4
        assert state2._atoms_file_size == size_after_second

    def test_create_vault(self, tmp_path):
        state, mek = create_vault(tmp_path, "passphrase", "alice")
        assert (tmp_path / "meta.json").exists()
        assert (tmp_path / "current.vmem").exists()
        assert (tmp_path / "current.atoms").exists()   # split working copy
        assert (tmp_path / "snapshots").is_dir()
        assert state.owner == "alice"


# ── Retrieval ────────────────────────────────────────────────────────────────

class TestRetrieval:
    def _make_atoms_dict(self, n: int) -> dict[str, MemoryObject]:
        atoms = {}
        for i in range(n):
            atom = make_atom(content=f"memory {i}", seed=i)
            atoms[atom.id] = atom
        return atoms

    def test_returns_top_k(self):
        atoms = self._make_atoms_dict(20)
        results = search(unit_vec(99), atoms, top_k=5)
        assert len(results) <= 5

    def test_sorted_descending(self):
        atoms = self._make_atoms_dict(10)
        results = search(unit_vec(0), atoms, top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_type_filter(self):
        atoms = {}
        for mtype, n_seeds in [
            (MemoryType.EPISODIC, range(0, 3)),
            (MemoryType.SEMANTIC, range(3, 6)),
        ]:
            for i in n_seeds:
                atom = make_atom(mtype=mtype, seed=i)
                atoms[atom.id] = atom
        results = search(unit_vec(0), atoms, memory_type=MemoryType.EPISODIC)
        assert all(r.atom.type == MemoryType.EPISODIC for r in results)

    def test_churned_excluded(self):
        atoms = self._make_atoms_dict(5)
        first_id = next(iter(atoms))
        atoms[first_id].is_churned = True
        results = search(unit_vec(0), atoms, top_k=10)
        returned_ids = {r.atom.id for r in results}
        assert first_id not in returned_ids

    def test_no_embedding_excluded(self):
        atoms = self._make_atoms_dict(3)
        first_id = next(iter(atoms))
        atoms[first_id].embedding = None
        results = search(unit_vec(0), atoms, top_k=10)
        returned_ids = {r.atom.id for r in results}
        assert first_id not in returned_ids

    def test_no_duplicate_ids(self):
        # Mix AFFINITY + COMPOSITE + ATOM; each should appear once
        atoms = {}
        for gran, seed in [
            (Granularity.AFFINITY, 0),
            (Granularity.COMPOSITE, 1),
            (Granularity.ATOM, 2),
        ]:
            atom = make_atom(granularity=gran, seed=seed)
            atom.significance = 0.8
            atoms[atom.id] = atom
        results = search(unit_vec(0), atoms, top_k=10)
        ids = [r.atom.id for r in results]
        assert len(ids) == len(set(ids))

    def test_affinity_tier_label(self):
        atoms = {}
        atom = make_atom(granularity=Granularity.AFFINITY, seed=0)
        atom.significance = 0.9
        atoms[atom.id] = atom
        results = search(unit_vec(0), atoms)
        assert results[0].tier == "AFFINITY"


# ── Session ───────────────────────────────────────────────────────────────────

class TestSession:
    def test_create_and_open(self, tmp_path):
        with VaultSession.create(
            tmp_path, "pass", "alice", embedder=NullEmbedder()
        ) as s:
            a = s.add("test memory")
            assert a.id
            assert s.atom_count == 1

        with VaultSession.open(tmp_path, "pass", embedder=NullEmbedder()) as s:
            assert s.atom_count == 1

    def test_wrong_passphrase(self, tmp_path):
        with VaultSession.create(tmp_path, "correct", "bob", embedder=NullEmbedder()):
            pass
        with pytest.raises(WrongPassphraseError):
            VaultSession.open(tmp_path, "wrong", embedder=NullEmbedder())

    def test_double_open_lock(self, tmp_path):
        with VaultSession.create(tmp_path, "pass", "carol", embedder=NullEmbedder()):
            pass
        s1 = VaultSession.open(tmp_path, "pass", embedder=NullEmbedder())
        try:
            with pytest.raises(VaultLockedError):
                VaultSession.open(tmp_path, "pass", embedder=NullEmbedder())
        finally:
            s1.close()

    def test_close_is_idempotent(self, tmp_path):
        with VaultSession.create(tmp_path, "pass", "dave", embedder=NullEmbedder()) as s:
            pass
        s.close()  # second call must not raise

    def test_flush_persists(self, tmp_path):
        with VaultSession.create(tmp_path, "p", "eve", embedder=NullEmbedder()) as s:
            s.add("memory before flush")
            s.flush()
            s.add("memory after flush")
            # Not flushed yet — but close() will flush

        with VaultSession.open(tmp_path, "p", embedder=NullEmbedder()) as s:
            assert s.atom_count == 2

    def test_search_returns_results(self, tmp_path):
        with VaultSession.create(tmp_path, "p", "frank", embedder=NullEmbedder()) as s:
            for i in range(5):
                s.add(f"memory content {i}")
            results = s.search("memory")
            # NullEmbedder → all-zero embeddings → equal cosine scores
            assert len(results) == 5

    def test_auto_classify(self, tmp_path):
        with VaultSession.create(tmp_path, "p", "grace", embedder=NullEmbedder()) as s:
            a = s.add("I met Dan yesterday at the summit")
            assert a.type == MemoryType.EPISODIC

    def test_repr(self, tmp_path):
        with VaultSession.create(tmp_path, "p", "hank", embedder=NullEmbedder()) as s:
            r = repr(s)
            assert "hank" in r
            assert "OPEN" in r


# ── Models / significance ─────────────────────────────────────────────────────

class TestModels:
    def test_significance_range(self):
        import time
        now = int(time.time())
        sig = compute_significance(5, now, DataClass.GENERAL)
        assert 0.0 < sig <= 1.0

    def test_significance_increases_with_frequency(self):
        import time
        now = int(time.time())
        s1 = compute_significance(1, now)
        s5 = compute_significance(5, now)
        s20 = compute_significance(20, now)
        assert s1 < s5 < s20

    def test_significance_decays_over_time(self):
        import time
        now = int(time.time())
        old = now - 365 * 86400  # 1 year ago
        sig_recent = compute_significance(5, now)
        sig_old    = compute_significance(5, old)
        assert sig_recent > sig_old

    def test_estimate_tokens(self):
        assert estimate_tokens("hello") == 1
        assert estimate_tokens("hello world test") >= 3
