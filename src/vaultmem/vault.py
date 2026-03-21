"""
VaultMem .vmem file format — read/write.

Working copy layout (file_type=0x01):
    current.vmem  — 48B header + encrypted index block (header + index only)
    current.atoms — encrypted atom blocks, append-only

Snapshot / export layout (single file, file_type=0x02 or 0x03):
    [48B header, plaintext]
    [credential block, 80B, plaintext — exports only]
    [encrypted index block: IV + len + index_json + tag, AAD=header]
    [encrypted atom blocks: per-atom IV + len + atom_json + tag, AAD=uuid_bytes]

The key difference: working copies separate atom storage from the index so that
checkpoints only append new atoms (O(new)) rather than rewriting all atoms
(O(total)). Atoms are immutable once written; only the index changes on flush.

Header format (48 bytes, big-endian):
    4B  magic            = 0x564D454D ("VMEM")
    2B  format_version   = 0x0001
    1B  encryption_algo  = 0x01 (AES-256-GCM)
    1B  kdf              = 0x01 (Argon2id)
    1B  file_type        = 0x01 | 0x02 | 0x03
    1B  reserved_1       = 0x00
    2B  reserved_2       = 0x0000
    4B  index_offset     (byte offset of index block start)
    4B  index_size       (total size of index block in bytes: IV+len+ct+tag)
    4B  atom_count
    4B  format_flags     (bit 0: key_wrapping, bit 1: compression, bit 2: per_atom_iv)
    16B file_id          (vault UUID, bytes)
    4B  reserved_3       = 0x00000000
"""
from __future__ import annotations

import json
import os
import struct
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .crypto import (
    encrypt_atom, decrypt_atom,
    encrypt_index, decrypt_index,
    pack_encrypted_block, unpack_encrypted_block,
    encrypted_block_size,
    wrap_mek, unwrap_mek,
    random_salt, random_mek,
    IV_LEN, TAG_LEN,
)
from .exceptions import VaultTamperedError
from .models import MemoryObject


# ---------------------------------------------------------------------------
# Header constants
# ---------------------------------------------------------------------------

VMEM_MAGIC       = b"VMEM"
FORMAT_VERSION   = 0x0001
ALGO_AES256_GCM  = 0x01
KDF_ARGON2ID     = 0x01
HEADER_SIZE      = 48

FILE_TYPE_WORKING  = 0x01
FILE_TYPE_SNAPSHOT = 0x02
FILE_TYPE_EXPORT   = 0x03

FORMAT_FLAG_KEY_WRAPPING  = 0b001
FORMAT_FLAG_COMPRESSION   = 0b010
FORMAT_FLAG_PER_ATOM_IV   = 0b100
DEFAULT_FORMAT_FLAGS = FORMAT_FLAG_KEY_WRAPPING | FORMAT_FLAG_COMPRESSION | FORMAT_FLAG_PER_ATOM_IV

# Credential block size for exports:
# 16B kdf_salt + 12B wrapped_mek_iv + 4B ct_len + 32B ct + 16B tag = 80 bytes
CREDENTIAL_BLOCK_SIZE = 80

HEADER_STRUCT = struct.Struct(">4sHBBBBHIIII16sI")
assert HEADER_STRUCT.size == HEADER_SIZE


# ---------------------------------------------------------------------------
# MemoryState — in-RAM representation of a vault session
# ---------------------------------------------------------------------------

@dataclass
class MemoryState:
    """Complete decrypted vault contents, held in RAM during a session."""
    owner: str
    file_id: str                               # UUID of the vault file
    atoms: dict[str, MemoryObject] = field(default_factory=dict)  # id → MemoryObject
    created_at: int = 0
    schema_version: int = 1
    _dirty: bool = field(default=False, repr=False)
    # Tracks which atoms are already written to current.atoms and their byte offsets.
    # Atoms not present here are "new" and must be appended on the next checkpoint.
    _atom_offsets: dict[str, tuple[int, int]] = field(default_factory=dict, repr=False)
    # Total bytes currently in current.atoms (sum of all written block sizes).
    _atoms_file_size: int = field(default=0, repr=False)

    @property
    def active_atoms(self) -> dict[str, MemoryObject]:
        """Non-churned atoms only."""
        return {k: v for k, v in self.atoms.items() if not v.is_churned}

    def mark_dirty(self) -> None:
        self._dirty = True

    def mark_clean(self) -> None:
        self._dirty = False

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def new_atom_ids(self) -> list[str]:
        """Atom IDs not yet written to current.atoms (in insertion order)."""
        return [aid for aid in self.atoms if aid not in self._atom_offsets]

    def to_index_dict(self) -> dict:
        """Serialize to the index-block JSON used by the split working copy format.

        Requires all atoms to have been written to current.atoms first
        (i.e., new_atom_ids() must be empty).
        """
        return {
            "owner": self.owner,
            "file_id": self.file_id,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
            "atom_map": [
                {
                    "id": aid,
                    "offset": self._atom_offsets[aid][0],
                    "size": self._atom_offsets[aid][1],
                }
                for aid in self.atoms
            ],
        }

    @classmethod
    def new(cls, owner: str) -> "MemoryState":
        import time
        return cls(owner=owner, file_id=str(uuid.uuid4()), created_at=int(time.time()))


# ---------------------------------------------------------------------------
# Header packing / unpacking
# ---------------------------------------------------------------------------

def pack_header(
    file_type: int,
    index_offset: int,
    index_size: int,
    atom_count: int,
    file_id: str,
    format_flags: int = DEFAULT_FORMAT_FLAGS,
) -> bytes:
    """Build the 48-byte plaintext header."""
    file_id_bytes = uuid.UUID(file_id).bytes
    return HEADER_STRUCT.pack(
        VMEM_MAGIC,
        FORMAT_VERSION,
        ALGO_AES256_GCM,
        KDF_ARGON2ID,
        file_type,
        0,               # reserved_1
        0,               # reserved_2
        index_offset,
        index_size,
        atom_count,
        format_flags,
        file_id_bytes,
        0,               # reserved_3
    )


def unpack_header(data: bytes) -> dict:
    """Parse the 48-byte header. Returns a dict of fields."""
    if len(data) < HEADER_SIZE:
        raise VaultTamperedError(f"Header too short: {len(data)} bytes")
    fields = HEADER_STRUCT.unpack(data[:HEADER_SIZE])
    magic, version, algo, kdf, file_type, _, _, index_offset, index_size, atom_count, \
        format_flags, file_id_bytes, _ = fields
    if magic != VMEM_MAGIC:
        raise VaultTamperedError(f"Invalid magic bytes: {magic!r}")
    if version != FORMAT_VERSION:
        raise VaultTamperedError(f"Unsupported format version: {version}")
    return {
        "format_version": version,
        "encryption_algo": algo,
        "kdf": kdf,
        "file_type": file_type,
        "index_offset": index_offset,
        "index_size": index_size,
        "atom_count": atom_count,
        "format_flags": format_flags,
        "file_id": str(uuid.UUID(bytes=file_id_bytes)),
    }


# ---------------------------------------------------------------------------
# Credential block (exports only)
# ---------------------------------------------------------------------------

def pack_credential_block(kdf_salt: bytes, wrapped_iv: bytes, wrapped_ct: bytes, wrapped_tag: bytes) -> bytes:
    """Pack the 80-byte credential block for export files."""
    ct_len = struct.pack(">I", len(wrapped_ct))
    block = kdf_salt + wrapped_iv + ct_len + wrapped_ct + wrapped_tag
    assert len(block) == CREDENTIAL_BLOCK_SIZE, f"Credential block wrong size: {len(block)}"
    return block


def unpack_credential_block(data: bytes, offset: int = HEADER_SIZE) -> tuple[bytes, bytes, bytes, bytes]:
    """Unpack credential material from an export file."""
    kdf_salt   = data[offset : offset + 16]
    wrapped_iv = data[offset + 16 : offset + 28]
    ct_len     = struct.unpack(">I", data[offset + 28 : offset + 32])[0]
    wrapped_ct = data[offset + 32 : offset + 32 + ct_len]
    wrapped_tag = data[offset + 32 + ct_len : offset + 32 + ct_len + 16]
    return kdf_salt, wrapped_iv, wrapped_ct, wrapped_tag


# ---------------------------------------------------------------------------
# Single-file format (exports and snapshots)
# ---------------------------------------------------------------------------

def write_vmem(
    mek: bytes,
    state: MemoryState,
    file_type: int = FILE_TYPE_WORKING,
    credential_info: Optional[tuple[bytes, bytes, bytes, bytes]] = None,
) -> bytes:
    """
    Serialize and encrypt a MemoryState into a self-contained .vmem file.

    Used for exports (file_type=FILE_TYPE_EXPORT) and named snapshots
    (file_type=FILE_TYPE_SNAPSHOT). Working copies use the split format
    (write_index_file + append_atoms_to_file) instead.

    Returns:
        Complete .vmem file bytes, ready to write to disk.
    """
    # Compute the section start for atoms
    credential_block_bytes = b""
    if file_type == FILE_TYPE_EXPORT:
        if credential_info is None:
            raise ValueError("credential_info required for export files")
        kdf_salt, wrapped_iv, wrapped_ct, wrapped_tag = credential_info
        credential_block_bytes = pack_credential_block(kdf_salt, wrapped_iv, wrapped_ct, wrapped_tag)
        index_offset = HEADER_SIZE + CREDENTIAL_BLOCK_SIZE
    else:
        index_offset = HEADER_SIZE

    # Encrypt all atoms, track their offsets within this file
    ordered_ids = list(state.atoms.keys())
    atom_blocks: list[bytes] = []
    index_size_placeholder = encrypted_block_size(1)  # will recompute
    atoms_start = index_offset + encrypted_block_size(1)  # placeholder

    # First pass: encrypt atoms and build the atom_map with real offsets.
    # We need index_size to compute atoms_start, but index_size depends on atom_map.
    # Resolve by computing index_size from an approximate atom_map first,
    # then adjusting. In practice: build atom blocks, then figure out offsets.
    atom_block_list: list[bytes] = []
    for atom_id in ordered_ids:
        atom = state.atoms[atom_id]
        atom_json = json.dumps(atom.to_dict(), ensure_ascii=False).encode("utf-8")
        atom_compressed = zlib.compress(atom_json)
        atom_id_bytes = uuid.UUID(atom_id).bytes
        iv, ct, tag = encrypt_atom(mek, atom_compressed, atom_id_bytes)
        atom_block_list.append(pack_encrypted_block(iv, ct, tag))

    # Build atom_map with section-relative offsets (0-based within the atom section).
    # This eliminates the circular dependency between index size and atom positions:
    # the index does not reference the absolute file position of atoms, so the
    # index can be built once without iteration.
    # Readers compute: abs_offset = atom_section_start + rel_offset, where
    # atom_section_start = index_offset + index_block_size (from the header).
    rel_offset = 0
    atom_map = []
    for i, atom_id in enumerate(ordered_ids):
        size = len(atom_block_list[i])
        atom_map.append({"id": atom_id, "offset": rel_offset, "size": size})
        rel_offset += size

    index_dict = {
        "owner": state.owner,
        "file_id": state.file_id,
        "created_at": state.created_at,
        "schema_version": state.schema_version,
        "atom_map": atom_map,
    }
    index_compressed = zlib.compress(
        json.dumps(index_dict, ensure_ascii=False).encode("utf-8")
    )

    # Build header
    header = pack_header(
        file_type=file_type,
        index_offset=index_offset,
        index_size=encrypted_block_size(len(index_compressed)),
        atom_count=len(ordered_ids),
        file_id=state.file_id,
    )

    # Encrypt index with header as AAD
    idx_iv, idx_ct, idx_tag = encrypt_index(mek, index_compressed, header)
    index_block = pack_encrypted_block(idx_iv, idx_ct, idx_tag)

    return header + credential_block_bytes + index_block + b"".join(atom_block_list)


def read_vmem(data: bytes, mek: bytes) -> MemoryState:
    """
    Decrypt and deserialize a self-contained .vmem file into a MemoryState.

    Used for exports and named snapshots. Working copies use read_working_copy().

    Raises:
        VaultTamperedError: If any GCM authentication tag fails.
    """
    header_fields = unpack_header(data)
    header_bytes  = data[:HEADER_SIZE]
    file_type     = header_fields["file_type"]
    index_offset  = header_fields["index_offset"]
    atom_count    = header_fields["atom_count"]
    file_id       = header_fields["file_id"]

    # Decrypt the index block; atom_section_start is immediately after it.
    idx_iv, idx_ct, idx_tag, atom_section_start = unpack_encrypted_block(data, index_offset)
    index_compressed = decrypt_index(mek, idx_iv, idx_ct, idx_tag, header_bytes)
    index_json = zlib.decompress(index_compressed)
    index_dict = json.loads(index_json)

    owner: str = index_dict.get("owner", "")
    created_at: int = index_dict.get("created_at", 0)
    schema_version: int = index_dict.get("schema_version", 1)
    atom_map: list[dict] = index_dict.get("atom_map", [])

    if len(atom_map) != atom_count:
        raise VaultTamperedError(
            f"atom_count in header ({atom_count}) does not match index ({len(atom_map)})"
        )

    # Decrypt atom blocks using section-relative offsets
    atoms: dict[str, MemoryObject] = {}
    for entry in atom_map:
        atom_id = entry["id"]
        offset  = atom_section_start + entry["offset"]  # absolute file offset
        atom_id_bytes = uuid.UUID(atom_id).bytes
        atm_iv, atm_ct, atm_tag, _ = unpack_encrypted_block(data, offset)
        atom_compressed = decrypt_atom(mek, atm_iv, atm_ct, atm_tag, atom_id_bytes)
        atom_json_bytes = zlib.decompress(atom_compressed)
        atom_dict = json.loads(atom_json_bytes)
        atoms[atom_id] = MemoryObject.from_dict(atom_dict)

    # _atom_offsets is empty: this state came from a single-file export, not a
    # split working copy. If saved via write_index_file, all atoms will be treated
    # as new and written to current.atoms.
    return MemoryState(
        owner=owner,
        file_id=file_id,
        atoms=atoms,
        created_at=created_at,
        schema_version=schema_version,
    )


# ---------------------------------------------------------------------------
# Split working copy format
# ---------------------------------------------------------------------------

def append_atoms_to_file(
    atoms_path: Path,
    mek: bytes,
    atoms: dict[str, MemoryObject],
    atom_ids: list[str],
    start_offset: int,
) -> dict[str, tuple[int, int]]:
    """
    Encrypt and append new atom blocks to current.atoms.

    Existing atom blocks are never touched — atoms are immutable once written.
    The file is fsynced before returning so the data is durable before the
    index is updated.

    Returns:
        Dict mapping each atom_id to (byte_offset, block_size) within current.atoms.
    """
    offsets: dict[str, tuple[int, int]] = {}
    with open(atoms_path, "ab") as fh:
        offset = start_offset
        for atom_id in atom_ids:
            atom = atoms[atom_id]
            atom_json = json.dumps(atom.to_dict(), ensure_ascii=False).encode("utf-8")
            atom_compressed = zlib.compress(atom_json)
            atom_id_bytes = uuid.UUID(atom_id).bytes
            iv, ct, tag = encrypt_atom(mek, atom_compressed, atom_id_bytes)
            block = pack_encrypted_block(iv, ct, tag)
            fh.write(block)
            offsets[atom_id] = (offset, len(block))
            offset += len(block)
        fh.flush()
        os.fsync(fh.fileno())
    return offsets


def write_index_file(
    index_path: Path,
    mek: bytes,
    state: MemoryState,
    file_type: int = FILE_TYPE_WORKING,
) -> None:
    """
    Write the header + encrypted index block for a working copy atomically.

    Uses temp-file-and-rename so the previous index is never corrupted by a
    crash mid-write. Atom blocks live in current.atoms and are not touched here.

    Precondition: state.new_atom_ids() must be empty (all atoms already in
    current.atoms and tracked in state._atom_offsets).
    """
    index_dict = state.to_index_dict()
    index_json = json.dumps(index_dict, ensure_ascii=False).encode("utf-8")
    index_compressed = zlib.compress(index_json)
    index_size = encrypted_block_size(len(index_compressed))

    # For working copies, the index sits immediately after the header.
    # No credential block (credentials live in meta.json).
    header = pack_header(
        file_type=file_type,
        index_offset=HEADER_SIZE,
        index_size=index_size,
        atom_count=len(state.atoms),
        file_id=state.file_id,
    )

    idx_iv, idx_ct, idx_tag = encrypt_index(mek, index_compressed, header)
    index_block = pack_encrypted_block(idx_iv, idx_ct, idx_tag)

    tmp = Path(str(index_path) + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(header + index_block)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.rename(index_path)


def read_working_copy(
    vmem_path: Path,
    atoms_path: Path,
    mek: bytes,
) -> MemoryState:
    """
    Read a split working copy (current.vmem + current.atoms) into a MemoryState.

    If current.atoms contains orphaned blocks beyond what the index references
    (e.g. from a crash between atom-append and index-rename), the extra bytes
    are silently truncated — they were not committed to the index.

    Raises:
        VaultTamperedError: If any GCM authentication tag fails.
    """
    # Read and decrypt the index
    with open(vmem_path, "rb") as fh:
        vmem_data = fh.read()

    header_fields = unpack_header(vmem_data)
    header_bytes  = vmem_data[:HEADER_SIZE]
    index_offset  = header_fields["index_offset"]
    atom_count    = header_fields["atom_count"]
    file_id       = header_fields["file_id"]

    idx_iv, idx_ct, idx_tag, _ = unpack_encrypted_block(vmem_data, index_offset)
    index_compressed = decrypt_index(mek, idx_iv, idx_ct, idx_tag, header_bytes)
    index_json = zlib.decompress(index_compressed)
    index_dict = json.loads(index_json)

    owner: str = index_dict.get("owner", "")
    created_at: int = index_dict.get("created_at", 0)
    schema_version: int = index_dict.get("schema_version", 1)
    atom_map: list[dict] = index_dict.get("atom_map", [])

    if len(atom_map) != atom_count:
        raise VaultTamperedError(
            f"atom_count in header ({atom_count}) does not match index ({len(atom_map)})"
        )

    # Read atom data, truncating any orphaned tail
    with open(atoms_path, "rb") as fh:
        atoms_data = fh.read()

    committed_size = sum(entry["size"] for entry in atom_map)
    if len(atoms_data) > committed_size:
        # Orphaned atoms from a crash — truncate to last committed state
        with open(atoms_path, "r+b") as fh:
            fh.truncate(committed_size)
        atoms_data = atoms_data[:committed_size]

    # Decrypt atoms using offsets from atom_map
    atoms: dict[str, MemoryObject] = {}
    atom_offsets: dict[str, tuple[int, int]] = {}
    for entry in atom_map:
        atom_id = entry["id"]
        offset  = entry["offset"]
        size    = entry["size"]
        atom_id_bytes = uuid.UUID(atom_id).bytes
        block = atoms_data[offset : offset + size]
        atm_iv, atm_ct, atm_tag, _ = unpack_encrypted_block(block, 0)
        atom_compressed = decrypt_atom(mek, atm_iv, atm_ct, atm_tag, atom_id_bytes)
        atom_json_bytes = zlib.decompress(atom_compressed)
        atom_dict = json.loads(atom_json_bytes)
        atoms[atom_id] = MemoryObject.from_dict(atom_dict)
        atom_offsets[atom_id] = (offset, size)

    return MemoryState(
        owner=owner,
        file_id=file_id,
        atoms=atoms,
        created_at=created_at,
        schema_version=schema_version,
        _atom_offsets=atom_offsets,
        _atoms_file_size=committed_size,
    )


# ---------------------------------------------------------------------------
# Create a new vault (on-disk initialization)
# ---------------------------------------------------------------------------

def create_vault(
    vault_dir: "Path | str",
    passphrase: str,
    owner: str,
    data_class: str = "GENERAL",
) -> tuple[MemoryState, bytes]:
    """
    Initialize a new VaultMem vault directory.

    Creates:
        vault_dir/current.vmem   — header + empty index (working copy)
        vault_dir/current.atoms  — empty atom store (append-only)
        vault_dir/meta.json      — credential metadata (kdf_salt, wrapped_MEK)
        vault_dir/snapshots/     — empty snapshots directory

    Returns:
        (initial_state, mek) — caller must zero MEK after session ends.
    """
    import time
    from .crypto import derive_kek, ARGON2_MEMORY_COST, ARGON2_MEMORY_COST_MEDICAL
    import base64

    vault_dir = Path(vault_dir)
    vault_dir.mkdir(parents=True, exist_ok=True)
    (vault_dir / "snapshots").mkdir(exist_ok=True)

    # Generate credential material
    kdf_salt = random_salt()
    mek = random_mek()

    memory_cost = (
        ARGON2_MEMORY_COST_MEDICAL if data_class == "MEDICAL"
        else ARGON2_MEMORY_COST
    )
    kek = derive_kek(passphrase, kdf_salt, memory_cost=memory_cost)
    try:
        wrapped_iv, wrapped_ct, wrapped_tag = wrap_mek(kek, mek)
    finally:
        del kek

    # Write meta.json
    meta = {
        "schema_version": 1,
        "data_class": data_class,
        "memory_cost": memory_cost,
        "credential_slots": [
            {
                "slot_id": "primary",
                "kdf_salt": base64.b64encode(kdf_salt).decode(),
                "wrapped_mek_iv": base64.b64encode(wrapped_iv).decode(),
                "wrapped_mek_ct": base64.b64encode(wrapped_ct).decode(),
                "wrapped_mek_tag": base64.b64encode(wrapped_tag).decode(),
                "created_at": int(time.time()),
                "expires_at": None,
            }
        ],
        "created_at": int(time.time()),
        "owner": owner,
    }
    with open(vault_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Create empty split working copy
    state = MemoryState.new(owner)

    # current.atoms — empty (no atoms yet)
    (vault_dir / "current.atoms").write_bytes(b"")

    # current.vmem — header + empty index
    write_index_file(vault_dir / "current.vmem", mek, state)

    return state, mek
