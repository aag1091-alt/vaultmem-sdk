"""
VaultMem cryptographic layer.

Implements:
  - Argon2id key derivation (KEK from passphrase)
  - AES-256-GCM authenticated encryption/decryption
  - MEK wrapping / unwrapping (key envelope protocol)
  - CSPRNG IV generation

Design:
  KEK = Argon2id(passphrase, kdf_salt)         — in RAM during credential ops only
  MEK = AES-256-GCM-decrypt(KEK, wrapped_MEK)  — in RAM for session lifetime
  atom_ct = AES-256-GCM(MEK, atom_json, AAD=uuid_bytes)
  index_ct = AES-256-GCM(MEK, index_json, AAD=header_bytes)

The MEK never exists outside session RAM.
The KEK is derived once at session open, immediately zeroed after MEK unwrap.
"""
from __future__ import annotations

import os
import struct
from typing import Optional

from argon2.low_level import hash_secret_raw, Type as Argon2Type
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .exceptions import WrongPassphraseError, VaultTamperedError


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEY_LEN   = 32   # 256-bit AES key
IV_LEN    = 12   # 96-bit GCM nonce (NIST recommended)
TAG_LEN   = 16   # 128-bit GCM authentication tag

# Argon2id parameters (RFC 9106 §4 "first recommended option set")
ARGON2_TIME_COST     = 3
ARGON2_MEMORY_COST   = 65536   # 64 MiB (in KiB)
ARGON2_PARALLELISM   = 4
ARGON2_MEMORY_COST_MEDICAL = 131072  # 128 MiB — doubled for MEDICAL data class

# AAD namespace constants (prevent cross-context decryption)
AAD_MEK_WRAP = b"vmem-mek-v1"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def random_iv() -> bytes:
    """Generate a cryptographically random 12-byte IV from OS CSPRNG."""
    return os.urandom(IV_LEN)


def random_mek() -> bytes:
    """Generate a new 32-byte Master Encryption Key from OS CSPRNG."""
    return os.urandom(KEY_LEN)


def random_salt() -> bytes:
    """Generate a 16-byte Argon2id KDF salt."""
    return os.urandom(16)


def zero_bytes(buf: bytearray) -> None:
    """Overwrite a bytearray with zeros (best-effort key erasure)."""
    for i in range(len(buf)):
        buf[i] = 0


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

def derive_kek(
    passphrase: str,
    kdf_salt: bytes,
    memory_cost: int = ARGON2_MEMORY_COST,
) -> bytes:
    """
    Derive a Key Encryption Key (KEK) from a passphrase using Argon2id.

    Args:
        passphrase: User-supplied passphrase (UTF-8 encoded).
        kdf_salt: 16-byte random salt stored in meta.json.
        memory_cost: Argon2id memory parameter in KiB.
                     Use ARGON2_MEMORY_COST_MEDICAL for MEDICAL vaults.

    Returns:
        32-byte KEK (bytes). Caller is responsible for zeroing after use.
    """
    return hash_secret_raw(
        secret=passphrase.encode("utf-8"),
        salt=kdf_salt,
        time_cost=ARGON2_TIME_COST,
        memory_cost=memory_cost,
        parallelism=ARGON2_PARALLELISM,
        hash_len=KEY_LEN,
        type=Argon2Type.ID,
    )


# ---------------------------------------------------------------------------
# AES-256-GCM primitives
# ---------------------------------------------------------------------------

def _gcm_encrypt(key: bytes, plaintext: bytes, aad: bytes) -> tuple[bytes, bytes, bytes]:
    """
    Encrypt plaintext with AES-256-GCM.

    Returns:
        (iv, ciphertext, tag) — each as bytes.
        ciphertext length == plaintext length (stream cipher, no padding).
    """
    iv = random_iv()
    aesgcm = AESGCM(key)
    ct_and_tag = aesgcm.encrypt(iv, plaintext, aad)
    # cryptography library appends the tag to the ciphertext
    ct  = ct_and_tag[:-TAG_LEN]
    tag = ct_and_tag[-TAG_LEN:]
    return iv, ct, tag


def _gcm_decrypt(
    key: bytes,
    iv: bytes,
    ciphertext: bytes,
    tag: bytes,
    aad: bytes,
    tamper_error_class: type[Exception] = VaultTamperedError,
) -> bytes:
    """
    Decrypt and authenticate AES-256-GCM ciphertext.

    Raises:
        tamper_error_class: If the authentication tag is invalid.
    """
    aesgcm = AESGCM(key)
    try:
        return aesgcm.decrypt(iv, ciphertext + tag, aad)
    except InvalidTag as exc:
        raise tamper_error_class("GCM authentication tag verification failed") from exc


# ---------------------------------------------------------------------------
# MEK wrapping / unwrapping
# ---------------------------------------------------------------------------

def wrap_mek(kek: bytes, mek: bytes) -> tuple[bytes, bytes, bytes]:
    """
    Encrypt (wrap) the MEK with the KEK.

    Uses AAD=b"vmem-mek-v1" to bind the ciphertext to its purpose.

    Returns:
        (iv, wrapped_ct, tag) — suitable for storage in meta.json.
    """
    return _gcm_encrypt(kek, mek, AAD_MEK_WRAP)


def unwrap_mek(kek: bytes, iv: bytes, wrapped_ct: bytes, tag: bytes) -> bytes:
    """
    Decrypt (unwrap) the MEK using the KEK.

    A wrong passphrase produces an invalid KEK, causing GCM tag verification
    to fail. This is detected here and re-raised as WrongPassphraseError.

    Returns:
        32-byte MEK.

    Raises:
        WrongPassphraseError: If the passphrase was incorrect.
    """
    return _gcm_decrypt(kek, iv, wrapped_ct, tag, AAD_MEK_WRAP, WrongPassphraseError)


# ---------------------------------------------------------------------------
# Atom encryption / decryption
# ---------------------------------------------------------------------------

def encrypt_atom(mek: bytes, atom_json: bytes, atom_id_bytes: bytes) -> tuple[bytes, bytes, bytes]:
    """
    Encrypt a serialized atom.

    AAD = atom_id_bytes (16-byte UUID) — prevents ciphertext transplantation.

    Returns:
        (iv, ciphertext, tag)
    """
    return _gcm_encrypt(mek, atom_json, atom_id_bytes)


def decrypt_atom(
    mek: bytes,
    iv: bytes,
    ciphertext: bytes,
    tag: bytes,
    atom_id_bytes: bytes,
) -> bytes:
    """
    Decrypt and authenticate a serialized atom.

    Raises:
        VaultTamperedError: If the atom ciphertext or AAD has been modified.
    """
    return _gcm_decrypt(mek, iv, ciphertext, tag, atom_id_bytes)


# ---------------------------------------------------------------------------
# Index block encryption / decryption
# ---------------------------------------------------------------------------

def encrypt_index(mek: bytes, index_json: bytes, header_bytes: bytes) -> tuple[bytes, bytes, bytes]:
    """
    Encrypt the vault index block.

    AAD = header_bytes[0:48] — binds the index to the file header, providing
    header tamper detection at zero additional key material cost.

    Returns:
        (iv, ciphertext, tag)
    """
    return _gcm_encrypt(mek, index_json, header_bytes[:48])


def decrypt_index(
    mek: bytes,
    iv: bytes,
    ciphertext: bytes,
    tag: bytes,
    header_bytes: bytes,
) -> bytes:
    """
    Decrypt and authenticate the vault index block.

    Raises:
        VaultTamperedError: If the index or header has been modified.
    """
    return _gcm_decrypt(mek, iv, ciphertext, tag, header_bytes[:48])


# ---------------------------------------------------------------------------
# Serialized block packing (for .vmem file I/O)
# ---------------------------------------------------------------------------

def pack_encrypted_block(iv: bytes, ciphertext: bytes, tag: bytes) -> bytes:
    """
    Pack an encrypted block into the on-disk format:
        [12B IV] [4B ct_len] [N bytes ciphertext] [16B auth tag]
    """
    ct_len = struct.pack(">I", len(ciphertext))
    return iv + ct_len + ciphertext + tag


def unpack_encrypted_block(data: bytes, offset: int = 0) -> tuple[bytes, bytes, bytes, int]:
    """
    Unpack an encrypted block from the on-disk format.

    Returns:
        (iv, ciphertext, tag, next_offset)
    """
    iv      = data[offset : offset + IV_LEN]
    ct_len  = struct.unpack(">I", data[offset + IV_LEN : offset + IV_LEN + 4])[0]
    ct_start = offset + IV_LEN + 4
    ct      = data[ct_start : ct_start + ct_len]
    tag     = data[ct_start + ct_len : ct_start + ct_len + TAG_LEN]
    next_offset = ct_start + ct_len + TAG_LEN
    return iv, ct, tag, next_offset


def encrypted_block_size(plaintext_len: int) -> int:
    """Compute the total on-disk size of an encrypted block for a given plaintext length.

    AES-GCM ciphertext length == plaintext length (stream mode, no padding).
    Total = IV (12) + len_prefix (4) + ciphertext (N) + tag (16).
    """
    return IV_LEN + 4 + plaintext_len + TAG_LEN
