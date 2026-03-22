"""
VaultMem blob storage layer.

BlobStore     — abstract interface for encrypted atom content.
FileBlobStore — local filesystem, one .enc file per atom.
S3BlobStore   — AWS S3 (or S3-compatible) storage. Requires: pip install "vaultmem[s3]"

Each blob is a raw pack_encrypted_block() output:
    [12B IV] [4B ct_len] [N bytes ciphertext] [16B auth tag]

Migration between backends requires no decryption — blobs are self-contained
ciphertext that can be copied opaquely:

    file_store.migrate_to(s3_store)   # copies encrypted bytes only

Use migrate_vault() to move both the blob store and the search index together.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .index import SearchIndex


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BlobStore(ABC):
    """Abstract encrypted atom blob store."""

    @abstractmethod
    def put(self, atom_id: str, data: bytes) -> None:
        """Write an encrypted atom blob. Idempotent (overwrites silently)."""

    @abstractmethod
    def get(self, atom_id: str) -> bytes:
        """
        Read an encrypted atom blob.

        Raises:
            KeyError: atom_id not found.
        """

    @abstractmethod
    def delete(self, atom_id: str) -> None:
        """Delete an encrypted atom blob. Idempotent (no-op if missing)."""

    @abstractmethod
    def list_ids(self) -> list[str]:
        """Return all stored atom IDs in undefined order."""

    def migrate_to(self, other: "BlobStore") -> None:
        """
        Copy all blobs to *other* without decrypting.

        The caller is responsible for closing this store after migration if
        desired. *other* is not closed — it remains open for further writes.
        """
        for atom_id in self.list_ids():
            other.put(atom_id, self.get(atom_id))


# ---------------------------------------------------------------------------
# S3 implementation
# ---------------------------------------------------------------------------

class S3BlobStore(BlobStore):
    """
    AWS S3 (or S3-compatible) blob store.

    Requires: ``pip install "vaultmem[s3]"``

    Layout::

        s3://{bucket}/{prefix}{atom_id}.enc

    The prefix defaults to ``"atoms/"`` and should include a trailing slash.
    Pass a pre-configured boto3 S3 client via *client* to use custom
    credentials, endpoints (e.g. MinIO, Cloudflare R2), or session config.

    Example::

        import boto3
        s3 = boto3.client("s3", region_name="us-east-1")
        store = S3BlobStore("my-bucket", client=s3)
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "atoms/",
        *,
        client=None,
        **boto3_kwargs,
    ) -> None:
        try:
            import boto3
            from botocore.exceptions import ClientError as _ClientError
            self._ClientError = _ClientError
        except ImportError:
            raise ImportError(
                'S3BlobStore requires boto3: pip install "vaultmem[s3]"'
            )
        self._bucket = bucket
        self._prefix = prefix
        self._s3 = client or boto3.client("s3", **boto3_kwargs)

    def _key(self, atom_id: str) -> str:
        return f"{self._prefix}{atom_id}.enc"

    def put(self, atom_id: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self._bucket, Key=self._key(atom_id), Body=data)

    def get(self, atom_id: str) -> bytes:
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=self._key(atom_id))
            return resp["Body"].read()
        except self._ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                raise KeyError(atom_id)
            raise

    def delete(self, atom_id: str) -> None:
        # S3 delete_object is always a no-op for missing keys — idempotent by default
        self._s3.delete_object(Bucket=self._bucket, Key=self._key(atom_id))

    def list_ids(self) -> list[str]:
        ids: list[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        suffix = ".enc"
        prefix_len = len(self._prefix)
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if key.endswith(suffix):
                    ids.append(key[prefix_len : -len(suffix)])
        return ids


# ---------------------------------------------------------------------------
# Filesystem implementation
# ---------------------------------------------------------------------------

class FileBlobStore(BlobStore):
    """
    Local filesystem blob store — one .enc file per atom.

    Layout::

        {vault_dir}/atoms/
            {atom_id}.enc   ← AES-256-GCM encrypted atom blob

    Writes are atomic on POSIX: each put() writes to a .enc.tmp file then
    renames it over the target, matching the crash-safety pattern used by
    write_index_file() in vault.py.
    """

    def __init__(self, vault_dir: "Path | str") -> None:
        self._atoms_dir = Path(vault_dir) / "atoms"
        self._atoms_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, atom_id: str) -> Path:
        return self._atoms_dir / f"{atom_id}.enc"

    def put(self, atom_id: str, data: bytes) -> None:
        tmp = self._atoms_dir / f"{atom_id}.enc.tmp"
        tmp.write_bytes(data)
        tmp.rename(self._path(atom_id))  # atomic on POSIX

    def get(self, atom_id: str) -> bytes:
        p = self._path(atom_id)
        if not p.exists():
            raise KeyError(atom_id)
        return p.read_bytes()

    def delete(self, atom_id: str) -> None:
        try:
            self._path(atom_id).unlink()
        except FileNotFoundError:
            pass

    def list_ids(self) -> list[str]:
        # glob("*.enc") won't match *.enc.tmp — no temp-file contamination
        return [p.stem for p in self._atoms_dir.glob("*.enc")]


# ---------------------------------------------------------------------------
# Migration helper
# ---------------------------------------------------------------------------

def migrate_vault(
    source_blob: BlobStore,
    source_index: "SearchIndex",
    target_blob: BlobStore,
    target_index: "SearchIndex",
) -> None:
    """
    Move a complete vault (blobs + index) from one backend pair to another.

    No decryption is performed — blobs are copied as opaque ciphertext.
    The source backends are not closed after migration.

    Example — local files → S3 + Postgres::

        from vaultmem import FileBlobStore, SQLiteSearchIndex, S3BlobStore, PostgresSearchIndex, migrate_vault

        migrate_vault(
            source_blob=FileBlobStore("./vault"),
            source_index=SQLiteSearchIndex("./vault/search_index.db"),
            target_blob=S3BlobStore("my-bucket"),
            target_index=PostgresSearchIndex("postgresql://user:pass@host/db"),
        )
    """
    source_blob.migrate_to(target_blob)
    source_index.migrate_to(target_index)
