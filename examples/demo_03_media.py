"""
VaultMem SDK — Example 03: Multi-Modal Media Ingestion

Shows how platform developers can integrate media memory:
  1. ImageExtractor  — pull EXIF date, GPS, optional OCR from photos
  2. MediaIngester   — unified dispatcher across file types
  3. VaultSession.add_media() — encrypt raw file + create searchable atom
  4. VaultSession.get_media() — decrypt and retrieve original bytes
  5. add_media_batch() — bulk import with progress callback
  6. Graceful degradation when optional packages are missing

The raw media file is encrypted end-to-end (same AES-256-GCM + MEK as atoms).
The platform developer never has access to plaintext content or files.

Requires for full functionality:
    pip install "vaultmem[media]"  (Pillow, piexif, PyMuPDF, openai-whisper, ffmpeg-python)

This demo creates synthetic test files inline so it runs with Pillow only.

Run:
    PYTHONPATH=src .venv/bin/python examples/demo_03_media.py
"""

import shutil
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from vaultmem import (
    VaultSession,
    NullEmbedder,
    ImageExtractor,
    MediaIngester,
    MediaExtractionResult,
    TimeQueryParser,
)

# ── Config ────────────────────────────────────────────────────────────────────
VAULT_DIR  = Path("/tmp/vaultmem_media")
PASSPHRASE = "hunter2"
OWNER      = "alice"
EMBEDDER   = NullEmbedder()

SEP = "─" * 60
def hr(title: str = "") -> None:
    print(f"\n{SEP}")
    if title:
        print(f"  {title}")
        print(SEP)

def fmt(unix_ts) -> str:
    if unix_ts is None:
        return "—"
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

# ── Helpers: create synthetic test images with EXIF ───────────────────────────

def make_jpeg_with_exif(
    out_path: Path,
    exif_datetime: str,           # "YYYY:MM:DD HH:MM:SS"
    gps_lat: float = None,
    gps_lon: float = None,
    width: int = 64,
    height: int = 64,
) -> Path:
    """
    Create a minimal JPEG with DateTimeOriginal EXIF.
    Uses Pillow's native EXIF API (no piexif required).
    GPS coordinates written via piexif if available.
    """
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(100, 149, 237))  # cornflower blue

    # Tag IDs: 36867=DateTimeOriginal, 36868=DateTimeDigitized, 271=Make, 272=Model
    exif = img.getexif()
    exif[36867] = exif_datetime   # DateTimeOriginal
    exif[36868] = exif_datetime   # DateTimeDigitized
    exif[271]   = "VaultCam"      # Make
    exif[272]   = "Demo v1"       # Model

    # GPS via piexif (optional)
    exif_bytes = exif.tobytes()
    if gps_lat is not None and gps_lon is not None:
        try:
            import piexif

            def to_dms(deg: float):
                d = int(abs(deg))
                m = int((abs(deg) - d) * 60)
                s = round(((abs(deg) - d) * 60 - m) * 60 * 100)
                return ((d, 1), (m, 1), (s, 100))

            exif_dict = piexif.load(exif_bytes)
            exif_dict["GPS"] = {
                piexif.GPSIFD.GPSLatitudeRef:  b"N" if gps_lat >= 0 else b"S",
                piexif.GPSIFD.GPSLatitude:     to_dms(gps_lat),
                piexif.GPSIFD.GPSLongitudeRef: b"E" if gps_lon >= 0 else b"W",
                piexif.GPSIFD.GPSLongitude:    to_dms(gps_lon),
            }
            exif_bytes = piexif.dump(exif_dict)
        except ImportError:
            pass  # GPS skipped — piexif not installed

    img.save(out_path, "JPEG", exif=exif_bytes)
    return out_path


def make_plain_file(out_path: Path, content: str) -> Path:
    """Create a plain text file (simulates a newspaper article or document scan)."""
    out_path.write_text(content, encoding="utf-8")
    return out_path


# ── Setup ─────────────────────────────────────────────────────────────────────
if VAULT_DIR.exists():
    shutil.rmtree(VAULT_DIR)

# ── 1. ImageExtractor — EXIF metadata extraction ─────────────────────────────
hr("1. ImageExtractor — EXIF metadata extraction")

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # Photo from Tokyo trip (April 2019)
    tokyo_jpg = make_jpeg_with_exif(
        tmp / "tokyo_temple.jpg",
        exif_datetime="2019:04:08 14:23:00",
        gps_lat=35.7148,   # Senso-ji Temple
        gps_lon=139.7967,
    )

    extractor = ImageExtractor()
    result = extractor.extract(tokyo_jpg)

    print(f"  File         : {tokyo_jpg.name}")
    print(f"  content_type : {result.content_type}")
    print(f"  captured_at  : {fmt(result.captured_at)}  ← from EXIF DateTimeOriginal")
    print(f"  location     : {result.location}")
    print(f"  transcript   : {result.transcript!r}  ← OCR (pytesseract, if installed)")
    print(f"  metadata     : {list(result.metadata.keys())[:5]} ...")

# ── 2. MediaIngester — unified dispatcher ─────────────────────────────────────
hr("2. MediaIngester — dispatches to the right extractor by file type")

def _pkg_installed(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False

print("  Extractor coverage (installed packages):")
test_types = [
    ("photo.jpg",  "image/jpeg      → ImageExtractor   (Pillow + piexif)",  "PIL"),
    ("scan.pdf",   "application/pdf → DocumentExtractor (PyMuPDF)",         "fitz"),
    ("memo.mp3",   "audio/mpeg      → AudioExtractor   (openai-whisper)",   "whisper"),
    ("clip.mp4",   "video/mp4       → VideoExtractor   (ffmpeg-python)",    "ffmpeg"),
]
for fname, description, pkg in test_types:
    status = "installed" if _pkg_installed(pkg) else "not installed — pip install 'vaultmem[media]'"
    print(f"    {fname:<12}  {description}  [{status}]")

print()
print("  Files with no matching extractor fall back to:")
print("    MediaExtractionResult(content_type=..., transcript='', captured_at=file_mtime)")

# ── 3. add_media — encrypt file + create searchable atom ─────────────────────
hr("3. VaultSession.add_media() — encrypt raw file + build searchable atom")

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # Create several synthetic media files with different dates
    photos = [
        ("tokyo_temple.jpg",      "2019:04:08 14:23:00", 35.7148,  139.7967, "Senso-ji temple visit"),
        ("graduation_day.jpg",    "2019:06:15 11:00:00", 51.5074,  -0.1278,  "MSc graduation ceremony"),
        ("berlin_arrival.jpg",    "2019:09:01 09:45:00", 52.5200,   13.4050, "First day in Berlin"),
        ("neurips_poster.jpg",    "2019:12:10 15:00:00", 37.7749, -122.4194, "NeurIPS poster session"),
        ("croatia_diving.jpg",    "2021:07:14 10:30:00", 43.5081,   16.4402, "First scuba dive"),
        ("halfmarathon_finish.jpg","2023:05:21 09:04:00", 52.5200,   13.4050, "Crossing the finish line"),
    ]

    with VaultSession.create(VAULT_DIR, PASSPHRASE, OWNER, embedder=EMBEDDER) as s:
        atoms = []
        for fname, exif_dt, lat, lon, caption in photos:
            fpath = make_jpeg_with_exif(tmp / fname, exif_dt, lat, lon)
            atom = s.add_media(fpath, caption=caption)
            atoms.append(atom)

        s.flush()

        print(f"  Added {len(atoms)} media atoms to vault")
        print()
        print(f"  {'File':<28}  {'captured_at':<12}  {'content_type':<12}  media_blob_id?")
        print(f"  {'-'*28}  {'-'*12}  {'-'*12}  {'-'*14}")
        for i, (fname, *_) in enumerate(photos):
            atom = atoms[i]
            print(
                f"  {fname:<28}  "
                f"{fmt(atom.captured_at)[:10]:<12}  "
                f"{atom.content_type:<12}  "
                f"{'yes (' + atom.media_blob_id[:8] + '…)' if atom.media_blob_id else 'no'}"
            )

# ── 4. get_media — retrieve and decrypt original file ─────────────────────────
hr("4. VaultSession.get_media() — decrypt and retrieve original bytes")

with VaultSession.open(VAULT_DIR, PASSPHRASE, embedder=EMBEDDER) as s:
    # search_by_time to find the graduation photo atom
    from_ts = int(datetime(2019, 6, 1, tzinfo=timezone.utc).timestamp())
    to_ts   = int(datetime(2019, 7, 1, tzinfo=timezone.utc).timestamp())
    june_atoms = s.search_by_time(from_ts, to_ts)

    print(f"  Atoms from June 2019: {len(june_atoms)}")
    for atom in june_atoms:
        print(f"    captured={fmt(atom.captured_at)[:10]}  {atom.content[:60]}")
        print(f"    media_source_path: {atom.media_source_path}")
        print(f"    location: {atom.location}")

        if atom.media_blob_id:
            raw_bytes = s.get_media(atom.media_blob_id)
            # Verify it's a valid JPEG (starts with FF D8)
            is_jpeg = raw_bytes[:2] == b"\xff\xd8"
            print(f"    Retrieved {len(raw_bytes)} bytes, valid JPEG: {is_jpeg}")
            print(f"    (Platform developer cannot do this — they don't have the passphrase)")

# ── 5. Vault media directory layout ───────────────────────────────────────────
hr("5. Vault directory layout with media")

print(f"  {VAULT_DIR}/")
for p in sorted(VAULT_DIR.rglob("*")):
    rel = p.relative_to(VAULT_DIR)
    if p.is_file():
        size = p.stat().st_size
        note = " ← encrypted atom blobs (append-only)" if p.name == "current.atoms" \
               else " ← encrypted index" if p.name == "current.vmem" \
               else " ← encrypted raw media file" if p.parent.name == "media" \
               else ""
        print(f"  ├── {str(rel):<40} {size:>8,} B{note}")

# ── 6. add_media_batch with progress callback ─────────────────────────────────
hr("6. add_media_batch() — bulk import with progress callback")

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # Simulate a photo library directory
    batch_photos = [
        ("holiday_2022_001.jpg", "2022:08:10 10:00:00"),
        ("holiday_2022_002.jpg", "2022:08:11 14:30:00"),
        ("holiday_2022_003.jpg", "2022:08:12 09:15:00"),
        ("birthday_2022.jpg",    "2022:11:05 19:00:00"),
        ("new_year_2023.jpg",    "2023:01:01 00:05:00"),
    ]
    paths = [
        make_jpeg_with_exif(tmp / fname, exif_dt)
        for fname, exif_dt in batch_photos
    ]

    def on_progress(current: int, total: int, path: Path) -> None:
        pct = int(100 * current / total)
        bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
        print(f"\r    [{bar}] {pct:3d}%  {path.name:<35}", end="", flush=True)
        if current == total:
            print()

    with VaultSession.open(VAULT_DIR, PASSPHRASE, embedder=EMBEDDER) as s:
        before = s.atom_count
        added = s.add_media_batch(paths, progress_callback=on_progress)
        s.flush()
        print(f"  Imported {len(added)} photos. Vault now has {s.atom_count} atoms (was {before}).")

# ── 7. End-to-end timeline browse ─────────────────────────────────────────────
hr("7. End-to-end: browse the full vault timeline")

with VaultSession.open(VAULT_DIR, PASSPHRASE, embedder=EMBEDDER) as s:
    all_atoms = s.search_by_time(0, int(datetime.now(tz=timezone.utc).timestamp()), top_k=500)

    # Group by year
    from collections import defaultdict
    by_year: dict[int, list] = defaultdict(list)
    for atom in all_atoms:
        ts_val = atom.captured_at or atom.created_at
        year = datetime.fromtimestamp(ts_val, tz=timezone.utc).year
        by_year[year].append(atom)

    print(f"  Full vault: {len(all_atoms)} memories\n")
    for year in sorted(by_year.keys()):
        group = by_year[year]
        print(f"  {year}  ({len(group)} memories)")
        for atom in sorted(group, key=lambda a: a.captured_at or a.created_at):
            ts_val = atom.captured_at or atom.created_at
            icon = {"image/jpeg": "[img]", "text/plain": "[txt]"}.get(atom.content_type, "[?]")
            print(f"    {fmt(ts_val)[:10]}  {icon}  {atom.content[:65]}")

print(f"\n{SEP}\nDone.\n")
