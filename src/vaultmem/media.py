"""
VaultMem media ingestion pipeline.

Provides pluggable extractors for photos, audio, PDFs, and video, plus a
pure-Python TimeQueryParser for natural-language time phrases.

All extractors use lazy imports so the SDK core stays dependency-free.
Install optional extras to enable each extractor:

    pip install "vaultmem[media]"
    # Pillow + piexif, PyMuPDF, openai-whisper, ffmpeg-python

Extractor | Package         | Media type
----------|-----------------|---------------------------
ImageExtractor    | Pillow (+ piexif) | JPEG, PNG, HEIC, TIFF
AudioExtractor    | openai-whisper   | MP3, WAV, M4A, OGG, FLAC
DocumentExtractor | PyMuPDF (fitz)   | PDF, EPUB, XPS
VideoExtractor    | ffmpeg-python    | MP4, MOV, MKV, AVI

Files with no matching extractor fall back to a plain result with the
file mtime as captured_at and an empty transcript.
"""
from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# MediaExtractionResult
# ---------------------------------------------------------------------------

@dataclass
class MediaExtractionResult:
    """Normalised output from any MediaExtractor."""
    content_type: str                          # MIME type, e.g. "image/jpeg"
    transcript: str = ""                       # Human-readable text extracted from the file
    metadata: dict = field(default_factory=dict)
    captured_at: Optional[int] = None         # Real-world capture timestamp (EXIF / metadata)
    location: Optional[dict] = None           # {lat, lon, place} if available


# ---------------------------------------------------------------------------
# MediaExtractor protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class MediaExtractor(Protocol):
    def can_handle(self, path: Path) -> bool: ...
    def extract(self, path: Path) -> MediaExtractionResult: ...


# ---------------------------------------------------------------------------
# ImageExtractor  (Pillow required; piexif optional for GPS)
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".tiff", ".tif", ".webp", ".bmp", ".gif"}


def _parse_exif_datetime(raw: str) -> Optional[int]:
    """Parse EXIF datetime string 'YYYY:MM:DD HH:MM:SS' → unix timestamp."""
    try:
        from datetime import datetime, timezone
        dt = datetime.strptime(raw.strip(), "%Y:%m:%d %H:%M:%S")
        return int(dt.replace(tzinfo=timezone.utc).timestamp())
    except (ValueError, AttributeError):
        return None


def _parse_gps_info(exif_dict: dict) -> Optional[dict]:
    """Extract GPS lat/lon from piexif GPS IFD dict."""
    try:
        import piexif
        gps = exif_dict.get("GPS", {})
        if not gps:
            return None

        def dms_to_deg(dms):
            d, m, s = dms
            return d[0] / d[1] + m[0] / m[1] / 60 + s[0] / s[1] / 3600

        lat = dms_to_deg(gps[piexif.GPSIFD.GPSLatitude])
        if gps.get(piexif.GPSIFD.GPSLatitudeRef) in (b"S", "S"):
            lat = -lat
        lon = dms_to_deg(gps[piexif.GPSIFD.GPSLongitude])
        if gps.get(piexif.GPSIFD.GPSLongitudeRef) in (b"W", "W"):
            lon = -lon
        return {"lat": round(lat, 6), "lon": round(lon, 6)}
    except Exception:
        return None


class ImageExtractor:
    """Extract EXIF date, GPS, and optional OCR text from image files."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in _IMAGE_EXTS

    def extract(self, path: Path) -> MediaExtractionResult:
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for ImageExtractor: pip install Pillow")

        mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        captured_at: Optional[int] = None
        location: Optional[dict] = None
        metadata: dict = {}

        with Image.open(path) as img:
            # Pillow native EXIF API (no piexif required for date)
            try:
                exif = img.getexif()
                if exif:
                    # 36867 = DateTimeOriginal, 36868 = DateTimeDigitized, 306 = DateTime
                    for tag_id in (36867, 36868, 306):
                        raw_dt = exif.get(tag_id)
                        if raw_dt:
                            captured_at = _parse_exif_datetime(str(raw_dt))
                            if captured_at:
                                break
                    # Camera make/model
                    make  = exif.get(271)
                    model = exif.get(272)
                    if make:
                        metadata["make"] = str(make)
                    if model:
                        metadata["model"] = str(model)
            except Exception:
                pass

            # GPS via piexif (optional — only if exif bytes are available)
            if location is None:
                try:
                    import piexif
                    exif_bytes = img.info.get("exif", b"")
                    if exif_bytes:
                        exif_dict = piexif.load(exif_bytes)
                        location = _parse_gps_info(exif_dict)
                except Exception:
                    pass

            metadata["size"] = img.size
            metadata["mode"] = img.mode

        # Fallback to file mtime if no EXIF date found
        if captured_at is None:
            captured_at = int(path.stat().st_mtime)

        # Optional OCR via pytesseract
        transcript = ""
        try:
            import pytesseract
            from PIL import Image as _Image
            with _Image.open(path) as img_ocr:
                transcript = pytesseract.image_to_string(img_ocr).strip()
        except ImportError:
            pass
        except Exception:
            pass

        return MediaExtractionResult(
            content_type=mime,
            transcript=transcript,
            metadata=metadata,
            captured_at=captured_at,
            location=location,
        )


# ---------------------------------------------------------------------------
# AudioExtractor  (openai-whisper required)
# ---------------------------------------------------------------------------

_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".opus"}

_whisper_model_cache: dict[str, object] = {}


class AudioExtractor:
    """Transcribe audio files using OpenAI Whisper (local, offline)."""

    def __init__(self, model_name: str = "base") -> None:
        self._model_name = model_name

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in _AUDIO_EXTS

    def extract(self, path: Path) -> MediaExtractionResult:
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper is required for AudioExtractor: "
                "pip install openai-whisper"
            )

        mime = mimetypes.guess_type(str(path))[0] or "audio/mpeg"

        if self._model_name not in _whisper_model_cache:
            _whisper_model_cache[self._model_name] = whisper.load_model(self._model_name)
        model = _whisper_model_cache[self._model_name]

        result = model.transcribe(str(path))
        transcript = result.get("text", "").strip()
        language = result.get("language", "")

        return MediaExtractionResult(
            content_type=mime,
            transcript=transcript,
            metadata={"whisper_language": language, "model": self._model_name},
            captured_at=int(path.stat().st_mtime),
        )


# ---------------------------------------------------------------------------
# DocumentExtractor  (PyMuPDF required)
# ---------------------------------------------------------------------------

_DOCUMENT_EXTS = {".pdf", ".epub", ".xps", ".oxps", ".cbz"}


def _parse_pdf_date(raw: str) -> Optional[int]:
    """Parse PDF date string 'D:YYYYMMDDHHmmSS' → unix timestamp."""
    try:
        from datetime import datetime, timezone
        s = raw.strip()
        if s.startswith("D:"):
            s = s[2:]
        # Take first 14 chars: YYYYMMDDHHMMSS
        dt = datetime.strptime(s[:14], "%Y%m%d%H%M%S")
        return int(dt.replace(tzinfo=timezone.utc).timestamp())
    except (ValueError, AttributeError):
        return None


class DocumentExtractor:
    """Extract text from PDF and other document formats via PyMuPDF."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in _DOCUMENT_EXTS

    def extract(self, path: Path) -> MediaExtractionResult:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for DocumentExtractor: pip install PyMuPDF"
            )

        mime = mimetypes.guess_type(str(path))[0] or "application/pdf"
        captured_at: Optional[int] = None
        metadata: dict = {}

        doc = fitz.open(str(path))
        try:
            meta = doc.metadata or {}
            metadata["page_count"] = doc.page_count
            if meta.get("title"):
                metadata["title"] = meta["title"]
            if meta.get("author"):
                metadata["author"] = meta["author"]

            raw_date = meta.get("creationDate") or meta.get("modDate")
            if raw_date:
                captured_at = _parse_pdf_date(raw_date)

            # Extract text from all pages (truncated to ~8000 chars)
            pages_text = []
            for page in doc:
                pages_text.append(page.get_text())
            transcript = "\n".join(pages_text)[:8000].strip()
        finally:
            doc.close()

        if captured_at is None:
            captured_at = int(path.stat().st_mtime)

        return MediaExtractionResult(
            content_type=mime,
            transcript=transcript,
            metadata=metadata,
            captured_at=captured_at,
        )


# ---------------------------------------------------------------------------
# VideoExtractor  (ffmpeg-python required)
# ---------------------------------------------------------------------------

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv", ".m4v"}


class VideoExtractor:
    """Extract audio track + first frame from video files via ffmpeg."""

    def __init__(self, audio_model: str = "base") -> None:
        self._audio_extractor = AudioExtractor(model_name=audio_model)

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in _VIDEO_EXTS

    def extract(self, path: Path) -> MediaExtractionResult:
        try:
            import ffmpeg
        except ImportError:
            raise ImportError(
                "ffmpeg-python is required for VideoExtractor: pip install ffmpeg-python"
            )

        import tempfile

        mime = mimetypes.guess_type(str(path))[0] or "video/mp4"

        # Probe metadata (duration, creation_time)
        captured_at: Optional[int] = None
        metadata: dict = {}
        try:
            probe = ffmpeg.probe(str(path))
            fmt = probe.get("format", {})
            tags = fmt.get("tags", {})
            metadata["duration_s"] = float(fmt.get("duration", 0))
            raw_ct = tags.get("creation_time")
            if raw_ct:
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(raw_ct.replace("Z", "+00:00"))
                captured_at = int(dt.timestamp())
        except Exception:
            pass

        if captured_at is None:
            captured_at = int(path.stat().st_mtime)

        # Extract audio → transcribe
        transcript = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            ffmpeg.input(str(path)).output(str(tmp_path), acodec="mp3", ac=1, ar="16000").run(
                quiet=True, overwrite_output=True
            )
            result = self._audio_extractor.extract(tmp_path)
            transcript = result.transcript
            metadata["whisper_language"] = result.metadata.get("whisper_language", "")
        except Exception:
            pass
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        return MediaExtractionResult(
            content_type=mime,
            transcript=transcript,
            metadata=metadata,
            captured_at=captured_at,
        )


# ---------------------------------------------------------------------------
# MediaIngester — dispatcher
# ---------------------------------------------------------------------------

_DEFAULT_EXTRACTORS: list[MediaExtractor] = []


def _get_default_extractors() -> list[MediaExtractor]:
    """Build default extractor list (lazily so imports stay optional)."""
    global _DEFAULT_EXTRACTORS
    if not _DEFAULT_EXTRACTORS:
        _DEFAULT_EXTRACTORS = [
            ImageExtractor(),
            DocumentExtractor(),
            AudioExtractor(),
            VideoExtractor(),
        ]
    return _DEFAULT_EXTRACTORS


class MediaIngester:
    """Dispatch a file to the right extractor based on extension."""

    def __init__(self, extractors: Optional[list[MediaExtractor]] = None) -> None:
        self._extractors = extractors or _get_default_extractors()

    def ingest(self, path: "str | Path") -> MediaExtractionResult:
        """
        Extract content from path using the first matching extractor.

        Falls back to a bare result (file mtime, empty transcript) when no
        extractor handles the file type or when the matching extractor's
        package is not installed.
        """
        path = Path(path)
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

        for extractor in self._extractors:
            if extractor.can_handle(path):
                try:
                    return extractor.extract(path)
                except ImportError:
                    # Package not installed — fall through to next extractor
                    pass
                except Exception:
                    pass  # Extraction failed — return bare result

        # Fallback: no extractor matched or all failed
        return MediaExtractionResult(
            content_type=mime,
            transcript="",
            captured_at=int(path.stat().st_mtime),
        )


# ---------------------------------------------------------------------------
# TimeQueryParser — pure Python, no dependencies
# ---------------------------------------------------------------------------

import re
import time as _time
from datetime import datetime, timezone


_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_SEASONS = {
    "spring": (3, 6),   # Mar–May
    "summer": (6, 9),   # Jun–Aug
    "fall":   (9, 12),  # Sep–Nov
    "autumn": (9, 12),
    "winter": (12, 3),  # Dec–Feb (wraps year)
}

# Patterns — ordered most-specific to least-specific
_SEASON_YEAR   = re.compile(r"\b(spring|summer|fall|autumn|winter)\s+(\d{4})\b", re.IGNORECASE)
_MONTH_YEAR    = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})\b",
    re.IGNORECASE,
)
_BARE_YEAR     = re.compile(r"\b(20\d{2}|19\d{2})\b")
_LAST_YEAR     = re.compile(r"\blast\s+year\b", re.IGNORECASE)
_THIS_YEAR     = re.compile(r"\bthis\s+year\b", re.IGNORECASE)
_THIS_MONTH    = re.compile(r"\bthis\s+month\b", re.IGNORECASE)
_LAST_MONTH    = re.compile(r"\blast\s+month\b", re.IGNORECASE)
_YESTERDAY     = re.compile(r"\byesterday\b", re.IGNORECASE)
_LAST_N_DAYS   = re.compile(r"\blast\s+(\d+)\s+days?\b", re.IGNORECASE)


def _utc_ts(year: int, month: int = 1, day: int = 1) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


class TimeQueryParser:
    """
    Extract a time range and semantic remainder from a natural-language query.

    Usage::

        from_ts, to_ts, remainder = TimeQueryParser.parse("what happened summer 2019")
        # from_ts = 2019-06-01 00:00:00 UTC
        # to_ts   = 2019-09-01 00:00:00 UTC
        # remainder = "what happened"

    Returns (None, None, original_query) when no time phrase is found.
    """

    @staticmethod
    def parse(query: str) -> tuple[Optional[int], Optional[int], str]:
        """Return (from_ts, to_ts, remainder). Timestamps are Unix seconds (UTC)."""
        now = datetime.now(tz=timezone.utc)

        # Season + year  e.g. "summer 2019"
        m = _SEASON_YEAR.search(query)
        if m:
            season = m.group(1).lower()
            year   = int(m.group(2))
            start_m, end_m = _SEASONS[season]
            if season == "winter":
                from_ts = _utc_ts(year, start_m)
                to_ts   = _utc_ts(year + 1, end_m)
            else:
                from_ts = _utc_ts(year, start_m)
                to_ts   = _utc_ts(year, end_m)
            remainder = (query[: m.start()] + query[m.end() :]).strip()
            return from_ts, to_ts, remainder

        # Month + year  e.g. "March 2020"
        m = _MONTH_YEAR.search(query)
        if m:
            month_name = m.group(1).lower()
            year       = int(m.group(2))
            month      = _MONTHS[month_name]
            from_ts    = _utc_ts(year, month)
            # End = first of next month
            next_year  = year + (1 if month == 12 else 0)
            next_month = 1 if month == 12 else month + 1
            to_ts      = _utc_ts(next_year, next_month)
            remainder  = (query[: m.start()] + query[m.end() :]).strip()
            return from_ts, to_ts, remainder

        # Bare year  e.g. "2019"
        m = _BARE_YEAR.search(query)
        if m:
            year      = int(m.group(1))
            from_ts   = _utc_ts(year)
            to_ts     = _utc_ts(year + 1)
            remainder = (query[: m.start()] + query[m.end() :]).strip()
            return from_ts, to_ts, remainder

        # yesterday
        if _YESTERDAY.search(query):
            from datetime import timedelta
            start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end   = start + timedelta(days=1)
            remainder = _YESTERDAY.sub("", query).strip()
            return int(start.timestamp()), int(end.timestamp()), remainder

        # last N days
        m = _LAST_N_DAYS.search(query)
        if m:
            n = int(m.group(1))
            from datetime import timedelta
            from_ts   = int((now - timedelta(days=n)).timestamp())
            to_ts     = int(now.timestamp())
            remainder = (query[: m.start()] + query[m.end() :]).strip()
            return from_ts, to_ts, remainder

        # last year
        if _LAST_YEAR.search(query):
            year      = now.year - 1
            from_ts   = _utc_ts(year)
            to_ts     = _utc_ts(year + 1)
            remainder = _LAST_YEAR.sub("", query).strip()
            return from_ts, to_ts, remainder

        # this year
        if _THIS_YEAR.search(query):
            year      = now.year
            from_ts   = _utc_ts(year)
            to_ts     = _utc_ts(year + 1)
            remainder = _THIS_YEAR.sub("", query).strip()
            return from_ts, to_ts, remainder

        # last month
        if _LAST_MONTH.search(query):
            year  = now.year
            month = now.month - 1
            if month == 0:
                month = 12
                year -= 1
            from_ts   = _utc_ts(year, month)
            next_year  = year + (1 if month == 12 else 0)
            next_month = 1 if month == 12 else month + 1
            to_ts      = _utc_ts(next_year, next_month)
            remainder  = _LAST_MONTH.sub("", query).strip()
            return from_ts, to_ts, remainder

        # this month
        if _THIS_MONTH.search(query):
            from_ts   = _utc_ts(now.year, now.month)
            next_year  = now.year + (1 if now.month == 12 else 0)
            next_month = 1 if now.month == 12 else now.month + 1
            to_ts      = _utc_ts(next_year, next_month)
            remainder  = _THIS_MONTH.sub("", query).strip()
            return from_ts, to_ts, remainder

        return None, None, query
