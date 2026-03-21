"""
VaultMem deterministic memory type classifier.

Assigns MemoryType based on four binary linguistic feature axes (F1–F4).
No LLM or external service is called — classification is rule-based, reproducible,
and privacy-preserving.

Decision tree (priority order: EPISODIC > PERSONA > PROCEDURAL > SEMANTIC):
    F1 (temporal) AND F4 (self-ref)  → EPISODIC
    F2 (habitual)  AND F4 (self-ref) → PERSONA
    F3 (procedural)                  → PROCEDURAL
    default                          → SEMANTIC
"""
from __future__ import annotations

import re

from .models import MemoryType


# ---------------------------------------------------------------------------
# Feature lexicons
# ---------------------------------------------------------------------------

# F1 — Temporal anchoring
_TEMPORAL_MARKERS = re.compile(
    r"\b("
    r"yesterday|today|tomorrow|tonight|"
    r"last\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"this\s+(morning|afternoon|evening|week|month|year)|"
    r"just\s+now|recently|earlier|"
    r"on\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"in\s+\d{4}|"
    r"ago\b|"
    r"\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?"  # date patterns
    r")\b",
    re.IGNORECASE,
)

# Past-tense narrative: first-person pronoun within 5 tokens of a past-tense verb
_PAST_TENSE_VERBS = re.compile(
    r"\b(met|went|said|told|learned|discovered|realized|found|saw|heard|"
    r"had|was|were|got|made|did|came|took|gave|thought|felt|decided|started|"
    r"finished|completed|worked|talked|discussed|called|wrote|read|ran|"
    r"happened|occurred|remembered|forgot|missed|attended|joined|left)\b",
    re.IGNORECASE,
)
_FIRST_PERSON = re.compile(r"\b(I|me|my|we|our|us)\b", re.IGNORECASE)


# F2 — Habitual/dispositional framing
_HABITUAL_MARKERS = re.compile(
    r"\b(always|usually|never|often|tend\s+to|every\s+time|"
    r"regularly|frequently|sometimes|rarely|mostly|typically|generally)\b",
    re.IGNORECASE,
)
_PREFERENCE_VERBS = re.compile(
    r"\b(prefer|like|love|hate|dislike|enjoy|avoid|"
    r"prefer\s+to|like\s+to|love\s+to|hate\s+to)\b",
    re.IGNORECASE,
)
_IDENTITY_FRAMING = re.compile(
    r"\b(I\s+am\s+someone\s+who|I'?m\s+the\s+kind\s+of|"
    r"I\s+believe|I\s+always|I'?m\s+(patient|direct|careful|detail|focused|"
    r"organized|creative|analytical|introvert|extrovert))\b",
    re.IGNORECASE,
)


# F3 — Sequential/procedural structure
_SEQUENTIAL_MARKERS = re.compile(
    r"\b(first|then|next|finally|after\s+that|lastly|"
    r"step\s+\d+|step\s+one|step\s+two|step\s+three|"
    r"1\.|2\.|3\.)\b",
    re.IGNORECASE,
)
_IMPERATIVE_VERBS = re.compile(
    r"^(run|open|click|navigate|execute|install|configure|"
    r"create|delete|update|deploy|start|stop|check|verify|"
    r"copy|paste|select|enter|type|go\s+to|look\s+for)\b",
    re.IGNORECASE | re.MULTILINE,
)
_PROCEDURAL_FRAMING = re.compile(
    r"\b(to\s+do\s+this|to\s+set\s+up|the\s+way\s+to|"
    r"the\s+process\s+(is|for)|in\s+order\s+to|how\s+to)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_f1(content: str) -> bool:
    """F1: Temporal anchoring — explicit time references OR past-tense narrative."""
    if _TEMPORAL_MARKERS.search(content):
        return True
    # Past-tense verb with first-person subject: look for both within same sentence
    sentences = re.split(r"[.!?]", content)
    for sent in sentences:
        if _PAST_TENSE_VERBS.search(sent) and _FIRST_PERSON.search(sent):
            return True
    return False


def _extract_f2(content: str) -> bool:
    """F2: Habitual/dispositional framing — frequency adverbs, preference verbs, identity."""
    return bool(
        _HABITUAL_MARKERS.search(content)
        or _PREFERENCE_VERBS.search(content)
        or _IDENTITY_FRAMING.search(content)
    )


def _extract_f3(content: str) -> bool:
    """F3: Sequential/procedural structure — ≥2 sequential markers OR imperative cluster."""
    sequential_hits = len(_SEQUENTIAL_MARKERS.findall(content))
    if sequential_hits >= 2:
        return True
    imperative_hits = len(_IMPERATIVE_VERBS.findall(content))
    if imperative_hits >= 1 and _PROCEDURAL_FRAMING.search(content):
        return True
    return False


def _extract_f4(content: str) -> bool:
    """F4: Self-reference — first-person subject in non-narrative context."""
    return bool(_FIRST_PERSON.search(content))


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------

def classify(content: str) -> MemoryType:
    """
    Classify a memory content string into a MemoryType.

    Priority order: EPISODIC > PERSONA > PROCEDURAL > SEMANTIC.
    SEMANTIC is the residual category.

    This function is deterministic: identical input always yields identical output
    across all platforms, Python versions, and VaultMem versions.

    Args:
        content: The memory text to classify.

    Returns:
        A MemoryType value.
    """
    f1 = _extract_f1(content)
    f2 = _extract_f2(content)
    f3 = _extract_f3(content)
    f4 = _extract_f4(content)

    if f1 and f4:
        return MemoryType.EPISODIC
    if f2 and f4:
        return MemoryType.PERSONA
    if f3:
        return MemoryType.PROCEDURAL
    return MemoryType.SEMANTIC


def classify_with_features(content: str) -> tuple[MemoryType, dict[str, bool]]:
    """
    Classify with full feature visibility (useful for debugging and audit).

    Returns:
        (MemoryType, {"F1": bool, "F2": bool, "F3": bool, "F4": bool})
    """
    f1 = _extract_f1(content)
    f2 = _extract_f2(content)
    f3 = _extract_f3(content)
    f4 = _extract_f4(content)

    features = {"F1": f1, "F2": f2, "F3": f3, "F4": f4}

    if f1 and f4:
        return MemoryType.EPISODIC, features
    if f2 and f4:
        return MemoryType.PERSONA, features
    if f3:
        return MemoryType.PROCEDURAL, features
    return MemoryType.SEMANTIC, features
