"""
VaultMem context sanitizer — PII stripping before LLM injection.

Uses Microsoft Presidio to detect and replace sensitive entities before memory
context is sent to a cloud LLM.  Two NER backends are supported:

    "spacy"        — lightweight (~12 MB model).  Good default for cloud
                     deployments or anywhere torch is unavailable.
                     Install: pip install 'vaultmem[spacy]'
                              python -m spacy download en_core_web_sm

    "transformers" — higher accuracy (~400 MB model, requires torch).
                     Best for local / high-privacy deployments.
                     Install: pip install 'vaultmem[presidio]'

Person names become natural-sounding pseudonyms (e.g. "Avinash" → "Jordan")
so the LLM addresses the user fluently and the response reads naturally after
restoration.  Structured PII (emails, phones, SSNs, IPs, credit cards, URLs)
becomes typed tokens ([EMAIL_1], [PHONE_1], …).

Usage::

    from vaultmem.sanitize import Sanitizer

    # Lightweight — works on Streamlit Cloud, CI, etc.
    san = Sanitizer(backend="spacy", owner_pseudonym="Jordan")

    # High-accuracy — local / privacy-critical deployments
    san = Sanitizer(backend="transformers", owner_pseudonym="Jordan")

    sanitized, rmap = san.sanitize(context, owner_name="Avinash")
    # "Avinash met Sarah at Microsoft last week, email him at a@ex.com"
    # → "Jordan met Casey at Acme last week, email him at [EMAIL_1]"

    response = llm.chat(f"Memories: {sanitized}\\nQuery: {query}")
    # LLM responds naturally: "Jordan, based on what Casey said at Acme..."

    clean = san.restore(response, rmap)
    # → "Avinash, based on what Sarah said at Microsoft..."

Notes:

    - The same ``Sanitizer`` instance is session-scoped: identical real values
      always produce the same pseudonym/token across multiple memories, so the
      LLM can reason about repeated entities as one object.

    - Pseudonym pools are sized for typical sessions.  If a pool is exhausted
      the sanitizer falls back to typed tokens ([PERSON_21] etc.).

    - Restoration replaces longest tokens first to avoid partial collisions
      (e.g. "Jordan Smith" is restored before "Jordan").
"""
from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Pseudonym pools
# ---------------------------------------------------------------------------

_PERSON_POOL: list[str] = [
    "Jordan", "Casey", "Morgan", "Riley", "Avery",
    "Quinn", "Sage", "Rowan", "Finley", "Blair",
    "Drew", "Logan", "Reese", "Skyler", "Parker",
    "Cameron", "Jamie", "Kendall", "Taylor", "Hayden",
]

_ORG_POOL: list[str] = [
    "Acme", "Globex", "Initech", "Pied Piper", "Hooli",
    "Initrode", "Massive Dynamic", "Umbrella", "Veridian",
]

_LOC_POOL: list[str] = [
    "Springfield", "Shelbyville", "Riverdale", "Maplewood",
    "Sunnydale", "Storybrooke", "Eagleton",
]


# ---------------------------------------------------------------------------
# Entity type configuration
# ---------------------------------------------------------------------------

# Types that are never PII — skip entirely regardless of backend
_SKIP_ENTITY_TYPES: frozenset[str] = frozenset({
    "DATE_TIME",   # dates, times, durations — not identifying
    "CARDINAL",    # numbers
    "ORDINAL",     # first, second, …
    "PERCENT",     # percentages
    "MONEY",       # monetary values
    "QUANTITY",    # measurements
    "LANGUAGE",    # programming/spoken languages
})


# ---------------------------------------------------------------------------
# Restoration helper
# ---------------------------------------------------------------------------

def _apply_restoration(text: str, restoration_map: dict[str, str]) -> str:
    """Replace pseudonyms / tokens with real values.
    Processes longest keys first to avoid partial-match collisions.
    """
    for pseudo, real in sorted(restoration_map.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(pseudo, real)
    return text


# ---------------------------------------------------------------------------
# Sanitizer
# ---------------------------------------------------------------------------

class Sanitizer:
    """
    Session-scoped context sanitizer for privacy-preserving LLM injection.

    Args:
        owner_pseudonym: Placeholder name used in place of the vault owner's
                         real name.  Defaults to ``"Jordan"``.
        backend:         NER backend to use.
                         ``"spacy"`` (default) — lightweight, no torch required;
                         install ``vaultmem[spacy]`` + ``en_core_web_sm``.
                         ``"transformers"`` — higher accuracy, requires torch;
                         install ``vaultmem[presidio]``.
    """

    def __init__(
        self,
        owner_pseudonym: str = "Jordan",
        backend: str = "spacy",
    ) -> None:
        if backend not in ("spacy", "transformers"):
            raise ValueError(f"backend must be 'spacy' or 'transformers', got {backend!r}")

        self._owner_pseudonym = owner_pseudonym
        self._backend = backend

        # real_value → pseudonym/token  (stable within this instance)
        self._forward: dict[str, str] = {}

        # counters for typed tokens ([EMAIL_1], [PHONE_2], …)
        self._token_counters: dict[str, int] = {}

        # next index into each pseudonym pool
        self._pool_idx: dict[str, int] = {"PERSON": 0, "ORG": 0, "LOC": 0}

        self._init_presidio()

    # ------------------------------------------------------------------
    # Presidio initialisation
    # ------------------------------------------------------------------

    def _init_presidio(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import-untyped]
            from presidio_analyzer.nlp_engine import NlpEngineProvider  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "presidio-analyzer is required for the Sanitizer.\n"
                "Install with: pip install 'vaultmem[spacy]' or 'vaultmem[presidio]'"
            ) from exc

        try:
            if self._backend == "spacy":
                config = {
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                }
                install_hint = (
                    "pip install 'vaultmem[spacy]' && "
                    "python -m spacy download en_core_web_sm"
                )
            else:
                config = {
                    "nlp_engine_name": "transformers",
                    "models": [{"lang_code": "en", "model_name": {
                        "spacy": "en_core_web_sm",
                        "transformers": "dslim/bert-base-NER",
                    }}],
                }
                install_hint = "pip install 'vaultmem[presidio]'"

            provider = NlpEngineProvider(nlp_configuration=config)
            self._analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise Presidio '{self._backend}' NLP engine.\n"
                f"Install with: {install_hint}"
            ) from exc

    # ------------------------------------------------------------------
    # Internal assignment helpers
    # ------------------------------------------------------------------

    def _assign(self, real: str, entity_type: str) -> str:
        """Return stable pseudonym/token for *real*, creating one if new."""
        if real in self._forward:
            return self._forward[real]

        used = set(self._forward.values())
        pools: dict[str, list[str]] = {
            "PERSON": _PERSON_POOL,
            "ORG":    _ORG_POOL,
            "LOC":    _LOC_POOL,
        }

        if entity_type in pools:
            pool = pools[entity_type]
            start = self._pool_idx[entity_type]
            for offset in range(len(pool)):
                candidate = pool[(start + offset) % len(pool)]
                if candidate not in used:
                    self._pool_idx[entity_type] = (start + offset + 1) % len(pool)
                    self._forward[real] = candidate
                    return candidate
            # Pool exhausted — fall through to typed token

        n = self._token_counters.get(entity_type, 0) + 1
        self._token_counters[entity_type] = n
        token = f"[{entity_type}_{n}]"
        self._forward[real] = token
        return token

    # ------------------------------------------------------------------
    # Public: sanitize
    # ------------------------------------------------------------------

    def sanitize(
        self,
        text: str,
        owner_name: Optional[str] = None,
    ) -> tuple[str, dict[str, str]]:
        """
        Sanitize *text* for safe injection into a cloud LLM.

        Args:
            text:       Memory context string (plaintext, decrypted from vault).
            owner_name: The vault owner's real name.  Replaced with
                        ``owner_pseudonym`` so the LLM can address the user
                        naturally and the response can be restored client-side.

        Returns:
            ``(sanitized_text, restoration_map)``

            Pass *restoration_map* unchanged to :meth:`restore` after
            receiving the LLM response.
        """
        # Run Presidio on original text so span offsets are valid
        results = self._analyzer.analyze(text=text, language="en")

        # Replace right-to-left to avoid offset drift
        results = sorted(results, key=lambda r: -r.start)

        for res in results:
            # Skip non-PII entity types entirely
            if res.entity_type in _SKIP_ENTITY_TYPES:
                continue

            real = text[res.start:res.end]

            if owner_name and real.lower() == owner_name.lower():
                pseudo = self._owner_pseudonym
                self._forward.setdefault(real, pseudo)
            else:
                etype = res.entity_type
                pool_type = (
                    "PERSON" if etype == "PERSON" else
                    "ORG"    if etype in ("ORG", "ORGANIZATION") else
                    "LOC"    if etype in ("LOCATION", "GPE") else
                    etype
                )
                pseudo = self._assign(real, pool_type)

            text = text[: res.start] + pseudo + text[res.end:]

        restoration_map = {pseudo: real for real, pseudo in self._forward.items()}
        return text, restoration_map

    # ------------------------------------------------------------------
    # Public: restore
    # ------------------------------------------------------------------

    def restore(self, text: str, restoration_map: dict[str, str]) -> str:
        """
        Restore real entity values in the LLM's response.

        Args:
            text:             Raw LLM response containing pseudonyms / tokens.
            restoration_map:  The map returned by :meth:`sanitize`.

        Returns:
            Response with all pseudonyms / tokens replaced by real values.
        """
        return _apply_restoration(text, restoration_map)
