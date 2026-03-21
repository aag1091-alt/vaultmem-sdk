"""
VaultMem embedding layer.

Default: LocalEmbedder using sentence-transformers/all-MiniLM-L6-v2.
No memory content is transmitted to any external service by default.

Protocol: any class with .embed(text: str) -> list[float] and
          .embed_batch(texts: list[str]) -> list[list[float]] qualifies.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Protocol (structural typing — any conforming class works)
# ---------------------------------------------------------------------------

@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding providers."""

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a float list."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Returns a list of float lists."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...


# ---------------------------------------------------------------------------
# LocalEmbedder — sentence-transformers, no external API calls
# ---------------------------------------------------------------------------

class LocalEmbedder:
    """
    Local embedder using sentence-transformers.

    Default model: all-MiniLM-L6-v2 (384-dim, ~22M params, ~80ms/sentence on CPU).
    No text is transmitted to any external service.

    The reference model (all-MiniLM-L6-v2) must be used for consistent cosine
    thresholds. Using alternative models with different vector spaces will
    produce shifted thresholds and unpredictable affinity detection behavior.
    """

    REFERENCE_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = REFERENCE_MODEL) -> None:
        self._model_name = model_name
        self._model = None  # Lazy load

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    def embed(self, text: str) -> list[float]:
        """Embed a single text. Returns 384-dim float list."""
        self._load()
        vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently. Returns list of 384-dim float lists."""
        if not texts:
            return []
        self._load()
        vecs = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    @property
    def dimension(self) -> int:
        return 384


# ---------------------------------------------------------------------------
# NullEmbedder — for testing without sentence-transformers installed
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# OllamaEmbedder — remote Ollama embedding server, no local GPU required
# ---------------------------------------------------------------------------

class OllamaEmbedder:
    """
    Embedder backed by a remote Ollama instance.

    Default model: all-minilm (all-MiniLM-L6-v2, 384-dim) — identical vector
    space to LocalEmbedder, so cosine thresholds are directly comparable.

    Note: memory content is transmitted to the Ollama host over the network.
    Use only when the Ollama host is trusted (e.g. your own machine via
    Tailscale).  For air-gapped privacy use LocalEmbedder instead.

    Args:
        base_url: Ollama API root, e.g. "http://100.118.247.106:11434".
        model:    Ollama model name.  Must produce 384-dim output to match
                  the rest of the VaultMem pipeline.
        timeout:  HTTP request timeout in seconds.
    """

    REFERENCE_MODEL = "all-minilm"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = REFERENCE_MODEL,
        timeout: int = 30,
    ) -> None:
        self._base    = base_url.rstrip("/")
        self._model   = model
        self._timeout = timeout

    def _call_batch(self, texts: list[str]) -> list[list[float]]:
        """Call Ollama /api/embed (Ollama ≥ 0.3) — accepts a list of inputs."""
        import urllib.request, json
        payload = json.dumps({"model": self._model, "input": texts}).encode()
        req = urllib.request.Request(
            self._base + "/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())["embeddings"]

    def embed(self, text: str) -> list[float]:
        return self._call_batch([text])[0]

    def embed_batch(self, texts: list[str], chunk_size: int = 20) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), chunk_size):
            results.extend(self._call_batch(texts[i : i + chunk_size]))
        return results

    @property
    def dimension(self) -> int:
        return 384


class NullEmbedder:
    """
    Embedder that returns zero vectors. For testing only.
    Search will not return meaningful results.
    """
    DIM = 384

    def embed(self, text: str) -> list[float]:
        return [0.0] * self.DIM

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.DIM for _ in texts]

    @property
    def dimension(self) -> int:
        return self.DIM
