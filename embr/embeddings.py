"""Turn memory text into a vector for semantic relevance.

Two implementations behind one `Embedder` interface:

  * `DeterministicEmbedder` - dependency-free, hashes words into a fixed-length bag-of-words
    vector. No GPU, no download, fully reproducible; captures lexical overlap (not deep
    semantics). This is the default so the core and the tests run anywhere.
  * `SentenceTransformerEmbedder` - real semantic embeddings from a compact model, behind
    the `[ml]` extra. Drop-in for the eval hardware.

Both are pure with respect to a given text: the same string always encodes to the same
vector, which is what lets embeddings be cached and persisted.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

_WORD = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens. Shared by the embedder and the relevance signal so there is
    one tokenizer, not two that can drift apart."""
    return _WORD.findall(text.lower())


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns a string into a fixed-length vector."""

    dim: int

    def encode(self, text: str) -> list[float]: ...


class DeterministicEmbedder:
    """Hashes words into a fixed-length, unit-normalised bag-of-words vector.

    Stable across processes (uses a content hash, not Python's salted `hash()`), so a vector
    written to the store today matches one computed tomorrow.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> int:
        # Content hash -> a fixed bucket, identical in every process and run.
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.dim

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            vector[self._bucket(token)] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector  # empty text -> all zeros (cosine treats it as neutral)
        return [value / norm for value in vector]


class SentenceTransformerEmbedder:
    """Real semantic embeddings from a compact sentence-transformers model.

    The model is imported and loaded lazily on first use, so importing this module never
    pulls in the heavy `[ml]` dependencies until an embedding is actually requested.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None  # loaded on first encode
        self.dim = 384  # all-MiniLM-L6-v2 output size; corrected once the model loads

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy: needs [ml] extra

            self._model = SentenceTransformer(self.model_name)
            # sentence-transformers 5.x renamed this and deprecated the old spelling, but
            # the `ml` extra allows 2.2 upward, so both names have to work.
            dimension = getattr(self._model, "get_embedding_dimension", None) or (
                self._model.get_sentence_embedding_dimension
            )
            self.dim = dimension()

    def encode(self, text: str) -> list[float]:
        self._ensure_model()
        assert self._model is not None
        return [float(value) for value in self._model.encode(text)]
