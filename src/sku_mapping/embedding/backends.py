"""Replaceable, lazy local embedding backends."""

from __future__ import annotations

import importlib.metadata
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sku_mapping.config import EmbeddingConfig

LOGGER = logging.getLogger(__name__)

# A configured model is not "available" until it actually produces a usable
# vector.  This probe text is fixed so the check is deterministic and cheap; it
# is never used as a real offer or master text.
HEALTH_CHECK_PROBE_TEXT = "embedding health check probe 400 g"


class EmbeddingBackendError(RuntimeError):
    """Raised when a configured embedding backend is unavailable or unsafe."""


class EmbeddingBackend(Protocol):
    """Minimal interface independent from the LightGBM predictor."""

    @property
    def model_id(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def device(self) -> str: ...

    def ensure_available(self) -> None: ...

    def encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray: ...


class LocalSentenceTransformerBackend:
    """Lazy sentence-transformers backend using a free local model."""

    _models: dict[
        tuple[str, str, str | None, bool, int], object
    ] = {}
    _lock = threading.RLock()

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._model: object | None = None
        self._resolved_version: str | None = None

    @property
    def model_id(self) -> str:
        return f"sentence-transformers:{self.config.model_name}"

    @property
    def model_version(self) -> str:
        self.ensure_available()
        if self._resolved_version is None:
            raise EmbeddingBackendError("Embedding model version is unresolved")
        return self._resolved_version

    @property
    def device(self) -> str:
        if self._model is None:
            return self.config.device
        return str(getattr(self._model, "device", self.config.device))

    def _resolve_version(self, model: object) -> str:
        if self.config.model_version:
            return self.config.model_version
        package_version = importlib.metadata.version("sentence-transformers")
        revision = None
        first_module = getattr(model, "_first_module", lambda: None)()
        auto_model = getattr(first_module, "auto_model", None)
        model_config = getattr(auto_model, "config", None)
        revision = getattr(model_config, "_commit_hash", None)
        if not revision:
            card_data = getattr(model, "_model_card_vars", {})
            if isinstance(card_data, dict):
                revision = card_data.get("model_revision")
        if not revision:
            raise EmbeddingBackendError(
                "Cannot resolve the local model revision safely; configure "
                "embedding.model_version explicitly before caching"
            )
        return f"sentence-transformers-{package_version}+{revision}"

    def ensure_available(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingBackendError(
                "sentence-transformers is unavailable; install the optional "
                "embedding dependencies or select another local backend"
            ) from error
        device = "" if self.config.device == "auto" else self.config.device
        key = (
            self.config.model_name,
            device,
            self.config.model_version,
            self.config.local_files_only,
            self.config.max_sequence_length,
        )
        with self._lock:
            model = self._models.get(key)
            if model is None:
                kwargs = {"device": device} if device else {}
                kwargs["local_files_only"] = self.config.local_files_only
                model = SentenceTransformer(self.config.model_name, **kwargs)
                self._models[key] = model
            self._model = model
            if hasattr(model, "max_seq_length"):
                model.max_seq_length = self.config.max_sequence_length
            self._resolved_version = self._resolve_version(model)

    def encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        self.ensure_available()
        result = self._model.encode(  # type: ignore[union-attr]
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(result, dtype=np.float32)


class LocalHashingEmbeddingBackend:
    """Dependency-light deterministic local embedding for tests and dry runs."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._vectorizer = None

    @property
    def model_id(self) -> str:
        return f"local-hashing:{self.config.model_name}"

    @property
    def model_version(self) -> str:
        if self.config.model_version:
            return self.config.model_version
        sklearn_version = importlib.metadata.version("scikit-learn")
        return f"sklearn-{sklearn_version}+hashing-word-1-2-384-v1"

    @property
    def device(self) -> str:
        return "cpu"

    def ensure_available(self) -> None:
        if self._vectorizer is not None:
            return
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vectorizer = HashingVectorizer(
            n_features=384,
            alternate_sign=False,
            norm=None,
            lowercase=False,
            ngram_range=(1, 2),
        )

    def encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        del batch_size
        self.ensure_available()
        matrix = self._vectorizer.transform(list(texts))  # type: ignore[union-attr]
        return np.asarray(matrix.toarray(), dtype=np.float32)


BackendFactory = Callable[[EmbeddingConfig], EmbeddingBackend]
_BACKEND_FACTORIES: dict[str, BackendFactory] = {
    "local_sentence_transformer": LocalSentenceTransformerBackend,
    "local_hashing": LocalHashingEmbeddingBackend,
}
_FACTORY_LOCK = threading.RLock()


def register_embedding_backend(name: str, factory: BackendFactory) -> None:
    """Register or replace a backend factory explicitly."""
    if not name.strip() or not callable(factory):
        raise ValueError("A non-empty backend name and callable are required")
    with _FACTORY_LOCK:
        _BACKEND_FACTORIES[name] = factory


def create_embedding_backend(config: EmbeddingConfig) -> EmbeddingBackend:
    """Construct the configured backend without loading its model."""
    with _FACTORY_LOCK:
        factory = _BACKEND_FACTORIES.get(config.backend)
    if factory is None:
        raise EmbeddingBackendError(
            f"Unknown embedding backend: {config.backend!r}"
        )
    return factory(config)
