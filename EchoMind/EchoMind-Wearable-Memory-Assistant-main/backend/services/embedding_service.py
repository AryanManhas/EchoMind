from __future__ import annotations

from typing import Any


class EmbeddingService:
    def __init__(self, model_name: str, enabled: bool = False) -> None:
        self.model_name = model_name
        self._enabled_config = enabled
        self._model = None
        self._error: str | None = None
        if self._enabled_config:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)

    @property
    def enabled(self) -> bool:
        return self._enabled_config and self._model is not None

    def embed_text(self, text: str) -> list[float] | None:
        if not self._enabled_config:
            return None
        if self._model is None:
            self._load_model()
        if self._model is None:
            return None
        vector = self._model.encode([text])[0]
        return vector.tolist()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "configured": self._enabled_config,
            "error": self._error,
        }
