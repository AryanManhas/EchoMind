from __future__ import annotations

import json
from typing import Any
from urllib import error, request


class LLMService:
    def __init__(
        self, enabled: bool, provider: str, model: str, endpoint_url: str
    ) -> None:
        self.enabled = enabled
        self.provider = provider
        self.model = model
        self.endpoint_url = endpoint_url

    def answer_with_context(self, query: str, memories: list[dict[str, Any]]) -> dict[str, Any]:
        citations = self._extract_citations(memories)
        context_lines = [
            f"- text: {m.get('text', '')}; type: {m.get('type')}; person: {m.get('person')}; time: {m.get('time')}"
            for m in memories[:5]
        ]
        context_block = "\n".join(context_lines) if context_lines else "- no related memories found"

        if self.enabled and self.provider == "ollama":
            llm_answer = self._ask_ollama(query, context_block)
            if llm_answer:
                return {"answer": llm_answer, "source": "ollama", "citations": citations}

        # deterministic fallback when no LLM available
        fallback = self._fallback_answer(query, memories)
        return {"answer": fallback, "source": "fallback", "citations": citations}

    def _ask_ollama(self, query: str, context_block: str) -> str | None:
        prompt = (
            "You are a memory assistant. Use ONLY provided memory context.\n"
            "If uncertain, say you are not sure.\n\n"
            f"Memory context:\n{context_block}\n\n"
            f"User query: {query}\n"
            "Give a concise answer in 1-2 sentences."
        )

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            req = request.Request(
                self.endpoint_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=25) as response:  # noqa: S310
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            return ((parsed.get("message") or {}).get("content") or "").strip() or None
        except (error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _fallback_answer(query: str, memories: list[dict[str, Any]]) -> str:
        if not memories:
            return "I could not find a related memory yet."
        top = memories[0]
        person = top.get("person")
        time = top.get("time")
        if person and time:
            return f"The most relevant memory is about {person} at {time}."
        if person:
            return f"The most relevant memory is about {person}."
        return f"The most relevant memory is: {top.get('text', '')}"

    @staticmethod
    def _extract_citations(memories: list[dict[str, Any]]) -> list[int]:
        citations: list[int] = []
        for item in memories[:5]:
            memory_id = item.get("id")
            if isinstance(memory_id, int):
                citations.append(memory_id)
        return citations
