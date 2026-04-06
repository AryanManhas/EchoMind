from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any


class NLPService:
    ACTION_RULES: dict[str, str] = {
        "meet": "meeting",
        "meeting": "meeting",
        "call": "call",
        "send": "reminder",
    }

    def __init__(self) -> None:
        self.nlp = self._load_spacy_pipeline()
        self._analyzer = None
        self._anonymizer = None
        self._init_presidio()

    def _init_presidio(self):
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
        except ImportError:
            pass

    def _redact_pii(self, text: str) -> str:
        if not self._analyzer or not self._anonymizer:
            return text
        results = self._analyzer.analyze(
            text=text, 
            entities=["CREDIT_CARD", "US_SSN", "PHONE_NUMBER", "EMAIL_ADDRESS", "IBAN_CODE"], 
            language='en'
        )
        anonymized_result = self._anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized_result.text

    @staticmethod
    def _load_spacy_pipeline():
        try:
            import spacy  # type: ignore

            return spacy.load("en_core_web_sm")
        except Exception:  # noqa: BLE001
            try:
                import spacy  # type: ignore

                return spacy.blank("en")
            except Exception:  # noqa: BLE001
                return None

    def extract_memory(self, text: str) -> dict[str, Any]:
        text = self._redact_pii(text)
        lowered = text.lower()
        mem_type = next(
            (mem_type for keyword, mem_type in self.ACTION_RULES.items() if keyword in lowered),
            "general",
        )

        person = None
        time_value = None
        date_entity = None
        clock_time_entity = None

        # Prefer complete expressions like "tomorrow at 4 PM" before other extraction paths.
        full_time_match = re.search(
            r"\b((?:today|tomorrow|tonight|this morning|this afternoon|this evening|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+at\s+\d{1,2}(?::\d{2})?\s?(?:AM|PM|am|pm))\b",
            text,
            flags=re.IGNORECASE,
        )
        if full_time_match:
            time_value = full_time_match.group(1)

        if self.nlp is not None:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON" and not person:
                    person = ent.text
                if ent.label_ == "DATE" and not date_entity:
                    date_entity = ent.text
                if ent.label_ == "TIME" and not clock_time_entity:
                    clock_time_entity = ent.text
                if ent.label_ in {"DATE", "TIME"} and not time_value:
                    time_value = ent.text

        if not time_value and date_entity and clock_time_entity:
            time_value = f"{date_entity} at {clock_time_entity}"

        # fallback regex if spaCy model is unavailable or misses entities
        if not person:
            person_match = re.search(r"\bwith\s+([A-Z][a-zA-Z]+)\b", text)
            if person_match:
                person = person_match.group(1)
        if not person:
            # Handle imperative phrasing like "Meet Rahul tomorrow at 4 PM"
            person_match = re.search(
                r"\b(?:meet|call|send)\s+([A-Z][a-zA-Z]+)\b",
                text,
                flags=re.IGNORECASE,
            )
            if person_match:
                person = person_match.group(1)

        if not time_value:
            time_match = re.search(
                r"\b((?:today|tomorrow)(?:\s+at\s+\d{1,2}(?::\d{2})?\s?(?:AM|PM|am|pm)?)?|\d{1,2}(?::\d{2})?\s?(?:AM|PM|am|pm))\b",
                text,
            )
            if time_match:
                time_value = time_match.group(1)

        reminder_meta = self._detect_reminder_meta(
            text=text,
            mem_type=mem_type,
            has_person=bool(person),
            has_time=bool(time_value),
        )
        due_time = self._infer_due_time_iso(time_value)

        return {
            "type": mem_type,
            "person": person,
            "time": time_value,
            "text": text.strip(),
            "is_reminder": reminder_meta["is_reminder"],
            "priority": reminder_meta["priority"],
            "status": "pending" if reminder_meta["is_reminder"] else "captured",
            "due_time": due_time,
            "importance_score": reminder_meta["score"],
        }

    @staticmethod
    def _detect_reminder_meta(
        text: str, mem_type: str, has_person: bool, has_time: bool
    ) -> dict[str, Any]:
        lowered = text.lower()
        score = 0.0
        keywords = [
            "remember",
            "remind",
            "don't forget",
            "deadline",
            "urgent",
            "must",
            "need to",
            "follow up",
            "call",
            "meet",
            "send",
            "tomorrow",
            "today",
        ]
        score += sum(0.12 for kw in keywords if kw in lowered)
        if mem_type in {"meeting", "call", "reminder"}:
            score += 0.2
        if has_person:
            score += 0.1
        if has_time:
            score += 0.18

        normalized = min(score, 1.0)
        is_reminder = normalized >= 0.33

        if normalized >= 0.7:
            priority = "high"
        elif normalized >= 0.45:
            priority = "medium"
        else:
            priority = "low"
        return {"is_reminder": is_reminder, "priority": priority, "score": normalized}

    @staticmethod
    def _infer_due_time_iso(time_value: str | None) -> str | None:
        if not time_value:
            return None
        text = time_value.lower().strip()
        now = datetime.utcnow()

        # Basic local parsing for "today/tomorrow at 4 PM" style phrases.
        hour_match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
        hour = 9
        minute = 0
        if hour_match:
            hour = int(hour_match.group(1))
            minute = int(hour_match.group(2) or 0)
            marker = hour_match.group(3)
            if marker == "pm" and hour < 12:
                hour += 12
            if marker == "am" and hour == 12:
                hour = 0

        base = now
        if "tomorrow" in text:
            base = now + timedelta(days=1)
        elif "today" in text:
            base = now
        elif any(day in text for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
            # Keep simple prototype behavior for weekday references.
            base = now + timedelta(days=1)

        due = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if due < now and "today" in text:
            due = due + timedelta(days=1)
        return due.isoformat()

    @staticmethod
    def to_concise_response(memory: dict[str, Any]) -> str:
        fragments = []
        if memory.get("type") and memory["type"] != "general":
            fragments.append(memory["type"].capitalize())
        if memory.get("person"):
            fragments.append(f"with {memory['person']}")
        if memory.get("time"):
            time_text = str(memory["time"]).strip()
            day_words = {
                "today",
                "tomorrow",
                "tonight",
                "this morning",
                "this afternoon",
                "this evening",
            }
            if " at " in time_text.lower():
                fragments.append(time_text)
            elif time_text.lower() in day_words:
                fragments.append(time_text)
            else:
                fragments.append(f"at {time_text}")
        if not fragments:
            return memory.get("text", "Memory saved")
        return " ".join(fragments)
