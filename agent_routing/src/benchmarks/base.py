"""Common types for normalized benchmark rows."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def question_hash(question: str) -> str:
    """Stable content hash of a question text.

    example_id values are assigned by load order and silently change whenever a
    normalized cache is rebuilt; cross-run bookkeeping (e.g. excluding SFT rows
    from GRPO training) must key on this hash instead.
    """
    norm = re.sub(r"\s+", " ", str(question).strip().lower())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


@dataclass
class StandardRow:
    """Unified representation of a benchmark example.

    All loaders must produce instances of this shape.
    """
    example_id: int
    benchmark_name: str           # e.g. "medqa"
    task_subtype: str             # e.g. "us_4options"
    question: str
    choices: Dict[str, str]       # {"A": "...", "B": "...", ...}
    ground_truth: str             # canonical key matching choices, e.g. "B"
    context: str = ""             # optional long context (medqa typically empty)
    metadata: Dict[str, Any] = field(default_factory=dict)
    split: str = ""               # "train" | "dev" | "test"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": int(self.example_id),
            "benchmark_name": self.benchmark_name,
            "task_subtype": self.task_subtype,
            "question": self.question,
            "choices": dict(self.choices),
            "ground_truth": self.ground_truth,
            "context": self.context,
            "metadata": dict(self.metadata),
            "split": self.split,
        }


def normalize_choices(raw: Any) -> Dict[str, str]:
    """Coerce raw choice fields into {key: text} with sorted keys."""
    if isinstance(raw, dict):
        return {
            str(k).strip(): str(v).strip()
            for k, v in sorted(raw.items(), key=lambda kv: str(kv[0]))
            if str(k).strip()
        }
    if isinstance(raw, (list, tuple)):
        out: Dict[str, str] = {}
        for i, v in enumerate(raw):
            if i >= 26:
                break
            out[chr(ord("A") + i)] = str(v).strip()
        return out
    return {}


def label_to_token(label: str) -> str:
    """Normalize an answer label into an `ANSWER_<TOKEN>` safe form."""
    s = str(label).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError(f"Invalid label for tokenization: {label!r}")
    return s.upper()


def resolve_mcq_answer(raw_answer: Any, choices: Dict[str, str]) -> str:
    """Map a raw answer field into one of the choice keys."""
    if not choices:
        return ""
    raw = "" if raw_answer is None else str(raw_answer).strip()
    if not raw:
        return ""
    if raw in choices:
        return raw
    upper = raw.upper()
    if upper in choices:
        return upper
    if raw.isdigit():
        # AMBIGUOUS for 1 <= idx < len(keys): we assume 0-based indexing (the
        # HF convention). A 1-based dataset would be silently off by one —
        # when adding such a source, convert its answers to letter keys in the
        # loader instead of relying on this fallback.
        idx = int(raw)
        keys = list(choices.keys())
        if 0 <= idx < len(keys):
            return keys[idx]
        if 1 <= idx <= len(keys):
            return keys[idx - 1]
    for k, v in choices.items():
        if raw == v or raw.lower() == v.lower():
            return k
    return ""