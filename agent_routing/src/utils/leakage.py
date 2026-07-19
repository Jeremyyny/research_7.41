"""GT leakage auditor for teacher-generated SFT data.

Strategy:
1. Build keyword set from ground_truth (canonical label, token form, choice value text).
2. Recursively scan all string fields in the generated JSON.
3. Optionally allow safe contexts (e.g. "the question asks whether X is true").

We err on the side of false positives — flagged samples should be retried or dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set, Tuple


SAFE_TEMPLATE_PATTERNS = (
    # Allow the prompt's restated question to mention the answer text.
    re.compile(r"the question asks", re.IGNORECASE),
)


@dataclass
class LeakageAuditResult:
    leaked: bool
    matches: List[Tuple[str, str]]  # (field_path, matched_keyword)


class LeakageAuditor:
    """Scan teacher-generated objects for GT leakage."""

    def __init__(self, min_keyword_len: int = 3) -> None:
        self.min_keyword_len = int(min_keyword_len)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _build_keywords(
        self,
        ground_truth_label: str,
        ground_truth_text: str = "",
        token_form: str = "",
        extra: Iterable[str] = (),
    ) -> Set[str]:
        kws: Set[str] = set()
        for raw in [ground_truth_label, ground_truth_text, token_form, *extra]:
            if not raw:
                continue
            n = self._normalize(str(raw))
            if len(n) >= self.min_keyword_len:
                kws.add(n)
        return kws

    def _scan(self, obj: Any, keywords: Set[str], path: str = "") -> List[Tuple[str, str]]:
        hits: List[Tuple[str, str]] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                hits.extend(self._scan(v, keywords, f"{path}.{k}" if path else str(k)))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                hits.extend(self._scan(v, keywords, f"{path}[{i}]"))
        elif isinstance(obj, str):
            t = self._normalize(obj)
            if any(p.search(t) for p in SAFE_TEMPLATE_PATTERNS):
                return hits
            for kw in keywords:
                # Whole-word/phrase boundary match: raw substring matching
                # flagged e.g. "yes" inside "yesterday", which both rejects
                # clean samples and biases the surviving data.
                if kw and re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", t):
                    hits.append((path or "<root>", kw))
                    break
        return hits

    def audit(
        self,
        generated: Dict[str, Any],
        ground_truth_label: str,
        ground_truth_text: str = "",
        token_form: str = "",
        extra_keywords: Iterable[str] = (),
        all_choice_texts: Iterable[str] = (),
    ) -> LeakageAuditResult:
        """Audit a generated object for GT leakage.

        all_choice_texts: full label set of the row (when known). If ANY choice
        text is too short to audit (< min_keyword_len, e.g. "No" in a Yes/No
        task), choice-TEXT auditing is skipped for the whole row — otherwise
        only the long labels get filtered, and "the label never mentioned" in
        the surviving data becomes a negative-space signal for the answer.
        The canonical label and ANSWER_<TOKEN> form are always audited.
        """
        choice_texts = [self._normalize(str(c)) for c in all_choice_texts if c]
        skip_choice_text = any(
            len(c) < self.min_keyword_len for c in choice_texts
        ) if choice_texts else False

        kws = self._build_keywords(
            ground_truth_label=ground_truth_label,
            ground_truth_text=("" if skip_choice_text else ground_truth_text),
            token_form=token_form,
            extra=(() if skip_choice_text else extra_keywords),
        )
        if not kws:
            return LeakageAuditResult(leaked=False, matches=[])
        matches = self._scan(generated, kws)
        return LeakageAuditResult(leaked=bool(matches), matches=matches)