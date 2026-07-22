#!/usr/bin/env python3
"""Summarize binary-reward routing traces, including collapse over time."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    with_draft = [r for r in rows if r.get("initial_draft") is not None]
    draft_wrong = [r for r in with_draft if not bool(r.get("initial_draft_correct"))]
    draft_correct = [r for r in with_draft if bool(r.get("initial_draft_correct"))]

    def call_rate(items: List[Dict[str, Any]]) -> float:
        return sum(int(r.get("tool_calls", 0)) > 0 for r in items) / max(1, len(items))

    return {
        "n": n,
        "accuracy": sum(bool(r.get("correct")) for r in rows) / max(1, n),
        "tool_call_rate": call_rate(rows),
        "avg_tool_calls": sum(int(r.get("tool_calls", 0)) for r in rows) / max(1, n),
        "initial_draft_coverage": len(with_draft) / max(1, n),
        "initial_draft_accuracy": sum(bool(r.get("initial_draft_correct")) for r in with_draft) / max(1, len(with_draft)),
        "call_rate_given_draft_wrong": call_rate(draft_wrong),
        "call_rate_given_draft_correct": call_rate(draft_correct),
        "draft_conditioned_call_gap": call_rate(draft_wrong) - call_rate(draft_correct),
        "correction_rate": sum(bool(r.get("corrected_by_tools")) for r in rows) / max(1, n),
        "corruption_rate": sum(bool(r.get("corrupted_by_tools")) for r in rows) / max(1, n),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_jsonl", type=Path)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.trace_jsonl)
    report = summarize(rows)
    if args.window > 0:
        report["windows"] = [
            {
                "start": start,
                "end": min(start + args.window, len(rows)),
                **summarize(rows[start : start + args.window]),
            }
            for start in range(0, len(rows), args.window)
        ]

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
