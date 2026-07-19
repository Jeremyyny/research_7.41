"""MedQA loader.

Supports two modes:
  1. HuggingFace: bigbio/med_qa or GBaker/MedQA-USMLE-4-options (default).
  2. Local JSONL: pass a directory containing train.jsonl / dev.jsonl / test.jsonl,
     or a single .jsonl/.json file.

Returns a list of StandardRow.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .base import StandardRow, normalize_choices, resolve_mcq_answer
from ..utils.io import read_json, read_jsonl


HF_DEFAULT_DATASET = "GBaker/MedQA-USMLE-4-options"


def _iter_records(raw: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(raw, list):
        for ex in raw:
            if isinstance(ex, dict):
                yield ex
    elif isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, dict):
                yield v


def _from_record(rec: Dict[str, Any], idx: int, default_split: str) -> Optional[StandardRow]:
    question = str(rec.get("question") or rec.get("prompt") or "").strip()
    choices_raw = rec.get("options") or rec.get("choices") or rec.get("candidates")
    choices = normalize_choices(choices_raw)
    if not question or len(choices) < 2:
        return None

    # First present (non-None) field wins. An `or` chain would skip falsy but
    # valid values like integer 0 (answer_idx of the first choice).
    answer_raw = next(
        (rec[k] for k in ("answer_idx", "answer", "label", "answer_label", "ground_truth")
         if rec.get(k) is not None and str(rec[k]).strip() != ""),
        None,
    )
    gt = resolve_mcq_answer(answer_raw, choices)
    if not gt or gt not in choices:
        return None

    split = str(rec.get("split") or rec.get("_source_split") or default_split or "").lower().strip()
    if split == "validation":
        split = "dev"

    metadata = {}
    for k in ("meta_info", "subject_name", "topic_name", "exam"):
        if k in rec and rec[k] is not None:
            metadata[k] = rec[k]

    return StandardRow(
        example_id=idx,
        benchmark_name="medqa",
        task_subtype=str(rec.get("subset") or "us_4options"),
        question=question,
        choices=choices,
        ground_truth=gt,
        context="",  # MedQA is closed-book
        metadata=metadata,
        split=split,
    )


def _load_from_local(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() == ".jsonl":
            rows = read_jsonl(str(p))
        else:
            rows = list(_iter_records(read_json(str(p))))
        return [{"_source_split": "", **r} for r in rows]

    if not p.is_dir():
        raise FileNotFoundError(f"MedQA local path not found: {path}")

    file_specs = []
    seen_files = set()
    for split_name in ("train", "dev", "validation", "test"):
        candidates = []
        for ext in (".jsonl", ".json"):
            exact = p / f"{split_name}{ext}"
            if exact.exists():
                candidates.append(exact)

        # Some MedQA releases name split files with a prefix, e.g.
        # phrases_no_exclude_train.jsonl under US/4_options.
        if not candidates:
            for ext in (".jsonl", ".json"):
                candidates.extend(sorted(p.glob(f"*_{split_name}{ext}")))

        norm = "dev" if split_name == "validation" else split_name
        for fp in candidates:
            resolved = fp.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            file_specs.append((norm, fp))

    if not file_specs:
        raise FileNotFoundError(f"No MedQA files under {path}")

    rows: List[Dict[str, Any]] = []
    for split_name, fp in file_specs:
        if fp.suffix.lower() == ".jsonl":
            data = read_jsonl(str(fp))
        else:
            data = list(_iter_records(read_json(str(fp))))
        for r in data:
            rows.append({"_source_split": split_name, **r})
    return rows


def _load_from_hf(dataset_name: str, cache_dir: Optional[str]) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, cache_dir=cache_dir)
    rows: List[Dict[str, Any]] = []
    for split_name in ds.keys():
        norm = "dev" if split_name == "validation" else split_name
        for rec in ds[split_name]:
            rows.append({"_source_split": norm, **dict(rec)})
    return rows


def load_medqa(
    source: str = "hf",
    hf_dataset: str = HF_DEFAULT_DATASET,
    local_path: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    max_examples: int = 0,
) -> List[StandardRow]:
    """Load MedQA into a list of StandardRow.

    Args:
        source: "hf" or "local".
        hf_dataset: HF dataset name (used when source == "hf").
        local_path: directory or file path (used when source == "local").
        hf_cache_dir: optional HF cache directory.
        max_examples: cap output size; 0 means no cap.
    """
    if source == "hf":
        raw_rows = _load_from_hf(hf_dataset, hf_cache_dir)
    elif source == "local":
        if not local_path:
            raise ValueError("source='local' requires local_path")
        raw_rows = _load_from_local(local_path)
    else:
        raise ValueError(f"Unknown source: {source}")

    rows: List[StandardRow] = []
    for i, rec in enumerate(raw_rows):
        sr = _from_record(rec, i, default_split=str(rec.get("_source_split", "")))
        if sr is not None:
            rows.append(sr)

    # Reassign contiguous example_ids after filtering
    for new_id, r in enumerate(rows):
        r.example_id = new_id

    if max_examples > 0 and len(rows) > max_examples:
        rows = rows[:max_examples]

    return rows
