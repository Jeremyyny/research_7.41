"""Synthesize SFT data for the three subagents using a teacher LLM.

Pipeline per sample:
  1. Pick a benchmark example.
  2. Build the teacher prompt for the target agent kind.
  3. Call teacher (with disk cache).
  4. Extract first JSON object from response.
  5. Validate against pydantic schema.
  6. Run leakage audit (rejects samples that mention GT label/text).
  7. On any failure, retry up to N times with a slight temperature bump.
  8. On final failure, log and skip.

Each successful row is written as:
  {
    "example_id": int,
    "benchmark_name": str,
    "agent_kind": str,
    "teacher_provider": str,
    "teacher_model": str,
    "prompt": [<chat messages for SUBAGENT runtime>],
    "response": "<JSON string of validated schema>"
  }

The "prompt" field uses the RUNTIME system prompt, not the teacher prompt.
This is what the subagent will be SFT'd to produce.
"""
from __future__ import annotations

import json
import os
import queue
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError
from tqdm import tqdm

from ..benchmarks.base import StandardRow, question_hash as _question_hash
from ..teachers.base import TeacherClient, TeacherResponse
from ..utils.cache import TeacherCallCache
from ..utils.io import append_jsonl, write_json, write_jsonl
from ..utils.leakage import LeakageAuditor

from .prompts.extractor import build_extractor_synth_prompt
from .prompts.reasoner import build_reasoner_synth_prompt
from .prompts.verifier import build_verifier_synth_prompt
from .prompts.runtime_prompts import build_runtime_messages
from .schemas import (
    AgentKind,
    ExtractorOutput,
    ReasonerOutput,
    VerifierOutput,
    SCHEMA_REGISTRY,
)


JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.find("{")
    if s == -1:
        return None
    # Greedy outermost brace match — fine for our schemas which are flat.
    e = text.rfind("}")
    if e <= s:
        return None
    chunk = text[s : e + 1]
    try:
        obj = json.loads(chunk)
        return obj if isinstance(obj, dict) else None
    except Exception:
        # Try a regex fallback for nested-prose responses.
        m = JSON_BLOCK_RE.search(text)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def _build_teacher_prompt(
    kind: AgentKind,
    row: StandardRow,
    candidate_answer: str = "",
) -> List[Dict[str, str]]:
    if kind == AgentKind.EXTRACTOR:
        # Extractor is GT-blind by design.
        return build_extractor_synth_prompt(row.question, row.context, row.choices)
    if kind == AgentKind.REASONER:
        return build_reasoner_synth_prompt(row.question, row.context, row.choices)
    if kind == AgentKind.VERIFIER:
        return build_verifier_synth_prompt(
            row.question, row.context, row.choices, candidate_answer=candidate_answer
        )
    raise ValueError(f"Unknown kind: {kind}")


def _sample_verifier_candidate(kind: AgentKind, row: StandardRow, seed: int) -> str:
    """Half of verifier SFT samples audit a random candidate choice key.

    Deterministic per (seed, example_id). A uniformly random candidate keeps the
    audit signal unbiased w.r.t. the ground truth (candidate is correct with
    probability 1/n_choices), so audits cannot become a leakage side-channel.
    """
    if kind != AgentKind.VERIFIER or not row.choices:
        return ""
    rng = random.Random((int(seed) << 32) ^ (int(row.example_id) * 2654435761 & 0xFFFFFFFF))
    if rng.random() < 0.5:
        return rng.choice(list(row.choices.keys()))
    return ""




def _validate_schema(kind: AgentKind, obj: Dict[str, Any]):
    schema_cls = SCHEMA_REGISTRY[kind]
    return schema_cls(**obj)


def _gt_audit_keywords(row: StandardRow) -> Dict[str, str]:
    """Build keyword set for leakage audit."""
    gt_label = row.ground_truth
    gt_text = row.choices.get(row.ground_truth, "") if row.choices else ""
    # Token form e.g. "ANSWER_B"
    token_form = f"ANSWER_{gt_label.upper()}" if gt_label else ""
    return {
        "ground_truth_label": gt_label,
        "ground_truth_text": gt_text,
        "token_form": token_form,
    }


def _reasoner_choice_coverage_check(
    kind: AgentKind,
    obj: Dict[str, Any],
    row: StandardRow,
) -> Tuple[bool, str]:
    """For Reasoner output, ensure candidate_considerations covers all choices."""
    if kind != AgentKind.REASONER:
        return True, ""
    if not row.choices:
        return True, ""
    ca = obj.get("candidate_considerations", [])
    if not isinstance(ca, list):
        return False, "candidate_considerations must be a list"
    seen_keys = {str(item.get("choice_key", "")).strip() for item in ca if isinstance(item, dict)}
    expected = set(row.choices.keys())
    missing = expected - seen_keys
    if missing:
        return False, f"candidate_considerations missing keys: {sorted(missing)}"
    return True, ""


@dataclass
class SynthStats:
    requested: int = 0
    succeeded: int = 0
    json_parse_fail: int = 0
    schema_fail: int = 0
    leakage_fail: int = 0
    balance_fail: int = 0
    teacher_error: int = 0


def _agent_default_max_tokens(kind: AgentKind) -> int:
    if kind == AgentKind.EXTRACTOR:
        return 1200
    if kind == AgentKind.REASONER:
        return 2200
    if kind == AgentKind.VERIFIER:
        return 1000
    return 1500


def synthesize_subagent_data(
    rows: List[StandardRow],
    agent_kind: AgentKind,
    teacher: TeacherClient,
    out_path: str,
    cache: Optional[TeacherCallCache] = None,
    auditor: Optional[LeakageAuditor] = None,
    n_samples: int = 500,
    base_temperature: float = 0.4,
    max_retries_per_sample: int = 2,
    seed: int = 42,
    log_path: Optional[str] = None,
    max_workers: int = 8,
    symmetric_leakage: bool = False,
) -> SynthStats:
    """Synthesize SFT data for one subagent.

    Args:
        rows: Pool of benchmark rows to draw from.
        agent_kind: Which subagent we are synthesizing for.
        teacher: TeacherClient instance.
        out_path: JSONL output path (one SFT sample per line).
        cache: Optional teacher-call disk cache.
        auditor: Optional leakage auditor (recommended).
        All subagent teacher prompts are GT-blind. Ground truth is used only
        for leakage auditing and downstream evaluation.
        n_samples: Target number of accepted samples.
        base_temperature: Starting temperature; bumped on retry.
        max_retries_per_sample: Number of retries before giving up on a row.
        seed: Reproducibility seed for row sampling.
        log_path: Optional JSONL path for per-attempt logs.
        symmetric_leakage: If True, audit against ALL choice texts, not only the
            ground-truth text. This removes the "negative space" bias where the
            one choice never restated verbatim is exactly the answer.
    """
    if auditor is None:
        auditor = LeakageAuditor()

    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)

    stats = SynthStats(requested=n_samples)
    _lock = threading.Lock()
    _succeeded_count = [0]  # mutable int for thread-safe check
    progress = tqdm(total=n_samples, desc=f"synth/{agent_kind.value}", ncols=100)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8"):
        pass
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w", encoding="utf-8"):
            pass

    def _process_one(row: StandardRow) -> Optional[Dict[str, Any]]:
        """Try up to max_retries_per_sample+1 times. Return sft_row dict or None."""
        candidate = _sample_verifier_candidate(agent_kind, row, seed)
        for attempt in range(max_retries_per_sample + 1):
            # Stop early if we already have enough successes
            with _lock:
                if _succeeded_count[0] >= n_samples:
                    return None

            temperature = min(0.95, base_temperature + 0.15 * attempt)
            messages = _build_teacher_prompt(agent_kind, row, candidate_answer=candidate)

            cache_key = None
            cached_resp: Optional[Dict[str, Any]] = None
            if cache is not None:
                cache_key = TeacherCallCache.make_key(
                    teacher.provider, teacher.model, messages, temperature,
                    max_tokens=_agent_default_max_tokens(agent_kind),
                )
                cached_resp = cache.get(cache_key)

            if cached_resp is not None:
                text = cached_resp.get("text", "")
            else:
                try:
                    resp: TeacherResponse = teacher.chat(
                        messages,
                        temperature=temperature,
                        max_tokens=_agent_default_max_tokens(agent_kind),
                    )
                    text = resp.text
                    if cache is not None and cache_key:
                        cache.put(cache_key, {"text": text, "raw": resp.raw})
                except Exception as e:
                    with _lock:
                        stats.teacher_error += 1
                    if log_path:
                        with _lock:
                            append_jsonl(log_path, [{
                                "ts": int(time.time()),
                                "example_id": row.example_id,
                                "agent_kind": agent_kind.value,
                                "attempt": attempt,
                                "error": f"teacher_error: {e}",
                            }])
                    continue

            obj = _extract_first_json(text)
            if obj is None:
                with _lock:
                    stats.json_parse_fail += 1
                if log_path:
                    with _lock:
                        append_jsonl(log_path, [{
                            "ts": int(time.time()),
                            "example_id": row.example_id,
                            "agent_kind": agent_kind.value,
                            "attempt": attempt,
                            "error": "json_parse_fail",
                            "text_preview": text[:400],
                        }])
                continue

            try:
                validated = _validate_schema(agent_kind, obj)
            except ValidationError as e:
                with _lock:
                    stats.schema_fail += 1
                if log_path:
                    with _lock:
                        append_jsonl(log_path, [{
                            "ts": int(time.time()),
                            "example_id": row.example_id,
                            "agent_kind": agent_kind.value,
                            "attempt": attempt,
                            "error": "schema_fail",
                        }])
                continue

            ok_balance, balance_msg = _reasoner_choice_coverage_check(agent_kind, obj, row)
            if not ok_balance:
                with _lock:
                    stats.balance_fail += 1
                if log_path:
                    with _lock:
                        append_jsonl(log_path, [{
                            "ts": int(time.time()),
                            "example_id": row.example_id,
                            "agent_kind": agent_kind.value,
                            "attempt": attempt,
                            "error": f"balance_fail: {balance_msg}",
                        }])
                continue

            kw = _gt_audit_keywords(row)
            extra_kws: List[str] = []
            if symmetric_leakage and row.choices:
                extra_kws = [
                    v for k, v in row.choices.items()
                    if k != row.ground_truth and v
                ]
            audit = auditor.audit(
                generated=obj,
                ground_truth_label=kw["ground_truth_label"],
                ground_truth_text=kw["ground_truth_text"],
                token_form=kw["token_form"],
                extra_keywords=extra_kws,
                all_choice_texts=list(row.choices.values()),
            )
            if audit.leaked:
                with _lock:
                    stats.leakage_fail += 1
                if log_path:
                    with _lock:
                        append_jsonl(log_path, [{
                            "ts": int(time.time()),
                            "example_id": row.example_id,
                            "agent_kind": agent_kind.value,
                            "attempt": attempt,
                            "error": f"leakage_fail: {audit.matches[:3]}",
                        }])
                continue

            # Success
            success_obj = validated.model_dump()
            runtime_prompt = build_runtime_messages(
                agent_kind=agent_kind.value,
                question=row.question,
                context=row.context,
                choices=row.choices,
                candidate_answer=candidate,
            )
            return {
                "example_id": int(row.example_id),
                "question_hash": _question_hash(row.question),
                "benchmark_name": row.benchmark_name,
                "agent_kind": agent_kind.value,
                "teacher_provider": teacher.provider,
                "teacher_model": teacher.model,
                "candidate_answer": candidate,
                "prompt": runtime_prompt,
                "response": json.dumps(success_obj, ensure_ascii=False),
            }

        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_one, row): row for row in pool}
        for future in as_completed(futures):
            with _lock:
                if _succeeded_count[0] >= n_samples:
                    future.cancel()
                    continue
            sft_row = future.result()
            if sft_row is not None:
                with _lock:
                    if _succeeded_count[0] < n_samples:
                        append_jsonl(out_path, [sft_row])
                        stats.succeeded += 1
                        _succeeded_count[0] += 1
                        progress.update(1)

    progress.close()

    # Sidecar metadata
    meta_path = out_path + ".meta.json"
    write_json(meta_path, {
        "agent_kind": agent_kind.value,
        "teacher_provider": teacher.provider,
        "teacher_model": teacher.model,
        "n_requested": n_samples,
        "n_pool": len(pool),
        "n_accepted": stats.succeeded,
        "stats": asdict(stats),
        "gt_visible_to_teacher": False,
    })

    return stats
