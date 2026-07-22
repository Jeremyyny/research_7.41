"""Evolve loop: turn manager failures into new SFT data, then SFT-train manager.

Three steps (called separately or together via pipeline.stages):
  1. build_manager_sft_from_failures: read fail_buffer.jsonl, run subagents
     and an optional teacher to construct multi-turn SFT trajectories.
  2. train_manager_sft: do per-turn SFT on the constructed jsonl.
  3. (back to GRPO with the SFT'd model as init — handled at pipeline level)

The teacher's job here is to PICK A TOOL SEQUENCE (0-3 tools) for each failed
example. The teacher does NOT generate the final answer text; we use the
ground truth to construct the final ANSWER_<TOKEN> line.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from ..benchmarks.base import StandardRow, question_hash as _question_hash
from ..subagents.runtime import FrozenSubagent, SubagentPool
from ..teachers.base import TeacherClient
from ..utils.io import read_jsonl, write_jsonl, write_json
from ..utils.seed import set_seed

try:
    from peft import LoraConfig, PeftModel, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

from .prompt import build_manager_system_prompt, build_manager_user_message


_ALLOWED_TOOLS = ("extractor_tool", "reasoner_tool", "verifier_tool")
_TOOL_NAME_TO_KIND = {
    "extractor_tool": "extractor",
    "reasoner_tool": "reasoner",
    "verifier_tool": "verifier",
}


def _label_to_token(label: str) -> str:
    s = str(label).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.upper()


def _final_answer_str(gt: str) -> str:
    return f"ANSWER_{_label_to_token(gt)}"


def _teacher_choose_tool_sequence(
    teacher: Optional[TeacherClient],
    question: str,
    context: str,
    choices: Dict[str, str],
    available_kinds: List[str],
    fallback_seq: Optional[List[str]] = None,
) -> List[str]:
    """Ask the teacher which tool sequence (length 0-3) would best help solve this.

    The teacher does NOT see the GT here — we want it to recommend a sequence
    that a confused-on-this-example manager should follow.

    Returns: list of tool names from _ALLOWED_TOOLS, deduplicated, length<=3.
    """
    available_tools = [k + "_tool" for k in available_kinds if (k + "_tool") in _ALLOWED_TOOLS]

    if teacher is None:
        if fallback_seq is not None:
            return [t for t in fallback_seq if t in available_tools][:3]
        # Heuristic: long context -> extractor first; MCQ -> reasoner; default reasoner.
        seq: List[str] = []
        if context and len(context) > 800 and "extractor_tool" in available_tools:
            seq.append("extractor_tool")
        if "reasoner_tool" in available_tools:
            seq.append("reasoner_tool")
        return seq[:3]

    sys_msg = (
        "You design tool-use plans for a manager agent that must solve multiple-choice questions.\n"
        f"Available tools: {available_tools}.\n"
        "Choose a sequence of 0 to 3 tools (no repeats) to create DIVERSE, HIGH-QUALITY training data.\n"
        "Guidelines:\n"
        "  - extractor_tool: use when the question has dense context, complex wording, or requires isolating key facts.\n"
        "  - reasoner_tool: use when multi-step inference or per-choice analysis is needed.\n"
        "  - verifier_tool: use when domain principles matter or reasoning errors are likely.\n"
        "  - 0 tools: only for trivially obvious questions where a manager could answer confidently without any help.\n"
        "  - 3 tools: for hard questions involving specialized knowledge, multi-step reasoning, AND verification risk.\n"
        "Target mix across many examples: ~10% k=0, ~25% k=1, ~45% k=2, ~20% k=3.\n"
        "Return ONLY JSON: {\"tool_sequence\": [\"tool_a\", \"tool_b\"]}"
    )
    choices_block = ""
    if choices:
        lines = [f"  {k}. {v}" for k, v in choices.items()]
        choices_block = "CHOICES:\n" + "\n".join(lines) + "\n\n"
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"{choices_block}"
        f"CONTEXT:\n{context if context else '(no context)'}\n"
    )
    try:
        resp = teacher.chat(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            temperature=0.2, max_tokens=200,
        )
        text = resp.text
        s = text.find("{")
        e = text.rfind("}")
        if s == -1 or e <= s:
            raise ValueError("no JSON in teacher response")
        obj = json.loads(text[s:e + 1])
        seq = obj.get("tool_sequence", [])
        if not isinstance(seq, list):
            raise ValueError("tool_sequence not a list")
        out: List[str] = []
        for item in seq:
            t = str(item).strip()
            if t in available_tools and t not in out:
                out.append(t)
            if len(out) >= 3:
                break
        return out
    except Exception:
        return fallback_seq or (
            ["extractor_tool", "reasoner_tool"]
            if context and len(context) > 800
            else ["reasoner_tool"]
        )


def _tool_call_message(
    tool_name: str,
    eid: int,
    call_id: str,
    binding_mode: str,
    content: str = "",
    extra_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    args: Dict[str, Any] = {"example_id": int(eid)} if binding_mode == "argument" else {}
    if extra_args:
        args.update(extra_args)
    msg: Dict[str, Any] = {
        "role": "assistant",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": tool_name, "arguments": json.dumps(args, ensure_ascii=False)},
        }],
    }
    if content:
        msg["content"] = content
    return msg


def _draft_answer_str(gt: str) -> str:
    return f"DRAFT_ANSWER_{_label_to_token(gt)}"


@dataclass
class EvolveSFTConfig:
    base_model: str
    extractor_adapter: Optional[str]
    reasoner_adapter: Optional[str]
    verifier_adapter: Optional[str]
    rows: List[StandardRow]
    fail_buffer_jsonl: str
    out_dir: str
    teacher: Optional[TeacherClient] = None
    seed: int = 42
    max_fail_samples: int = 1500
    binding_mode: str = "environment"
    task_description: str = ""


def _register_available_subagents(
    base_model: str,
    extractor_adapter: Optional[str],
    reasoner_adapter: Optional[str],
    verifier_adapter: Optional[str],
    device: str,
) -> tuple[SubagentPool, List[str]]:
    pool = SubagentPool()
    available_kinds: List[str] = []
    if extractor_adapter:
        pool.register(FrozenSubagent(base_model, extractor_adapter, "extractor", device))
        available_kinds.append("extractor")
    if reasoner_adapter:
        pool.register(FrozenSubagent(base_model, reasoner_adapter, "reasoner", device))
        available_kinds.append("reasoner")
    if verifier_adapter:
        pool.register(FrozenSubagent(base_model, verifier_adapter, "verifier", device))
        available_kinds.append("verifier")
    return pool, available_kinds


def _coldstart_fallback_sequence(idx: int, context: str, available_kinds: List[str]) -> List[str]:
    available_tools = {k + "_tool" for k in available_kinds}
    if context and len(context) > 800 and "extractor_tool" in available_tools:
        seq = ["extractor_tool", "reasoner_tool"]
    elif idx % 5 == 0 and "extractor_tool" in available_tools:
        seq = ["extractor_tool", "reasoner_tool"]
    elif idx % 5 == 1 and "verifier_tool" in available_tools:
        seq = ["verifier_tool", "reasoner_tool"]
    else:
        seq = ["reasoner_tool"]
    return [t for t in seq if t in available_tools][:3]


def _build_manager_tool_sft_rows(
    rows: List[StandardRow],
    pool: SubagentPool,
    available_kinds: List[str],
    teacher: Optional[TeacherClient],
    binding_mode: str,
    task_description: str,
    cache_namespace: str,
    initial_draft_by_eid: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """Build per-turn manager SFT trajectories.

    initial_draft_by_eid: optional map example_id -> choice key used as the
    manager's stated draft on tool-calling turns (e.g. the manager's actual
    failed prediction from the GRPO fail buffer). The final turn always states
    the corrected GT draft + answer, so the trace teaches a W→C revision.
    When absent, drafts default to the GT (plain teacher forcing).
    The verifier is asked to audit the draft actually stated at that turn —
    never the raw ground truth — so verifier tool outputs in training traces
    match what the frozen verifier will see at inference time.
    """
    try:
        from tqdm import tqdm
        _iter = tqdm(rows, desc=f"[{cache_namespace}] building SFT rows", unit="ex")
    except ImportError:
        _iter = rows

    sft_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for idx, row in enumerate(_iter):
        eid = int(row.example_id)
        sys_prompt = build_manager_system_prompt(
            label_keys=list(row.choices.keys()),
            task_description=task_description,
        )
        user_msg = build_manager_user_message(
            example_id=eid,
            question=row.question,
            context=row.context,
            choices=row.choices,
            binding_mode=binding_mode,
        )
        base_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]

        fallback_seq = _coldstart_fallback_sequence(idx, row.context, available_kinds)
        seq = _teacher_choose_tool_sequence(
            teacher=teacher,
            question=row.question,
            context=row.context,
            choices=row.choices,
            available_kinds=available_kinds,
            fallback_seq=fallback_seq,
        )

        # Draft stated on tool-calling turns: the manager's actual (possibly
        # wrong) prediction when known, else the GT. The final turn always
        # corrects to GT.
        initial_draft = (initial_draft_by_eid or {}).get(eid, "")
        if initial_draft not in row.choices:
            initial_draft = row.ground_truth

        tool_outputs: Dict[str, str] = {}
        for tname in seq:
            kind = _TOOL_NAME_TO_KIND[tname]
            if not pool.has(kind):
                continue
            tool_outputs[tname] = pool.call(
                agent_kind=kind,
                example_id=eid,
                question=row.question,
                context=row.context,
                choices=row.choices,
                cache_namespace=cache_namespace,
                candidate_answer=(initial_draft if kind == "verifier" else ""),
            )

        final_text = _final_answer_str(row.ground_truth)
        draft_text = _draft_answer_str(row.ground_truth)
        turn_draft_text = _draft_answer_str(initial_draft)
        qhash = _question_hash(row.question)
        if not seq:
            sft_rows.append({
                "example_id": eid,
                "question_hash": qhash,
                "prompt": base_messages,
                "response": [{"role": "assistant", "content": f"{draft_text}\n{final_text}"}],
            })
            continue

        history = list(base_messages)
        for i, tname in enumerate(seq):
            call_id = f"call_{eid}_{i+1}"
            # ADC policy: every tool-calling turn states the current draft answer;
            # verifier calls pass that same draft so it audits that hypothesis.
            extra_args = {"current_draft": initial_draft} if tname == "verifier_tool" else None
            asst_call = _tool_call_message(
                tname, eid, call_id, binding_mode,
                content=turn_draft_text, extra_args=extra_args,
            )
            sft_rows.append({
                "example_id": eid,
                "question_hash": qhash,
                "prompt": list(history),
                "response": [asst_call],
            })
            tool_msg = {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tname,
                "content": tool_outputs.get(tname, '{"error":"tool_not_available"}'),
            }
            history = history + [asst_call, tool_msg]

        sft_rows.append({
            "example_id": eid,
            "question_hash": qhash,
            "prompt": list(history),
            "response": [{"role": "assistant", "content": f"{draft_text}\n{final_text}"}],
        })
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{cache_namespace}] {idx+1}/{len(rows)} examples | {len(sft_rows)} SFT turns | {elapsed:.0f}s elapsed")
    return sft_rows


def build_manager_sft_from_failures(cfg: EvolveSFTConfig) -> str:
    """Read fail buffer, build per-turn SFT trajectories, write to disk.

    Output is a JSONL where each row is a per-turn (prompt, response) pair:
      - turn 1: user message -> first tool_call (or final answer if seq is empty)
      - turn 2: turn1 + tool output -> second tool_call (or final answer)
      - turn 3+: ... up to 3 tools, then final answer turn
    """
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pool, available_kinds = _register_available_subagents(
        cfg.base_model,
        cfg.extractor_adapter,
        cfg.reasoner_adapter,
        cfg.verifier_adapter,
        device,
    )

    row_index = {int(r.example_id): r for r in cfg.rows}
    # example_id values silently change whenever a normalized cache is rebuilt
    # (see benchmarks/base.py). Prefer question_hash matching when the fail
    # buffer carries it; fall back to example_id for older buffers.
    row_by_hash = {_question_hash(r.question): r for r in cfg.rows}

    # Read failures, dedupe per example, cap. Also collect the manager's
    # failed prediction so the SFT trace can state it as the initial draft.
    fails: List[int] = []
    seen = set()
    initial_draft_by_eid: Dict[int, str] = {}
    n_id_fallback = 0
    if not os.path.exists(cfg.fail_buffer_jsonl):
        raise FileNotFoundError(f"fail_buffer not found: {cfg.fail_buffer_jsonl}")
    for row in read_jsonl(cfg.fail_buffer_jsonl):
        matched: Optional[StandardRow] = None
        qh = row.get("question_hash")
        if qh:
            matched = row_by_hash.get(str(qh))
        if matched is None:
            try:
                matched = row_index.get(int(row.get("example_id")))
                if matched is not None and qh:
                    # hash present but unknown -> row not in cfg.rows; skip
                    matched = None
                elif matched is not None:
                    n_id_fallback += 1
            except Exception:
                matched = None
        if matched is None:
            continue
        eid = int(matched.example_id)
        if eid in seen:
            continue
        seen.add(eid)
        fails.append(eid)
        pred = row.get("pred")
        if isinstance(pred, str) and pred in matched.choices:
            initial_draft_by_eid[eid] = pred
        if len(fails) >= cfg.max_fail_samples:
            break

    print(
        f"[EVOLVE] {len(fails)} unique failed examples selected from buffer "
        f"(example_id fallback matches: {n_id_fallback})."
    )

    selected_rows = [row_index[eid] for eid in fails]
    sft_rows = _build_manager_tool_sft_rows(
        rows=selected_rows,
        pool=pool,
        available_kinds=available_kinds,
        teacher=cfg.teacher,
        binding_mode=cfg.binding_mode,
        task_description=cfg.task_description,
        cache_namespace="evolve",
        initial_draft_by_eid=initial_draft_by_eid,
    )

    out_path = os.path.join(cfg.out_dir, "manager_sft_from_failures.jsonl")
    write_jsonl(out_path, sft_rows)
    write_json(os.path.join(cfg.out_dir, "evolve_run_config.json"), {
        "n_failed_examples": len(fails),
        "n_sft_rows": len(sft_rows),
        "available_kinds": available_kinds,
        "binding_mode": cfg.binding_mode,
        "teacher_provider": cfg.teacher.provider if cfg.teacher else "heuristic",
        "teacher_model": cfg.teacher.model if cfg.teacher else "",
    })
    print(f"[EVOLVE] wrote {len(sft_rows)} SFT rows -> {out_path}")
    return out_path


@dataclass
class ColdStartSFTConfig:
    base_model: str
    extractor_adapter: Optional[str]
    reasoner_adapter: Optional[str]
    verifier_adapter: Optional[str]
    rows: List[StandardRow]
    out_dir: str
    teacher: Optional[TeacherClient] = None
    seed: int = 42
    n_samples: int = 300
    binding_mode: str = "environment"
    task_description: str = ""


def build_manager_sft_from_rows(cfg: ColdStartSFTConfig) -> str:
    """Build manager tool-call SFT rows from ordinary training examples."""
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool, available_kinds = _register_available_subagents(
        cfg.base_model,
        cfg.extractor_adapter,
        cfg.reasoner_adapter,
        cfg.verifier_adapter,
        device,
    )
    if not available_kinds:
        raise ValueError("No subagent adapters available for cold-start SFT.")

    sample = list(cfg.rows)
    random.Random(cfg.seed).shuffle(sample)
    if cfg.n_samples > 0:
        sample = sample[:cfg.n_samples]

    print(f"[COLDSTART] building SFT data for {len(sample)} examples | subagents={available_kinds}")
    sft_rows = _build_manager_tool_sft_rows(
        rows=sample,
        pool=pool,
        available_kinds=available_kinds,
        teacher=cfg.teacher,
        binding_mode=cfg.binding_mode,
        task_description=cfg.task_description,
        cache_namespace="coldstart",
    )

    out_path = os.path.join(cfg.out_dir, "manager_sft_coldstart.jsonl")
    write_jsonl(out_path, sft_rows)
    write_json(os.path.join(cfg.out_dir, "coldstart_run_config.json"), {
        "n_examples": len(sample),
        "n_sft_rows": len(sft_rows),
        "available_kinds": available_kinds,
        "binding_mode": cfg.binding_mode,
        "teacher_provider": cfg.teacher.provider if cfg.teacher else "heuristic",
        "teacher_model": cfg.teacher.model if cfg.teacher else "",
    })
    print(f"[COLDSTART] wrote {len(sft_rows)} SFT rows from {len(sample)} examples -> {out_path}")
    return out_path


def make_diverse_sequences(
    rows: List[StandardRow],
    available_kinds: List[str],
    seed: int = 42,
) -> Dict[int, List[str]]:
    """Assign a force-balanced tool sequence to each row without calling any model.

    Deterministic given seed. Sequences use only available agent kinds.
    Distribution when all 3 agents present:
      k=0 (direct answer):           10%
      k=1 reasoner only:             25%
      k=1 extractor only:             5%
      k=2 extractor→reasoner:        25%
      k=2 reasoner→verifier:         20%
      k=3 extractor→reasoner→verifier: 15%
    """
    has_ext = "extractor" in available_kinds
    has_rsn = "reasoner" in available_kinds
    has_vrf = "verifier" in available_kinds

    # (weight, sequence) — only include sequences whose agents are available
    _slots = [
        (10, []),
        (25, ["reasoner_tool"] if has_rsn else []),
        (5,  ["extractor_tool"] if has_ext else []),
        (25, ["extractor_tool", "reasoner_tool"] if (has_ext and has_rsn) else (["reasoner_tool"] if has_rsn else [])),
        (20, ["reasoner_tool", "verifier_tool"] if (has_rsn and has_vrf) else (["reasoner_tool"] if has_rsn else [])),
        (15, ["extractor_tool", "reasoner_tool", "verifier_tool"] if (has_ext and has_rsn and has_vrf) else (["reasoner_tool"] if has_rsn else [])),
    ]

    # Build a flat template list proportional to weights
    template: List[List[str]] = []
    for weight, seq in _slots:
        template.extend([seq] * weight)  # total = 100 slots

    rng = random.Random(seed)
    shuffled_rows = list(rows)
    rng.shuffle(shuffled_rows)

    sequences: Dict[int, List[str]] = {}
    for i, row in enumerate(shuffled_rows):
        sequences[int(row.example_id)] = list(template[i % len(template)])

    return sequences


def build_manager_sft_from_sequences(
    cfg: ColdStartSFTConfig,
    sequences: Dict[int, List[str]],
    out_path: Optional[str] = None,
) -> str:
    """Build manager SFT trajectories using pre-computed tool sequences.

    Intended for the offline batch workflow:
      1. export_manager_coldstart_prompts -> prompts.jsonl
      2. generate_openai_jsonl.py (or any batch API) -> responses.jsonl
      3. import_manager_coldstart_responses calls this with parsed sequences

    Subagents are still run locally to produce tool outputs; only the tool
    *selection* decision comes from the pre-computed sequences.
    """
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool, available_kinds = _register_available_subagents(
        cfg.base_model,
        cfg.extractor_adapter,
        cfg.reasoner_adapter,
        cfg.verifier_adapter,
        device,
    )
    available_tools = {k + "_tool" for k in available_kinds}

    selected = [r for r in cfg.rows if int(r.example_id) in sequences]
    print(f"[COLDSTART_IMPORT] {len(selected)} examples | subagents={available_kinds}")

    try:
        from tqdm import tqdm
        _iter = tqdm(selected, desc="[coldstart_import] building SFT rows", unit="ex")
    except ImportError:
        _iter = selected

    sft_rows: List[Dict[str, Any]] = []
    for row in _iter:
        eid = int(row.example_id)
        seq = [t for t in sequences.get(eid, []) if t in available_tools][:3]

        sys_prompt = build_manager_system_prompt(
            label_keys=list(row.choices.keys()),
            task_description=cfg.task_description,
        )
        user_msg = build_manager_user_message(
            example_id=eid,
            question=row.question,
            context=row.context,
            choices=row.choices,
            binding_mode=cfg.binding_mode,
        )
        base_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]

        tool_outputs: Dict[str, str] = {}
        for tname in seq:
            kind = _TOOL_NAME_TO_KIND[tname]
            if not pool.has(kind):
                continue
            tool_outputs[tname] = pool.call(
                agent_kind=kind,
                example_id=eid,
                question=row.question,
                context=row.context,
                choices=row.choices,
                cache_namespace="coldstart_import",
                candidate_answer=(row.ground_truth if kind == "verifier" else ""),
            )

        final_text = _final_answer_str(row.ground_truth)
        draft_text = _draft_answer_str(row.ground_truth)
        qhash = _question_hash(row.question)
        if not seq:
            sft_rows.append({
                "example_id": eid,
                "question_hash": qhash,
                "prompt": base_messages,
                "response": [{"role": "assistant", "content": f"{draft_text}\n{final_text}"}],
            })
            continue

        history = list(base_messages)
        for i, tname in enumerate(seq):
            call_id = f"call_{eid}_{i+1}"
            extra_args = {"current_draft": row.ground_truth} if tname == "verifier_tool" else None
            asst_call = _tool_call_message(
                tname, eid, call_id, cfg.binding_mode,
                content=draft_text, extra_args=extra_args,
            )
            sft_rows.append({
                "example_id": eid,
                "question_hash": qhash,
                "prompt": list(history),
                "response": [asst_call],
            })
            tool_msg = {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tname,
                "content": tool_outputs.get(tname, '{"error":"tool_not_available"}'),
            }
            history = history + [asst_call, tool_msg]

        sft_rows.append({
            "example_id": eid,
            "question_hash": qhash,
            "prompt": list(history),
            "response": [{"role": "assistant", "content": f"{draft_text}\n{final_text}"}],
        })

    if out_path is None:
        out_path = os.path.join(cfg.out_dir, "coldstart_from_sequences_sft.jsonl")
    write_jsonl(out_path, sft_rows)
    write_json(out_path + ".meta.json", {
        "n_examples": len(selected),
        "n_sft_rows": len(sft_rows),
        "available_kinds": available_kinds,
        "binding_mode": cfg.binding_mode,
    })
    print(f"[COLDSTART_IMPORT] {len(sft_rows)} SFT turns from {len(selected)} examples -> {out_path}")
    return out_path


# -------------- Manager SFT --------------

@dataclass
class ManagerSFTConfig:
    base_model: str
    train_jsonl: str
    out_dir: str
    init_model_or_adapter: Optional[str] = None
    seed: int = 42
    max_seq_len: int = 4096
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_steps: int = -1
    bf16: bool = True


def _render_chat(tokenizer, messages, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        )


def _mask_prefix_len(prompt_ids: List[int], full_ids: List[int]) -> int:
    """Common token prefix of the prompt-only and full renders. See
    subagents/train.py: len(prompt_ids) is wrong for templates (e.g. Qwen3 with
    enable_thinking=False) whose generation prompt is not a strict prefix of
    the full render — it would mask the first response tokens."""
    n = min(len(prompt_ids), len(full_ids))
    i = 0
    while i < n and prompt_ids[i] == full_ids[i]:
        i += 1
    return i


def _tokenize_manager_sft(rows: List[Dict[str, Any]], tok, max_seq_len: int) -> Dataset:
    eos = tok.eos_token or ""

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt_msgs = ex["prompt"]
        response_msgs = ex["response"]
        if isinstance(response_msgs, dict):
            response_msgs = [response_msgs]
        elif isinstance(response_msgs, str):
            response_msgs = [{"role": "assistant", "content": response_msgs}]

        prompt_text = _render_chat(tok, prompt_msgs, add_generation_prompt=True)
        full_text = _render_chat(tok, prompt_msgs + response_msgs, add_generation_prompt=False)
        if eos and not full_text.rstrip().endswith(eos):
            full_text = full_text + eos

        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        full = tok(full_text, add_special_tokens=False)
        input_ids = full["input_ids"][:max_seq_len]
        attention_mask = full["attention_mask"][:max_seq_len]
        plen = min(_mask_prefix_len(prompt_ids, full["input_ids"]), max_seq_len)
        labels = ([-100] * plen) + input_ids[plen:]
        labels = labels[:max_seq_len]
        if len(labels) < len(input_ids):
            labels += [-100] * (len(input_ids) - len(labels))

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    ds = Dataset.from_list(rows)
    return ds.map(_map, remove_columns=ds.column_names)


def train_manager_sft(cfg: ManagerSFTConfig) -> None:
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.use_lora and not PEFT_AVAILABLE:
        raise RuntimeError("peft is required when manager SFT is configured with LoRA.")

    init_source = cfg.init_model_or_adapter or cfg.base_model
    tok = AutoTokenizer.from_pretrained(init_source, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    dtype = torch.bfloat16 if (cfg.bf16 and device == "cuda") else torch.float32
    is_adapter_init = bool(
        cfg.init_model_or_adapter
        and os.path.isdir(cfg.init_model_or_adapter)
        and os.path.exists(os.path.join(cfg.init_model_or_adapter, "adapter_config.json"))
    )
    is_full_init = bool(
        cfg.init_model_or_adapter
        and os.path.isdir(cfg.init_model_or_adapter)
        and os.path.exists(os.path.join(cfg.init_model_or_adapter, "config.json"))
        and not is_adapter_init
    )
    if is_adapter_init:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is required to continue SFT from a manager adapter.")
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        model = PeftModel.from_pretrained(
            base, cfg.init_model_or_adapter, is_trainable=cfg.use_lora
        ).to(device)
        if not cfg.use_lora:
            model = model.merge_and_unload().to(device)
        print(f"[MANAGER_SFT] continuing from adapter -> {cfg.init_model_or_adapter}")
    elif is_full_init:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.init_model_or_adapter, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        print(f"[MANAGER_SFT] continuing from full checkpoint -> {cfg.init_model_or_adapter}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
    model.config.use_cache = False
    if not cfg.use_lora:
        for param in model.parameters():
            param.requires_grad_(True)

    if cfg.use_lora and PEFT_AVAILABLE and not is_adapter_init:
        candidate = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        present = {n.split(".")[-1] for n, _ in model.named_modules()}
        target = [m for m in candidate if m in present] or ["q_proj", "v_proj"]
        lconf = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=target,
        )
        model = get_peft_model(model, lconf)
        print(f"[MANAGER_SFT/LoRA] r={cfg.lora_r} alpha={cfg.lora_alpha} target_modules={target}")

    rows = read_jsonl(cfg.train_jsonl)
    if not rows:
        raise ValueError(f"No rows in {cfg.train_jsonl}")
    print(f"[MANAGER_SFT] tokenizing {len(rows)} rows ...")
    train_ds = _tokenize_manager_sft(rows, tok, cfg.max_seq_len)
    total_steps = (len(train_ds) // (cfg.per_device_batch_size * cfg.gradient_accumulation_steps)) * cfg.num_train_epochs
    if cfg.max_steps > 0:
        total_steps = min(total_steps, cfg.max_steps)
    print(f"[MANAGER_SFT] {len(train_ds)} train examples | ~{total_steps} steps | lr={cfg.learning_rate} | epochs={cfg.num_train_epochs}")
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, return_tensors="pt")

    args = TrainingArguments(
        output_dir=cfg.out_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=1,
        save_strategy="epoch",
        bf16=(cfg.bf16 and device == "cuda"),
        fp16=False,
        report_to=[],
        seed=cfg.seed,
        remove_unused_columns=False,
        max_steps=(cfg.max_steps if cfg.max_steps > 0 else -1),
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
    trainer.train()
    os.makedirs(cfg.out_dir, exist_ok=True)
    trainer.model.save_pretrained(cfg.out_dir)
    tok.save_pretrained(cfg.out_dir)
    print(f"[MANAGER_SFT] saved -> {cfg.out_dir}")
