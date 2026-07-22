"""Counterfactual marginal-value data for manager routing.

The manager is evaluated from one shared, model-generated draft.  We then
intervene on the next routing action by forcing each still-available advisor,
ask the manager to revise its answer, and continue breadth-first until a
correct path is found or ``max_depth`` is reached.

Ground truth is used only to *select* a shortest successful trajectory.  It is
never substituted for a model prediction.  This gives the routing policy the
lexicographic supervision we actually want:

  1. prefer a correct trajectory to an incorrect trajectory;
  2. among correct trajectories, prefer the one with fewer advisor calls;
  3. when neither branch is correct, do not teach a spurious no-call action.

GRPO can subsequently keep its plain binary terminal reward.  The otherwise
unidentifiable efficiency tie-break is learned here from paired
counterfactuals rather than from a global per-call penalty.
"""
from __future__ import annotations

import itertools
import json
import os
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ..benchmarks.base import StandardRow
from .prompt import (
    build_manager_system_prompt,
    build_manager_user_message,
    parse_draft_answer,
    parse_final_answer,
)


ADVISOR_KINDS: Tuple[str, ...] = ("extractor", "reasoner", "verifier")

_DIRECT_PROBE = (
    "Training-time counterfactual probe: do not call a tool. State your current "
    "best choice as DRAFT_ANSWER_<TOKEN>, then submit the same choice on the "
    "last line as ANSWER_<TOKEN>."
)

_REVISION_PROBE = (
    "Training-time counterfactual probe: using the evidence already in the "
    "conversation, do not call another tool. State your revised best choice as "
    "DRAFT_ANSWER_<TOKEN>, then submit it on the last line as ANSWER_<TOKEN>."
)


@dataclass
class MarginalValueConfig:
    base_model: str
    manager_dir: str
    rows: List["StandardRow"]
    out_dir: str
    extractor_adapter: Optional[str]
    reasoner_adapter: Optional[str]
    verifier_adapter: Optional[str]
    seed: int = 42
    n_samples: int = 300
    max_depth: int = 1
    max_new_tokens: int = 512
    temperature: float = 0.0
    binding_mode: str = "environment"
    task_description: str = ""
    max_commit_rescue_ratio: float = 1.0
    subagent_server_url: Optional[str] = None


@dataclass
class BranchState:
    sequence: Tuple[str, ...]
    messages: List[Dict[str, Any]]
    drafts: List[str]
    final_pred: Optional[str]
    valid: bool
    correct: bool
    trajectory: List[Dict[str, Any]]


def _answer_token(label: str) -> str:
    token = str(label).strip().upper()
    return "".join(ch if ch.isalnum() else "_" for ch in token).strip("_")


def _draft_and_final(label: str) -> str:
    token = _answer_token(label)
    return f"DRAFT_ANSWER_{token}\nANSWER_{token}"


def _draft_only(label: str) -> str:
    return f"DRAFT_ANSWER_{_answer_token(label)}"


def _tool_schemas(binding_mode: str) -> List[Dict[str, Any]]:
    required = ["example_id"] if binding_mode == "argument" else []
    properties: Dict[str, Any] = {}
    if binding_mode == "argument":
        properties["example_id"] = {
            "type": "integer",
            "description": "The current example ID.",
        }
    verifier_properties = dict(properties)
    verifier_properties["current_draft"] = {
        "type": "string",
        "description": "The current draft answer key.",
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "extractor_tool",
                "description": "Extract decision-relevant factual signals.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reasoner_tool",
                "description": "Produce a structured reasoning scaffold.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "verifier_tool",
                "description": "Audit the current draft for errors.",
                "parameters": {
                    "type": "object",
                    "properties": verifier_properties,
                    "required": required,
                },
            },
        },
    ]


def _tool_call_message(
    tool_kind: str,
    example_id: int,
    current_draft: str,
    call_id: str,
    binding_mode: str,
) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    if binding_mode == "argument":
        args["example_id"] = int(example_id)
    if tool_kind == "verifier":
        args["current_draft"] = current_draft
    return {
        "role": "assistant",
        "content": _draft_only(current_draft),
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": f"{tool_kind}_tool",
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }],
    }


def _render_chat(tokenizer: Any, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> str:
    kwargs = dict(
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _load_manager(cfg: MarginalValueConfig, device: str, dtype: Any):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = cfg.manager_dir or cfg.base_model
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    is_local_adapter = (
        os.path.isdir(source)
        and os.path.exists(os.path.join(source, "adapter_config.json"))
    )
    if is_local_adapter:
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        model = PeftModel.from_pretrained(base, source).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            source, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
    model.eval()
    return tokenizer, model


def _build_pool(cfg: MarginalValueConfig, device: str):
    from ..subagents.runtime import FrozenSubagent, RemoteSubagentPool, SubagentPool

    if cfg.subagent_server_url:
        return RemoteSubagentPool(
            server_url=cfg.subagent_server_url,
            registered_kinds=list(ADVISOR_KINDS),
        )

    pool = SubagentPool()
    adapters = {
        "extractor": cfg.extractor_adapter,
        "reasoner": cfg.reasoner_adapter,
        "verifier": cfg.verifier_adapter,
    }
    for kind, adapter in adapters.items():
        if adapter and os.path.exists(adapter):
            pool.register(FrozenSubagent(cfg.base_model, adapter, kind, device))
    if not any(pool.has(kind) for kind in ADVISOR_KINDS):
        raise FileNotFoundError("No subagent adapters are available for marginal-value collection.")
    return pool


def _generate_answer(
    tokenizer: Any,
    model: Any,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    choice_keys: List[str],
    probe: str,
    max_new_tokens: int,
    temperature: float,
    device: str,
) -> Tuple[Optional[str], str, bool]:
    import torch

    probe_messages = list(messages) + [{"role": "user", "content": probe}]
    prompt = _render_chat(tokenizer, probe_messages, tools)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    do_sample = temperature > 1e-6
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **({"temperature": max(temperature, 1e-6)} if do_sample else {}),
        )
    text = tokenizer.decode(
        generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    final = parse_final_answer(text, choice_keys)
    draft = parse_draft_answer(text, choice_keys)
    pred = final if final is not None else draft
    return pred, text, pred is not None


def choose_preferred_sequence(
    direct_correct: bool,
    branches: Sequence[Dict[str, Any]],
    tie_break_seed: Optional[int] = None,
) -> Optional[Tuple[str, ...]]:
    """Return the shortest successful sequence; commit if already correct."""
    if direct_correct:
        return tuple()
    successful = [
        tuple(str(x) for x in branch.get("sequence", []))
        for branch in branches
        if bool(branch.get("correct"))
    ]
    if not successful:
        return None
    min_depth = min(len(seq) for seq in successful)
    shortest = sorted(seq for seq in successful if len(seq) == min_depth)
    if tie_break_seed is None or len(shortest) == 1:
        return shortest[0]
    return random.Random(tie_break_seed).choice(shortest)


def summarize_counterfactuals(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute draft-conditioned oracle and per-advisor marginal statistics."""
    n = len(records)
    n_valid_direct = sum(bool(r.get("direct_valid")) for r in records)
    n_direct_correct = sum(bool(r.get("direct_correct")) for r in records)
    n_oracle_correct = sum(r.get("preferred_sequence") is not None for r in records)
    n_rescued = sum(
        (not bool(r.get("direct_correct")))
        and isinstance(r.get("preferred_sequence"), list)
        and len(r.get("preferred_sequence") or []) > 0
        for r in records
    )
    depth_counts: Dict[str, int] = {}
    first_tool_counts: Dict[str, int] = {}
    for row in records:
        seq = row.get("preferred_sequence")
        if not isinstance(seq, list):
            continue
        depth_counts[str(len(seq))] = depth_counts.get(str(len(seq)), 0) + 1
        if seq:
            first = str(seq[0])
            first_tool_counts[first] = first_tool_counts.get(first, 0) + 1

    by_advisor: Dict[str, Dict[str, Any]] = {}
    for kind in ADVISOR_KINDS:
        one_step = []
        for row in records:
            for branch in row.get("branches", []):
                if branch.get("sequence") == [kind]:
                    one_step.append((bool(row.get("direct_correct")), bool(branch.get("correct"))))
                    break
        rescue = sum((not direct_ok) and call_ok for direct_ok, call_ok in one_step)
        corruption = sum(direct_ok and (not call_ok) for direct_ok, call_ok in one_step)
        both_correct = sum(direct_ok and call_ok for direct_ok, call_ok in one_step)
        both_wrong = sum((not direct_ok) and (not call_ok) for direct_ok, call_ok in one_step)
        by_advisor[kind] = {
            "n": len(one_step),
            "rescue_count": rescue,
            "corruption_count": corruption,
            "both_correct_count": both_correct,
            "both_wrong_count": both_wrong,
            "rescue_rate": rescue / max(1, len(one_step)),
            "corruption_rate": corruption / max(1, len(one_step)),
            "net_marginal_rate": (rescue - corruption) / max(1, len(one_step)),
        }

    return {
        "n_examples": n,
        "direct_valid_rate": n_valid_direct / max(1, n),
        "direct_accuracy": n_direct_correct / max(1, n),
        "oracle_accuracy": n_oracle_correct / max(1, n),
        "oracle_gain": (n_oracle_correct - n_direct_correct) / max(1, n),
        "n_rescued": n_rescued,
        "n_unsolved": n - n_oracle_correct,
        "preferred_depth_counts": depth_counts,
        "preferred_first_tool_counts": first_tool_counts,
        "by_advisor_one_step": by_advisor,
    }


def _make_sft_rows(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    preferred = record.get("preferred_sequence")
    if not isinstance(preferred, list):
        return []
    base_messages = list(record["base_messages"])
    qhash = str(record["question_hash"])
    eid = int(record["example_id"])

    if not preferred:
        pred = str(record["direct_pred"])
        return [{
            "example_id": eid,
            "question_hash": qhash,
            "decision_type": "commit",
            "prompt": base_messages,
            "response": [{"role": "assistant", "content": _draft_and_final(pred)}],
        }]

    chosen = None
    for branch in record.get("branches", []):
        if branch.get("sequence") == preferred:
            chosen = branch
            break
    if chosen is None:
        raise ValueError(f"preferred branch missing for example_id={eid}: {preferred}")

    sft_rows: List[Dict[str, Any]] = []
    history = list(base_messages)
    trajectory = list(chosen.get("trajectory") or [])
    for event in trajectory:
        if event.get("role") != "assistant" or not event.get("tool_calls"):
            continue
        asst = {
            "role": "assistant",
            "content": str(event.get("content") or ""),
            "tool_calls": list(event["tool_calls"]),
        }
        sft_rows.append({
            "example_id": eid,
            "question_hash": qhash,
            "decision_type": "call",
            "preferred_sequence": list(preferred),
            "prompt": list(history),
            "response": [asst],
        })
        history.append(asst)
        tool_event = event.get("tool_event")
        if isinstance(tool_event, dict):
            tool_msg = dict(tool_event)
            history.append(tool_msg)

    final_pred = str(chosen.get("final_pred") or "")
    if not final_pred:
        return []
    sft_rows.append({
        "example_id": eid,
        "question_hash": qhash,
        "decision_type": "commit_after_call",
        "preferred_sequence": list(preferred),
        "prompt": list(history),
        "response": [{"role": "assistant", "content": _draft_and_final(final_pred)}],
    })
    return sft_rows


def _balance_records(
    records: List[Dict[str, Any]],
    max_commit_rescue_ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    rescue = [r for r in records if isinstance(r.get("preferred_sequence"), list) and r["preferred_sequence"]]
    commit = [r for r in records if r.get("preferred_sequence") == []]
    if max_commit_rescue_ratio >= 0:
        max_commit = int(max_commit_rescue_ratio * len(rescue))
        random.Random(seed).shuffle(commit)
        commit = commit[:max_commit]
    selected = rescue + commit
    random.Random(seed + 1).shuffle(selected)
    return selected


def build_marginal_value_sft(cfg: MarginalValueConfig) -> Dict[str, Any]:
    """Collect counterfactual branches and write shortest-success SFT data."""
    import torch
    from ..benchmarks.base import question_hash
    from ..utils.io import write_json, write_jsonl
    from ..utils.seed import set_seed

    if cfg.max_depth < 1 or cfg.max_depth > len(ADVISOR_KINDS):
        raise ValueError(f"max_depth must be in [1, {len(ADVISOR_KINDS)}]")
    if cfg.binding_mode not in {"argument", "environment"}:
        raise ValueError("binding_mode must be argument or environment")

    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer, model = _load_manager(cfg, device, dtype)
    pool = _build_pool(cfg, device)
    tools = _tool_schemas(cfg.binding_mode)
    available = [kind for kind in ADVISOR_KINDS if pool.has(kind)]

    sample = list(cfg.rows)
    random.Random(cfg.seed).shuffle(sample)
    if cfg.n_samples > 0:
        sample = sample[:cfg.n_samples]

    try:
        from tqdm import tqdm

        iterator = tqdm(sample, desc="marginal-value branches", unit="ex")
    except ImportError:
        iterator = sample

    records: List[Dict[str, Any]] = []
    branch_rows: List[Dict[str, Any]] = []
    for row in iterator:
        eid = int(row.example_id)
        choice_keys = list(row.choices.keys())
        base_messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": build_manager_system_prompt(
                    label_keys=choice_keys,
                    task_description=cfg.task_description,
                ),
            },
            {
                "role": "user",
                "content": build_manager_user_message(
                    example_id=eid,
                    question=row.question,
                    context=row.context,
                    choices=row.choices,
                    binding_mode=cfg.binding_mode,
                ),
            },
        ]

        direct_pred, direct_text, direct_valid = _generate_answer(
            tokenizer=tokenizer,
            model=model,
            messages=base_messages,
            tools=tools,
            choice_keys=choice_keys,
            probe=_DIRECT_PROBE,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            device=device,
        )
        direct_correct = bool(direct_valid and direct_pred == row.ground_truth)
        branches: List[Dict[str, Any]] = []

        if direct_valid and direct_pred is not None:
            frontier = [BranchState(
                sequence=tuple(),
                messages=list(base_messages),
                drafts=[direct_pred],
                final_pred=direct_pred,
                valid=True,
                correct=direct_correct,
                trajectory=[],
            )]
            for depth in range(1, cfg.max_depth + 1):
                next_frontier: List[BranchState] = []
                depth_success = False
                for state in frontier:
                    remaining = [kind for kind in available if kind not in state.sequence]
                    for kind in remaining:
                        current_draft = state.drafts[-1]
                        sequence = state.sequence + (kind,)
                        call_id = f"mv_{eid}_{'_'.join(sequence)}"
                        asst_call = _tool_call_message(
                            tool_kind=kind,
                            example_id=eid,
                            current_draft=current_draft,
                            call_id=call_id,
                            binding_mode=cfg.binding_mode,
                        )
                        tool_output = pool.call(
                            agent_kind=kind,
                            example_id=eid,
                            question=row.question,
                            context=row.context,
                            choices=row.choices,
                            cache_namespace="marginal_value",
                            candidate_answer=(current_draft if kind == "verifier" else ""),
                        )
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": f"{kind}_tool",
                            "content": tool_output,
                        }
                        messages = list(state.messages) + [asst_call, tool_msg]
                        pred, probe_text, valid = _generate_answer(
                            tokenizer=tokenizer,
                            model=model,
                            messages=messages,
                            tools=tools,
                            choice_keys=choice_keys,
                            probe=_REVISION_PROBE,
                            max_new_tokens=cfg.max_new_tokens,
                            temperature=cfg.temperature,
                            device=device,
                        )
                        next_draft = pred if pred is not None else current_draft
                        correct = bool(valid and pred == row.ground_truth)
                        trajectory = list(state.trajectory) + [{
                            "role": "assistant",
                            "content": asst_call["content"],
                            "tool_calls": asst_call["tool_calls"],
                            "tool_event": tool_msg,
                        }]
                        branch = {
                            "example_id": eid,
                            "question_hash": question_hash(row.question),
                            "sequence": list(sequence),
                            "depth": len(sequence),
                            "initial_draft": direct_pred,
                            "drafts": list(state.drafts) + [next_draft],
                            "final_pred": pred,
                            "valid": valid,
                            "correct": correct,
                            "probe_text": probe_text[:1200],
                            "trajectory": trajectory,
                        }
                        branches.append(branch)
                        branch_rows.append(branch)
                        next_frontier.append(BranchState(
                            sequence=sequence,
                            messages=messages,
                            drafts=list(state.drafts) + [next_draft],
                            final_pred=pred,
                            valid=valid,
                            correct=correct,
                            trajectory=trajectory,
                        ))
                        depth_success = depth_success or correct

                # Direct-correct examples need depth-1 branches only for
                # corruption diagnostics.  Incorrect examples stop at the
                # first successful depth, which is the shortest possible path.
                if direct_correct or depth_success:
                    break
                frontier = next_frontier

        preferred = choose_preferred_sequence(
            direct_correct, branches, tie_break_seed=cfg.seed + eid
        )
        record = {
            "example_id": eid,
            "question_hash": question_hash(row.question),
            "benchmark_name": row.benchmark_name,
            "ground_truth": row.ground_truth,
            "direct_pred": direct_pred,
            "direct_valid": direct_valid,
            "direct_correct": direct_correct,
            "direct_text": direct_text[:1200],
            "preferred_sequence": list(preferred) if preferred is not None else None,
            "base_messages": base_messages,
            "branches": branches,
        }
        records.append(record)

    report = summarize_counterfactuals(records)
    selected = _balance_records(records, cfg.max_commit_rescue_ratio, cfg.seed)
    sft_rows = list(itertools.chain.from_iterable(_make_sft_rows(row) for row in selected))
    report.update({
        "n_selected_decisions": len(selected),
        "n_selected_rescue_decisions": sum(bool(r.get("preferred_sequence")) for r in selected),
        "n_selected_commit_decisions": sum(r.get("preferred_sequence") == [] for r in selected),
        "n_sft_turns": len(sft_rows),
        "max_depth": cfg.max_depth,
        "max_commit_rescue_ratio": cfg.max_commit_rescue_ratio,
        "available_advisors": available,
        "manager_dir": cfg.manager_dir,
    })

    records_path = os.path.join(cfg.out_dir, "counterfactual_records.jsonl")
    branches_path = os.path.join(cfg.out_dir, "counterfactual_branches.jsonl")
    sft_path = os.path.join(cfg.out_dir, "manager_sft_marginal.jsonl")
    report_path = os.path.join(cfg.out_dir, "marginal_value_report.json")
    write_jsonl(records_path, records)
    write_jsonl(branches_path, branch_rows)
    write_jsonl(sft_path, sft_rows)
    write_json(report_path, report)
    print(
        "[MARGINAL_VALUE] "
        f"direct={report['direct_accuracy']:.3f} "
        f"oracle={report['oracle_accuracy']:.3f} "
        f"gain={report['oracle_gain']:.3f} "
        f"rescued={report['n_rescued']} sft_turns={len(sft_rows)}"
    )
    return {
        "records_jsonl": records_path,
        "branches_jsonl": branches_path,
        "sft_jsonl": sft_path,
        "report_json": report_path,
        "report": report,
    }
