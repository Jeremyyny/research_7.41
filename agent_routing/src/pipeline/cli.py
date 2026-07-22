"""CLI entry point.

Usage examples (see README for full walkthroughs):

  # Synthesize 500 reasoner samples with Claude as teacher
  python -m src.pipeline.cli synth_subagent \\
      --teacher_provider anthropic --teacher_model claude-sonnet-4-5 \\
      --teacher_id claude_sonnet_4_5 \\
      --agent_kind reasoner --n_samples 500

  # Train the reasoner subagent
  python -m src.pipeline.cli train_subagent \\
      --teacher_id claude_sonnet_4_5 --agent_kind reasoner

  # GRPO-train the manager
  python -m src.pipeline.cli train_manager_grpo \\
      --teacher_id claude_sonnet_4_5

  # One full evolve round
  python -m src.pipeline.cli evolve_round \\
      --teacher_id claude_sonnet_4_5
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from ..benchmarks.base import StandardRow
from ..utils.io import read_jsonl, write_jsonl
from . import stages


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agent_routing")
    parser.add_argument("stage", type=str, choices=[
        "load_medqa",
        "load_gpqa",
        "load_mmlu_pro",
        "export_legalbench_jsonl",
        "synth_subagent",
        "export_deepseek_jsonl",
        "import_deepseek_jsonl",
        "train_subagent",
        "train_manager_grpo",
        "build_marginal_sft",
        "manager_coldstart_sft",
        "export_manager_coldstart_prompts",
        "import_manager_coldstart_responses",
        "evolve_build_sft",
        "train_manager_sft",
        "evolve_round",
        "eval_subagents",
        "eval_manager",
        "eval_manager_tools",
        "eval_manager_forced",
    ])

    # Context-level flags
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher_id", type=str, default="default",
                        help="Logical id used to namespace outputs (e.g. mmlu_pro_gpt54).")
    parser.add_argument("--subagent_teacher_id", type=str, default="",
                        help="If set, load subagent adapters from this teacher_id's adapter dir instead of --teacher_id. "
                             "Use when reusing subagents trained under a different run (e.g. --teacher_id mmlu_pro_gpt54 "
                             "--subagent_teacher_id mmlu_pro_claude).")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binding_mode", type=str, default="auto",
                        choices=["auto", "environment", "argument"])

    # MedQA loading
    parser.add_argument("--medqa_source", type=str, default="hf", choices=["hf", "local"])
    parser.add_argument("--medqa_hf_dataset", type=str, default="GBaker/MedQA-USMLE-4-options")
    parser.add_argument("--medqa_local_path", type=str, default="")
    parser.add_argument("--medqa_hf_cache", type=str, default="")
    parser.add_argument("--medqa_max", type=int, default=0)
    parser.add_argument("--medqa_normalized_cache", type=str, default="")
    parser.add_argument("--medqa_refresh_cache", action="store_true",
                        help="Reload MedQA from the requested source/path and overwrite the normalized cache.")

    # LegalBench loading
    parser.add_argument("--legalbench_hf_dataset", type=str, default="nguha/legalbench")
    parser.add_argument("--legalbench_configs", type=str, default="",
                        help="Comma-separated LegalBench config names, e.g. 'abercrombie,hearsay', or 'all'.")
    parser.add_argument("--legalbench_split", type=str, default="test",
                        help="LegalBench split to draw examples from, usually test because train is few-shot.")
    parser.add_argument("--legalbench_hf_cache", type=str, default="")
    parser.add_argument("--legalbench_max", type=int, default=0)
    parser.add_argument("--legalbench_max_labels", type=int, default=12)
    parser.add_argument("--legalbench_normalized_cache", type=str, default="")
    parser.add_argument("--legalbench_refresh_cache", action="store_true")

    # GPQA loading
    parser.add_argument("--gpqa_hf_dataset", type=str, default="Idavidrein/gpqa")
    parser.add_argument("--gpqa_subsets", type=str, default="gpqa_diamond",
                        help="Comma-separated GPQA subset names: gpqa_main, gpqa_diamond, "
                             "gpqa_extended, or 'all'. Default: gpqa_diamond.")
    parser.add_argument("--gpqa_hf_cache", type=str, default="")
    parser.add_argument("--gpqa_max", type=int, default=0)
    parser.add_argument("--gpqa_answer_seed", type=int, default=42,
                        help="Seed for A/B/C/D answer shuffling (keeps mapping deterministic).")
    parser.add_argument("--gpqa_exclude_subsets", type=str, default="",
                        help="Comma-separated GPQA subsets whose questions are REMOVED from "
                             "the loaded rows. GPQA subsets are nested (diamond ⊆ main ⊆ "
                             "extended); pass 'gpqa_diamond' when training on main/extended "
                             "and evaluating on diamond to avoid contamination.")
    parser.add_argument("--gpqa_normalized_cache", type=str, default="")
    parser.add_argument("--gpqa_refresh_cache", action="store_true")

    # MMLU-Pro loading
    parser.add_argument("--mmlu_pro_hf_dataset", type=str, default="TIGER-Lab/MMLU-Pro")
    parser.add_argument("--mmlu_pro_categories", type=str, default="",
                        help="Comma-separated category names to keep, e.g. 'math,physics'. "
                             "Empty means all categories.")
    parser.add_argument("--mmlu_pro_hf_cache", type=str, default="")
    parser.add_argument("--mmlu_pro_max", type=int, default=0)
    parser.add_argument("--mmlu_pro_splits", type=str, default="test,validation",
                        help="Comma-separated HF split names to load.")
    parser.add_argument("--mmlu_pro_normalized_cache", type=str, default="")
    parser.add_argument("--mmlu_pro_refresh_cache", action="store_true")

    # Split sizes
    parser.add_argument("--train_size", type=int, default=600)
    parser.add_argument("--dev_size", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=200)

    # Synth
    parser.add_argument("--teacher_provider", type=str, default="",
                        choices=["", "anthropic", "claude", "openai", "gpt", "deepseek"])
    parser.add_argument("--teacher_model", type=str, default="")
    parser.add_argument("--agent_kind", type=str, default="",
                        choices=["", "extractor", "reasoner", "verifier"])
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--synth_temperature", type=float, default=0.4)
    parser.add_argument("--synth_max_retries", type=int, default=2)
    parser.add_argument("--synth_workers", type=int, default=8,
                        help="Parallel teacher API calls during synthesis (default 8).")
    parser.add_argument("--synth_no_cache", action="store_true")
    parser.add_argument("--synth_symmetric_leakage", action="store_true",
                        help="Leakage-audit against ALL choice texts instead of only the "
                             "ground-truth text. Removes the negative-space bias where the "
                             "one never-restated choice is exactly the answer.")
    parser.add_argument("--deepseek_prompt_jsonl", type=str, default="",
                        help="Prompt JSONL for local DeepSeek batch generation.")
    parser.add_argument("--deepseek_response_jsonl", type=str, default="",
                        help="Response JSONL produced by local DeepSeek generate_jsonl.py.")
    parser.add_argument("--deepseek_sft_jsonl", type=str, default="",
                        help="Optional imported SFT JSONL output path.")
    parser.add_argument("--deepseek_teacher_model", type=str, default="deepseek-local",
                        help="Metadata model name used when importing local DeepSeek responses.")
    parser.add_argument("--deepseek_import_raw_responses", action="store_true",
                        help="Import response text as-is, but pair it with runtime subagent prompts instead of validating/filtering.")

    # Subagent SFT
    parser.add_argument("--sft_epochs", type=int, default=3)
    parser.add_argument("--sft_lr", type=float, default=2e-4)
    parser.add_argument("--sft_max_seq_len", type=int, default=4096)
    parser.add_argument("--sft_bs", type=int, default=1)
    parser.add_argument("--sft_grad_accum", type=int, default=8)
    parser.add_argument("--sft_max_steps", type=int, default=-1)
    parser.add_argument("--sft_no_lora", action="store_true")
    parser.add_argument("--sft_train_jsonl", type=str, default="",
                        help="Optional explicit SFT JSONL path for train_subagent. Must contain prompt and response fields.")
    parser.add_argument("--sft_dev_jsonl", type=str, default="")

    # Manager GRPO
    parser.add_argument("--mgr_bs", type=int, default=2)
    parser.add_argument("--mgr_max_completion_length", type=int, default=2048)
    parser.add_argument("--mgr_temperature", type=float, default=0.9)
    parser.add_argument("--mgr_num_generations", type=int, default=6)
    parser.add_argument("--mgr_grpo_beta", type=float, default=0.01)
    parser.add_argument("--mgr_routing_efficiency_bonus", type=float, default=0.0)
    parser.add_argument("--mgr_tool_use_bonus", type=float, default=0.0,
                        help="Bonus added only when the final answer is correct and at least one native tool was called.")
    parser.add_argument("--mgr_ccr_mode", action="store_true",
                        help="(Legacy) Enable CCR reward (log scoring rule). "
                             "Requires p_low>0.5 to avoid reward inversion. "
                             "Prefer --mgr_adc_mode instead.")
    parser.add_argument("--mgr_ccr_p_high", type=float, default=0.9,
                        help="CCR implicit confidence when manager calls 0 tools (default 0.9).")
    parser.add_argument("--mgr_ccr_p_low", type=float, default=0.6,
                        help="CCR implicit confidence when manager calls k_max tools. "
                             "MUST be >0.5 to avoid reward inversion at k>=2 (default 0.6).")
    parser.add_argument("--mgr_ccr_k_max", type=int, default=3,
                        help="CCR k_max; must match max_tool_calling_iterations (default 3).")
    # ADC — legacy Adaptive Deliberation Control reward ablation
    parser.add_argument("--mgr_adc_mode", action="store_true",
                        help="Legacy ablation: enable ADC anytime reward: "
                             "+draft_bonus per CORRECT DRAFT_ANSWER_, final bonus, tool cost. "
                             "Retained only for comparison with older runs.")
    parser.add_argument("--mgr_adc_cost_per_tool", type=float, default=0.05,
                        help="ADC per-tool cost subtracted from reward (default 0.05). "
                             "Encourages the manager to stop calling tools when not helpful.")
    parser.add_argument("--mgr_adc_draft_bonus", type=float, default=0.2,
                        help="ADC bonus per CORRECT draft answer (default 0.2). "
                             "Honest best-guess drafts are the unique optimal policy.")
    parser.add_argument("--mgr_adc_missing_draft_penalty", type=float, default=0.1,
                        help="ADC penalty per tool call without an accompanying "
                             "DRAFT_ANSWER_ (default 0.1). Enforces the draft format.")
    parser.add_argument("--mgr_adc_final_bonus", type=float, default=1.0,
                        help="ADC bonus for final correct answer (default 1.0).")
    parser.add_argument("--mgr_adc_variant", type=str, default="anytime",
                        choices=["anytime", "transition", "sum"],
                        help="ADC process-reward variant. 'anytime' (default) is the "
                             "incentive-compatible design. 'transition' and 'sum' reproduce "
                             "the provably exploitable designs — ABLATION ARMS ONLY (RQ3): "
                             "transition pays for sandbagged first drafts, sum is farmable "
                             "by superfluous tool calls.")
    parser.add_argument("--mgr_full_parameter_rl", action="store_true",
                        help="Run full-parameter GRPO. If --mgr_init_adapter is set, merge it into the base model first.")
    parser.add_argument("--mgr_max_steps", type=int, default=-1)
    parser.add_argument("--mgr_output_dir", type=str, default="",
                        help="Optional explicit output directory for train_manager_grpo.")
    parser.add_argument("--mgr_use_wandb", action="store_true")
    parser.add_argument("--mgr_init_adapter", type=str, default="",
                        help="Optional manager LoRA adapter to initialize GRPO from, e.g. outputs/manager/<id>/sft_evolved.")
    parser.add_argument("--subagent_server_url", type=str, default="",
                        help="vLLM HTTP server URL for subagents (multi-GPU mode). "
                             "E.g. http://localhost:8000. When set, no subagent weights are loaded "
                             "into the training processes; use with accelerate + ZeRO Stage 3.")
    parser.add_argument("--wandb_project", type=str, default="agent_routing")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--task_description", type=str, default="")
    parser.add_argument("--mgr_exploration_hint", type=str, default="",
                        help="START-style hint injected into manager system prompt during GRPO training only. "
                             "Empty string disables. "
                             "Example: 'Hint: When uncertain, calling 2-3 tools often reveals what one tool misses.'")
    parser.add_argument("--mgr_clip_epsilon_high", type=float, default=0.0,
                        help="DAPO Clip-Higher: upper bound for importance-sampling ratio clip. "
                             "0 = standard symmetric clipping (epsilon=0.2 both sides). "
                             "Recommended: 0.28 (DAPO paper default).")
    parser.add_argument("--exclude_sft_example_ids", action="append", default=[],
                        help="JSONL path(s), comma-separated or repeated, whose example_id values are excluded from manager GRPO train rows.")

    # Counterfactual marginal-value routing data
    parser.add_argument("--mv_manager_dir", type=str, default="",
                        help="Manager checkpoint used to generate the shared initial draft and revise counterfactual branches. Empty uses --base_model.")
    parser.add_argument("--mv_n_samples", type=int, default=300,
                        help="Number of training questions used for marginal-value branch collection.")
    parser.add_argument("--mv_max_depth", type=int, default=1, choices=[1, 2, 3],
                        help="Maximum number of distinct advisors in a forced counterfactual branch. Start with 1; use 2/3 only if the one-step oracle leaves useful headroom.")
    parser.add_argument("--mv_max_new_tokens", type=int, default=512,
                        help="Manager token budget for each counterfactual answer probe.")
    parser.add_argument("--mv_temperature", type=float, default=0.0,
                        help="Counterfactual manager sampling temperature. Keep 0 for paired, deterministic comparisons.")
    parser.add_argument("--mv_max_commit_rescue_ratio", type=float, default=1.0,
                        help="Cap direct-correct commit decisions per rescued decision in marginal SFT. Negative keeps all commits; 1.0 gives a balanced cold start.")
    parser.add_argument("--mv_output_dir", type=str, default="",
                        help="Optional output directory for counterfactual records, report, and manager_sft_marginal.jsonl.")

    # Evolve
    parser.add_argument("--evolve_max_fail_samples", type=int, default=1500)
    parser.add_argument("--fail_buffer_jsonl", type=str, default="",
                        help="Optional explicit GRPO fail_buffer.jsonl path for evolve_build_sft.")
    parser.add_argument("--coldstart_n_samples", type=int, default=300)
    parser.add_argument("--coldstart_force_diverse", action="store_true",
                        help="Skip teacher model; assign balanced tool-sequence distribution "
                             "(k=0/1/2/3) across coldstart examples. No API calls needed.")
    parser.add_argument("--coldstart_prompt_jsonl", type=str, default="",
                        help="Prompt JSONL for import_manager_coldstart_responses (output of export_manager_coldstart_prompts).")
    parser.add_argument("--coldstart_response_jsonl", type=str, default="",
                        help="Response JSONL for import_manager_coldstart_responses (tool sequences from DeepSeek / batch API).")
    parser.add_argument("--coldstart_out_jsonl", type=str, default="",
                        help="Optional explicit output path for the SFT JSONL built by import_manager_coldstart_responses.")
    parser.add_argument("--manager_sft_train_jsonl", type=str, default="",
                        help="Optional explicit manager SFT JSONL for train_manager_sft.")
    parser.add_argument("--manager_sft_init_adapter", type=str, default="",
                        help="Optional existing manager adapter/full checkpoint to continue SFT from instead of restarting from --base_model.")
    parser.add_argument("--manager_sft_output_dir", type=str, default="",
                        help="Optional explicit output directory for train_manager_sft.")
    parser.add_argument("--manager_sft_lr", type=float, default=2e-5)
    parser.add_argument("--manager_sft_epochs", type=int, default=1)

    # Eval
    parser.add_argument("--eval_n_samples", type=int, default=100)
    parser.add_argument("--eval_kinds", type=str, default="extractor,reasoner,verifier")
    parser.add_argument("--eval_manager_dir", type=str, default="")
    parser.add_argument("--eval_temperature", type=float, default=0.0)
    parser.add_argument("--eval_max_new_tokens", type=int, default=1024)
    parser.add_argument("--eval_max_tool_calls", type=int, default=3)
    parser.add_argument("--eval_forced_tools", type=str, default="none",
                        help="Fixed delegation sequence for eval_manager_forced: comma-separated "
                             "advisor kinds, e.g. 'extractor,reasoner,verifier', or 'none' for "
                             "the zero-delegation baseline. Running every subset yields fixed-k "
                             "baselines and the per-question stopping oracle.")
    parser.add_argument("--eval_out_tag", type=str, default="",
                        help="Optional filename tag for eval_manager_forced outputs.")
    parser.add_argument("--eval_sc_k", type=int, default=1,
                        help="Self-consistency baseline for eval_manager: sample k completions "
                             "and majority-vote (k=1 disables; use as the matched-compute "
                             "resampling control).")
    parser.add_argument("--eval_sc_temperature", type=float, default=0.7,
                        help="Sampling temperature for the self-consistency baseline.")

    return parser.parse_args()


def _ctx_from(args) -> stages.StageContext:
    return stages.StageContext(
        base_model=args.base_model,
        teacher_id=args.teacher_id,
        output_root=args.output_root,
        seed=args.seed,
        binding_mode=args.binding_mode,
        subagent_teacher_id=getattr(args, "subagent_teacher_id", ""),
    )


def _load_or_split(args) -> dict:
    """Load MedQA, split into train/dev/test, also serialize splits to disk."""
    cache = args.medqa_normalized_cache or os.path.join(
        args.output_root, "data", "medqa_normalized.jsonl"
    )
    if args.medqa_refresh_cache or not os.path.exists(cache):
        rows = stages.run_load_medqa(
            source=args.medqa_source,
            hf_dataset=args.medqa_hf_dataset,
            local_path=(args.medqa_local_path or None),
            hf_cache_dir=(args.medqa_hf_cache or None),
            max_examples=args.medqa_max,
            cache_normalized_path=cache,
        )
    else:
        from ..benchmarks.base import StandardRow
        rows = [StandardRow(**r) for r in read_jsonl(cache)]
        print(f"[LOAD_MEDQA] loaded cached {len(rows)} rows -> {cache}")

    train, dev, test = stages._split_rows(
        rows=rows, train_size=args.train_size, dev_size=args.dev_size,
        test_size=args.test_size, seed=args.seed,
    )
    print(f"[SPLIT] train/dev/test = {len(train)}/{len(dev)}/{len(test)}")
    return {"all": rows, "train": train, "dev": dev, "test": test}


def _load_legalbench_or_cache(args) -> List[StandardRow]:
    cache = args.legalbench_normalized_cache or os.path.join(
        args.output_root, "data", "legalbench_normalized.jsonl"
    )
    if args.legalbench_refresh_cache or not os.path.exists(cache):
        rows = stages.run_load_legalbench(
            dataset_name=args.legalbench_hf_dataset,
            configs=args.legalbench_configs,
            split=args.legalbench_split,
            hf_cache_dir=(args.legalbench_hf_cache or None),
            max_examples=args.legalbench_max,
            max_labels=args.legalbench_max_labels,
            cache_normalized_path=cache,
        )
    else:
        rows = [StandardRow(**r) for r in read_jsonl(cache)]
        print(f"[LOAD_LEGALBENCH] loaded cached {len(rows)} rows -> {cache}")
    return rows


def _using_legalbench(args) -> bool:
    return bool(args.legalbench_normalized_cache or args.legalbench_configs)


def _using_gpqa(args) -> bool:
    # NOTE: --gpqa_subsets has a non-empty default, so only the cache path
    # (or the load_gpqa stage itself) activates the GPQA branch.
    return bool(getattr(args, "gpqa_normalized_cache", ""))


def _using_mmlu_pro(args) -> bool:
    return bool(
        getattr(args, "mmlu_pro_normalized_cache", "")
        or getattr(args, "mmlu_pro_categories", "") != ""
        # Explicit flag to use MMLU-Pro even with no category filter
        or getattr(args, "use_mmlu_pro", False)
    )


def _load_gpqa_or_cache(args) -> List[StandardRow]:
    cache = args.gpqa_normalized_cache or os.path.join(
        args.output_root, "data", "gpqa_normalized.jsonl"
    )
    if args.gpqa_refresh_cache or not os.path.exists(cache):
        rows = stages.run_load_gpqa(
            dataset_name=args.gpqa_hf_dataset,
            subsets=args.gpqa_subsets,
            hf_cache_dir=(args.gpqa_hf_cache or None),
            max_examples=args.gpqa_max,
            answer_seed=args.gpqa_answer_seed,
            cache_normalized_path=cache,
            exclude_subsets=args.gpqa_exclude_subsets,
        )
    else:
        rows = [stages.StandardRow(**r) for r in read_jsonl(cache)]
        print(f"[LOAD_GPQA] loaded cached {len(rows)} rows -> {cache}")
    return rows


def _load_mmlu_pro_or_cache(args) -> List[StandardRow]:
    cache = args.mmlu_pro_normalized_cache or os.path.join(
        args.output_root, "data", "mmlu_pro_normalized.jsonl"
    )
    if args.mmlu_pro_refresh_cache or not os.path.exists(cache):
        rows = stages.run_load_mmlu_pro(
            dataset_name=args.mmlu_pro_hf_dataset,
            categories=args.mmlu_pro_categories,
            hf_cache_dir=(args.mmlu_pro_hf_cache or None),
            max_examples=args.mmlu_pro_max,
            splits=args.mmlu_pro_splits,
            cache_normalized_path=cache,
        )
    else:
        rows = [stages.StandardRow(**r) for r in read_jsonl(cache)]
        print(f"[LOAD_MMLU_PRO] loaded cached {len(rows)} rows -> {cache}")
    return rows


def _load_benchmark_splits(args) -> dict:
    """Load the requested benchmark and return train/dev/test splits.

    Priority (first active wins): mmlu_pro > gpqa > legalbench > medqa.
    GPQA and MMLU-Pro have no predefined train/dev/test split, so rows are
    split deterministically using --train_size / --dev_size / --test_size.
    """
    # MMLU-Pro: active when --mmlu_pro_normalized_cache is set OR
    #           --mmlu_pro_categories is non-empty OR stage == load_mmlu_pro
    if getattr(args, "stage", "") == "load_mmlu_pro" or _using_mmlu_pro(args):
        rows = _load_mmlu_pro_or_cache(args)
        train, dev, test = stages._split_rows(
            rows=rows, train_size=args.train_size, dev_size=args.dev_size,
            test_size=args.test_size, seed=args.seed,
        )
        print(f"[SPLIT/MMLU_PRO] train/dev/test = {len(train)}/{len(dev)}/{len(test)}")
        return {"all": rows, "train": train, "dev": dev, "test": test}

    # GPQA: active when --gpqa_normalized_cache is set OR stage == load_gpqa
    if getattr(args, "stage", "") == "load_gpqa" or (
        _using_gpqa(args) and not _using_legalbench(args)
    ):
        rows = _load_gpqa_or_cache(args)
        train, dev, test = stages._split_rows(
            rows=rows, train_size=args.train_size, dev_size=args.dev_size,
            test_size=args.test_size, seed=args.seed,
        )
        print(f"[SPLIT/GPQA] train/dev/test = {len(train)}/{len(dev)}/{len(test)}")
        return {"all": rows, "train": train, "dev": dev, "test": test}

    if _using_legalbench(args):
        rows = _load_legalbench_or_cache(args)
        train, dev, test = stages._split_rows(
            rows=rows, train_size=args.train_size, dev_size=args.dev_size,
            test_size=args.test_size, seed=args.seed,
        )
        print(f"[SPLIT/LEGALBENCH] train/dev/test = {len(train)}/{len(dev)}/{len(test)}")
        return {"all": rows, "train": train, "dev": dev, "test": test}

    return _load_or_split(args)


def _load_eval_rows(args) -> List[StandardRow]:
    data = _load_benchmark_splits(args)
    return data["test"] or data["dev"]


def _exclude_sft_rows(rows: List[StandardRow], paths: List[str]) -> List[StandardRow]:
    from ..benchmarks.base import question_hash

    expanded: List[str] = []
    for item in paths or []:
        expanded.extend([p.strip() for p in item.split(",") if p.strip()])
    if not expanded:
        return rows

    exclude_ids = set()
    exclude_hashes = set()
    for path in expanded:
        if not os.path.exists(path):
            raise FileNotFoundError(f"exclude_sft_example_ids path not found: {path}")
        for row in read_jsonl(path):
            if row.get("example_id") is not None:
                exclude_ids.add(int(row["example_id"]))
            # question_hash survives normalized-cache rebuilds; example_id does not.
            if row.get("question_hash"):
                exclude_hashes.add(str(row["question_hash"]))

    kept = [
        r for r in rows
        if int(r.example_id) not in exclude_ids
        and question_hash(r.question) not in exclude_hashes
    ]
    print(
        f"[EXCLUDE_SFT] files={len(expanded)} ids={len(exclude_ids)} "
        f"hashes={len(exclude_hashes)} train_rows {len(rows)} -> {len(kept)}"
    )
    if not kept:
        raise ValueError("No manager training rows left after excluding SFT example_ids.")
    return kept


def main() -> None:
    args = _parse_args()
    ctx = _ctx_from(args)

    if args.stage == "load_medqa":
        _load_or_split(args)
        return

    if args.stage == "load_gpqa":
        rows = _load_gpqa_or_cache(args)
        train, dev, test = stages._split_rows(
            rows=rows, train_size=args.train_size, dev_size=args.dev_size,
            test_size=args.test_size, seed=args.seed,
        )
        print(f"[LOAD_GPQA] train/dev/test = {len(train)}/{len(dev)}/{len(test)}")
        return

    if args.stage == "load_mmlu_pro":
        rows = _load_mmlu_pro_or_cache(args)
        train, dev, test = stages._split_rows(
            rows=rows, train_size=args.train_size, dev_size=args.dev_size,
            test_size=args.test_size, seed=args.seed,
        )
        print(f"[LOAD_MMLU_PRO] train/dev/test = {len(train)}/{len(dev)}/{len(test)}")
        return

    if args.stage == "export_legalbench_jsonl":
        if not args.agent_kind:
            sys.exit("export_legalbench_jsonl requires --agent_kind")
        if not args.legalbench_configs:
            sys.exit("export_legalbench_jsonl requires --legalbench_configs")
        data = _load_benchmark_splits(args)
        rows = data["train"]
        if not rows:
            sys.exit("No LegalBench rows loaded. Check configs/split/max_labels.")
        out_path = args.deepseek_prompt_jsonl or os.path.join(
            ctx.sft_data_root,
            f"{args.agent_kind}_legalbench_prompts.jsonl",
        )
        result = stages.run_export_deepseek_subagent_prompts(
            ctx=ctx,
            rows=rows,
            agent_kind=args.agent_kind,
            out_path=out_path,
            n_samples=args.n_samples,
        )
        print("[EXPORT_LEGALBENCH_JSONL]", result)
        return

    if args.stage == "synth_subagent":
        if not (args.teacher_provider and args.teacher_model and args.agent_kind):
            sys.exit("synth_subagent requires --teacher_provider, --teacher_model, --agent_kind")
        data = _load_benchmark_splits(args)
        kind = args.agent_kind
        # Synthesize on the train pool
        result = stages.run_synthesize_subagent(
            ctx=ctx, rows=data["train"], agent_kind=kind,
            teacher_provider=args.teacher_provider, teacher_model=args.teacher_model,
            n_samples=args.n_samples,
            base_temperature=args.synth_temperature,
            max_retries=args.synth_max_retries,
            use_cache=(not args.synth_no_cache),
            max_workers=args.synth_workers,
            symmetric_leakage=args.synth_symmetric_leakage,
        )
        print("[SYNTH]", result)
        return

    if args.stage == "export_deepseek_jsonl":
        if not args.agent_kind:
            sys.exit("export_deepseek_jsonl requires --agent_kind")
        data = _load_benchmark_splits(args)
        kind = args.agent_kind
        result = stages.run_export_deepseek_subagent_prompts(
            ctx=ctx,
            rows=data["train"],
            agent_kind=kind,
            out_path=(args.deepseek_prompt_jsonl or None),
            n_samples=args.n_samples,
        )
        print("[EXPORT_DEEPSEEK_JSONL]", result)
        return

    if args.stage == "import_deepseek_jsonl":
        if not args.agent_kind:
            sys.exit("import_deepseek_jsonl requires --agent_kind")
        if not (args.deepseek_prompt_jsonl and args.deepseek_response_jsonl):
            sys.exit("import_deepseek_jsonl requires --deepseek_prompt_jsonl and --deepseek_response_jsonl")
        kind = args.agent_kind
        result = stages.run_import_deepseek_subagent_responses(
            ctx=ctx,
            agent_kind=kind,
            prompt_jsonl=args.deepseek_prompt_jsonl,
            response_jsonl=args.deepseek_response_jsonl,
            out_path=(args.deepseek_sft_jsonl or None),
            teacher_model=args.deepseek_teacher_model,
            raw_responses=args.deepseek_import_raw_responses,
        )
        print("[IMPORT_DEEPSEEK_JSONL]", result)
        return

    if args.stage == "train_subagent":
        if not args.agent_kind:
            sys.exit("train_subagent requires --agent_kind")
        kind = args.agent_kind
        result = stages.run_train_subagent(
            ctx=ctx, agent_kind=kind,
            train_jsonl=(args.sft_train_jsonl or None),
            dev_jsonl=(args.sft_dev_jsonl or None),
            epochs=args.sft_epochs, lr=args.sft_lr,
            max_seq_len=args.sft_max_seq_len,
            per_device_batch_size=args.sft_bs,
            gradient_accumulation_steps=args.sft_grad_accum,
            use_lora=(not args.sft_no_lora),
            max_steps=args.sft_max_steps,
        )
        print("[TRAIN_SUBAGENT]", result)
        return

    if args.stage == "train_manager_grpo":
        data = _load_benchmark_splits(args)
        train_rows = _exclude_sft_rows(data["train"], args.exclude_sft_example_ids)
        result = stages.run_train_manager_grpo(
            ctx=ctx, train_rows=train_rows,
            manager_adapter=(args.mgr_init_adapter or None),
            per_device_batch_size=args.mgr_bs,
            max_completion_length=args.mgr_max_completion_length,
            temperature=args.mgr_temperature,
            num_generations=args.mgr_num_generations,
            grpo_beta=args.mgr_grpo_beta,
            routing_efficiency_bonus=args.mgr_routing_efficiency_bonus,
            tool_use_bonus=args.mgr_tool_use_bonus,
            ccr_mode=args.mgr_ccr_mode,
            ccr_p_high=args.mgr_ccr_p_high,
            ccr_p_low=args.mgr_ccr_p_low,
            ccr_k_max=args.mgr_ccr_k_max,
            adc_mode=args.mgr_adc_mode,
            adc_cost_per_tool=args.mgr_adc_cost_per_tool,
            adc_draft_bonus=args.mgr_adc_draft_bonus,
            adc_missing_draft_penalty=args.mgr_adc_missing_draft_penalty,
            adc_final_bonus=args.mgr_adc_final_bonus,
            adc_variant=args.mgr_adc_variant,
            full_parameter_rl=args.mgr_full_parameter_rl,
            max_steps=args.mgr_max_steps,
            output_dir=(args.mgr_output_dir or None),
            use_wandb=args.mgr_use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            task_description=args.task_description,
            subagent_server_url=(args.subagent_server_url or None),
            exploration_hint=args.mgr_exploration_hint,
            clip_epsilon_high=args.mgr_clip_epsilon_high,
        )
        print("[TRAIN_MGR_GRPO]", result)
        return

    if args.stage == "evolve_build_sft":
        data = _load_benchmark_splits(args)
        result = stages.run_evolve_build_sft(
            ctx=ctx, rows=data["all"],
            teacher_provider=(args.teacher_provider or None),
            teacher_model=(args.teacher_model or None),
            fail_buffer_jsonl=(args.fail_buffer_jsonl or None),
            max_fail_samples=args.evolve_max_fail_samples,
            task_description=args.task_description,
        )
        print("[EVOLVE_BUILD_SFT]", result)
        return

    if args.stage == "build_marginal_sft":
        data = _load_benchmark_splits(args)
        train_rows = _exclude_sft_rows(data["train"], args.exclude_sft_example_ids)
        result = stages.run_build_marginal_sft(
            ctx=ctx,
            rows=train_rows,
            manager_dir=(args.mv_manager_dir or None),
            n_samples=args.mv_n_samples,
            max_depth=args.mv_max_depth,
            max_new_tokens=args.mv_max_new_tokens,
            temperature=args.mv_temperature,
            max_commit_rescue_ratio=args.mv_max_commit_rescue_ratio,
            task_description=args.task_description,
            output_dir=(args.mv_output_dir or None),
            subagent_server_url=(args.subagent_server_url or None),
        )
        print("[BUILD_MARGINAL_SFT]", result)
        return

    if args.stage == "export_manager_coldstart_prompts":
        data = _load_benchmark_splits(args)
        train_rows = _exclude_sft_rows(data["train"], args.exclude_sft_example_ids)
        result = stages.run_export_manager_coldstart_prompts(
            ctx=ctx,
            rows=train_rows,
            n_samples=args.coldstart_n_samples,
            out_path=(args.coldstart_prompt_jsonl or None),
        )
        print("[EXPORT_MANAGER_COLDSTART_PROMPTS]", result)
        return

    if args.stage == "import_manager_coldstart_responses":
        if not args.coldstart_prompt_jsonl or not args.coldstart_response_jsonl:
            sys.exit(
                "import_manager_coldstart_responses requires "
                "--coldstart_prompt_jsonl and --coldstart_response_jsonl"
            )
        result = stages.run_import_manager_coldstart_responses(
            ctx=ctx,
            prompt_jsonl=args.coldstart_prompt_jsonl,
            response_jsonl=args.coldstart_response_jsonl,
            out_path=(args.coldstart_out_jsonl or None),
        )
        print("[IMPORT_MANAGER_COLDSTART_RESPONSES]", result)
        return

    if args.stage == "manager_coldstart_sft":
        data = _load_benchmark_splits(args)
        train_rows = _exclude_sft_rows(data["train"], args.exclude_sft_example_ids)
        result = stages.run_manager_coldstart_sft(
            ctx=ctx,
            rows=train_rows,
            teacher_provider=(args.teacher_provider or None),
            teacher_model=(args.teacher_model or None),
            n_samples=args.coldstart_n_samples,
            task_description=args.task_description,
            epochs=args.manager_sft_epochs,
            lr=args.manager_sft_lr,
            max_seq_len=args.sft_max_seq_len,
            per_device_batch_size=args.sft_bs,
            gradient_accumulation_steps=args.sft_grad_accum,
            use_lora=(not args.sft_no_lora),
            max_steps=args.sft_max_steps,
            force_diverse=args.coldstart_force_diverse,
        )
        print("[MANAGER_COLDSTART_SFT]", result)
        return

    if args.stage == "train_manager_sft":
        result = stages.run_train_manager_sft(
            ctx=ctx,
            train_jsonl=(args.manager_sft_train_jsonl or None),
            init_model_or_adapter=(args.manager_sft_init_adapter or None),
            output_dir=(args.manager_sft_output_dir or None),
            epochs=args.manager_sft_epochs,
            lr=args.manager_sft_lr,
            max_seq_len=args.sft_max_seq_len,
            per_device_batch_size=args.sft_bs,
            gradient_accumulation_steps=args.sft_grad_accum,
            use_lora=(not args.sft_no_lora),
            max_steps=args.sft_max_steps,
        )
        print("[TRAIN_MGR_SFT]", result)
        return

    if args.stage == "evolve_round":
        data = _load_benchmark_splits(args)
        train_rows = _exclude_sft_rows(data["train"], args.exclude_sft_example_ids)
        grpo_kwargs = dict(
            manager_adapter=(args.mgr_init_adapter or None),
            per_device_batch_size=args.mgr_bs,
            max_completion_length=args.mgr_max_completion_length,
            temperature=args.mgr_temperature,
            num_generations=args.mgr_num_generations,
            grpo_beta=args.mgr_grpo_beta,
            routing_efficiency_bonus=args.mgr_routing_efficiency_bonus,
            tool_use_bonus=args.mgr_tool_use_bonus,
            ccr_mode=args.mgr_ccr_mode,
            ccr_p_high=args.mgr_ccr_p_high,
            ccr_p_low=args.mgr_ccr_p_low,
            ccr_k_max=args.mgr_ccr_k_max,
            adc_mode=args.mgr_adc_mode,
            adc_cost_per_tool=args.mgr_adc_cost_per_tool,
            adc_draft_bonus=args.mgr_adc_draft_bonus,
            adc_missing_draft_penalty=args.mgr_adc_missing_draft_penalty,
            adc_final_bonus=args.mgr_adc_final_bonus,
            adc_variant=args.mgr_adc_variant,
            full_parameter_rl=args.mgr_full_parameter_rl,
            max_steps=args.mgr_max_steps,
            output_dir=(args.mgr_output_dir or None),
            use_wandb=args.mgr_use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            task_description=args.task_description,
            exploration_hint=args.mgr_exploration_hint,
            clip_epsilon_high=args.mgr_clip_epsilon_high,
        )
        evolve_kwargs = dict(
            teacher_provider=(args.teacher_provider or None),
            teacher_model=(args.teacher_model or None),
            max_fail_samples=args.evolve_max_fail_samples,
            task_description=args.task_description,
        )
        sft_kwargs = dict(
            epochs=args.manager_sft_epochs,
            lr=args.manager_sft_lr,
            max_seq_len=args.sft_max_seq_len,
            per_device_batch_size=args.sft_bs,
            gradient_accumulation_steps=args.sft_grad_accum,
            use_lora=(not args.sft_no_lora),
            max_steps=args.sft_max_steps,
        )
        result = stages.run_evolve_round(
            ctx=ctx, train_rows=train_rows, full_rows=data["all"],
            grpo_kwargs=grpo_kwargs, evolve_kwargs=evolve_kwargs, sft_kwargs=sft_kwargs,
        )
        print("[EVOLVE_ROUND]", result)
        return

    if args.stage == "eval_subagents":
        data = _load_benchmark_splits(args)
        kinds = [k.strip() for k in args.eval_kinds.split(",") if k.strip()]
        result = stages.run_eval_subagents(
            ctx=ctx, rows=data["dev"] or data["test"], agent_kinds=kinds,
            n_samples=args.eval_n_samples,
        )
        print("[EVAL_SUBAGENTS]", result["by_agent"])
        return

    if args.stage == "eval_manager":
        result = stages.run_eval_manager(
            ctx=ctx, rows=_load_eval_rows(args),
            manager_dir=(args.eval_manager_dir or None),
            n_samples=args.eval_n_samples,
            temperature=args.eval_temperature,
            max_new_tokens=args.eval_max_new_tokens,
            task_description=args.task_description,
            sc_k=args.eval_sc_k,
            sc_temperature=args.eval_sc_temperature,
        )
        print("[EVAL_MANAGER]", result)
        return

    if args.stage == "eval_manager_forced":
        forced = [t.strip() for t in args.eval_forced_tools.split(",") if t.strip()]
        if forced == ["none"]:
            forced = []
        result = stages.run_eval_manager_forced(
            ctx=ctx, rows=_load_eval_rows(args),
            manager_dir=(args.eval_manager_dir or None),
            forced_tools=forced,
            n_samples=args.eval_n_samples,
            temperature=args.eval_temperature,
            max_new_tokens=args.eval_max_new_tokens,
            task_description=args.task_description,
            out_tag=args.eval_out_tag,
            subagent_server_url=(args.subagent_server_url or None),
        )
        print("[EVAL_MANAGER_FORCED]", result)
        return

    if args.stage == "eval_manager_tools":
        result = stages.run_eval_manager_tools(
            ctx=ctx, rows=_load_eval_rows(args),
            manager_dir=(args.eval_manager_dir or None),
            n_samples=args.eval_n_samples,
            temperature=args.eval_temperature,
            max_new_tokens=args.eval_max_new_tokens,
            max_tool_calls=args.eval_max_tool_calls,
            task_description=args.task_description,
            subagent_server_url=(args.subagent_server_url or None),
        )
        print("[EVAL_MANAGER_TOOLS]", result)
        return

    sys.exit(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
