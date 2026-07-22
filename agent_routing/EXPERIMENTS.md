# Experiment Plan — Learning When to Commit (8B, 4× GPU cluster)

> **Historical ADC plan.** The current main method uses counterfactual
> marginal-value SFT followed by binary-outcome GRPO. Follow
> [MARGINAL_VALUE_EXPERIMENTS.md](MARGINAL_VALUE_EXPERIMENTS.md) for all new
> runs. Keep the ADC commands below only for failure-analysis ablations.

Full execution plan for the three research questions:

- **RQ1** Does delegation buy accuracy — and is it structure or distillation?
- **RQ2** Is the learned policy a genuine stopping policy? (Pareto / regret-vs-oracle / zero-shot difficulty adaptation)
- **RQ3** Is incentive compatibility necessary? (reward ablation arms + sandbagging curve)

> **This file supersedes README.md and EXPERIMENTS_LEGACY.md** (both describe
> the old `rule_applier` pipeline). The `src/` tree here is the 2026-07-03
> sync: verifier rename, anytime ADC reward + ablation variants, draft-aware
> cold start, verifier `current_draft` chain, GPQA dedup, `question_hash`
> exclusion, forced-sequence eval (`eval_manager_forced`), self-consistency
> eval (`--eval_sc_k`).

---

## 0. Migration notes (read once before touching the cluster)

The code is NOT backward-compatible with artifacts from the old 6.8 snapshot:

| Artifact on cluster | Reusable? | Why |
|---|---|---|
| `outputs/data/medqa_us4_normalized.jsonl` | **yes** | loader unchanged |
| extractor SFT jsonl + adapter | acceptable | prompt/schema unchanged; rows lack `question_hash` — exclusion falls back to `example_id`, fine while the cache file is unchanged |
| reasoner SFT jsonl + adapter | **regenerate** | schema constraints changed since 6.8 |
| `rule_applier` SFT jsonl + adapter | **regenerate as `verifier`** | renamed + new `candidate_answer_audit` field + candidate sampling |
| manager cold-start data + adapter | **regenerate** | old format has NO `DRAFT_ANSWER_` demonstrations — it actively trains drafts away and starves the ADC reward |
| GRPO checkpoints | discard | reward changed |
| flags `--mgr_adc_correction_bonus` / `--mgr_adc_corruption_penalty` | removed | replaced by `--mgr_adc_draft_bonus`, `--mgr_adc_missing_draft_penalty`, `--mgr_adc_variant` |

Remote-specific bits preserved in this folder: `.venv` pinning in
`scripts/train_manager_grpo_multigpu.sh` (`/home/yizzhao/research_0703/.venv`),
`--override-generation-config '{"enable_thinking": false}'` in
`scripts/start_subagent_server.sh`, and the ZeRO **stage-2** accelerate config
that ran on the cluster (`configs/accelerate_zero3.yaml`; set `zero_stage: 3`
back only if stage 2 OOMs).

---

## 1. Hardware layout and environments

```
GPU 0  →  vLLM server: base model + 3 LoRA advisors (~27 GB)
GPU 1  ┐
GPU 2  ├→ accelerate + DeepSpeed (manager GRPO, full-parameter)
GPU 3  ┘
```

```bash
# training env (once)
source /home/yizzhao/research_0703/.venv/bin/activate
pip install -r requirements.txt accelerate deepspeed

# vLLM env (once, separate to avoid version conflicts)
conda create -n vllm_env python=3.11 -y && conda activate vllm_env && pip install vllm
```

Global config used by every command below (put in an `env.sh` and `source` it):

```bash
export PYTHONUTF8=1
export BASE_MODEL=Qwen/Qwen3-8B
export TEACHER_ID=commit_gpt_8b            # namespaces ALL outputs of this run family
export PROVIDER=openai
export MODEL=gpt-4o
export OPENAI_API_KEY=...
export MEDQA_CACHE=outputs/data/medqa_us4_normalized.jsonl
export TASK_DESC="You are a manager agent solving USMLE-style medical multiple-choice questions."
# split budget: 3x500 advisor SFT + 300 cold start + ~600 GRPO, all inside the train pool
export SPLIT="--train_size 1400 --dev_size 200 --test_size 500"
```

> Offline-teacher alternative: every `synth_subagent` below has an
> `export_deepseek_jsonl` → (run batch) → `import_deepseek_jsonl` equivalent.
> This plan uses the online teacher for brevity.

---

## 1.5 Benchmark roles and data budgets

One benchmark trains; the other three form the zero-shot difficulty gradient.
This is the standard "train one domain, probe a gradient" design — it maximizes
what the four benchmarks can jointly prove (transferable stopping) instead of
running four cramped in-domain pipelines.

| Benchmark | Role | Available pool | Why this role |
|---|---|---|---|
| **MedQA** | training domain (full pipeline) | train 10,178 / test 1,273 | only pool large enough for disjoint SFT/cold-start/GRPO/eval splits |
| **LegalBench** (5 tasks) | zero-shot probe, easy end | ~380 test rows total | too small to train honestly (3×SFT + coldstart + GRPO would overlap); binary labels = high base accuracy → expect lowest delegation depth |
| **MMLU-Pro** | zero-shot probe, harder (10 options) | 12,032 test | community evaluates on the full test split — training on any part breaks comparability |
| **GPQA-Diamond** | zero-shot probe, hardest | 198 | too small to train; expect max delegation depth |
| GPQA train pool | *optional* in-domain hard replication (§8.4) | 446 = 98 diamond + 348 non-diamond (`scripts/build_gpqa_splits.py`) | eval = 100 held-out diamond; leftover diamond goes INTO training — the hardest questions teach deep delegation |

**MedQA budget** (all training draws come from a fixed 1,400-row window of the
train split; dev is carved from the train tail — never from test):

| Pool | Rows | Disjointness |
|---|---|---|
| Advisor SFT (per kind ×3) | 500 accepted | the 3 kinds may share questions with each other |
| Manager cold-start | 300 | excluded from advisor-SFT ids/hashes |
| Manager GRPO | ≈600 | excluded from advisor-SFT ∪ cold-start |
| Dev (sanity only) | 200 | carved from train tail |
| Test (dev-phase) | 500 | HF test split |
| Test (final paper numbers) | 1,273 (full) | rerun Stage D + §7 rows once at the end |

**Zero-shot probe budgets** (eval only, `--train_size 0` — the split code now
honors loader labels, so the FULL probe is evaluated):

| Probe | Eval rows | Command sizes |
|---|---|---|
| LegalBench 5 tasks | ~380 (all; report per task) | `--train_size 0 --dev_size 0 --test_size 400` |
| MMLU-Pro | 500 sampled from 12k | `--train_size 0 --dev_size 0 --test_size 500` |
| GPQA-Diamond | 198 (all) | `--train_size 0 --dev_size 0 --test_size 198` |

## 1.6 Hyperparameter defaults (8B, experience-based)

| Stage | Setting | Value | Rationale |
|---|---|---|---|
| Synthesis | temperature / retries / workers | 0.4 / 2 / 8 | schema-validity sweet spot; retries bump temp +0.15 |
| Advisor SFT | lr / epochs / LoRA / batch | 5e-5 / 3 / r16 α32 / bs1×ga8, seq 4096 | lr ≥ 1e-4 destabilizes 8B; 500×3ep ≈ 190 updates |
| Cold start | n / epochs / lr | 300 / 2 / 5e-6 | format only; 1 epoch sometimes leaves `tools/call_frequency≈0`, 2 is safer at this lr |
| GRPO (main) | bs / gens / temp / completion | 2×3GPU / **6** / 1.0 / 3072 | gens must divide global batch (6); 3072 fits 3-advisor trajectories |
| GRPO (main) | beta / clip_high / steps | 0.01 / 0.28 / 300 | DAPO clip-higher for exploration; 300 steps = 300 questions × 6 rollouts ≈ ½ pass over 600 |
| ADC reward | draft / missing / final / cost | 0.2 / 0.1 / 1.0 / 0.05 | verified ordering: k=0 1.20 > k=2 honest 1.10 > k=3 1.05 > sandbag 1.03; drafts 1.10 > omitted 0.90 |
| Pareto grid | cost_per_tool | 0.02 / 0.05 / 0.10 / 0.20 | spans "almost free" → "one call must flip ~⅕ of answers to pay off" |
| GPQA in-domain (§8.4) | steps / beta | 120 / 0.02 | ~196 GRPO rows ⇒ ~4 passes; stronger KL against overfitting |
| Eval | decoding | greedy, max_new 1024 | deterministic; SC baseline uses temp 0.7 |

---

## 2. One-time data preparation

```bash
# MedQA (training benchmark)
python -m src.pipeline.cli load_medqa --base_model "$BASE_MODEL" \
    --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT

# LegalBench (easy-end probe; loads + caches on first use)
python -m src.pipeline.cli eval_manager --help >/dev/null  # (no dedicated load stage; the
# eval command in §8.3 passes --legalbench_configs and caches to the jsonl below)

# MMLU-Pro (zero-shot transfer probe ONLY — never train on it)
python -m src.pipeline.cli load_mmlu_pro --base_model "$BASE_MODEL" \
    --mmlu_pro_normalized_cache outputs/data/mmlu_pro_normalized.jsonl \
    --mmlu_pro_splits test --train_size 0 --dev_size 0 --test_size 500

# GPQA-Diamond (hard-end probe; all 198 rows are used — the split code honors
# loader labels when --train_size 0). Gated: accept terms on HF + login first.
python -m src.pipeline.cli load_gpqa --base_model "$BASE_MODEL" \
    --gpqa_subsets gpqa_diamond \
    --gpqa_normalized_cache outputs/data/gpqa_diamond_normalized.jsonl \
    --train_size 0 --dev_size 0 --test_size 198

# GPQA train/eval split for the §8.4 in-domain replication. GPQA is 546
# questions TOTAL (nested: diamond 198 ⊆ main 448 ⊆ extended 546). This holds
# out 100 diamond questions for eval and pools the remaining 98 diamond +
# 348 non-diamond = 446 for training (hash-disjoint, asserted):
python scripts/build_gpqa_splits.py --eval_n 100 --seed 42
# -> outputs/data/gpqa_diamond_eval100.jsonl  (100 rows, split=test)
# -> outputs/data/gpqa_train446.jsonl         (446 rows, split=train)
```

---

## 3. Stage A — build the three frozen advisors

Synthesis (~30–60 min per kind at 8 workers). The verifier automatically audits
a random candidate on ~50% of samples; `--synth_symmetric_leakage` audits ALL
choice texts, closing the negative-space leakage bias:

```bash
for KIND in extractor reasoner verifier; do
  python -m src.pipeline.cli synth_subagent \
      --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
      --teacher_provider "$PROVIDER" --teacher_model "$MODEL" \
      --agent_kind "$KIND" --n_samples 500 \
      --synth_symmetric_leakage \
      --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT \
      --task_description "$TASK_DESC"
done
# gate: outputs/sft_data/$TEACHER_ID/*_sft.jsonl.meta.json — if leakage_fail > 20%,
# lower --synth_temperature or switch teacher before spending GPU time.
```

LoRA SFT (1 GPU each, ~1–2 h each on H100; parallelize across GPUs 1–3):

```bash
for KIND in extractor reasoner verifier; do
  python -m src.pipeline.cli train_subagent \
      --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
      --agent_kind "$KIND" \
      --sft_epochs 3 --sft_lr 5e-5 --sft_bs 1 --sft_grad_accum 8
done
```

Gate before proceeding (`json_ok_rate` and `schema_ok_rate` > 0.9):

```bash
python -m src.pipeline.cli eval_subagents \
    --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
    --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT --eval_n_samples 50
```

---

## 4. Stage B — manager cold start (now demonstrates drafts)

Every tool-calling turn in the generated SFT data carries `DRAFT_ANSWER_<K>`
content, verifier calls carry `current_draft`, and the final turn is
`DRAFT_ANSWER_<K>\nANSWER_<K>`. Format teaching only — small LR:

```bash
python -m src.pipeline.cli manager_coldstart_sft \
    --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
    --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT \
    --exclude_sft_example_ids "outputs/sft_data/${TEACHER_ID}/extractor_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TEACHER_ID}/reasoner_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TEACHER_ID}/verifier_sft.jsonl" \
    --coldstart_n_samples 300 --coldstart_force_diverse \
    --manager_sft_epochs 2 --manager_sft_lr 5e-6 \
    --task_description "$TASK_DESC"
# adapter -> outputs/manager/$TEACHER_ID/sft_coldstart
# sanity: grep -c DRAFT_ANSWER outputs/manager/$TEACHER_ID/evolve/manager_sft_coldstart_diverse.jsonl   # >> 0
```

---

## 5. Stage C — GRPO with the anytime ADC reward (the main run)

Terminal A (GPU 0, keep alive for the whole training):

```bash
conda activate vllm_env
bash scripts/start_subagent_server.sh "$BASE_MODEL" "$TEACHER_ID"
curl http://localhost:8000/v1/models | python -m json.tool   # expect: extractor, reasoner, verifier
```

Terminal B (GPUs 1–3). The launch script pins the remote venv and sets
bs 2 × 3 GPUs = global batch 6 = `num_generations` (1 question × 6 rollouts
per step; completion budget 3072 for 3-advisor trajectories):

```bash
EXCL="--exclude_sft_example_ids outputs/sft_data/${TEACHER_ID}/extractor_sft.jsonl \
      --exclude_sft_example_ids outputs/sft_data/${TEACHER_ID}/reasoner_sft.jsonl \
      --exclude_sft_example_ids outputs/sft_data/${TEACHER_ID}/verifier_sft.jsonl \
      --exclude_sft_example_ids outputs/manager/${TEACHER_ID}/evolve/manager_sft_coldstart_diverse.jsonl"

bash scripts/train_manager_grpo_multigpu.sh "$TEACHER_ID" \
    --base_model "$BASE_MODEL" \
    --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT \
    $EXCL \
    --mgr_init_adapter "outputs/manager/${TEACHER_ID}/sft_coldstart" \
    --mgr_output_dir "outputs/manager/${TEACHER_ID}/grpo_anytime_c05" \
    --mgr_adc_mode --mgr_adc_variant anytime \
    --mgr_adc_draft_bonus 0.2 --mgr_adc_missing_draft_penalty 0.1 \
    --mgr_adc_final_bonus 1.0 --mgr_adc_cost_per_tool 0.05 \
    --mgr_clip_epsilon_high 0.28 \
    --mgr_max_steps 300 \
    --mgr_use_wandb --wandb_project agent_routing \
    --wandb_run_name "${TEACHER_ID}_anytime_c05" \
    --task_description "$TASK_DESC"
```

Health checks in the first 30 minutes:

```bash
# 1) tools are being called (cold-start adapter loaded)
grep -o '"tool_calls": [0-9]' outputs/manager/$TEACHER_ID/grpo_anytime_c05/train_raw_trace.jsonl | sort | uniq -c

# 2) drafts are being emitted (the ADC signal exists) — expect > 0.8
python - <<'PY'
import os
from src.utils.io import read_jsonl
r = read_jsonl(f"outputs/manager/{os.environ['TEACHER_ID']}/grpo_anytime_c05/train_raw_trace.jsonl")
adc = [x for x in r if str(x.get("reward_mode","")).startswith("adc")]
print("pct_with_drafts =", sum(1 for x in adc if x.get("n_drafts",0)>0)/max(1,len(adc)))
PY
# if ~0: the cold-start adapter was not loaded — check --mgr_init_adapter path.
```

~300 steps ≈ 6–10 h on 3× H100 (advisor outputs are cached per question, so
rollouts 2–6 on the same question pay no advisor cost).

---

## 6. Stage D — main evaluation (the learned orchestrator)

```bash
python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
    --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT \
    --eval_n_samples 500 \
    --eval_manager_dir "outputs/manager/${TEACHER_ID}/grpo_anytime_c05" \
    --task_description "$TASK_DESC"
# -> outputs/eval/$TEACHER_ID/manager_tool_eval_report.json (accuracy, avg_tool_calls, tool_counts)
```

---

## 7. RQ1 — the main table (six rows, MedQA test 500)

| # | Row | How |
|---|---|---|
| 1 | 8B zero-shot CoT (floor) | (a) |
| 2 | 8B + self-consistency@4 (matched-compute control) | (b) |
| 3 | 8B + direct CoT distillation (distillation-confound control) | (c) |
| 4 | 8B + all three advisors, forced (fixed-depth structure ceiling) | (d) |
| 5 | 8B + learned orchestrator | Stage D |
| 6 | Frontier reference (GPT-4o direct) | one-off API script, or cite reported numbers |

**(a) zero-shot floor** — note: needs a local snapshot of the base model
(eval loads `--eval_manager_dir` as a directory; `huggingface-cli download
Qwen/Qwen3-8B --local-dir models/qwen3-8b` once, or point at any full local copy):

```bash
python -m src.pipeline.cli eval_manager \
    --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
    --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT \
    --eval_n_samples 500 --eval_manager_dir models/qwen3-8b \
    --task_description "$TASK_DESC"
```

**(b) self-consistency@k.** Match its token budget to row 5's delegation
budget (avg_tool_calls × ~2k tokens per advisor round-trip; avg ≈ 1.5 → k ≈ 4):

```bash
python -m src.pipeline.cli eval_manager \
    --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
    --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT \
    --eval_n_samples 500 --eval_manager_dir models/qwen3-8b \
    --eval_sc_k 4 --eval_sc_temperature 0.7 \
    --task_description "$TASK_DESC"
# -> outputs/eval/$TEACHER_ID/manager_eval_report_sc4.json
```

**(c) distillation control** — same teacher, same question pool,
teacher-token budget matched to advisor-SFT + cold-start, distilled as plain
CoT into a single model:

```bash
# 1. build CoT prompts from the same train pool
python - <<'PY'
import os, random
from src.utils.io import read_jsonl, write_jsonl
rows = read_jsonl(os.environ["MEDQA_CACHE"])
train = [r for r in rows if r.get("split")=="train"][:1400]
random.Random(42).shuffle(train)
out=[]
for r in train[:800]:   # ~ matches the 3x500 + 300 teacher-token budget; recount after generation
    ch = "\n".join(f"  {k}. {v}" for k,v in r["choices"].items())
    out.append({"example_id": r["example_id"],
        "prompt":[{"role":"system","content":"Answer the medical MCQ. Reason step by step, then end with one line: ANSWER_<LETTER>."},
                  {"role":"user","content":f"{r['question']}\n\nChoices:\n{ch}"}],
        "ground_truth": r["ground_truth"], "choices": r["choices"]})
write_jsonl("outputs/sft_data/distill_cot_prompts.jsonl", out)
PY

# 2. generate with the SAME teacher
python scripts/generate_openai_jsonl.py \
    --input outputs/sft_data/distill_cot_prompts.jsonl \
    --output outputs/sft_data/distill_cot_responses.jsonl --model "$MODEL"

# 3. convert to manager-SFT rows (keep responses whose final line parses AND is correct)
python - <<'PY'
from src.utils.io import read_jsonl, write_jsonl
from src.manager.prompt import parse_final_answer
P={r["example_id"]:r for r in read_jsonl("outputs/sft_data/distill_cot_prompts.jsonl")}
out=[]
for resp in read_jsonl("outputs/sft_data/distill_cot_responses.jsonl"):
    src=P.get(resp.get("example_id")); text=str(resp.get("response") or "")
    if not src: continue
    if parse_final_answer(text, list(src["choices"].keys())) != src["ground_truth"]: continue
    out.append({"example_id":src["example_id"],"prompt":src["prompt"],
                "response":[{"role":"assistant","content":text.strip()}]})
write_jsonl("outputs/sft_data/distill_cot_sft.jsonl", out); print(len(out),"rows kept")
PY

# 4. SFT + eval under a separate namespace
python -m src.pipeline.cli train_manager_sft \
    --base_model "$BASE_MODEL" --teacher_id "${TEACHER_ID}_distill" \
    --manager_sft_train_jsonl outputs/sft_data/distill_cot_sft.jsonl \
    --manager_sft_epochs 2 --manager_sft_lr 1e-5
python -m src.pipeline.cli eval_manager \
    --base_model "$BASE_MODEL" --teacher_id "${TEACHER_ID}_distill" \
    --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT --eval_n_samples 500 \
    --eval_manager_dir "outputs/manager/${TEACHER_ID}_distill/sft_evolved" \
    --task_description "$TASK_DESC"
```

**(d) fixed all-three baseline** — forced delegation; the manager only writes
the final answer given all three advisor outputs:

```bash
python -m src.pipeline.cli eval_manager_forced \
    --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
    --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT --eval_n_samples 500 \
    --eval_manager_dir "outputs/manager/${TEACHER_ID}/grpo_anytime_c05" \
    --eval_forced_tools "extractor,reasoner,verifier" \
    --task_description "$TASK_DESC"
```

**Success criteria**: row5 ≥ row4 at ~half the delegations; row5 > row2 and
row3 (structure beats matched compute AND matched distillation); row5
approaches row6.

---

## 8. RQ2 — stopping quality

### 8.1 Fixed-k baselines + the stopping oracle (Fig. B; also the k-points of Fig. A)

Run the forced eval over all 8 advisor subsets (advisor outputs are cached —
after the first pass each run only pays manager generation):

```bash
for SEQ in none extractor reasoner verifier \
           "extractor,reasoner" "reasoner,verifier" "extractor,verifier" \
           "extractor,reasoner,verifier"; do
  python -m src.pipeline.cli eval_manager_forced \
      --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
      --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT --eval_n_samples 500 \
      --eval_manager_dir "outputs/manager/${TEACHER_ID}/grpo_anytime_c05" \
      --eval_forced_tools "$SEQ" \
      --task_description "$TASK_DESC"
done
```

Oracle + regret (offline, no GPU):

```bash
python - <<'PY'
import glob, os
from src.utils.io import read_jsonl
COST = 0.05
eval_dir = f"outputs/eval/{os.environ['TEACHER_ID']}"
oracle = {}
for f in glob.glob(f"{eval_dir}/manager_forced_*.jsonl"):
    for r in read_jsonl(f):
        rew = (1.0 if r["correct"] else 0.0) - COST * r["tool_calls"]
        eid = r["example_id"]
        if eid not in oracle or rew > oracle[eid]:
            oracle[eid] = rew
learned = {r["example_id"]: (1.0 if r["correct"] else 0.0) - COST * r["tool_calls"]
           for r in read_jsonl(f"{eval_dir}/manager_tool_eval.jsonl")}
common = sorted(set(oracle) & set(learned))
regret = sum(oracle[e] - learned[e] for e in common) / len(common)
print(f"n={len(common)}  oracle={sum(oracle[e] for e in common)/len(common):.4f}  "
      f"learned={sum(learned[e] for e in common)/len(common):.4f}  REGRET={regret:.4f}")
PY
```

### 8.2 Cost–accuracy Pareto family (Fig. A)

Retrain the main run at four prices (everything else identical; ≈ one Stage C
runtime each — schedule sequentially, or halve `--mgr_max_steps` first pass):

```bash
for C in 0.02 0.05 0.10 0.20; do
  TAG=$(echo $C | tr -d '.')
  bash scripts/train_manager_grpo_multigpu.sh "$TEACHER_ID" \
      --base_model "$BASE_MODEL" \
      --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT $EXCL \
      --mgr_init_adapter "outputs/manager/${TEACHER_ID}/sft_coldstart" \
      --mgr_output_dir "outputs/manager/${TEACHER_ID}/grpo_anytime_c${TAG}" \
      --mgr_adc_mode --mgr_adc_variant anytime --mgr_adc_cost_per_tool $C \
      --mgr_clip_epsilon_high 0.28 --mgr_max_steps 300 \
      --mgr_use_wandb --wandb_project agent_routing \
      --wandb_run_name "${TEACHER_ID}_anytime_c${TAG}" \
      --task_description "$TASK_DESC"
  python -m src.pipeline.cli eval_manager_tools \
      --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
      --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT --eval_n_samples 500 \
      --eval_manager_dir "outputs/manager/${TEACHER_ID}/grpo_anytime_c${TAG}" \
      --task_description "$TASK_DESC"
  cp outputs/eval/$TEACHER_ID/manager_tool_eval_report.json \
     outputs/eval/$TEACHER_ID/pareto_c${TAG}.json
done
# plot: x = avg_tool_calls, y = accuracy; overlay the fixed-k points from §8.1.
# expectation: the policy family dominates the fixed-k hull and shifts LEFT as C grows.
```

### 8.3 Zero-shot difficulty adaptation (Fig. C) + AURC

One MedQA-trained policy, four benchmarks, no retraining. Copy the two eval
output files to a per-benchmark name after each run (they overwrite):

```bash
MGR="outputs/manager/${TEACHER_ID}/grpo_anytime_c05"

# LegalBench (easy end) — eval-only: all ~380 rows across the 5 tasks
python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
    --legalbench_configs "abercrombie,hearsay,personal_jurisdiction,proa,successor_liability" \
    --legalbench_normalized_cache outputs/data/legalbench_5tasks.jsonl \
    --train_size 0 --dev_size 0 --test_size 400 --eval_n_samples 400 \
    --eval_manager_dir "$MGR" --task_description "$TASK_DESC"
for f in manager_tool_eval.jsonl manager_tool_eval_report.json; do
  cp outputs/eval/$TEACHER_ID/$f outputs/eval/$TEACHER_ID/transfer_legalbench_$f; done

# MedQA (in-domain): Stage D outputs — copy the same way (transfer_medqa_*)

# MMLU-Pro (harder, 10 options)
python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
    --mmlu_pro_normalized_cache outputs/data/mmlu_pro_normalized.jsonl \
    --train_size 0 --dev_size 0 --test_size 500 --eval_n_samples 500 \
    --eval_manager_dir "$MGR" --task_description "$TASK_DESC"
for f in manager_tool_eval.jsonl manager_tool_eval_report.json; do
  cp outputs/eval/$TEACHER_ID/$f outputs/eval/$TEACHER_ID/transfer_mmlupro_$f; done

# GPQA-Diamond (hardest)
python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
    --gpqa_normalized_cache outputs/data/gpqa_diamond_normalized.jsonl \
    --train_size 0 --dev_size 0 --test_size 198 --eval_n_samples 198 \
    --eval_manager_dir "$MGR" --task_description "$TASK_DESC"
for f in manager_tool_eval.jsonl manager_tool_eval_report.json; do
  cp outputs/eval/$TEACHER_ID/$f outputs/eval/$TEACHER_ID/transfer_gpqa_$f; done
```

AURC per benchmark:

```bash
python - <<'PY'
import glob, os
from src.utils.io import read_jsonl
from src.manager.reward import compute_risk_coverage
for f in sorted(glob.glob(f"outputs/eval/{os.environ['TEACHER_ID']}/transfer_*_manager_tool_eval.jsonl")):
    recs = read_jsonl(f)
    rc = compute_risk_coverage(recs)
    avg_k = sum(r["tool_calls"] for r in recs)/max(1,len(recs))
    print(f"{os.path.basename(f):55s} avg_tools={avg_k:.2f}  AURC={rc['aurc']:.4f}")
PY
# expectation: avg_tools monotone — LegalBench < MedQA < MMLU-Pro < GPQA-Diamond.
```

### 8.4 Optional — in-domain replication on GPQA (hard domain)

Replicates RQ1 rows 4–5 and the regret analysis on a much harder domain.
Split (from `scripts/build_gpqa_splits.py`, §2): **eval = 100 held-out diamond
questions; train pool = 446** (98 leftover diamond + 348 non-diamond). Budget
inside the 446: advisor SFT 160 (the three kinds share rows), cold start 50,
dev 40, GRPO ≈ 196 (the remainder, enforced by hash exclusions).

> The GPQA-trained manager is evaluated on `gpqa_diamond_eval100` ONLY — 98
> diamond questions sit in its training pool. The full-198 diamond probe in
> §8.3 remains valid for the *MedQA-trained* manager, which never saw GPQA.

```bash
export GPQA_ID=ds_gpqa
export GPQA_TASK="You are a manager agent solving expert-level graduate science multiple-choice questions."
export GPQA_TRAIN="--gpqa_normalized_cache outputs/data/gpqa_train446.jsonl --train_size 0 --dev_size 40 --test_size 0"

# 1) advisor SFT data via the offline DeepSeek bridge (160 per kind)
for KIND in extractor reasoner verifier; do
  python -m src.pipeline.cli export_deepseek_jsonl       --agent_kind "$KIND" --teacher_id "$GPQA_ID" $GPQA_TRAIN --n_samples 160
done
#   ...run DeepSeek on outputs/sft_data/$GPQA_ID/*_deepseek_prompts.jsonl...
for KIND in extractor reasoner verifier; do
  python -m src.pipeline.cli import_deepseek_jsonl       --agent_kind "$KIND" --teacher_id "$GPQA_ID"       --deepseek_prompt_jsonl  "outputs/sft_data/${GPQA_ID}/${KIND}_deepseek_prompts.jsonl"       --deepseek_response_jsonl "outputs/sft_data/${GPQA_ID}/${KIND}_deepseek_responses.jsonl"
  python -m src.pipeline.cli train_subagent       --base_model "$BASE_MODEL" --teacher_id "$GPQA_ID" --agent_kind "$KIND"       --sft_epochs 4 --sft_lr 5e-5 --sft_bs 1 --sft_grad_accum 8
done

# 2) cold start (50 rows, hash-excluded from advisor rows)
GPQA_EXCL="--exclude_sft_example_ids outputs/sft_data/${GPQA_ID}/extractor_sft.jsonl            --exclude_sft_example_ids outputs/sft_data/${GPQA_ID}/reasoner_sft.jsonl            --exclude_sft_example_ids outputs/sft_data/${GPQA_ID}/verifier_sft.jsonl"
python -m src.pipeline.cli manager_coldstart_sft     --base_model "$BASE_MODEL" --teacher_id "$GPQA_ID" $GPQA_TRAIN $GPQA_EXCL     --coldstart_n_samples 50 --coldstart_force_diverse     --manager_sft_epochs 2 --manager_sft_lr 5e-6     --task_description "$GPQA_TASK"

# 3) GRPO on the ~196 remaining rows
#    (terminal A now serves the GPQA advisors:
#     bash scripts/start_subagent_server.sh "$BASE_MODEL" "$GPQA_ID")
bash scripts/train_manager_grpo_multigpu.sh "$GPQA_ID"     --base_model "$BASE_MODEL" $GPQA_TRAIN $GPQA_EXCL     --exclude_sft_example_ids "outputs/manager/${GPQA_ID}/evolve/manager_sft_coldstart_diverse.jsonl"     --mgr_init_adapter "outputs/manager/${GPQA_ID}/sft_coldstart"     --mgr_output_dir "outputs/manager/${GPQA_ID}/grpo_anytime_c05"     --mgr_adc_mode --mgr_adc_variant anytime --mgr_adc_cost_per_tool 0.05     --mgr_grpo_beta 0.02 --mgr_clip_epsilon_high 0.28 --mgr_max_steps 120     --mgr_use_wandb --wandb_project agent_routing     --wandb_run_name "${GPQA_ID}_anytime_c05"     --task_description "$GPQA_TASK"

# 4) eval STRICTLY on the 100 held-out diamond questions
python -m src.pipeline.cli eval_manager_tools     --base_model "$BASE_MODEL" --teacher_id "$GPQA_ID"     --gpqa_normalized_cache outputs/data/gpqa_diamond_eval100.jsonl     --train_size 0 --dev_size 0 --test_size 100 --eval_n_samples 100     --eval_manager_dir "outputs/manager/${GPQA_ID}/grpo_anytime_c05"     --task_description "$GPQA_TASK"
```

Cheaper alternative: skip step 1 and transfer the MedQA advisors + cold start
(`--subagent_teacher_id "$TEACHER_ID"`, `--mgr_init_adapter` from the MedQA
cold start, server serving the MedQA adapters); then everything but dev goes
to GRPO (~406 rows, `--mgr_max_steps 200`).

---

## 9. RQ3 — incentive-compatibility ablation (reward arms + sandbagging curve)

Same data, same cold start, same steps; only the reward changes. Arm 1
(anytime) is the Stage C run, reused.

```bash
# arm 2: binary outcome only
bash scripts/train_manager_grpo_multigpu.sh "$TEACHER_ID" \
    --base_model "$BASE_MODEL" --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT $EXCL \
    --mgr_init_adapter "outputs/manager/${TEACHER_ID}/sft_coldstart" \
    --mgr_output_dir "outputs/manager/${TEACHER_ID}/grpo_binary" \
    --mgr_max_steps 300 --mgr_use_wandb --wandb_project agent_routing \
    --wandb_run_name "${TEACHER_ID}_binary" --task_description "$TASK_DESC"

# arm 3: CCR log-scoring (legacy depth-as-confidence; p_low > 0.5 required)
bash scripts/train_manager_grpo_multigpu.sh "$TEACHER_ID" \
    --base_model "$BASE_MODEL" --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT $EXCL \
    --mgr_init_adapter "outputs/manager/${TEACHER_ID}/sft_coldstart" \
    --mgr_output_dir "outputs/manager/${TEACHER_ID}/grpo_ccr" \
    --mgr_ccr_mode --mgr_ccr_p_high 0.9 --mgr_ccr_p_low 0.6 \
    --mgr_max_steps 300 --mgr_use_wandb --wandb_project agent_routing \
    --wandb_run_name "${TEACHER_ID}_ccr" --task_description "$TASK_DESC"

# arm 4: transition reward — the sandbagging exploit (0.5 = the old default coefficient)
bash scripts/train_manager_grpo_multigpu.sh "$TEACHER_ID" \
    --base_model "$BASE_MODEL" --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT $EXCL \
    --mgr_init_adapter "outputs/manager/${TEACHER_ID}/sft_coldstart" \
    --mgr_output_dir "outputs/manager/${TEACHER_ID}/grpo_transition" \
    --mgr_adc_mode --mgr_adc_variant transition --mgr_adc_draft_bonus 0.5 \
    --mgr_adc_cost_per_tool 0.05 \
    --mgr_max_steps 300 --mgr_use_wandb --wandb_project agent_routing \
    --wandb_run_name "${TEACHER_ID}_transition" --task_description "$TASK_DESC"

# arm 5 (optional): summed draft bonus — the farming exploit
#   ... --mgr_adc_variant sum --mgr_adc_draft_bonus 0.2 --mgr_adc_cost_per_tool 0.05 \
#       --mgr_output_dir outputs/manager/${TEACHER_ID}/grpo_sum
```

**The headline curve — sandbagging rate over training** (run per arm):

```bash
python - outputs/manager/$TEACHER_ID/grpo_transition/train_raw_trace.jsonl <<'PY'
import sys
from src.utils.io import read_jsonl
recs = [r for r in read_jsonl(sys.argv[1]) if r.get("y_hat_seq")]
W = 200  # sliding window in completions (6 completions per training step)
for i in range(0, max(1, len(recs)-W), W):
    win = recs[i:i+W]
    sb  = sum(1 for r in win
              if len(r["y_hat_seq"]) >= 2
              and r["y_hat_seq"][0] is not None
              and r["y_hat_seq"][0] != r["ground_truth"]
              and r.get("correct"))
    dr  = sum(1 for r in win if r.get("n_drafts", 0) > 0)
    print(f"step~{i//6:5d}  sandbag_rate={sb/len(win):.3f}  draft_rate={dr/len(win):.3f}")
PY
# expectation: the transition arm's sandbag_rate CLIMBS during training; the anytime
# arm stays near the model's natural zero-shot error rate.
# Complete the RQ3 table: run Stage D + the §8.1 regret snippet for every arm.
```

---

## 10. Robustness / reporting checklist

- **3 seeds** for Stage C and each RQ3 arm: add `--seed 43` / `--seed 44` and a
  `_s43` suffix on `--mgr_output_dir` / `--wandb_run_name`.
- **Paired tests**: accuracy deltas on the same 500 questions → McNemar; regret
  and AURC → paired bootstrap over questions.
- **Weak-teacher ablation** (RQ1 credibility): rerun Stage A+B with
  `--teacher_provider deepseek --teacher_model deepseek-chat --teacher_id
  commit_ds_8b`, then Stage C unchanged. RQ2/RQ3 conclusions must be
  teacher-invariant; RQ1 absolute numbers may drop — report honestly.
- **LegalBench**: report per-task (filter the eval jsonl by `task_subtype`).

### Where every paper number lives

| Paper item | File |
|---|---|
| RQ1 table rows 1–3 | `outputs/eval/<id>/manager_eval_report*.json` |
| RQ1 row 4 / fixed-k points | `outputs/eval/<id>/manager_forced_*_report.json` |
| RQ1 row 5 | `outputs/eval/<id>/manager_tool_eval_report.json` |
| Fig. A Pareto | `outputs/eval/<id>/pareto_c*.json` + forced reports |
| Fig. B regret | §8.1 snippet over `manager_forced_*.jsonl` + `manager_tool_eval.jsonl` |
| Fig. C transfer + AURC | `outputs/eval/<id>/transfer_*_manager_tool_eval*` |
| RQ3 sandbagging curve | per-arm `train_raw_trace.jsonl` |
| Advisor quality gates | `outputs/sft_data/<id>/*_sft.jsonl.meta.json`, `outputs/eval/<id>/subagent_eval_report.json` |
