# Draft-conditioned marginal-value routing: experiment plan

This is the main protocol for the current paper. It uses a binary terminal
correctness reward for GRPO. ADC, CCR, additive per-call penalties, and generic
tool-use bonuses are not part of the main method.

The central training signal is built before GRPO: from one model-generated
`DRAFT_ANSWER`, training-time interventions compare `COMMIT` with each advisor
call. Ground truth selects a shortest trajectory that actually becomes
correct. Incorrect trajectories are never preferred merely because they use
fewer calls.

## 1. What the new stage produces

`build_marginal_sft` writes four artifacts under
`outputs/manager/$TEACHER_ID/marginal_value/`:

- `counterfactual_records.jsonl`: one record per question, containing the
  direct draft, all evaluated branches, and the preferred shortest sequence;
- `counterfactual_branches.jsonl`: one row per forced advisor sequence;
- `manager_sft_marginal.jsonl`: per-turn manager SFT data;
- `marginal_value_report.json`: direct accuracy, oracle accuracy/gain,
  rescue/corruption rates, selected depths, and advisor distribution.

The selection rule is lexicographic:

1. correct beats incorrect;
2. if direct and advisor-assisted answers are both correct, commit;
3. if direct is wrong and an advisor path is correct, call a shortest such path;
4. if no evaluated path is correct, emit no routing SFT target for that question.

## 2. Fixed experimental setup

Run commands from `agent_routing/`.

```bash
export BASE_MODEL="Qwen/Qwen3-8B"
export TEACHER_ID="medqa_marginal_v1"
export MEDQA_CACHE="outputs/data/medqa_us4_normalized.jsonl"
export TASK_DESC="You are a manager agent solving a medical multiple-choice question."
export TRAIN_SIZE=1400
export DEV_SIZE=200
export TEST_SIZE=500
export SUBAGENT_SERVER_URL="http://localhost:8000"
```

Use the same data split, seed, advisor checkpoints, manager base model, and
generation limits for every main comparison. Do not choose thresholds or
checkpoints using the test split.

Before starting, verify that these exist:

```bash
for KIND in extractor reasoner verifier; do
  test -d "outputs/adapters/$TEACHER_ID/${KIND}_adapter" || echo "missing $KIND"
done
```

If advisors are stored under a different run namespace, add
`--subagent_teacher_id "$SUBAGENT_TEACHER_ID"` to every command.

For 8B experiments, start the existing LoRA advisor server on GPU 0 before
counterfactual collection and keep the collector/manager on another GPU:

```bash
# terminal A
bash scripts/start_subagent_server.sh "$BASE_MODEL" "$TEACHER_ID"
```

In terminal B, pin single-manager collection and evaluation to another GPU:

```bash
export CUDA_VISIBLE_DEVICES=1
```

The commands below use `--subagent_server_url`. Omit that flag only for a
small-model smoke test where all three local advisor instances fit in memory.

## 3. Step 0: advisor validity and direct baseline

First evaluate advisor schema validity. A malformed advisor cannot have
learnable marginal value.

```bash
python -m src.pipeline.cli eval_subagents \
  --base_model "$BASE_MODEL" \
  --teacher_id "$TEACHER_ID" \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size "$TEST_SIZE" \
  --eval_n_samples 100
```

Record JSON and schema validity for every advisor. Do not continue if schema
validity is below 95%; inspect advisor prompts/checkpoints first.

Measure the manager's direct baseline on dev:

```bash
python -m src.pipeline.cli eval_manager \
  --base_model "$BASE_MODEL" \
  --teacher_id "$TEACHER_ID" \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size 0 \
  --eval_n_samples "$DEV_SIZE" \
  --eval_manager_dir "$BASE_MODEL" \
  --task_description "$TASK_DESC"
```

## 4. Step 1: cheap counterfactual smoke test

Start with 24 questions and one-step branches. This checks model loading,
native tool messages, answer parsing, and whether at least one advisor can
rescue a direct error.

```bash
python -m src.pipeline.cli build_marginal_sft \
  --base_model "$BASE_MODEL" \
  --teacher_id "$TEACHER_ID" \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size "$TEST_SIZE" \
  --mv_manager_dir "$BASE_MODEL" \
  --mv_n_samples 24 \
  --mv_max_depth 1 \
  --mv_max_commit_rescue_ratio 1.0 \
  --subagent_server_url "$SUBAGENT_SERVER_URL" \
  --task_description "$TASK_DESC"
```

Inspect:

```bash
python -m json.tool \
  "outputs/manager/$TEACHER_ID/marginal_value/marginal_value_report.json"
```

Smoke-test gates:

- `direct_valid_rate >= 0.95`;
- all three advisors have `n > 0`;
- `n_sft_turns > 0`;
- inspect at least ten `counterfactual_records.jsonl` rows manually;
- a rescued row must contain a wrong model draft, an actual tool output, and a
  correct model revision. It must not contain a GT answer inserted as a draft.

## 5. Step 2: measure whether useful marginal value exists

Run the one-step diagnostic on 300–500 training questions. Use deterministic
manager generation (`temperature=0`) so paired differences are attributable to
the forced advisor rather than sampling noise.

```bash
python -m src.pipeline.cli build_marginal_sft \
  --base_model "$BASE_MODEL" \
  --teacher_id "$TEACHER_ID" \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size "$TEST_SIZE" \
  --mv_manager_dir "$BASE_MODEL" \
  --mv_n_samples 400 \
  --mv_max_depth 1 \
  --mv_max_commit_rescue_ratio 1.0 \
  --mv_temperature 0 \
  --subagent_server_url "$SUBAGENT_SERVER_URL" \
  --task_description "$TASK_DESC"
```

Decision gate:

- `oracle_gain >= 0.03`: proceed to routing SFT;
- `0.01 <= oracle_gain < 0.03`: proceed as a pilot, but advisor usefulness is
  likely the main bottleneck;
- `oracle_gain < 0.01`: stop manager RL work and improve advisors. A routing
  algorithm cannot learn useful calls if forced advisors almost never repair a
  direct error;
- for each advisor, inspect `rescue_rate`, `corruption_rate`, and
  `net_marginal_rate`. A negative net advisor should not be made mandatory.

The threshold is a practical go/no-go rule, not a reported statistical claim.
Report confidence intervals in the paper.

## 6. Step 3: test multi-advisor complementarity only if needed

If one-step gain is limited but qualitative inspection suggests complementary
advisor evidence, rerun with depth 2. The collector evaluates every one-step
branch, then expands only direct-wrong questions with no one-step success.

```bash
python -m src.pipeline.cli build_marginal_sft \
  --base_model "$BASE_MODEL" \
  --teacher_id "$TEACHER_ID" \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size "$TEST_SIZE" \
  --mv_manager_dir "$BASE_MODEL" \
  --mv_n_samples 400 \
  --mv_max_depth 2 \
  --mv_max_commit_rescue_ratio 1.0 \
  --mv_temperature 0 \
  --mv_output_dir "outputs/manager/$TEACHER_ID/marginal_value_d2" \
  --subagent_server_url "$SUBAGENT_SERVER_URL" \
  --task_description "$TASK_DESC"
```

Use depth 3 only if depth 2 produces a meaningful additional oracle gain.
Compare the report's depth counts and incremental oracle gain. Do not use
depth 3 merely to create longer demonstrations.

## 7. Step 4: marginal-value SFT

Train the first routing policy from the selected counterfactual decisions.
Use the depth that passed the previous gate.

```bash
export MV_DIR="outputs/manager/$TEACHER_ID/marginal_value"
export MV_SFT="$MV_DIR/manager_sft_marginal.jsonl"
export MV_ADAPTER="outputs/manager/$TEACHER_ID/sft_marginal"

python -m src.pipeline.cli train_manager_sft \
  --base_model "$BASE_MODEL" \
  --teacher_id "$TEACHER_ID" \
  --manager_sft_train_jsonl "$MV_SFT" \
  --manager_sft_output_dir "$MV_ADAPTER" \
  --manager_sft_epochs 1 \
  --manager_sft_lr 1e-5 \
  --sft_max_seq_len 4096 \
  --sft_bs 1 --sft_grad_accum 8
```

The default commit/rescue ratio is 1:1 at the question level. Ablate
0.5, 1.0, 2.0, and `-1` (keep every direct-correct commit), but choose the main
ratio on dev only.

## 8. Step 5: evaluate the SFT routing policy before RL

```bash
python -m src.pipeline.cli eval_manager_tools \
  --base_model "$BASE_MODEL" \
  --teacher_id "$TEACHER_ID" \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size 0 \
  --eval_n_samples "$DEV_SIZE" \
  --eval_manager_dir "$MV_ADAPTER" \
  --eval_max_tool_calls 3 \
  --subagent_server_url "$SUBAGENT_SERVER_URL" \
  --task_description "$TASK_DESC"
```

The report now contains:

- `call_rate_given_draft_wrong`;
- `call_rate_given_draft_correct`;
- `draft_conditioned_call_gap`;
- `correction_rate` and `corruption_rate`;
- final accuracy and average calls.

Required behavioral gate:

- tool call rate is neither 0 nor 1;
- `draft_conditioned_call_gap > 0`;
- correction rate exceeds corruption rate;
- accuracy is not materially below the direct baseline.

If this gate fails, do not start GRPO. Inspect the counterfactual records,
change the commit/rescue ratio, or improve advisor quality.

## 9. Step 6: binary-only GRPO

Initialize GRPO from the marginal SFT adapter. Keep every auxiliary reward off.
Use a moderately stronger KL anchor than the old runs so sparse binary updates
do not immediately erase the learned routing prior.

```bash
export GRPO_DIR="outputs/manager/$TEACHER_ID/grpo_binary_marginal"

bash scripts/train_manager_grpo_multigpu.sh "$TEACHER_ID" \
  --base_model "$BASE_MODEL" \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size "$TEST_SIZE" \
  --mgr_init_adapter "$MV_ADAPTER" \
  --mgr_output_dir "$GRPO_DIR" \
  --mgr_num_generations 6 \
  --mgr_temperature 0.9 \
  --mgr_grpo_beta 0.05 \
  --mgr_max_steps 100 \
  --mgr_routing_efficiency_bonus 0 \
  --mgr_tool_use_bonus 0 \
  --mgr_clip_epsilon_high 0.28 \
  --subagent_server_url "$SUBAGENT_SERVER_URL" \
  --exclude_sft_example_ids "$MV_DIR/counterfactual_records.jsonl" \
  --mgr_use_wandb \
  --wandb_project agent_routing \
  --wandb_run_name "${TEACHER_ID}_binary_mv" \
  --task_description "$TASK_DESC"
```

Do not pass `--mgr_adc_mode` or `--mgr_ccr_mode`.

Summarize routing behavior over training windows:

```bash
python scripts/summarize_routing_trace.py \
  "$GRPO_DIR/train_raw_trace.jsonl" \
  --window 100 \
  --out "$GRPO_DIR/routing_trace_report.json"
```

Abort a run if two consecutive windows satisfy either condition:

- tool call rate `< 0.03` while the marginal oracle gain was positive;
- tool call rate `> 0.95` and corruption rate is not falling.

For the first sweep, compare `beta = {0.02, 0.05, 0.10}` and
`max_steps = {50, 100, 200}`. Select on dev accuracy first, then average calls
among configurations within the chosen accuracy tolerance.

## 10. Step 7: post-GRPO dev evaluation

```bash
python -m src.pipeline.cli eval_manager_tools \
  --base_model "$BASE_MODEL" \
  --teacher_id "$TEACHER_ID" \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size 0 \
  --eval_n_samples "$DEV_SIZE" \
  --eval_manager_dir "$GRPO_DIR" \
  --eval_max_tool_calls 3 \
  --subagent_server_url "$SUBAGENT_SERVER_URL" \
  --task_description "$TASK_DESC"
```

Compare direct, MV-SFT, and MV-SFT+binary-GRPO on:

| Model | Accuracy | Avg calls | Call rate | Call given draft wrong | Call given draft correct | Correction | Corruption |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct base | | 0 | 0 | 0 | 0 | 0 | 0 |
| Marginal SFT | | | | | | | |
| Marginal SFT + binary GRPO | | | | | | | |

Choose a checkpoint only from dev behavior. Freeze all choices before the test
run.

## 11. Step 8: optional second marginal-value iteration

If GRPO improves answers but weakens routing selectivity, collect fresh
counterfactuals using the GRPO checkpoint and continue SFT from that checkpoint.
This is not a reset to the base model.

```bash
export MV2_DIR="outputs/manager/$TEACHER_ID/marginal_value_round2"
export MV2_ADAPTER="outputs/manager/$TEACHER_ID/sft_marginal_round2"

python -m src.pipeline.cli build_marginal_sft \
  --base_model "$BASE_MODEL" \
  --teacher_id "$TEACHER_ID" \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size "$TEST_SIZE" \
  --mv_manager_dir "$GRPO_DIR" \
  --mv_n_samples 400 --mv_max_depth 1 \
  --mv_max_commit_rescue_ratio 1.0 \
  --mv_output_dir "$MV2_DIR" \
  --subagent_server_url "$SUBAGENT_SERVER_URL" \
  --task_description "$TASK_DESC"

python -m src.pipeline.cli train_manager_sft \
  --base_model "$BASE_MODEL" \
  --teacher_id "$TEACHER_ID" \
  --manager_sft_train_jsonl "$MV2_DIR/manager_sft_marginal.jsonl" \
  --manager_sft_init_adapter "$GRPO_DIR" \
  --manager_sft_output_dir "$MV2_ADAPTER" \
  --manager_sft_epochs 1 --manager_sft_lr 5e-6
```

This continuation flag is important: without it, the old manager-SFT code
would restart from the base model and discard the GRPO policy.

## 12. Step 9: final test and transfer evaluation

After freezing the selected manager, run exactly once on MedQA test and then
zero-shot transfer benchmarks.

```bash
export FINAL_MANAGER="$GRPO_DIR"  # or the dev-selected round-2 checkpoint

python -m src.pipeline.cli eval_manager_tools \
  --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
  --medqa_normalized_cache "$MEDQA_CACHE" \
  --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size "$TEST_SIZE" \
  --eval_n_samples "$TEST_SIZE" \
  --eval_manager_dir "$FINAL_MANAGER" \
  --subagent_server_url "$SUBAGENT_SERVER_URL" \
  --task_description "$TASK_DESC"
```

Repeat the same command with the existing LegalBench, MMLU-Pro, and GPQA cache
flags. Do not retune the manager, KL coefficient, commit ratio, or stopping
threshold on transfer benchmarks.

## 13. Required baselines and ablations

Main baselines:

1. base manager, direct answer;
2. base manager + all advisors forced;
3. random advisor routing with matched average calls;
4. heuristic/teacher sequence cold start from the old pipeline;
5. binary GRPO without marginal-value SFT;
6. marginal-value SFT without GRPO;
7. marginal-value SFT + binary GRPO (main method).

Core ablations:

1. replace the real initial draft with GT (expected to damage routing);
2. select teacher/heuristic sequences without checking counterfactual outcome;
3. remove the correct-correct commit tie-break;
4. remove commit/rescue balancing;
5. one-step versus depth-2 marginal search;
6. `beta = 0.02/0.05/0.10`;
7. binary reward versus additive per-call cost, documenting no-call collapse;
8. omit `DRAFT_ANSWER` from routing turns.

Do not present ADC variants as the main method. If retained, they belong only
in a failure-analysis appendix.

## 14. Statistical reporting

- Use at least three random seeds for the main method and strongest baselines.
- Report mean and standard deviation for accuracy and average calls.
- Use paired bootstrap confidence intervals on per-question accuracy
  differences because systems are evaluated on the same questions.
- Bootstrap average-call differences and correction/corruption differences.
- Report oracle gain with a confidence interval; it defines the maximum
  exploitable value of the current advisors.
- Include the full call-count distribution (`k=0,1,2,3`), not only its mean.

The paper's stopping claim should be supported by both outcomes and behavior:
high `call_rate_given_draft_wrong`, low `call_rate_given_draft_correct`, positive
correction-minus-corruption, and competitive final accuracy at fewer calls.

## 15. Failure diagnosis order

When a run fails, inspect in this order:

1. **No oracle gain:** advisors do not repair drafts; fix advisors.
2. **Oracle gain but empty SFT:** parsing or branch materialization bug.
3. **SFT call rate is zero:** commit examples dominate or tool calls do not
   render correctly in the tokenizer chat template.
4. **SFT calls everything:** reduce rescue oversampling and check whether
   direct-correct commit rows are present.
5. **SFT is selective but GRPO collapses:** increase KL anchor, shorten GRPO,
   and select the last stable checkpoint; do not add a negative per-call cost.
6. **Calls occur but do not correct:** advisor evidence is not being integrated;
   inspect manager revisions and add those fresh failures to round 2.
7. **Accuracy rises but calls also rise:** verify that the correct-correct commit
   tie-break is present and collect fresh marginal pairs from the new policy.

This order separates advisor capability, counterfactual data quality, routing
initialization, and RL stability instead of trying to repair all four with one
reward coefficient.
