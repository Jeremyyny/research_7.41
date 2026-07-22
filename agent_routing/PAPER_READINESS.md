# Paper Readiness: Learning When to Commit

This document records the proposed paper narrative, the claims that the current
system can support, the code-paper alignment issues that must be resolved, and
the experiments required before the main paper is written.

## 1. Recommended paper thesis

The paper should not be presented as another general trainable multi-agent
framework. Its strongest and most specific question is:

> When should a manager seek additional specialist advice, and when should it
> commit to its current answer?

The system studies this question as cost-sensitive sequential stopping over
frozen specialist advisors. At each turn, the manager either calls one unused
advisor or commits to an answer. The manager also exposes its current answer
belief through an explicit draft. This makes it possible to measure whether
consultation corrects or corrupts the manager's belief.

The main technical contribution should be the reward design rather than the
existence of the extractor, reasoner, and verifier. Two natural process rewards
are exploitable:

1. A transition reward for wrong-to-correct changes can reward a policy for
   deliberately starting with a wrong draft and correcting it later.
2. A summed correct-draft bonus can reward unnecessary calls that repeat an
   already-correct draft.

The proposed anytime reward instead uses bounded average correctness over
explicit answer statements, adds final-answer correctness, and subtracts
consultation cost. The intended claim is that this design resists the two
identified exploit classes while supporting adaptive stopping.

## 2. Recommended title and terminology

Current safe title:

**Learning When to Commit: Exploit-Resistant Adaptive Deliberation for
Multi-Agent Reasoning**

A stronger title may use *incentive-compatible stopping* only after the formal
claim and implementation assumptions are made precise.

Preferred terms:

- **manager**: the only component allowed to submit the final answer;
- **advisor**: a frozen specialist that returns structured decision signals;
- **draft**: the manager's current answer belief before another consultation;
- **commit**: submit the final answer and end the episode;
- **consultation cost**: the cost assigned to an advisor call;
- **anytime reward**: the bounded reward over the sequence of explicit answer
  beliefs.

Avoid broad claims such as "the first cost-aware tool-use method" or "the first
trainable agent orchestrator." Existing work already studies both topics. The
paper's distinction is the combination of specialist consultation, explicit
belief evolution, stopping, and exploit-resistant process supervision.

## 3. Introduction outline

### Paragraph 1: Motivation and tension

Specialized agents can help by extracting evidence, structuring a problem, and
checking a candidate solution. More consultation is not always better. Easy
questions incur unnecessary cost, redundant advice adds little information,
and noisy advice can overturn a correct initial belief.

### Paragraph 2: Central question

Ask the paper's central question directly: when is another consultation worth
its cost? Define the manager's problem as choosing between an unused advisor and
commitment after each observation.

### Paragraph 3: Gap in existing methods

Separate the relevant literature into three groups:

- fixed multi-agent workflows use a predetermined depth or order;
- cost-aware tool-use RL optimizes final correctness and tool cost, but usually
  does not model the evolution of the manager's answer belief;
- trainable agentic systems optimize planning inside multi-turn execution, but
  outcome-only rewards do not directly distinguish useful belief revision from
  redundant deliberation.

Relevant comparisons include AgentFlow, OTC-PO, EMTIR-GRPO, budget-aware tool
use, and utility-guided orchestration.

### Paragraph 4: Why the reward is difficult

Introduce the two concrete reward failures early: transition sandbagging and
summed-draft farming. This is the main technical tension of the paper and should
appear before the method description.

### Paragraph 5: Method

Introduce the manager, the three frozen advisors, explicit drafts, the
delegate-or-commit action space, and on-policy GRPO training. State that advisors
return structured signals and do not own the final answer.

### Paragraph 6: Evaluation questions

Organize evaluation around research questions rather than around datasets:

- **RQ1:** Does specialist consultation improve accuracy beyond matched
  computation and teacher distillation?
- **RQ2:** Does the learned manager implement useful stopping and improve the
  cost-accuracy trade-off?
- **RQ3:** Do exploitable reward variants produce the predicted strategic
  behavior?

### Paragraph 7: Results and contributions

Do not write numerical claims until final evaluation artifacts exist. Report
final accuracy, average consultation count, Pareto behavior, transfer behavior,
and reward-exploit measurements only after all seeds and statistical tests are
complete.

## 4. Intended method definition

For a question with advisor set
`A = {extractor, reasoner, verifier}`, the state contains the question,
previous advisor outputs, used advisors, and the manager's current draft. The
action space at turn `t` should be:

`{COMMIT} union {a in A: a has not been used}`.

A consultation returns a deterministic structured observation from the selected
frozen advisor. The manager then updates its draft and chooses again. An episode
ends when the manager commits or reaches the maximum consultation budget.

The proposed anytime reward currently has four parts:

- final correctness;
- bounded average correctness over all valid answer statements;
- a penalty for a tool call without a corresponding draft;
- a per-consultation cost.

The paper should distinguish the training reward from the evaluation utility.
The primary evaluation utility should remain transparent, for example:

`utility = final_correctness - lambda * number_of_calls`.

## 5. P0: Code-paper alignment blockers

These items should be resolved before claiming strict sequential stopping or
incentive compatibility.

### P0.1 Enforce one sequential decision per turn

Relevant files:

- `src/manager/grpo_train.py`
- `src/pipeline/stages.py`
- `src/manager/prompt.py`

Current issue:

- environment binding guards repeated advisors, but argument-binding tool
  functions do not apply the same hard guard;
- evaluation can execute several tool calls emitted in one assistant turn;
- prompt instructions discourage repeats and parallel calls, but the action
  constraint is not enforced in every runtime path.

Required changes:

- [ ] Maintain a per-episode set of used advisors in both binding modes.
- [ ] Reject a second call to an already-used advisor.
- [ ] Permit at most one advisor call in each manager turn.
- [ ] Apply identical rules during GRPO rollout, local evaluation, remote
      evaluation, cold-start traces, and forced evaluation.
- [ ] Add tests for repeated calls, parallel calls, and calls after budget
      exhaustion.
- [ ] Log invalid action attempts separately from ordinary format failures.

Acceptance criterion: every successful trajectory is a sequence of single
advisor observations followed by one commit action.

### P0.2 Replace or rename the current "stopping oracle"

Relevant files:

- `EXPERIMENTS.md`
- `src/pipeline/stages.py`

Current issue:

- the experiment enumerates eight unordered advisor subsets, not all ordered
  sequences;
- three advisors produce 16 possible ordered prefixes:
  `1 + 3 + 6 + 6 = 16`;
- verifier output depends on the current draft, while the forced evaluator uses
  a generic verifier without that draft;
- the current score is therefore a best fixed-subset hindsight baseline, not a
  guaranteed oracle upper bound.

Required changes:

- [ ] Until fixed, rename the metric to **best fixed advisor subset in
      hindsight** and avoid "oracle regret."
- [ ] Decide whether advisor order is part of the policy being evaluated.
- [ ] If it is, enumerate all ordered sequences.
- [ ] Make forced trajectories reproduce the same draft-dependent verifier
      inputs used by the learned policy.
- [ ] Verify that the resulting oracle utility is never below the learned
      policy's utility on the same action space.
- [ ] Save per-question selected sequence, utility, and learned-policy regret.

### P0.3 Qualify the incentive-compatibility claim

Relevant files:

- `src/manager/reward.py`
- `src/manager/prompt.py`
- `src/subagents/prompts/verifier.py`

Current issue:

For fixed future observations and a fixed trajectory length, reporting the
highest-probability answer maximizes expected draft correctness. In the current
system, however, the verifier observation depends on `current_draft`. A manager
may report a different hypothesis to obtain a different audit. The draft is
therefore both a scored report and an environment-changing action.

Required changes:

- [ ] State the exact assumptions under which truthful MAP reporting is
      optimal.
- [ ] Decide whether to separate `reported_draft` from
      `verifier_audit_target`.
- [ ] If they remain coupled, weaken the claim to resistance against the two
      demonstrated exploit classes.
- [ ] Add a proposition for boundedness and non-farmability of the anytime
      reward.
- [ ] Add counterexamples for the transition and summed variants.
- [ ] Add unit tests that enumerate short trajectories and verify the reward
      ordering.
- [ ] Test whether policies strategically change the verifier target while
      keeping the scored draft fixed or incorrect.

### P0.4 Correct cross-domain task descriptions

Relevant file:

- `EXPERIMENTS.md`

Current issue:

Some transfer commands reuse the MedQA `TASK_DESC` for LegalBench, MMLU-Pro,
and GPQA. This introduces an avoidable prompt mismatch.

Required changes:

- [ ] Use a domain-neutral manager prompt for the primary transfer experiment,
      or use a correctly matched description for each benchmark.
- [ ] Keep the prompt policy fixed across compared methods.
- [ ] Record the exact task description in every evaluation report.
- [ ] Add a prompt-sensitivity ablation if domain-specific descriptions are
      retained.

## 6. P1: Experimental requirements

### P1.1 RQ1 — Does delegation add value?

Primary MedQA comparison:

- [ ] Base 8B direct reasoning.
- [ ] Base 8B self-consistency with matched inference compute.
- [ ] Direct chain-of-thought distillation with the same teacher and a matched
      teacher-output token budget.
- [ ] Fixed all-advisor workflow.
- [ ] Learned delegate-or-commit policy.
- [ ] Frontier model reference, clearly labeled as a reference rather than a
      matched training condition.

Controls:

- [ ] Match actual generated teacher tokens, not only the number of examples.
- [ ] Report manager and advisor parameter counts.
- [ ] Report total generated tokens, advisor calls, latency, and peak hardware
      usage where possible.
- [ ] Ensure all training and evaluation pools are hash-disjoint.
- [ ] Evaluate the final selected system once on the full MedQA test split.

### P1.2 RQ2 — Is the policy genuinely adaptive?

- [ ] Evaluate all valid fixed sequences or clearly defined fixed subsets.
- [ ] Train/evaluate a cost grid to obtain a cost-accuracy Pareto curve.
- [ ] Report accuracy, average calls, call distribution, and utility.
- [ ] Compare the learned policy with the best fixed policy at equal or lower
      average cost.
- [ ] Measure per-question hindsight regret only after the oracle action space
      matches the learned action space.
- [ ] Test whether call depth tracks empirical base-model difficulty rather
      than assuming a fixed ordering of benchmark difficulty.
- [ ] Run MedQA-to-LegalBench, MMLU-Pro, and GPQA transfer without retraining.
- [ ] Report per-task LegalBench results.

### P1.3 RQ3 — Does reward design change behavior?

Required arms:

- [ ] Binary final-outcome reward.
- [ ] CCR legacy reward.
- [ ] Transition reward.
- [ ] Summed-draft reward.
- [ ] Proposed anytime reward.

Required diagnostics:

- [ ] Initial-draft error rate.
- [ ] Wrong-to-correct and correct-to-wrong transition rates.
- [ ] Sandbagging rate over training.
- [ ] Redundant call rate.
- [ ] Correct-draft repetition rate.
- [ ] Missing-draft rate.
- [ ] Final accuracy and average calls.
- [ ] Reward versus evaluation utility.

The transition and summed arms should be presented as controlled counterexamples,
not competitive methods.

### P1.4 Statistical reporting

- [ ] Run at least three seeds for the main policy and every central reward
      ablation.
- [ ] Report mean and standard deviation.
- [ ] Use paired McNemar tests for accuracy on the same questions.
- [ ] Use paired bootstrap intervals for utility, call count, and regret.
- [ ] State all model-selection decisions and keep the final test split out of
      those decisions.
- [ ] Preserve raw trajectories and evaluation reports for reproducibility.

## 7. P1: Additional methodological checks

### Forced-policy distribution match

The forced all-advisor baseline currently injects advisor outputs and asks the
manager only for a final answer. Check whether this matches the contexts seen in
cold-start and GRPO training.

- [ ] Reproduce intermediate draft turns in forced trajectories.
- [ ] Use the same advisor order, tool schema, and verifier conditioning.
- [ ] Separate a fixed-workflow baseline from a hindsight best-sequence
      analysis.

### Advisor-output guarantees

The schemas prevent an explicit final-answer field, and synthesis is
ground-truth blind with leakage auditing. This does not prove the absence of
semantic answer hints.

- [ ] Report JSON validity and schema validity at runtime.
- [ ] Report explicit leakage audit rejection rates.
- [ ] Audit a sample manually for semantic preference leakage.
- [ ] Test manager performance when advisor outputs are shuffled across
      questions.
- [ ] Test each advisor alone and all advisor combinations.

### Risk-coverage terminology

The current risk-coverage analysis treats fewer calls as greater confidence,
but call depth is an action rather than an explicit calibrated probability.

- [ ] Prefer **depth-conditioned risk-coverage** unless a confidence
      interpretation is established.
- [ ] Report accuracy separately for 0, 1, 2, and 3 calls.
- [ ] Test monotonicity rather than assuming it.

## 8. Recommended paper figures and tables

### Figure 1: The central problem

Show three examples under the same manager:

- easy question: correct initial draft, immediate commit;
- medium question: one advisor changes a wrong draft to a correct draft;
- hard question: several non-redundant consultations before commit.

Include a small contrast showing how transition reward encourages a deliberately
wrong first draft.

### Figure 2: Method

Show the manager's sequential loop:

`draft -> choose advisor or commit -> structured observation -> updated draft`.

Clearly mark advisors as frozen and the manager as trainable.

### Table 1: Main effectiveness and efficiency results

Columns should include accuracy, average calls, total tokens, and evaluation
utility. Include direct reasoning, self-consistency, distillation, fixed
workflow, and the learned policy.

### Figure 3: Stopping quality

Plot cost versus accuracy for the learned policy family and fixed policies. Add
hindsight regret only after the oracle definition is corrected.

### Figure 4: Reward behavior

Plot sandbagging, draft farming, correction, and corruption rates over training
for the reward variants.

### Table 2: Cross-domain transfer

Report performance and consultation depth on MedQA, LegalBench, MMLU-Pro, and
GPQA. Interpret difficulty using observed base-model performance.

## 9. Claim discipline

Before results are available, the paper may claim that the work:

- formulates specialist consultation as a learned delegate-or-commit problem;
- exposes intermediate answer beliefs through drafts;
- identifies two concrete reward exploits;
- proposes a bounded anytime reward designed to resist those exploits;
- supplies an evaluation protocol for effectiveness, stopping, and reward
  behavior.

Do not yet claim that the method:

- outperforms fixed workflows;
- dominates the cost-accuracy Pareto frontier;
- generalizes across domains;
- approaches frontier-model accuracy;
- is fully incentive-compatible;
- attains low oracle regret.

Each of these claims requires completed experiments or stronger formal
conditions.

## 10. Recommended implementation order

1. [ ] Enforce the exact sequential action space in every runtime path.
2. [ ] Add trajectory-level tests for repeats, parallel calls, drafts, and
       commitment.
3. [ ] Resolve the scored-draft versus verifier-target coupling.
4. [ ] Finalize and test the reward proposition and counterexamples.
5. [ ] Correct cross-domain prompts.
6. [ ] Replace or rebuild the oracle evaluation.
7. [ ] Regenerate cold-start data under the final trajectory protocol.
8. [ ] Run advisor quality gates.
9. [ ] Run one small end-to-end GRPO smoke experiment.
10. [ ] Run RQ1 main baselines.
11. [ ] Run the cost grid and stopping analyses.
12. [ ] Run reward ablations.
13. [ ] Run cross-domain transfer.
14. [ ] Run final seeds and statistical tests.
15. [ ] Freeze numbers, then write the abstract and results paragraph of the
        Introduction.

## 11. Definition of paper readiness

The paper is ready for full drafting when:

- the implementation matches the stated sequential decision process;
- the reward claim has precise assumptions and tests;
- the oracle or hindsight baseline is correctly named and implemented;
- all main baselines use controlled data and compute budgets;
- three-seed main and reward-ablation results are complete;
- raw trajectories support every reported process metric;
- final tables and figures can be regenerated from saved artifacts;
- every numerical statement in the abstract and Introduction maps to a saved
  report.
