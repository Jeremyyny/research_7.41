"""Runtime system prompts (used at inference, AFTER subagent SFT).

These are stripped of teacher-facing meta-explanation. The trained subagent
should be able to follow these directly.

Verifier supports an optional candidate_answer parameter. When set, the verifier
audits whether the specific answer is well-supported, in addition to generating
general domain principles and error patterns. This enables Adaptive Deliberation
Control: the manager can request a targeted audit of its current draft answer.
"""
from __future__ import annotations

from typing import Dict, List, Optional


EXTRACTOR_RUNTIME_SYSTEM = """You are the Extractor sub-agent.

Given a question (and optional choices and context), extract decision-relevant signals. Output ONLY a JSON object with this schema:
{
  "key_evidence": [{"text": str, "relevance": float, "polarity": "support"|"oppose"|"neutral"}],
  "extracted_facts": [str],
  "missing_info": [str],
  "context_summary": str,
  "confidence": float
}

Rules:
- Output ONLY valid JSON, no extra text.
- Do NOT state the final answer.
- If context is empty, key_evidence=[] and use extracted_facts for concrete factual elements (entities, quantities, conditions) pulled from the question stem.
- Treat all answer choices fairly; do not favor any one.
"""


REASONER_RUNTIME_SYSTEM = """You are the Reasoner sub-agent.

Given a question (and choices, optional context), produce a short neutral scaffold. Output ONLY a JSON object with this schema:
{
  "case_facts": [str],
  "task_type": str,
  "decision_factors": [str],
  "knowledge_slots": [str],
  "candidate_considerations": [{"choice_key": str, "relevant_if": [str], "less_relevant_if": [str]}],
  "missing_information": [str],
  "format_confidence": float
}

Rules:
- Output ONLY valid JSON.
- NEVER state the final answer or which choice is correct.
- candidate_considerations must cover ALL choice keys.
- Use conditional relevant_if / less_relevant_if fields, not support/against.
- Keep the response short and neutral.
"""


VERIFIER_RUNTIME_SYSTEM = """You are the Verifier sub-agent.

Given a question (and optional context, choices), identify relevant domain principles, specify concrete checks for a solver's reasoning, and flag common mistake patterns.

If a candidate answer is provided, additionally audit whether that specific answer is well-supported by the evidence and principles.

Output ONLY a JSON object with this schema:
{
  "relevant_principles": [{"principle": str, "source": str}],
  "checks": [{"check": str, "status": "pass"|"fail"|"unclear", "note": str}],
  "potential_errors": [str],
  "candidate_answer_audit": str,
  "uncertainty_notes": [str],
  "confidence": float
}

Rules:
- Output ONLY valid JSON.
- Do NOT state the final answer.
- relevant_principles must apply regardless of which answer is correct.
- checks describe what to verify in the reasoning process.
- potential_errors describe mistake patterns a solver might make.
- candidate_answer_audit: if a candidate answer was given, briefly assess whether it is well-supported. If no candidate was given, set to "no candidate provided".
"""


def _format_choices_block(choices: Dict[str, str]) -> str:
    if not choices:
        return ""
    lines = [f"  {k}. {v}" for k, v in choices.items()]
    return "CHOICES:\n" + "\n".join(lines) + "\n\n"


def build_extractor_runtime_user(question: str, context: str, choices: Dict[str, str]) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"{_format_choices_block(choices)}"
        f"CONTEXT:\n{context if context else '(no context)'}\n\n"
        "Produce the JSON object."
    )


def build_reasoner_runtime_user(question: str, context: str, choices: Dict[str, str]) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"{_format_choices_block(choices)}"
        f"CONTEXT:\n{context if context else '(no context)'}\n\n"
        "Produce the JSON object."
    )


def build_verifier_runtime_user(
    question: str,
    context: str,
    choices: Dict[str, str],
    candidate_answer: str = "",
) -> str:
    candidate_block = ""
    if candidate_answer.strip():
        candidate_block = (
            f"CANDIDATE ANSWER TO AUDIT: {candidate_answer.strip()}\n"
            "Specifically assess whether this candidate is well-supported.\n\n"
        )
    return (
        f"QUESTION:\n{question}\n\n"
        f"{_format_choices_block(choices)}"
        f"CONTEXT:\n{context if context else '(no context)'}\n\n"
        f"{candidate_block}"
        "Produce the JSON object."
    )


def build_runtime_messages(
    agent_kind: str,
    question: str,
    context: str,
    choices: Dict[str, str],
    candidate_answer: str = "",
) -> List[Dict[str, str]]:
    """Build the prompt message list for the given subagent.

    Args:
        agent_kind: "extractor" | "reasoner" | "verifier"
        question: the question text
        context: optional context / passage
        choices: dict of choice key -> choice text
        candidate_answer: (verifier only) current draft answer to audit.
            When set, verifier will specifically evaluate this answer hypothesis.
    """
    if agent_kind == "extractor":
        return [
            {"role": "system", "content": EXTRACTOR_RUNTIME_SYSTEM},
            {"role": "user", "content": build_extractor_runtime_user(question, context, choices)},
        ]
    if agent_kind == "reasoner":
        return [
            {"role": "system", "content": REASONER_RUNTIME_SYSTEM},
            {"role": "user", "content": build_reasoner_runtime_user(question, context, choices)},
        ]
    if agent_kind == "verifier":
        return [
            {"role": "system", "content": VERIFIER_RUNTIME_SYSTEM},
            {"role": "user", "content": build_verifier_runtime_user(
                question, context, choices, candidate_answer=candidate_answer
            )},
        ]
    raise ValueError(f"Unknown agent_kind: {agent_kind}")
