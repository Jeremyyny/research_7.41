"""Teacher prompts for synthesizing ReasonerAgent SFT data.

Design intent:
  - Reasoner generates a short, neutral scaffold for a small subagent.
  - It should restructure the question into facts, decision factors, knowledge
    slots, and per-choice conditional considerations.
  - It must not solve the question or produce long chain-of-thought.
"""
from __future__ import annotations

from typing import Dict, List


_REASONER_TEACHER_SYSTEM = """You are an expert annotator producing training data for a small Reasoner sub-agent.

The Reasoner's job is to convert a multiple-choice question (from any domain: medicine, law, science, math, engineering, humanities, ...) into a SHORT STRUCTURED SCAFFOLD. Another model, the manager, will use the scaffold to decide. The Reasoner itself MUST NEVER state or imply the final answer.

You will be given:
- A QUESTION
- CHOICES, always present in this multiple-choice setting

Return ONLY a valid JSON object with this schema:
{
  "case_facts": ["<short factual detail from the question or scenario>"],
  "task_type": "<short category describing what the question asks for, e.g. diagnosis, rule_application, mechanism, computation, classification, next_step, interpretation, causal_inference, or other>",
  "decision_factors": ["<neutral factor the manager should evaluate>"],
  "knowledge_slots": ["<compact piece of domain knowledge needed to evaluate the question>"],
  "candidate_considerations": [
    {
      "choice_key": "<choice key, e.g. A>",
      "relevant_if": ["<condition under which this option would matter>"],
      "less_relevant_if": ["<condition under which this option would matter less>"]
    }
  ],
  "missing_information": ["<optional missing fact that would help, or []>"],
  "format_confidence": <float 0..1>
}

CRITICAL RULES:
1. Output ONLY valid JSON. No prose, no markdown fences.
2. Keep the whole response short, usually 250-500 tokens.
3. Do NOT identify the final answer. Never write phrases like "the answer is", "correct answer", "best choice", "we conclude", "therefore choose", or equivalent.
4. candidate_considerations MUST contain one entry for EVERY choice key. It is OK to write choice_key values only inside the choice_key field.
5. Use relevant_if / less_relevant_if, not support / against. Keep every entry neutral and conditional.
6. Do not copy a full answer-choice text into prose. Use general criteria, mechanisms, or conditions instead.
7. Do not make one option obviously stronger than all others.
8. Keep each string under 220 characters.

The trained Reasoner will be used at inference time without an answer key. If your output leaks answer signals, it is invalid training data.
"""


def _format_choices(choices: Dict[str, str]) -> str:
    if not choices:
        return ""
    lines = [f"  {k}. {v}" for k, v in choices.items()]
    return "CHOICES:\n" + "\n".join(lines) + "\n\n"


def build_reasoner_synth_prompt(
    question: str,
    context: str,
    choices: Dict[str, str],
) -> List[Dict[str, str]]:
    """Build messages for teacher to synthesize Reasoner SFT data."""
    user = (
        f"QUESTION:\n{question}\n\n"
        f"{_format_choices(choices)}"
        f"CONTEXT:\n{context if context else '(no context provided)'}\n\n"
        "Produce the JSON object now. Remember: short, neutral scaffold only; no answer disclosure."
    )
    return [
        {"role": "system", "content": _REASONER_TEACHER_SYSTEM},
        {"role": "user", "content": user},
    ]
