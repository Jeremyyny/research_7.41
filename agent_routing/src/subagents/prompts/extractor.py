"""Teacher prompts for synthesizing ExtractorAgent SFT data.

Design intent:
  - Extractor's job is OBJECTIVE evidence/fact extraction.
  - We do NOT show the teacher the ground truth label or the GT choice text.
    Reason: leakage avoidance + we want the teacher to surface evidence on
    BOTH sides (support and oppose), not only pro-GT evidence.
  - For closed-book MCQ with empty context (e.g. MedQA, GPQA), we pivot the
    agent's role to "extract concrete facts from the question stem itself".
"""
from __future__ import annotations

import json
from typing import Dict, List


_EXTRACTOR_TEACHER_SYSTEM = """You are an expert annotator producing training data for an Extractor sub-agent.

The Extractor's job is to surface DECISION-RELEVANT SIGNALS that manager agent will use to reason about an answer. The Extractor itself NEVER decides the final answer.

You will be given:
- A QUESTION (and CHOICES, if multiple-choice)
- A CONTEXT (which may be empty for closed-book questions)

You must produce a JSON object that exactly matches this schema:
{
  "key_evidence": [
    {"text": "<verbatim or near-verbatim sentence from context>", "relevance": <float 0..1>, "polarity": "support" | "oppose" | "neutral"}
  ],
  "extracted_facts": ["<concrete fact pulled from the question or context, one per item>"],
  "missing_info": ["<information that would be helpful but is not provided>"],
  "context_summary": "<<= 400 char neutral summary of the context (or question stem if no context)>",
  "confidence": <float 0..1, your confidence the extraction is complete and unbiased>
}

CRITICAL RULES:
1. Output ONLY valid JSON. No prose, no markdown fences.
2. Do NOT state, hint, or imply which answer choice is correct.
3. When choices are present, your evidence and facts must be useful for evaluating ALL choices, not selectively favor any one.
4. If context is empty (closed-book), set key_evidence=[] and use extracted_facts to enumerate concrete pieces of information FROM THE QUESTION STEM (e.g. "patient age 45", "contract was signed before delivery", "reaction occurs at 300 K", "array is sorted in ascending order").
5. extracted_facts entries must be self-contained and de-contextualized (a separate agent will read them without seeing the original question).
6. Polarity is relative to the QUESTION's directional claim, not to any choice. If unclear, use "neutral".
7. Keep each text field under the schema's character limits.
"""


def _format_choices(choices: Dict[str, str]) -> str:
    if not choices:
        return ""
    lines = [f"  {k}. {v}" for k, v in choices.items()]
    return "CHOICES:\n" + "\n".join(lines) + "\n\n"


def build_extractor_synth_prompt(
    question: str,
    context: str,
    choices: Dict[str, str],
) -> List[Dict[str, str]]:
    """Build (system, user) messages for teacher to synthesize Extractor SFT.

    Note: ground_truth is NOT passed in. Extractor is GT-blind by design.
    """
    user = (
        f"QUESTION:\n{question}\n\n"
        f"{_format_choices(choices)}"
        f"CONTEXT:\n{context if context else '(no context provided; this is a closed-book question)'}\n\n"
        "Produce the JSON object now."
    )
    return [
        {"role": "system", "content": _EXTRACTOR_TEACHER_SYSTEM},
        {"role": "user", "content": user},
    ]