"""FrozenAgent: load a SFT'd subagent and run it as a non-trainable tool.

Used at:
  - manager GRPO training time (subagents are tools called from manager rollouts)
  - manager evaluation time
  - manager evolve_round (to produce tool outputs for SFT trace construction)

Key behaviors:
  - Loads base model + LoRA adapter (PEFT). If adapter_path points to a full
    save_pretrained dir (no adapter_config.json), loads as a full model.
  - Greedy decoding by default (deterministic tool outputs, important for
    GRPO group-relative advantage computation).
  - Caches outputs by (agent_kind, example_id) so repeated calls on the same
    example during multi-rollout GRPO are free.

For multi-GPU full-parameter GRPO, use RemoteSubagentPool instead of SubagentPool.
RemoteSubagentPool calls subagents via a vLLM HTTP server running on a dedicated GPU,
so no subagent weights are loaded into the training processes.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts.runtime_prompts import build_runtime_messages

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


def _render_chat(tokenizer, messages, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


@dataclass
class FrozenSubagent:
    base_model: str
    adapter_path: Optional[str]
    agent_kind: str             # "extractor" | "reasoner" | "verifier"
    device: str = "cuda"
    max_new_tokens: int = 1024
    dtype_str: str = "bfloat16"

    _tok: Any = field(init=False, default=None)
    _model: Any = field(init=False, default=None)

    def __post_init__(self):
        if self.adapter_path:
            p = self.adapter_path
            if os.sep in p or p.count("/") > 1 or p.startswith("."):
                self.adapter_path = os.path.abspath(p)
        self._tok = AutoTokenizer.from_pretrained(
            self.adapter_path or self.base_model, trust_remote_code=True
        )
        if self._tok.pad_token_id is None and self._tok.eos_token_id is not None:
            self._tok.pad_token_id = self._tok.eos_token_id
        self._tok.padding_side = "left"

        dtype = torch.bfloat16 if self.dtype_str == "bfloat16" and self.device == "cuda" else torch.float32

        is_full_save = (
            self.adapter_path
            and os.path.isdir(self.adapter_path)
            and not os.path.exists(os.path.join(self.adapter_path, "adapter_config.json"))
            and os.path.exists(os.path.join(self.adapter_path, "config.json"))
        )

        if is_full_save:
            model = AutoModelForCausalLM.from_pretrained(
                self.adapter_path, torch_dtype=dtype, trust_remote_code=True
            ).to(self.device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model, torch_dtype=dtype, trust_remote_code=True
            ).to(self.device)
            if self.adapter_path:
                if not PEFT_AVAILABLE:
                    raise RuntimeError("peft not available; cannot load adapter.")
                model = PeftModel.from_pretrained(model, self.adapter_path).to(self.device)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model

    @torch.no_grad()
    def generate(
        self,
        question: str,
        context: str,
        choices: Dict[str, str],
        temperature: float = 0.0,
        candidate_answer: str = "",
    ) -> str:
        messages = build_runtime_messages(
            agent_kind=self.agent_kind,
            question=question,
            context=context,
            choices=choices,
            candidate_answer=candidate_answer,
        )
        prompt = _render_chat(self._tok, messages, add_generation_prompt=True)
        inputs = self._tok(prompt, return_tensors="pt").to(self.device)

        do_sample = temperature > 1e-6
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tok.pad_token_id,
            "eos_token_id": self._tok.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-6)

        out = self._model.generate(**inputs, **gen_kwargs)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True).strip()


class SubagentPool:
    """Holds up to three FrozenSubagent instances and routes calls by kind.

    Provides per-(kind, example_id) output caching: during GRPO with N rollouts
    per example, the manager may call the same tool multiple times across
    rollouts; we want the tool output to be deterministic and cheap.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, FrozenSubagent] = {}
        self._cache: Dict[str, str] = {}
        self._call_log: List[Dict[str, Any]] = []

    def register(self, agent: FrozenSubagent) -> None:
        self._agents[agent.agent_kind] = agent

    def has(self, agent_kind: str) -> bool:
        return agent_kind in self._agents

    def call(
        self,
        agent_kind: str,
        example_id: int,
        question: str,
        context: str,
        choices: Dict[str, str],
        cache_namespace: str = "default",
        candidate_answer: str = "",
    ) -> str:
        # candidate_answer bypasses cache (different context = different output)
        cache_key_suffix = f"::{candidate_answer}" if candidate_answer else ""
        key = f"{cache_namespace}::{agent_kind}::{int(example_id)}{cache_key_suffix}"
        if key in self._cache:
            self._call_log.append({
                "ts": int(time.time()),
                "agent_kind": agent_kind,
                "example_id": int(example_id),
                "cache_hit": True,
            })
            return self._cache[key]

        if agent_kind not in self._agents:
            raise KeyError(f"Subagent not registered: {agent_kind}")

        text = self._agents[agent_kind].generate(
            question, context, choices, candidate_answer=candidate_answer
        )
        self._cache[key] = text
        self._call_log.append({
            "ts": int(time.time()),
            "agent_kind": agent_kind,
            "example_id": int(example_id),
            "cache_hit": False,
            "output_len": len(text),
        })
        return text

    def clear_cache(self) -> None:
        self._cache.clear()

    def drain_log(self) -> List[Dict[str, Any]]:
        log = self._call_log
        self._call_log = []
        return log


class RemoteSubagentPool:
    """Calls subagents via a vLLM HTTP server instead of loading models locally.

    Drop-in replacement for SubagentPool for multi-GPU full-parameter GRPO.
    Subagents run on a dedicated GPU (GPU 0) via vLLM with --enable-lora;
    the adapter name is used as the model identifier in the OpenAI-compatible API.

    Usage:
        pool = RemoteSubagentPool("http://localhost:8000")
        text = pool.call("extractor", example_id=42, question=..., context=..., choices=...)
    """

    def __init__(
        self,
        server_url: str,
        registered_kinds: Optional[List[str]] = None,
        max_new_tokens: int = 1024,
        timeout: int = 120,
    ) -> None:
        if not REQUESTS_AVAILABLE:
            raise RuntimeError(
                "requests is required for RemoteSubagentPool. pip install requests"
            )
        self._server_url = server_url.rstrip("/")
        self._kinds: set = set(registered_kinds or ["extractor", "reasoner", "verifier"])
        self._max_new_tokens = max_new_tokens
        self._timeout = timeout
        self._cache: Dict[str, str] = {}
        self._call_log: List[Dict[str, Any]] = []

    def has(self, agent_kind: str) -> bool:
        return agent_kind in self._kinds

    def call(
        self,
        agent_kind: str,
        example_id: int,
        question: str,
        context: str,
        choices: Dict[str, str],
        cache_namespace: str = "default",
        candidate_answer: str = "",
    ) -> str:
        cache_key_suffix = f"::{candidate_answer}" if candidate_answer else ""
        key = f"{cache_namespace}::{agent_kind}::{int(example_id)}{cache_key_suffix}"
        if key in self._cache:
            self._call_log.append({
                "ts": int(time.time()),
                "agent_kind": agent_kind,
                "example_id": int(example_id),
                "cache_hit": True,
            })
            return self._cache[key]

        messages = build_runtime_messages(
            agent_kind=agent_kind,
            question=question,
            context=context,
            choices=choices,
            candidate_answer=candidate_answer,
        )
        # NOTE: "extra_body" is an OpenAI *Python SDK* client-side argument, not
        # an HTTP field. When POSTing raw JSON to vLLM, chat_template_kwargs
        # must sit at the top level of the payload or thinking is NOT disabled.
        payload = {
            "model": agent_kind,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self._max_new_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = _requests.post(
            f"{self._server_url}/v1/chat/completions",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()

        self._cache[key] = text
        self._call_log.append({
            "ts": int(time.time()),
            "agent_kind": agent_kind,
            "example_id": int(example_id),
            "cache_hit": False,
            "output_len": len(text),
        })
        return text

    def clear_cache(self) -> None:
        self._cache.clear()

    def drain_log(self) -> List[Dict[str, Any]]:
        log = self._call_log
        self._call_log = []
        return log