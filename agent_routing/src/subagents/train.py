"""Subagent LoRA SFT training. Runtime classes (FrozenSubagent, SubagentPool)
live in runtime.py — do not duplicate them here."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.io import read_jsonl
from ..utils.seed import set_seed

try:
    from peft import LoraConfig, get_peft_model
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
class SFTConfig:
    base_model: str
    train_jsonl: str
    out_dir: str
    dev_jsonl: Optional[str] = None
    seed: int = 42
    max_seq_len: int = 4096
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_steps: int = -1
    bf16: bool = True


def _mask_prefix_len(prompt_ids: List[int], full_ids: List[int]) -> int:
    """Length of the common token prefix between the prompt-only render and the
    full (prompt+response) render.

    Using len(prompt_ids) directly is WRONG for templates where the generation
    prompt is not a strict prefix of the full render — e.g. Qwen3 with
    enable_thinking=False appends an empty <think></think> block to the
    generation prompt that does not appear before the assistant content in the
    full render. That off-by-N would mask the first response tokens.
    """
    n = min(len(prompt_ids), len(full_ids))
    i = 0
    while i < n and prompt_ids[i] == full_ids[i]:
        i += 1
    return i


def _tokenize_subagent_sft(rows: List[Dict[str, Any]], tok, max_seq_len: int) -> Any:
    from datasets import Dataset

    eos = tok.eos_token or ""

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt_msgs = ex["prompt"]
        response = ex["response"]
        response_msgs = [{"role": "assistant", "content": response}]

        prompt_text = _render_chat(tok, prompt_msgs, add_generation_prompt=True)
        full_text = _render_chat(tok, prompt_msgs + response_msgs, add_generation_prompt=False)
        # Most chat templates already close the last turn with the EOS token;
        # only append it when missing to avoid training a doubled EOS.
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


def _lora_target_modules(model) -> List[str]:
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    return [name for name in candidates if name in present] or ["q_proj", "v_proj"]


def train_subagent_sft(cfg: SFTConfig) -> None:
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    if cfg.use_lora and not PEFT_AVAILABLE:
        raise RuntimeError("peft is required for LoRA subagent SFT training.")

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    dtype = torch.bfloat16 if (cfg.bf16 and device == "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.config.use_cache = False

    if cfg.use_lora:
        target = _lora_target_modules(model)
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target,
        )
        model = get_peft_model(model, lora_cfg)
        print(f"[SUBAGENT_SFT/LoRA] r={cfg.lora_r} alpha={cfg.lora_alpha} target_modules={target}")

    train_rows = read_jsonl(cfg.train_jsonl)
    if not train_rows:
        raise ValueError(f"No rows in {cfg.train_jsonl}")
    train_ds = _tokenize_subagent_sft(train_rows, tok, cfg.max_seq_len)

    eval_ds = None
    if cfg.dev_jsonl:
        dev_rows = read_jsonl(cfg.dev_jsonl)
        eval_ds = _tokenize_subagent_sft(dev_rows, tok, cfg.max_seq_len) if dev_rows else None

    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, return_tensors="pt")
    args = TrainingArguments(
        output_dir=cfg.out_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        bf16=(cfg.bf16 and device == "cuda"),
        fp16=False,
        report_to=[],
        seed=cfg.seed,
        remove_unused_columns=False,
        max_steps=(cfg.max_steps if cfg.max_steps > 0 else -1),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()
    os.makedirs(cfg.out_dir, exist_ok=True)
    trainer.model.save_pretrained(cfg.out_dir)
    tok.save_pretrained(cfg.out_dir)
    print(f"[SUBAGENT_SFT] saved -> {cfg.out_dir}")
