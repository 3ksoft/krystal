#!/usr/bin/env python
"""Train LFM2.5-1.2B-Instruct reserved tokens as structural tokens.

No vocabulary growth: we reuse the model's existing ``<|reserved_N|>`` tokens, so
the 2^16 = 65,536 vocab geometry that the Chomato WebGPU runtime depends on
(ADA-0006 dense masks, WQ4 layout, TS tokenizer) is preserved.

Pipeline
--------
1. discover ``<|reserved_N|>`` tokens and assign each a structural role
   (roles.json)
2. mean-initialize their embedding rows from semantically similar tokens
3. freeze the base model; train ONLY the reserved rows (tied LM head follows)
4. save the trained rows as a small adapter (reserved_rows.safetensors) plus
   roles.json for the runtime's structured decoder

The runtime already treats reserved tokens as single special tokens, so a
trained model can emit them in free generation; constrained decoding (ADA-0006)
enforces exactness on top.

Usage
-----
# quick structural check (loads model, inits rows, prints stats, exits)
uv run python python/train_reserved_tokens.py --model ../models/safetensors --check

# full training run
uv run python python/train_reserved_tokens.py --model ../models/safetensors \
    --out out/reserved --steps 600 --epochs 3 --batch 4 --lr 2e-4

# custom role -> token mapping (optional)
uv run python python/train_reserved_tokens.py --model ../models/safetensors \
    --roles roles.json --data my_data.jsonl --out out/reserved
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

# --- NixOS workaround ---------------------------------------------------
# Triton probes `/sbin/ldconfig -p` to find libcuda.so, but NixOS has no
# /sbin/ldconfig. Point Triton at the driver directory from the flake's
# LD_LIBRARY_PATH instead via TRITON_LIBCUDA_PATH; env vars propagate into
# torch.compile's worker subprocesses, so this fixes kernel compilation too.
_libcuda_dir = next(
    (d for d in os.getenv("LD_LIBRARY_PATH", "").split(":")
     if d and os.path.exists(os.path.join(d, "libcuda.so.1"))),
    None,
)
if _libcuda_dir and not os.getenv("TRITON_LIBCUDA_PATH"):
    os.environ["TRITON_LIBCUDA_PATH"] = _libcuda_dir
# ------------------------------------------------------------------------

import torch
from datasets import Dataset
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from unsloth import FastLanguageModel

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Default structural roles -> reserved token strings. The strings must exist in
# the LFM2.5 vocab (checked at runtime); override with --roles.
DEFAULT_ROLES = {
    "json":    "<|reserved_10|>",  # begin structured block
    "obj":     "<|reserved_11|>",  # object start
    "arr":     "<|reserved_12|>",  # array start
    "key":     "<|reserved_13|>",  # field name follows
    "str":     "<|reserved_14|>",  # string value follows
    "num":     "<|reserved_15|>",  # number value follows
    "bool":    "<|reserved_16|>",  # boolean value follows
    "null":    "<|reserved_17|>",  # null value follows
    "end_obj": "<|reserved_18|>",  # object end
    "end_arr": "<|reserved_19|>",  # array end
    "end":     "<|reserved_20|>",  # end structured block
}

# Seed text used to mean-initialize each role's embedding row (tokenized, then
# averaged over the resulting token ids). Gives each row a sensible prior.
SEED_TOKENS = {
    "json":    ["{"],
    "obj":     ["{", "}"],
    "arr":     ["[", "]"],
    "key":     ['"', ":"],
    "str":     ['"'],
    "num":     ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."],
    "bool":    ["true", "false"],
    "null":    ["null"],
    "end_obj": ["}"],
    "end_arr": ["]"],
    "end":     ["<|im_end|>"],
    # GPU-token dataset delimiters (reserved_100/101) get a start/end prior.
    "gpu_start": ["<|im_start|>"],
    "gpu_end":   ["<|im_end|>"],
}

SYSTEM_PROMPT = (
    "You output structured data as a compact token stream. "
    "Objects start with <|reserved_11|>, arrays with <|reserved_12|>, keys with "
    "<|reserved_13|>, strings with <|reserved_14|>, numbers with <|reserved_15|>, "
    "booleans with <|reserved_16|> and null with <|reserved_17|>. "
    "End objects with <|reserved_18|>, arrays with <|reserved_19|> and the whole "
    "block with <|reserved_20|>."
)

# Built-in synthetic examples: (user prompt, python value).
BUILTIN_EXAMPLES: list[tuple[str, object]] = [
    ("Return a JSON object with keys name, age and active.", {"name": "Alice", "age": 30, "active": True}),
    ("Give me a JSON object with title and tags.", {"title": "Report", "tags": ["a", "b", "c"]}),
    ("Output JSON: city, population, area.", {"city": "Kraków", "population": 804237, "area": 326.8}),
    ("JSON with coordinates.", {"x": 1.5, "y": -2.25, "z": 0}),
    ("A JSON object where value is null.", {"note": None}),
    ("Return an empty object.", {}),
    ("Return an empty array.", []),
    ("Return a list of two JSON objects.", [{"id": 1, "ok": True}, {"id": 2, "ok": False}]),
    ("JSON with a nested object.", {"user": {"name": "Bob", "meta": {"level": 5, "vip": False}}}),
    ("Return a JSON array of numbers.", [1, 2, 3, 4, 5]),
    ("JSON with boolean fields only.", {"on": True, "off": False}),
    ("Return JSON with a long string field.", {"message": "hello world, this is a long string value"}),
    ("JSON: product, price, inStock.", {"product": "Widget", "price": 9.99, "inStock": True}),
    ("Return a nested array.", {"matrix": [[1, 0], [0, 1]]}),
    ("JSON with a single numeric field.", {"count": 42}),
    ("Return JSON describing a person.", {"person": {"name": "Carol", "age": 27, "hobbies": ["reading", "hiking"], "employed": True}}),
    ("JSON with email and phone.", {"email": "a@b.co", "phone": "+48123456789"}),
    ("Return an object with a null field and a number.", {"value": None, "delta": -7}),
    ("JSON array of strings.", ["red", "green", "blue"]),
    ("Return JSON for a book.", {"book": {"title": "Dune", "year": 1965, "authors": ["Frank Herbert"]}}),
]

RESERVED_RE = re.compile(r"^<\|reserved_(\d+)\|>$")


def parse_args() -> argparse.Namespace:
    # default to <repo>/models/safetensors regardless of CWD
    default_model = str(Path(__file__).resolve().parents[3] / "models" / "safetensors")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=default_model,
                   help="HF safetensors dir (or HF model id) of LFM2.5-1.2B-Instruct")
    p.add_argument("--out", default="out/reserved", help="output dir for adapter + roles.json")
    p.add_argument("--roles", default=None, help="optional JSON file mapping role -> <|reserved_N|> string")
    p.add_argument("--data", default=None, help="optional JSONL file of {\"input\": ..., \"output\": ...} examples")
    p.add_argument("--seq-len", type=int, default=512, help="max sequence length")
    p.add_argument("--batch", type=int, default=4, help="per-device train batch size")
    p.add_argument("--lr", type=float, default=2e-4, help="embedding learning rate")
    p.add_argument("--steps", type=int, default=600, help="max training steps (0 = train for --epochs)")
    p.add_argument("--epochs", type=float, default=3.0, help="max epochs")
    p.add_argument("--lora", action="store_true", help="also train a small LoRA on attention/MLP")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval", default=None,
                   help="JSONL validation file; after training, report reserved-token emission metrics")
    p.add_argument("--check", action="store_true",
                   help="load model, mean-init rows, (optionally build LoRA), print stats and exit (no training)")
    return p.parse_args()


# --------------------------------------------------------------------------
# Roles / vocab helpers
# --------------------------------------------------------------------------

def discover_reserved_ids(tokenizer) -> dict[str, int]:
    """Return {token_string: id} for every <|reserved_N|> token in the vocab."""
    vocab = tokenizer.get_vocab()
    return {t: i for t, i in vocab.items() if RESERVED_RE.match(t)}


def resolve_roles(tokenizer, roles: dict[str, str]) -> dict[str, int]:
    """Resolve {role: token_string} -> {role: token_id}, validating existence."""
    vocab = tokenizer.get_vocab()
    resolved: dict[str, int] = {}
    for role, tok in roles.items():
        if tok not in vocab:
            raise ValueError(f"role '{role}': token {tok!r} not in vocab (use an existing <|reserved_N|> token)")
        resolved[role] = vocab[tok]
    return resolved


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def to_structural(value: object, roles: dict[str, str]) -> list[str]:
    """Render a python value as the structural token stream (token list)."""
    out: list[str] = []

    def scalar(v: object) -> None:
        if isinstance(v, bool):
            out.append(roles["bool"]); out.append("true" if v else "false")
        elif v is None:
            out.append(roles["null"])
        elif isinstance(v, int):
            out.append(roles["num"]); out.append(str(v))
        elif isinstance(v, float):
            out.append(roles["num"]); out.append(repr(v))
        else:  # str
            out.append(roles["str"]); out.append(json.dumps(str(v)))

    def walk(v: object) -> None:
        if isinstance(v, dict):
            out.append(roles["obj"])
            for k, val in v.items():
                out.append(roles["key"]); out.append(json.dumps(k))
                walk(val)
            out.append(roles["end_obj"])
        elif isinstance(v, list):
            out.append(roles["arr"])
            for item in v:
                walk(item)
            out.append(roles["end_arr"])
        else:
            scalar(v)

    out.append(roles["json"])
    walk(value)
    out.append(roles["end"])
    return out


def _load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def encode_chat_example(tokenizer, messages: list[dict], seq_len: int) -> dict:
    """OpenAI-style messages -> ChatML with assistant-only labels.

    System/user tokens are masked with -100; only assistant content (plus the
    trailing <|im_end|>) is trained on. If too long, leading system/user turns
    are dropped before the assistant content is touched.
    """
    segments: list[tuple[list[int], bool]] = []  # (ids, is_assistant)
    for i, m in enumerate(messages):
        text = f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        seg_ids = tokenizer.encode(text, add_special_tokens=False)
        is_assistant = m["role"] == "assistant"
        if is_assistant or i < len(messages) - 1:  # keep all; drop from the front later if needed
            segments.append((seg_ids, is_assistant))
    if not any(is_a for _, is_a in segments):
        raise ValueError("dataset example has no assistant turn")

    overflow = sum(len(s[0]) for s in segments) + 1 - seq_len  # +1 for BOS
    while overflow > 0 and segments and not segments[0][1]:
        overflow -= len(segments.pop(0)[0])
    if overflow > 0 and segments:
        seg_ids, is_assistant = segments[0]
        if len(seg_ids) > overflow:
            # keep the segment start (leading delimiters live there); cut the tail
            keep = len(seg_ids) - overflow
            if is_assistant:
                print(f"  [warn] truncated assistant turn {len(seg_ids)} -> {keep} tokens "
                      f"(seq_len {seq_len} too small for this example)")
            segments[0] = (seg_ids[:keep], is_assistant)
        else:
            segments.pop(0)

    ids: list[int] = [tokenizer.bos_token_id]
    labels: list[int] = [-100]
    for seg_ids, is_assistant in segments:
        ids.extend(seg_ids)
        labels.extend(seg_ids if is_assistant else [-100] * len(seg_ids))
    return {"input_ids": ids[:seq_len], "labels": labels[:seq_len]}


def encode_io_example(tokenizer, prompt: str, completion: str, seq_len: int) -> dict:
    """{input, output} -> ChatML with completion-only labels."""
    prompt_ids = tokenizer.encode(
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        add_special_tokens=True,
    )
    comp_ids = tokenizer.encode(completion + "<|im_end|>\n", add_special_tokens=False)
    if len(prompt_ids) + len(comp_ids) > seq_len:
        overflow = len(prompt_ids) + len(comp_ids) - seq_len
        prompt_ids = prompt_ids[min(overflow, len(prompt_ids)):]
    if len(prompt_ids) + len(comp_ids) > seq_len:
        comp_ids = comp_ids[:max(seq_len - len(prompt_ids), 0)]
    ids = prompt_ids + comp_ids
    labels = [-100] * len(prompt_ids) + comp_ids
    return {"input_ids": ids, "labels": labels}


def build_dataset(tokenizer, role_tokens: dict[str, str], data_path: str | None, seq_len: int):
    """Load a JSONL dataset into tokenized examples with masked labels.

    Auto-detects the format per line:
      * {"messages": [{role, content}, ...]} -> ChatML, assistant-only labels
      * {"input": ..., "output": ...}        -> completion-only labels
    Without --data, built-in synthetic examples use the default structural roles.
    """
    if data_path:
        raw = _load_jsonl(data_path)
    else:
        raw = [{"input": q, "output": to_structural(v, DEFAULT_ROLES)} for q, v in BUILTIN_EXAMPLES]

    examples = []
    for ex in raw:
        if "messages" in ex:
            examples.append(encode_chat_example(tokenizer, ex["messages"], seq_len))
        else:
            comp = ex.get("output", "")
            if isinstance(comp, list):
                comp = " ".join(comp)
            examples.append(encode_io_example(tokenizer, ex.get("input", ""), comp, seq_len))
    return Dataset.from_list(examples)


class ChatLMCollator:
    """Pad input_ids (pad_token) and labels (-100) to the batch max length."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


# --------------------------------------------------------------------------
# Embedding init / freezing
# --------------------------------------------------------------------------

def mean_init_rows(model, tokenizer, roles: dict[str, int]) -> None:
    """Mean-initialize the reserved rows from semantically similar tokens."""
    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        for role, rid in roles.items():
            seeds = SEED_TOKENS.get(role, [])
            if not seeds:
                continue
            seed_ids = []
            for s in seeds:
                seed_ids.extend(tokenizer.encode(s, add_special_tokens=False))
            if not seed_ids:
                continue
            seed_ids = torch.tensor(sorted(set(seed_ids)))
            # compute in fp32 to be safe, then cast back
            mean_row = emb[seed_ids].float().mean(0)
            emb[rid] = mean_row.to(emb.dtype)


def freeze_except_reserved_rows(model, roles: dict[str, int]) -> list[torch.Tensor]:
    """Freeze everything except LoRA params and the reserved embedding rows.

    On a PEFT-wrapped model, LoRA parameters stay trainable; every other param
    is frozen. The reserved rows (and their tied LM-head rows) are then
    unfrozen and grad-masked so only those rows update.

    Returns the list of distinct trainable weight tensors that got the
    row-mask hook (for reporting).
    """
    is_peft = hasattr(model, "base_model") or hasattr(model, "peft_config")
    for name, p in model.named_parameters():
        if is_peft and ("lora_" in name or "modules_to_save" in name):
            continue  # keep LoRA parameters trainable
        p.requires_grad = False

    target_ids = torch.tensor(sorted(roles.values()), dtype=torch.long)

    def row_mask_hook(grad: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(grad)
        mask[target_ids.to(grad.device)] = 1.0
        return grad * mask

    hooked: list[torch.Tensor] = []
    seen: set[int] = set()
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        if module is None:
            continue
        w = module.weight
        if id(w) in seen:
            continue
        seen.add(id(w))
        w.requires_grad = True
        w.register_hook(row_mask_hook)
        hooked.append(w)
    return hooked


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------

def save_adapter(out_dir: Path, model, roles: dict[str, int], role_tokens: dict[str, str]) -> None:
    """Write reserved_rows.safetensors (k x hidden) + roles.json."""
    rows = model.get_input_embeddings().weight.detach().to(torch.bfloat16)
    role_ids = {role: int(rid) for role, rid in roles.items()}
    ids_tensor = torch.tensor(sorted(role_ids.values()), dtype=torch.int64)
    row_tensor = rows[ids_tensor]  # (k, hidden)

    from safetensors.torch import save_file
    save_file({"reserved_ids": ids_tensor, "reserved_rows": row_tensor}, out_dir / "reserved_rows.safetensors")
    (out_dir / "roles.json").write_text(json.dumps({
        "roles": role_ids,
        "tokens": {role: role_tokens[role] for role in role_ids},
        "hidden_size": row_tensor.shape[1],
        "n_rows": len(role_ids),
    }, indent=2))
    print(f"  adapter -> {out_dir / 'reserved_rows.safetensors'} "
          f"({row_tensor.shape[0]} rows x {row_tensor.shape[1]}) + roles.json")


class AdapterCheckpointCallback(TrainerCallback):
    """Periodically dump the tiny reserved-rows adapter (not full checkpoints)."""

    def __init__(self, out_dir, model, roles, role_tokens, every: int):
        self.out_dir, self.model = out_dir, model
        self.roles, self.role_tokens = roles, role_tokens
        self.every = max(1, every)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every == 0:
            save_adapter(self.out_dir, self.model, self.roles, self.role_tokens)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def verify(model, tokenizer, role_tokens: dict[str, str], prompt: str = "Return a JSON object with keys name and age.") -> None:
    from unsloth import FastLanguageModel as FLM

    FLM.for_inference(model)
    sys_text = SYSTEM_PROMPT
    text = (f"<|im_start|>system\n{sys_text}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    ids = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor([ids]).to(model.device)
    out = model.generate(input_ids=input_ids, max_new_tokens=48, do_sample=False)
    generated = tokenizer.decode(out[0][len(ids):], skip_special_tokens=False)
    print("\n--- verification generation ---")
    print(text)
    print(generated)
    role_hits = {role: generated.count(tok) for role, tok in role_tokens.items()}
    print("role token hits:", {k: v for k, v in role_hits.items() if v > 0} or "none yet (expected after more steps)")


def evaluate(model, tokenizer, val_path: str, roles: dict[str, int], max_new: int = 32) -> None:
    """Greedy-decode a JSONL validation set and report token-emission metrics.

    Uses each example's ``metadata.kind``:
      * gpu    -> want the reserved tokens emitted (+ a payload)
      * normal -> want NO reserved tokens emitted
    """
    from collections import Counter
    from unsloth import FastLanguageModel as FLM

    FLM.for_inference(model)
    raw = _load_jsonl(val_path)
    role_ids = set(roles.values())
    stats = Counter()

    for ex in raw:
        kind = ex.get("metadata", {}).get("kind", "normal")
        messages = ex["messages"]
        prompt = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            for m in messages[:-1]
        ) + "<|im_start|>assistant\n"
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        out = model.generate(
            input_ids=torch.tensor([ids]).to(model.device),
            max_new_tokens=max_new,
            do_sample=False,
        )
        gen_ids = out[0][len(ids):].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        emitted = [rid for rid in sorted(role_ids) if rid in gen_ids]

        if kind == "gpu":
            stats["gpu_total"] += 1
            stats["gpu_any_token"] += 1 if emitted else 0
            stats["gpu_all_tokens"] += 1 if len(emitted) == len(role_ids) else 0
            stats["gpu_payload"] += 1 if "simulate" in gen_text else 0
        else:
            stats["normal_total"] += 1
            stats["normal_clean"] += 1 if not emitted else 0

    def pct(num: int, den: int) -> str:
        return f"{100.0 * num / den:.1f}%" if den else "n/a"

    print("\n--- validation metrics (greedy decode) ---")
    print(f"  gpu examples: {stats['gpu_total']} | emitted any role token: "
          f"{pct(stats['gpu_any_token'], stats['gpu_total'])} | emitted ALL: "
          f"{pct(stats['gpu_all_tokens'], stats['gpu_total'])} | payload 'simulate': "
          f"{pct(stats['gpu_payload'], stats['gpu_total'])}")
    print(f"  normal examples: {stats['normal_total']} | clean (no role token): "
          f"{pct(stats['normal_clean'], stats['normal_total'])}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # transformers >= 5 rejects relative paths as repo ids; resolve to absolute.
    model_dir = str(Path(args.model).resolve())
    print(f"Loading tokenizer + model from {model_dir} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=args.seq_len,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )

    reserved = discover_reserved_ids(tokenizer)
    print(f"discovered {len(reserved)} <|reserved_N|> tokens "
          f"(ids {min(reserved.values())}..{max(reserved.values())})")

    roles_file = json.loads(Path(args.roles).read_text()) if args.roles else DEFAULT_ROLES
    role_tokens = {role: str(tok) for role, tok in roles_file.items()}  # role -> <|reserved_N|> string
    roles = resolve_roles(tokenizer, role_tokens)                       # role -> token id
    print("role -> token map:")
    for role, rid in sorted(roles.items(), key=lambda kv: kv[1]):
        print(f"  {role:8s} -> {tokenizer.convert_ids_to_tokens(rid)!r} (id {rid})")

    print("mean-initializing reserved rows ...")
    mean_init_rows(model, tokenizer, roles)

    # LoRA / training prep first, then freeze: unsloth's get_peft_model would
    # otherwise re-freeze our unfrozen embedding rows.
    if args.lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_r * 2,
            lora_dropout=0,
            use_gradient_checkpointing=False,
            random_state=args.seed,
        )
        print(f"LoRA enabled (r={args.lora_r}); embeddings remain trainable via grad mask")
    else:
        model = FastLanguageModel.for_training(model)

    hooked = freeze_except_reserved_rows(model, roles)
    print(f"frozen base; trainable weight tensors with row-mask: {[tuple(w.shape) for w in hooked]}")

    if args.check:
        if args.data:
            ds = build_dataset(tokenizer, role_tokens, args.data, args.seq_len)
            ex0 = ds[0]
            print(f"--check: dataset {len(ds)} examples; first example "
                  f"n_input={len(ex0['input_ids'])} n_train_labels="
                  f"{sum(1 for l in ex0['labels'] if l != -100)}")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"--check mode: not training. trainable params: {trainable:,}")
        return

    print("building dataset ...")
    dataset = build_dataset(tokenizer, role_tokens, args.data, args.seq_len)
    print(f"train examples: {len(dataset)}")

    train_kwargs = dict(
        output_dir=str(out_dir / "runs"),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        weight_decay=0.0,          # row-masked embedding: keep decay off so pretrained rows don't shrink
        bf16=True,
        warmup_ratio=0.05,
        logging_steps=25,
        save_strategy="no",        # full checkpoints are 2.4GB; the tiny adapter is saved via callback
        report_to=[],
        seed=args.seed,
        dataloader_drop_last=False,
        remove_unused_columns=False,
    )
    if args.steps > 0:
        train_kwargs["max_steps"] = args.steps
    else:
        train_kwargs["num_train_epochs"] = args.epochs
    training_args = TrainingArguments(**train_kwargs)

    collator = ChatLMCollator(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[AdapterCheckpointCallback(out_dir, model, roles, role_tokens, every=args.steps // 5 or 1)],
    )

    print("training ...")
    trainer.train()

    save_adapter(out_dir, model, roles, role_tokens)
    if args.eval:
        evaluate(model, tokenizer, args.eval, roles)
    verify(model, tokenizer, role_tokens)
    print("\ndone.")


if __name__ == "__main__":
    main()
