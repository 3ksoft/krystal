# Finetune

LoRA training tooling for Chomato's reference model **`LiquidAI/LFM2.5-1.2B-Instruct`**,
built on [Unsloth](https://github.com/unslothai/unsloth).

## Python environment (uv + Unsloth)

The training venv lives in this package (`.venv`) and is managed by `uv`.

Prerequisites (provided by the repo's nix devenv at the repo root — `flake.nix` / `.envrc`):

* `uv`
* NVIDIA driver with CUDA exposed via the flake's `LD_LIBRARY_PATH`
  (`/run/opengl-driver/lib` on NixOS)

Setup:

```bash
cd packages/finetune
uv sync            # creates .venv, pins everything in uv.lock
```

This installs the full Unsloth training stack on Python 3.13:

| Package        | Version            | Note                                   |
| -------------- | ------------------ | -------------------------------------- |
| unsloth        | 2026.8.2           | native `Lfm2` patching                 |
| torch          | 2.11.0+cu130       | CUDA build, RTX 3060 (sm_86) verified  |
| transformers   | 5.5.0              |                                       |
| trl            | 0.24.0             | SFT/GRPO trainers                      |
| peft           | 0.20.0             | LoRA adapters                          |
| bitsandbytes   | 0.50.0             | 4-bit QLoRA                            |
| accelerate     | 1.14.0             |                                       |
| datasets       | 4.3.0              |                                       |
| xformers       | 0.0.35             | attention backend                      |

Activate:

```bash
source .venv/bin/activate   # or: uv run python ...
```

Smoke test (model load, 4-bit, on GPU):

```bash
uv run python - <<'EOF'
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "LiquidAI/LFM2.5-1.2B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_training(model)
print(model, tokenizer)
EOF
```

### NixOS notes

* Unsloth prints *"CUDA is not linked properly"* on NixOS — harmless. The driver's
  `libcuda.so.1` is found via the flake's `LD_LIBRARY_PATH`, so `torch.cuda.is_available()`
  returns `True`. Ignore the `sudo ldconfig /usr/lib64-nvidia` suggestion.
* Triton probes `/sbin/ldconfig` (absent on NixOS) to find `libcuda.so`. The training
  script sets `TRITON_LIBCUDA_PATH` from `LD_LIBRARY_PATH` automatically; for any other
  tool, `export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib` does the same.
* If `HF_HOME` points at a path that doesn't exist (e.g. an unmounted `/mnt/ssd/hf`),
  Unsloth redirects model downloads to a temp dir. Point `HF_HOME` at a real, writable
  location to cache `LFM2.5-1.2B-Instruct` between runs.

## Reserved-token training (structured output)

The LFM2.5 vocab has **377 `\<|reserved_N|>` tokens** (ids 14–395) plus 1,136 unused
slots (ids 64,400–65,535), all inside the 65,536 vocab — so structural tokens can be
trained **without growing the vocabulary** (the runtime's `2^16` masks / WQ4 layout stay
valid, and the TS tokenizer already treats reserved tokens as single special tokens).

`python/train_reserved_tokens.py` trains a handful of reserved ids as structural tokens:

```bash
# quick check (loads model, mean-inits rows, prints the role map, exits)
uv run python python/train_reserved_tokens.py --check

# real run: embedding-only, base frozen, tied LM head follows the rows
uv run python python/train_reserved_tokens.py \
    --out out/reserved --steps 600 --epochs 3 --batch 4 --lr 2e-4
```

What it does:

1. assigns roles (json/obj/arr/key/str/num/bool/null/end…) to reserved ids,
2. mean-initializes those rows from semantically similar tokens (`{`, `[`, digits, …),
3. freezes the base and trains only the selected rows (grad-masked, so pretrained rows
   stay untouched; `weight_decay=0` for the same reason),
4. saves `reserved_rows.safetensors` (k×2048) + `roles.json` for the runtime's
   structured decoder, then runs a quick generation check.

Custom role map (`--roles roles.json`) and dataset (`--data data.jsonl`) are
supported; `--data` auto-detects OpenAI `messages` format (assistant-only labels)
vs `{"input", "output"}`. Pass `--lora` to also train a small LoRA on
attention/MLP, and `--eval val.jsonl` for post-training emission metrics.

### GPU-token dataset run

`src/generate_gpu_token_dataset.ts` builds a synthetic dataset that teaches the
model to emit `<|reserved_100|>` / `<|reserved_101|>` as GPU-command delimiters
(`<|reserved_100|>simulate<|reserved_101|>`), with hard negatives
that mention simulation but must NOT emit tokens. Regenerate with:

```bash
deno run --allow-write src/generate_gpu_token_dataset.ts
```

Full run (prepared, not executed automatically):

```bash
bash run_gpu_tokens.sh
# = uv run python python/train_reserved_tokens.py --model ../models/safetensors \
#     --data gpu-token-dataset/train.jsonl --eval gpu-token-dataset/validation.jsonl \
#     --roles roles/gpu-tokens.json --out out/gpu-tokens \
#     --steps 600 --epochs 4 --batch 4 --seq-len 512 --lr 2e-4 --lora --lora-r 16
```

This trains LoRA on attention/MLP **and** the two delimiter embedding rows
(grad-masked), on assistant turns only, saves the LoRA under `out/gpu-tokens/lora/`,
then reports validation metrics
(% examples emitting the tokens correctly, % clean on normal examples).

Dry-run sanity check (loads the model, builds LoRA, reports stats, no training):

```bash
uv run python python/train_reserved_tokens.py \
  --data gpu-token-dataset/train.jsonl --roles roles/gpu-tokens.json --lora --check
```

## Package layout

* `python/` — intended home for training scripts and dataset generation code.
* `tokenizer.json` — LFM2.5 tokenizer copy used by the TS/WebGPU runtime.

## Exploration areas

* **Structured output tokens**
  Use reserved tokenizer tokens as a compact structural vocabulary for schema-driven generation and constrained decoding.

* **Knowledge adapters**
  Build LoRA adapters as versioned knowledge packages, for example:

  * `node-25.31.1`
  * `eu-vat-tax-2026`
  * project-specific APIs or documentation

* **Dataset generation**
  Generate training and evaluation datasets from structured sources such as documentation, schemas, and examples.

* **Evaluation and packaging**
  Measure adapter quality and package trained adapters together with the metadata required by the runtime.

This package depends on Python-based ML tooling but remains isolated from the inference runtime.
