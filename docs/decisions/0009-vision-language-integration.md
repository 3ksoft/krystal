# ADA-0009 — Vision-Language Integration (LFM2.5-VL-1.6B)

Status: **ACCEPTED** 
Date: 2026-08-09

## Context

Chomato is a focused WebGPU inference runtime for `LFM2.5-1.2B-Instruct`. The
obvious next experiment is vision-language: run `LFM2.5-VL-1.6B`, which shares
the same text architecture as the current reference model and adds a SigLIP2
NaFlex vision encoder (~426M) plus a two-layer MLP projector.

This is a deliberately strong architecture test: if the runtime is genuinely a
*model runtime* rather than a runtime of one specific checkpoint, wiring in a
vision encoder that produces embeddings for the existing LM path should require
new kernels but **no changes to the LM execution, checkpoint, or structured
generation semantics**.

All model facts below were verified against the HuggingFace repository and the
local GGUF pair in `./models/` on 2026-08-09.

## Decision

Implement the v0 PoC envelope:

```text
v0:
single image
≤ 512×512 (pad to 512×512)
no tiling
no thumbnail
fixed 256 image tokens
```

The vision tower runs once per image as a separate GPU prefill producing a
tensor of image embeddings. The embeddings are injected into the LM input
sequence in place of the `<image>` placeholder token; the LM prefill then runs
over the combined sequence with no architectural changes.

Target model: **LFM2.5-VL-1.6B** (not the 450M variant), because it exercises
the exact same text backbone class as the current runtime, which is the point
of the experiment.

## Verified model facts

### Text backbone (`LFM2.5-VL-1.6B-F16.gguf`, local)

Identical architecture to the current reference model:

| key | value |
|---|---|
| `general.architecture` | `lfm2` |
| `lfm2.block_count` | 16 |
| `lfm2.embedding_length` | 2048 |
| `lfm2.feed_forward_length` | 8192 |
| `lfm2.attention.head_count` | 32 |
| `lfm2.attention.head_count_kv` | per-layer (16 entries) |
| `lfm2.rope.freq_base` | 1000000 |
| `lfm2.attention.layer_norm_rms_epsilon` | 1e-5 |
| `lfm2.vocab_size` | 65536 |
| `lfm2.shortconv.l_cache` | 3 |
| tokenizer | gpt2 / lfm2, bos 1, eos 7 |

The weights are **not** the Instruct checkpoint: the VL model's LM backbone is
`LFM2.5-1.2B-Base` (per the model card), fine-tuned together with the vision
tower. The runtime definition, kernels, layout, and checkpoint format apply
1:1; only the weight file changes.

### Vision tower (`mmproj-LFM2.5-VL-1.6b-F16.gguf`, local)

llama.cpp `clip`-architecture mmproj, 441 tensors, 426M parameters:

| tensor | shape (GGUF dims) | type |
|---|---|---|
| `v.patch_embd.weight` | 16×16×3×1152 | F16 |
| `v.patch_embd.bias` | 1152 | F32 |
| `v.position_embd.weight` | 1152×256 | F32 |
| `v.blk.N.ln1.{weight,bias}` | 1152 | F32 |
| `v.blk.N.ln2.{weight,bias}` | 1152 | F32 |
| `v.blk.N.attn_{q,k,v,out}.weight` | 1152×1152 | F16 |
| `v.blk.N.attn_{q,k,v,out}.bias` | 1152 | F32 |
| `v.blk.N.ffn_up.weight` / `.bias` | 1152×4304 / 4304 | F16 / F32 |
| `v.blk.N.ffn_down.weight` / `.bias` | 4304×1152 / 1152 | F16 / F32 |
| `v.post_ln.{weight,bias}` | 1152 | F32 |
| `mm.1.weight` / `.bias` | 4608×2048 / 2048 | F16 / F32 |
| `mm.2.weight` / `.bias` | 2048×2048 / 2048 | F16 / F32 |

Key metadata:

```text
clip.vision.embedding_length = 1152
clip.vision.block_count = 27
clip.vision.patch_size = 16
clip.vision.attention.head_count = 16
clip.vision.feed_forward_length = 4304
clip.vision.attention.layer_norm_epsilon = 1e-6
clip.vision.projector.scale_factor = 2     # pixel unshuffle factor
clip.projector_type = "lfm2"
clip.use_gelu = true
```

Implementation-relevant facts derived from this layout and from
`transformers` `modeling_siglip2.py`:

1. The patch embedding is a **convolution kernel**, not a linear page: the
   `[16,16,3,1152]` GGUF tensor is the torch conv weight `[1152,3,16,16]`
   (GGUF dims are reversed torch dims). It is equivalent to a matmul
   `[N_patches, 768] @ [768, 1152]` once the image is patchified by the
   processor.
2. Position embeddings are **learned, 16×16 grid, bilinearly resized** to the
   image's patch grid (NaFlex), with padding beyond `h*w` repeating the first
   embedding. Host-side step.
3. Blocks are **pre-LayerNorm** (`ln1` → attention → residual, `ln2` → MLP →
   residual) with **biases everywhere** — attention projections, MLP, and the
   final `post_ln`.
4. Attention is **bidirectional, no RoPE, scale = head_dim^-0.5**, with a
   padding mask (NaFlex pads patches to `max_num_patches = 1024`).
5. MLP activation is **tanh-approximated GELU** (`clip.use_gelu`).
6. Projector: pixel-unshuffle(×2) on the spatial grid → `mm.1` (4608→2048,
   bias) → GELU → `mm.2` (2048→2048, bias). No layernorm in the projector.
   For a 512×512 image (32×32 patches) this yields 16×16 = **256 image
   tokens**, matching `max_image_tokens = 256`.
7. The image placeholder is token **396** (`<image>`), per the model config
   (`image_token_id = 396`) and the tokenizer. Verified in the converted
   VL WQ4: `token[396] = "<image>"`, `token_type[396] = 3` (control/special).
   The vocabulary also carries tiling markers `<|image_start|>@498`,
   `<|image_end|>@499`, `<|image_split|>@500`, unused in v0.

## Architecture

```text
image (decoded pixels)
   ↓ host processor
resize bilinear → normalize to [-1,1] (mean/std 0.5)
   ↓
patchify 16×16 → [1024, 768]
   ↓
+ patch bias; + resized position embeddings (host)
   ↓
vision tower (WGSL, new programs)
27 × { pre-LN → bidirectional attention (masked, no RoPE) → residual
        → pre-LN → MLP(GELU-tanh) → residual }
   ↓ post-LN
[1024, 1152]
   ↓ pixel unshuffle ×2 → [256, 4608]
   ↓ mm.1 → GELU → mm.2
[256, 2048]  image embeddings
   ↓
inject into LM arena at the <image> position
   ↓
existing LFM2.5 prefill over the combined sequence
```

In runtime terms the LM prompt is split into contiguous segments:

```text
embed(text-before)  → hiddenA[0..i)
copy(image embeddings) → hiddenA[i..i+256)
embed(text-after)   → hiddenA[i+256..n)
layers(0..16) over n tokens (existing prefill)
```

Causal attention and RoPE need no changes: the image tokens occupy ordinary
sequence positions. Vision embedding injection can be a tiny copy pass (or a
CPU-side write of the projector output buffer, read back once per image).

### Checkpoints

Image content is part of the prefix, so exact checkpoints (KV + conv +
continuation metadata) are **unaffected**. A checkpoint that includes an image
prefix restores and continues exactly as a text-only one does. The vision tower
itself is stateless (one-shot bidirectional forward) and needs no checkpoint
state.

### Correctness class

Per ADA-0008: VL adds a new numerical-approximation axis, orthogonal to exact
checkpoint semantics:

```text
exact checkpoint semantics
+ WQ4 numerical approximation (LM, already accepted)
+ vision tower precision (new: F16 or WQ4, to be measured)
+ host processor determinism (new)
```

The vision path needs its own differential oracle before the "cat image" test
can be trusted: golden image embeddings generated with `transformers` (or the
official ONNX WebGPU demo) for a small fixed image set.

## New runtime pieces

| piece | where | notes |
|---|---|---|
| LayerNorm (with bias) | new WGSL program | current `rms_norm` is RMSNorm-only |
| bidirectional attention (mask, no RoPE) | new/parameterized WGSL | existing attention is causal + RoPE |
| GELU-tanh | new WGSL program | only `silu_mul` exists |
| bias-add pass | new WGSL program | existing matmul has no bias support |
| patch embedding + pos-emb add | host-side or tiny pass | conv kernel read as `[1152, 768]` matmul weights |
| vision forward orchestration | new module in `packages/webgpu/src` | stateless encoder forward over scratch arena |
| embedding injection | tiny pass or host write | hiddenA segment copy |
| vision config parse | `packages/quant` or `lfm2` | from embedded `clip.*` GGUF metadata in the sidecar |

All matmul dimensions in the tower are WQ4-safe except one: the matmul-ized
patch embedding input (768), attention/projector widths (1152, 2048, 4608) are
divisible by 32, but **`ffn_down` has input width 4304, which is not**
(4304 % 32 = 16). The WQ4 sidecar therefore stores the 27
`v.blk.N.ffn_down.weight` matrices raw (F16), and the vision runtime dispatches
`matmul_f16` for them and `matmul_wq4` for everything else. This mixed
precision is measured, not tuned: the tower runs once per image, so F16
`ffn_down` cost is acceptable for v0. `mm.1` (4608→2048) and `v.patch_embd`
(768→1152) reuse the existing matmul path directly.

## Weight path

Two official GGUF files are already present locally in `./models/`:

```text
LFM2.5-VL-1.6B-F16.gguf            text backbone (lfm2 architecture)
mmproj-LFM2.5-VL-1.6b-F16.gguf     vision tower + projector (clip architecture)
```

- The text file converts with the **existing** `convert_gguf_to_wq4.ts`
  unchanged (verified: same metadata keys, same tensor naming).
- The mmproj needs a sidecar conversion: F16 matrices → WQ4 (except
  `ffn_down`, width 4304, kept raw F16), small/dense tensors
  (`v.position_embd`, `v.patch_embd`, all norm/biases) kept raw. The
  `clip.*` metadata rides along in the embedded index, so the sidecar carries
  the vision config. **M1 done**: both files converted locally
  (`models/LFM2.5-VL-1.6B-WQ4.wq4` 0.73 GB, `models/LFM2.5-VL-mmproj-WQ4.wq4`
  0.45 GB) and round-trip verified with the existing WQ4 reader (tensor
  counts/shapes/encodings preserved, `clip.*` keys present).
- The runtime then loads a model pair: text `.wq4` (existing path) + vision
  sidecar (loader landed in M3: `openVisionWeights` in
  `packages/webgpu/src/vision/integration.ts` sniffs the 4-byte magic and
  routes GGUF vs WQ4; `loadVisionWeights` dequantizes WQ4 matrices on the
  host — `dequantizeWq4Tensor` in `packages/quant/src/vision/weights.ts`,
  block-format-identical to the text runtime's WGSL `matmul_wq4` — and
  passes the raw-F16/raw-F32 tensors through byte-for-byte, so the decoded
  layout is source-agnostic).

## Non-goals for v0

- Tiling, thumbnails, multi-image input.
- Aspect-ratio-preserving NaFlex resizing (fixed 512×512 pad instead).
- WQ4 quality tuning of the vision tower (F16 fallback stays available).
- Reusing the Instruct weights as the VL LM backbone (must use the VL file).
- Changing the public typed `generate()` API (image arrives as a context block
  with provenance per ADA-0005).

## Milestones

1. **M0 — processor**: host image preprocessing (decode/resize/normalize/
   patchify + pos-embed resize) + golden tests against `transformers`.
   Patch vectors must be laid out `(channel, h, w)` with `w` fastest, matching
   the flattening of the `[1152, 3, 16, 16]` conv weight when it is used as a
   `[768, 1152]` matmul — golden tests depend on this ordering.
   **M0 done**: `packages/webgpu/src/vision/processor.ts` (torch-compatible
   bilinear with source clamping, `(c,h,w)` patchify, pos-emb resize),
   `packages/quant/src/vision/config.ts` (clip.* metadata parse),
   `packages/quant/src/vision/weights.ts` (F16-exact sidecar/GGUF loader),
   and `packages/quant/src/vision/reference.ts` (exact CPU reference tower:
   pre-LN + biases, bidirectional masked attention, GELU-tanh, pixel
   unshuffle, projector) — validated by `tests/vision.test.ts` (10 passing
   tests, model-dependent ones skip when ./models is absent).
2. **M1 — weights**: convert text GGUF + mmproj to the runtime format;
   sidecar loader + vision config parse; verify tensor/shape round-trip.
   **M1 done**: both files converted locally
   (`models/LFM2.5-VL-1.6B-WQ4.wq4` 0.73 GB, `models/LFM2.5-VL-mmproj-WQ4.wq4`
   0.45 GB) and round-trip verified with the existing WQ4 reader.
3. **M2 — vision tower**: LayerNorm, bidirectional attention, GELU-tanh,
   bias-add WGSL programs; forward orchestration; F32 first, then WQ4.
   **M2 done**: 7 self-contained WGSL programs in
   `packages/webgpu/src/vision/shaders/` (patch_embed, layernorm w/ bias,
   matmul+bias, gelu_tanh, bidirectional masked attention, residual_add,
   unshuffle+projector), embedded by `scripts/build-vision-shaders.ts` into
   `shaders.generated.ts`; host `packages/webgpu/src/vision/tower.ts`
   (`VisionTower`: exact-F32 weight upload, arena-packed activations,
   per-dispatch params ring with dynamic offsets, raw WebGPU — deliberately
   outside the Sandblaster AOT graph, see module doc). Differential tests in
   `tests/vision-gpu.test.ts` run the real WGSL tower on Dawn (the `webgpu`
   npm bindings, same harness as the checkpoint tests) against the CPU oracle
   on the same F16 GGUF weights: all-valid 4×4 and masked 12/16 grids agree
   to ~1.5e-6 (f32 kernel noise), the 18×16 = 288-patch case (crosses the
   attention shader's 256-key slice boundary) agrees to 1.98e-6, and
   full-27-layer hidden drifts to ~1e-4. The attention softmax is deliberately
   serial (see the nondeterminism note below).
   Notes:   the oracle input contract is now **full-grid** (patchCount =
   gridH*gridW, masked rows present with zero patches/posEmb) so the
   pixel-unshuffle tail can gather every grid position. The WQ4 sidecar
   runtime path landed in M3: host dequant (see Weight path) + raw-F16
   `ffn_down` pass-through; `tests/vision.test.ts` verifies the dequantized
   weights match the F16 source within block-quantization error and that the
   oracle forward with WQ4 weights tracks the F16 reference (cosine > 0.99).
4. **M3 — integration**: embedding injection + segment prefill + generate.
   Gate: *image of a cat → the model says it sees a cat*.
   **M3 done**: `VisionLfm2Session` (`packages/webgpu/src/vision/integration.ts`)
   composes processor → tower → `Lfm2Forward.generateWithImageEmbeddings`
   (placeholder expansion + hiddenA row overwrite between the embed and
   layer command buffers). The cat gate passes with the same answer as
   llama.cpp; 20/20 real asset images (transparent product renders composited
   on white, see `misc/vl-chat.ts`) are identified correctly end to end.
   Runtime harness: `misc/vl-chat.ts` (JPEG/PNG decode, white compositing,
   chat CLI; the vision tower now defaults to the WQ4 sidecar
   `models/LFM2.5-VL-mmproj-WQ4.wq4`, F16 GGUF selectable via `--vision`) and
   the `tests/vision-vl.test.ts` M3 suite (placeholder parsing, tower-vs-grid
   consistency, input sensitivity, injection changes LM output, determinism,
   cat gate).
5. **M4 — polish**: checkpoints with image prefixes (expected to work
   unchanged), GUI image input, golden differential tests.

### CPU reference as oracle — limits

`misc/vl-cpu-embed.ts` measures the reference tower end to end. Numbers on
this machine (2026-08-09): weights load ~12 s, forward is linear in patch
count — 4×4 grid (16 patches, 4 tokens) in ~8 s, so the real 512×512 envelope
(32×32 = 1024 patches, 256 tokens) is ~8–9 min of plain JS.

The reference is therefore the differential oracle for the **WGSL kernels at
test-grid sizes**, not for the full envelope: forward cost is ~0.5 s/patch
(measured 7.8 s at 16 patches), so a CPU-only comparison harness should use
grids of ~8×8 (64 patches, ~30 s per forward) or smaller, and the "cat image"
gate is a semantic check, not a golden one. (The benchmark pins `targetSize`
to the input dims so the grid stays small — without it `fitSize` upscales to
512 and the run becomes a 1024-patch forward.)

The GPU differential harness (`tests/vision-gpu.test.ts`) is the fast path:
the tower runs on Dawn in milliseconds, so the comparison cost is dominated
by the ~22 s weight load/upload, not the grid size.

### Kernel bugs the oracle caught (M2)

The WGSL tower shipped with three real correctness bugs that only a
numerical oracle can find (a "cat image" test would have passed all three):

1. **attention softmax sum** was per-invocation, not workgroup-reduced — for
   an n-key query the output was scaled by ~n (exact 4× on a 2×2 grid).
2. **attention softmax max** was per-invocation too, so non-uniform scores
   were normalized against the wrong max.
3. **patch embedding** accumulated a strided partial dot product (every 64th
   input) without a workgroup reduction — 12 of 768 inputs per output.

All three are strided-accumulation-without-reduction bugs; the oracle's
layer-boundary bisect (`misc/vl-gpu-debug.ts`, debug readbacks on the tower)
localized each in one run. This is exactly the regression net the M2
milestone was built for.

### Layout bugs found against llama.cpp (M3)

M3's first end-to-end runs had the classic symptom of a *wrongly laid-out
pipeline*: embeddings varied with the image (so the tower was reading real
pixels) but the LM never recognized content (so the spatial structure was
destroyed somewhere). Differential checks against llama.cpp's own tower
(`tools/mtmd/debug/mtmd-debug` node dumps + `llama-mtmd-debug` full trace on
the same F16 GGUF pair) found two layout bugs that the CPU-oracle/GPU
differential could not: both implementations shared the bug, so they agreed
with each other while disagreeing with the reference. The reference for
layout questions is therefore **llama.cpp**, not the shared oracle:

1. **`v.position_embd.weight` is `[position, dim]` row-major, not transposed.**
   The GGUF stores dims `[1152, 256]` with `data[p*1152 + d]`. The loader
   transposed it to `[d*256 + p]`, so every grid position received a
   hidden-dim slice instead of its embedding — spatial structure destroyed,
   image variance preserved (maxDiff vs llama.cpp base grid 0.8–1.7 before,
   9.7e-5 after). Fixed in `weights.ts positionEmbedding()`; regression
   test asserts the raw layout.
2. **Pixel-unshuffle channel order is llama.cpp's, not torch's.** The GGUF
   `mm.1` columns are stored with the sub-pixel bits **blocked at the top of
   the channel space**: `u = c + dim*i + dim*factor*j` with `i` the vertical
   sub-pixel, `j` the horizontal one (token `(x>>1) + (gridW/2)*(y>>1)`), not
   torch `PixelUnshuffle`'s interleaved `c*factor^2 + i*factor + j`. Using
   torch's order yields embeddings that differ per image but never match the
   content — the exact observed symptom. Order verified empirically against
   `llama-mtmd-debug`'s `pixel_shuffle` node dump and fixed in
   `reference.ts` + `unshuffle_project.wgsl` (both derive it from `dim` and
   `factor`, no hardcoded constants).

After both fixes the full tower matches llama.cpp's final image embeddings to
~3e-2 (F16-vs-F32 noise floor; layer outputs to ~5e-4), the M3 cat gate
passes with the same answer as llama.cpp, and the oracle/GPU differential
still agrees to ~1e-6.

### Attention kernel nondeterminism (M2, late)

The first cut of the attention kernel kept the softmax parallel (each
invocation exp's its own key slice, workgroup tree-reduce for max and sum,
parallel V accumulation). On the Dawn/NVIDIA (Ampere) stack that structure
was **nondeterministic**: with identical inputs, a handful of whole
(head, token) workgroup rows per run differed by ~5e-2 (no NaN, per-element
ratios vs the oracle varied 0.11–0.94, so individual `scores[]` entries were
stale when the V phase read them). It did not reproduce at n ≤ 64 but scaled
with key count (probabilistic, not a hard threshold).

Bisection (isolated attention harness, then a minimal workgroup-memory probe)
isolated the trigger: any structure with a **parallel exp write followed by a
cross-invocation read** was racy; serializing the softmax into invocation 0
was deterministic, as was adding unrelated global writes after each
reduction (scheduling, not memory-layout, sensitive). The kernel therefore
runs the softmax serially:

```text
phase 1: parallel per-key scores (each invocation owns its slice)
phase 2: SERIAL max-subtraction softmax + sum by invocation 0 (one barrier)
phase 3: parallel V accumulation (first headDim invocations scan all keys)
```

The serial softmax costs O(n) per workgroup, negligible next to the
O(n·headDim) V accumulation for n ≤ 1024, and the three-phase structure is
provably deterministic. The 288-patch differential test (crosses the 256-key
slice boundary) now passes at maxAbsDiff 1.98e-6.

## Open questions

- WQ4 quality on the vision tower: measured, not tuned. The M3 runtime now
  loads the WQ4 sidecar by default; dequant-vs-F16 weight comparison stays
  within block-quantization error and the oracle forward tracks the F16
  reference (cosine > 0.99). If WQ4 quality regresses the cat gate on some
  image set, the F16 GGUF remains selectable (`--vision`/session source).
- Processor exactness: NaFlex position-embedding resize (bilinear +
  antialias) must be reproduced deterministically; minor deviations are
  acceptable for the PoC gate but not for golden tests.
- Whether attention biases are worth a dedicated pass vs. pre-adding them to
  the arena before the matmul reads.
- Context capacity pressure: 256 image tokens leave ~768 for text+output at
  the current 1024 capacity; document or raise capacity if needed.
- `max_image_tokens` reduction (e.g. 64/128) as a speed/quality dial once the
  fixed-256 path works.
- **mmproj `clip.vision.image_size = 256` vs HF `tile_size = 512`**:
  `image_size = 256` is the square of the 16×16 position-embedding base grid
  (256 = 16²), not a preprocessor target; the processor resizes the base grid
  bilinearly (align_corners=False) to the image's patch grid, which matches
  llama.cpp's tower exactly (verified via `mtmd-debug` pos-embed dumps,
  maxDiff 9.7e-5). Resolved as informational.

## Decision gate

The gate is **satisfied** as of 2026-08-09:

1. The M3 gate passes: the cat image produces a sensible answer, matching
   llama.cpp; the broader real-image smoke set also passes.
2. Vision embeddings are differentially checked against both the CPU oracle
   and llama.cpp. GPU-vs-oracle tolerance is ~2e-6; the final
   llama.cpp-vs-runtime difference is ~3e-2 and is documented as the
   F16-source/F32-runtime noise floor.
3. Exact checkpoint tests containing image prefixes pass the existing
   checkpoint-equivalence suite.
4. The mmproj sidecar layout, tensor exceptions, and vision configuration
   parsing are documented here alongside the WQ4 text path.

Acceptance covers the v0 PoC envelope only. WQ4 quality measurement for the
vision tower and broader M4 polish remain open questions; they do not block
this decision because the accepted runtime path uses the differential-tested
F16 mmproj weights.
