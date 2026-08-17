# Krystal shader sources

These are the shaders the runtime actually runs. They are linked at build time
into `src/krystal.artifact.generated.ts` by `scripts/build-krystal-artifact.ts`,
and the runtime loads that artifact — it never reads this directory.

Each `.wgsl` file contains **only the compute function body**. Sandblaster
generates the entry point, the builtin parameters and the resource declarations
from the `engine.compute({...})` call in `src/krystal-definition.ts`. `includes/`
holds shared helpers and workgroup scratch declarations, and is also where any
`var<workgroup>` must live — WGSL does not allow one in a function body, which
is what the body-only convention would otherwise tempt you into.

The header comment at the top of each body (`// entryPoint:`,
`// workgroupSize:`, `// builtins:`) is **documentation, not configuration**.
Nothing parses it. The values that take effect are in `krystal-definition.ts`,
so if the two disagree, the header is the one that is wrong.

## Where the truth lives

| question | file |
|---|---|
| which includes, which resources, which workgroup size | `src/krystal-definition.ts` |
| how many workgroups a dispatch launches | `src/krystal-layout.ts` (`defineKrystalPasses`) |
| which shader files exist | the `*_SHADER_NAMES` lists in `src/krystal-layout.ts` |
| which *programs* exist | `KRYSTAL_PROGRAM_NAMES` — the union of the lists above |

A per-shader table of includes used to live here and silently went stale, so it
is deliberately gone. `krystal-definition.ts` is short and it is the thing that
executes.

## Shaders

### Core elementwise/utility (`src/shaders/`)

| shader | WG | notes |
|---|---:|---|
| `matmul_f32` | 64 | y = x @ W^T against a weight32 page (row-reduce over input dim) |
| `residual_add` | 256 | arena[out] = arena[in] + arena[aux] |
| `arena_copy` | 256 | arena[out] = arena[in] |

### M1 training (`src/shaders/training/`)

| shader | WG | notes |
|---|---:|---|
| `embedding_f32` | 256 | hidden = E[tokens] against a weight32 page |
| `zero_f32` | 256 | zero a region |
| `cross_entropy_forward_backward` | 64 | fused CE forward + dLogits |
| `loss_reduce` | 64 | mean scalar loss telemetry |
| `matmul_backward_input` | 256 | dInput = dOutput @ W against a weight32 page |
| `matmul_backward_weight` | 256 | dW = dOutput^T @ input |
| `embedding_backward` | 256 | scatter-add of dHidden onto token rows |
| `sgd_step` | 256 | in-place parameter update against a weight32 page |
| `attention_forward` | 64 | persists softmax probs for backward |
| `attention_backward_scores` | 64 | softmax-score gradient |
| `attention_backward_qkv` | 256 | dQ/dK/dV |

### M2b Krystal forward (`src/shaders/training/`)

| shader | WG | notes |
|---|---:|---|
| `krystal_field_embed` | 256 | SoA frame -> field states (concatenated tables, weight32) |
| `krystal_attention_forward` | 64 | local record attention (bidirectional, host-masked) |
| `relu` | 256 | FFN activation |
| `krystal_pool` | 64 | learned-query pool (bank + query) |
| `krystal_selector` | 64 | catalog selection + soft gather |
| `krystal_decision_head` | 64 | typed route-kind logits (weight32) |

### M3 Krystal backward (`src/shaders/training/`)

| shader | WG | notes |
|---|---:|---|
| `relu_backward` | 256 | |
| `krystal_attention_backward_scores` | 64 | softmax-score gradient |
| `krystal_attention_backward_qkv` | 256 | dQ/dK/dV |
| `krystal_field_embed_backward` | 256 | scatter-add into the concatenated tables |
| `krystal_pool_backward` | 64 | pool softmax/score gradients + dPool partials |
| `krystal_pool_dpool` | 256 | reduce dPool partials over records |
| `krystal_selector_backward_scores` | 64 | soft-gather score gradient + pointer loss |
| `krystal_selector_backward_qkv` | 256 | dQProj/dKProj/dValue |
| `krystal_decision_head_backward` | 256 | dCtx (3 parts) + dWh |
