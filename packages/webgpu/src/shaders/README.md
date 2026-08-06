# LFM2 shader sources

This directory is a **non-wired migration staging area**. The running LFM2 runtime still uses `packages/lfm2/src/shaders/runtime.wgsl`.

Each top-level `.wgsl` file contains only the compute function body. Sandblaster is expected to generate the entry-point wrapper, builtin parameters and resource declarations. `includes/` contains shared WGSL helpers/scratch declarations.

| shader | WG | builtins | logical resources | includes |
|---|---:|---|---|---|
| `embedding` | 256 | global_invocation_id -> gid | op, runtime, tokens, arena, weightRaw | arena-index, weights-f16 |
| `embedding_wq4` | 256 | global_invocation_id -> gid | op, runtime, tokens, arena, weightRaw | arena-index, weights-wq4 |
| `rms_norm` | 64 | workgroup_id -> wid; local_invocation_id -> lid | op, arena, weight32 | constants, arena-index, reduce-f32 |
| `matmul_f16` | 64 | workgroup_id -> wid; local_invocation_id -> lid | op, arena, weightRaw | constants, arena-index, weights-f16, reduce-f32 |
| `matmul_f32` | 64 | workgroup_id -> wid; local_invocation_id -> lid | op, arena, weight32 | constants, arena-index, reduce-f32 |
| `matmul_wq4` | 64 | workgroup_id -> wid; local_invocation_id -> lid | op, arena, weightRaw | constants, reduce-f32 |
| `residual_add` | 256 | global_invocation_id -> gid | op, arena | - |
| `silu_mul` | 256 | global_invocation_id -> gid | op, arena | - |
| `shortconv_prefill` | 256 | global_invocation_id -> gid | op, arena, convCache, weight32 | - |
| `shortconv_continue` | 1 | workgroup_id -> wid | op, arena, convCache, weight32 | - |
| `shortconv_decode` | 256 | global_invocation_id -> gid | op, arena, convCache, weight32 | - |
| `qk_norm_rope` | 64 | workgroup_id -> wid; local_invocation_id -> lid | op, runtime, arena, weight32 | constants, token-position, rope, reduce-f32 |
| `kv_store` | 256 | global_invocation_id -> gid | op, runtime, arena, kvCache | constants, token-position |
| `attention` | 64 | workgroup_id -> wid; local_invocation_id -> lid | op, runtime, arena, kvCache | constants, token-position, attention-scores |
| `arena_copy` | 256 | global_invocation_id -> gid | op, arena | - |
| `argmax_candidates` | 256 | local_invocation_id -> lid | op, runtime, tokens, arena, candidateTokens, decodeTelemetry | constants, reduce-f32, reduce-u32, telemetry |
| `argmax` | 256 | local_invocation_id -> lid | op, runtime, tokens, arena, decodeTelemetry | constants, reduce-f32, reduce-u32, telemetry |

Notes:

- Resource names are the names used by the current monolithic shader; they are not binding assignments.
- `op` is the current per-dispatch parameter block. Sandblaster can later own its uniform declaration.
- `telemetry.wgsl` is copied mechanically from the current runtime. If the telemetry schema changes, regenerate/adapt this helper before wiring the new build.
- No runtime import/path has been changed by this split.
