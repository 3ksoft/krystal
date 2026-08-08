# LFM2 shader sources

These are the shaders the runtime actually runs. They are linked at build time
into `src/lfm2.artifact.generated.ts` by `scripts/build-lfm2-artifact.ts`, and
the runtime loads that artifact — it never reads this directory.

Each top-level `.wgsl` file contains **only the compute function body**.
Sandblaster generates the entry point, the builtin parameters and the resource
declarations from the `engine.compute({...})` call in `src/lfm2-definition.ts`.
`includes/` holds shared helpers and workgroup scratch declarations, and is also
where any `var<workgroup>` must live — WGSL does not allow one in a function
body, which is what the body-only convention would otherwise tempt you into.

The header comment at the top of each body (`// entryPoint:`,
`// workgroupSize:`, `// builtins:`) is **documentation, not configuration**.
Nothing parses it. The values that take effect are in `lfm2-definition.ts`, so
if the two disagree, the header is the one that is wrong.

## Where the truth lives

| question | file |
|---|---|
| which includes, which resources, which workgroup size | `src/lfm2-definition.ts` |
| how many workgroups a dispatch launches | `src/lfm2-layout.ts` (`defineLfm2Passes`) |
| which shader files exist | `LFM2_SHADER_NAMES` in `src/lfm2-layout.ts` |
| which *programs* exist | `LFM2_PROGRAM_NAMES` — not the same set, see below |

A per-shader table of includes used to live here and silently went stale, so it
is deliberately gone. `lfm2-definition.ts` is short and it is the thing that
executes.

## Shaders

| shader | WG | notes |
|---|---:|---|
| `embedding` | 256 | F16 weights |
| `embedding_wq4` | 256 | tied `token_embd`, WQ4 |
| `rms_norm` | 64 | |
| `matmul_f16` | 64 | fallback path |
| `matmul_f32` | 64 | fallback path |
| `matmul_wq4` | 64 | the hot kernel — compiled twice, see below |
| `residual_add` | 256 | |
| `silu_mul` | 256 | |
| `shortconv_prefill` | 256 | |
| `shortconv_continue` | 1 | resumes a conv window from cached state |
| `shortconv_decode` | 256 | |
| `qk_norm_rope` | 64 | |
| `kv_store` | 256 | |
| `attention` | 64 | |
| `arena_copy` | 256 | |
| `argmax_candidates` | 256 | sparse candidate set |
| `argmax` | 256 | full vocabulary |
| `constraint_mask` | 64 | structured decoding: legal-token mask |
| `constraint_argmax` | 256 | structured decoding: masked argmax |

`schema.wgsl` is generated from the schema package for reference and is **not**
one of these — it is not linked and must not be included.

## One body, two programs

`matmul_wq4.wgsl` is compiled twice, into `matmul_wq4` and `matmul_wq4_wide`.
They differ only in the `MATMUL_ROWS` constant, supplied by the
`matmul-rows` / `matmul-rows-wide` include, and `matmulWq4Program(outputDim)`
in `src/lfm2-layout.ts` picks between them per call. The measured reason for
two is in `includes/matmul-rows-wide.wgsl`.

This is why `MATMUL_ROWS` is not in `includes/common.wgsl` with the other
constants, and why program names are no longer 1:1 with file names. It could not
be a pipeline-overridable constant: WGSL requires a const-expression for the
size of the function-scope accumulator array.

Two things must agree or output rows are silently dropped: the include's
`MATMUL_ROWS` and the divisor in that program's `defineLfm2Passes` entry.
