# Benchmark report

Measured results, kept here rather than in `docs/experiments/` so the experiment
notes stay records of a question and this stays a record of an answer.

Environment for everything below: RTX 3060 (12 GB, 360 GB/s = **335.3 GiB/s**
spec peak), NVIDIA driver 610.43.03, Linux, bun + Dawn, `LFM2.5-1.2B-Instruct`
in WQ4 (699.5 MiB of weights). Date: 2026-08-08.

Every number has a test that produces it. If a number here disagrees with the
test, the test is right.

| what | test |
|---|---|
| achievable memory bandwidth | `tests/bandwidth-floor.test.ts` |
| decode step stage breakdown | `tests/decode-profile.test.ts` |
| WQ4 matmul kernel variants | `tests/matmul-variants.test.ts` |
| CPU encode share of wall time | `tests/decode-overhead.test.ts` |
| structured-generation overhead | `tests/structured-benchmark.test.ts` |

## 1. What the memory system can do

Streaming the same 80 MiB buffer the LM head reads:

| access | time | GiB/s | % of spec peak |
|---|---:|---:|---:|
| scalar | 0.262 ms | 298.0 | 88.9% |
| `vec4` | 0.244 ms | 320.6 | 95.6% |

This is the ceiling any weight-streaming kernel is measured against. It also
means vectorizing loads is worth at most ~7.6% on top of a kernel that is
already streaming well.

## 2. Where a decode step goes

One token, pass-boundary timestamps (the shipped schedule runs prefill and every
decode step in **one** pass, so this split exists only in the test):

| stage | time | share |
|---|---:|---:|
| embed | 0.008 ms | 0.1% |
| block stack | 6.13 ms | ~90% |
| LM head | 0.543 ms | ~8% |
| argmax | 0.096 ms | 1.4% |
| **total (GPU)** | **~6.8 ms** | |

Wall clock is ~8.3 ms/token; the difference is CPU-side command encoding, which
`decode-overhead` measures at **21.5% of wall time** over a 128-token run.

The whole step moves ~700 MiB in ~6.8 ms, about **98 GiB/s** — roughly a third
of §1. See §4 before concluding that the missing 3× is available.

**This table caps several ideas at once.** Anything that makes the LM head
cheaper — sparse rows, vocabulary pruning, a smaller output width — is bounded
by 8%, no matter how perfect. A vocabulary-pruning scheme that drops 24% of rows
is bounded by 2%.

## 3. WQ4 matmul kernel

The original kernel computed one output row per workgroup and reached ~22 GiB/s.
It was never memory-bound: it was launch- and reduction-bound, 52k workgroups
each producing one scalar. Tiling rows amortizes both.

GiB/s of weight traffic, `tokenCount = 1` (the decode regime):

| shape | 1 row/wg | rows8 | rows16 |
|---|---:|---:|---:|
| `lm_head` 2048×52428 | 23.3 | 118.3 | **156.5** |
| `ffn_gate/up` 2048×8192 | 23.8 | 113.5 | **134.3** |
| `conv_in_proj` 2048×6144 | 22.9 | 111.8 | **121.2** |
| `ffn_down` 8192×2048 | 28.3 | **138.2** | 134.3 |
| `attn_q/out` 2048×2048 | 22.7 | **85.1** | 76.9 |

No single tiling wins everywhere: wide tiles pay off until the row count stops
covering the launch, and at 2048 rows they stop. Both are linked, and
`matmulWq4Program(outputDim)` picks per call.

**Numerics.** Hoisting the block scale out of the inner loop changes the
association of the 32 additions, so results are f32-equivalent rather than
bit-identical: worst observed `maxAbsDiff` 6.7e-4 at `inputDim` 8192, and 0 at
2048. The exact-output checkpoint and structured-generation suites cover whether
that ever moves a token; they pass.

## 4. Isolated matmul numbers do not predict the block stack

This is the most important entry on the page, and it is a negative result.

Switching `ffn_gate/up` from rows8 to rows16 is **+18% in isolation** (§3), and
that shape carries 46% of the model's weight bytes — which predicts roughly 7%
off the block stack. Matched A/B, four samples each, same tree, medians:

| stage | all rows8 | per-shape | delta |
|---|---:|---:|---:|
| LM head | 0.679 ms | 0.543 ms | **−20%** |
| block stack | 6.213 ms | 6.133 ms | −1.3% |
| **step total** | **7.145 ms** | **6.777 ms** | **−5.2%** |

The LM head — one large dispatch — delivers its isolated gain in full. The block
stack — ~90% of the step, ~92 smaller matmuls in a dependency chain — does not.
`tests/matmul-tiling.test.ts` confirms the wide program really is selected there,
and `matmul-variants` runs at `tokenCount = 1` like decode does, so this is
neither a wiring bug nor a regime mismatch. **The gap is unexplained.**

Consequences, until it is explained:

- Treat isolated matmul throughput as non-predictive for the block stack.
- Do not spend on kernel or layout work aimed at it on the strength of an
  isolated benchmark. A `vec4`-aligned WQ4 block layout was considered on
  exactly that reasoning and dropped.
- The measurement that would settle it is a pass-boundary timing of one in-model
  `ffn_gate` dispatch, the way `decode-profile` already isolates the LM head.

## 5. Structured generation overhead

| | range |
|---|---|
| `constraint_mask` | 0.07–0.30 ms |
| `constraint_argmax` | 0.14–0.16 ms |
| unconstrained baseline | 8.37–8.81 ms/token |
| constrained − unconstrained | −1.5% to +2.8% |

The end-to-end difference is inside run-to-run noise. Combined with the 8% LM
head ceiling from §2, sparse LM-head execution has no case at present.
