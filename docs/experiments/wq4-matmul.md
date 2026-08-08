# Experiment — WQ4 Matmul

Status: **ACTIVE / IMPLEMENTATION EVOLVING**

## Goal

Determine whether the custom WQ4 math and GPU kernel are worthwhile after comparing against a stronger F16/reference implementation and accounting for whole-model memory.

## Current WQ4 block

Current converter layout:

```text
32 weights
4 × u32 packed signed 4-bit values = 16 bytes
1 × i32 exponent                     = 4 bytes
-------------------------------------------
20 bytes / 32 weights
```

Effective raw storage:

```text
160 bits / 32 weights = 5 bits/weight
```

Relative to F16 raw block storage:

```text
64 bytes / 20 bytes = 3.2× smaller
```

## Current quantizer

The current implementation:

- finds block max absolute value,
- chooses a clamped power-of-two exponent,
- quantizes signed values to `[-8, 7]`,
- packs nibbles into four `u32` words,
- stores one exponent word per 32 values.

## Current candidate evidence

Recent benchmark candidates show the WQ4 kernel can beat the current F16 reference in both decode-like (`M=1`) and larger-prefill (`M>1`) cases while maintaining very small candidate-vs-reference numerical drift.

However, the F16 comparator has received only limited optimization, so this benchmark is evidence that WQ4 is viable — not proof that the current WQ4 kernel/layout is globally optimal.

## Measured results

Kernel and end-to-end numbers live in [../benchmarks.md](../benchmarks.md), per
this pack's rule that measured results go to a benchmark report rather than
growing an experiment note. The short version, as it bears on the questions
below:

- Row tiling took the kernel from ~22 GiB/s to 85–156 GiB/s depending on shape,
  and no single tiling wins on every shape, so two are linked and picked per
  call by output width.
- The achievable bandwidth on the same GPU is 298–321 GiB/s, and a whole decode
  step runs at ~98 GiB/s — so question 2 below is *not* closed.
- But an isolated +18% on the shape carrying 46% of the model's bytes produced
  ~1% in the model. **Isolated matmul benchmarks are currently not predictive
  for the block stack**, which is the single most important caveat on the
  benchmark matrix below.

## Do not conflate three questions

### 1. Quantization quality

Does the 5-bit-effective block representation preserve model quality adequately?

### 2. Kernel performance

Does direct/repacked WQ4 matmul outperform a properly optimized F16/mixed-precision path for target shapes?

### 3. Whole-model memory

Does the runtime representation actually reduce resident VRAM after all repacking/duplicate buffers?

The third question is tracked in `wq4-model-layout.md` and may invalidate a layout that looks excellent in an isolated matmul benchmark.

## Benchmark matrix

Read the caveat in *Measured results* first: an isolated matmul benchmark has
already been shown to mispredict the in-model block stack by an order of
magnitude. Anything measured with this matrix needs an in-model A/B before it is
acted on.

Keep at least:

```text
M=1      decode-oriented
M≈block/prefill sizes
representative K/N from LFM2.5 layers
```

Compare:

```text
current WQ4 candidate
improved F16 baseline
other practical packed layouts if introduced
```

Measure:

- kernel time,
- effective GFLOP/s where meaningful,
- bytes read/written,
- dequant/repack overhead,
- numerical error,
- end-to-end decode TPS,
- end-to-end prefill throughput.

## Correctness

Require:

- tensor-level comparison against F16/F32 reference,
- model-level regression tests,
- no reliance on matching one final greedy output as the sole quality metric.

## Format status

The current converter/container remains experimental.

A production format still needs:

- tensor directory,
- names/shapes/strides,
- alignment rules,
- quantization metadata,
- version/compatibility policy,
- mixed-precision support if used.

## Priority

Do not over-optimize isolated matmul while the larger architectural unknowns are still being closed.

At the same time, whole-model layout work is important enough that WQ4 cannot be treated as a solved subsystem. The declared buffer allocation is ~946 MiB — 699.5 weights, 128 `OpParams`, 88.9 activation arena, 24 KV, ~6 the rest — which is a computed figure, not measured residency; actual VRAM including driver overhead has not been re-measured since the layout changed.
