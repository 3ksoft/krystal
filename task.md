# Task: make the WQ4 matmul kernel faster

Propose the best solutions for the problem below. Rank your proposals by
expected speedup per unit of implementation risk, and say explicitly which
measurements would confirm or refute each one.

## Context

Chomato runs LFM2.5-1.2B inference in **WebGPU / WGSL** (no CUDA, no
subgroup-specific vendor extensions assumed). Weights are 4-bit block
quantized (**WQ4**). All matmuls in the model go through one kernel.

Hardware measured on: **NVIDIA RTX 3060, 360 GB/s** theoretical memory
bandwidth, via Dawn native.

## Measured problem

One **decode step** (single token, all 16 blocks), GPU timestamps at pass
boundaries:

| stage | time | share |
|---|---|---|
| embed | 0.007 ms | 0.0% |
| 16 transformer blocks | 26.42 ms | 88.2% |
| LM head (2048 x 65536) | 3.60 ms | 12.0% |
| argmax | 0.10 ms | 0.3% |
| **GPU total** | **29.95 ms** | |
| wall clock | 33.29 ms | |

Decode reads essentially every weight once per token: **~700 MiB per token**.

    700 MiB / 29.95 ms  =>  ~23 GiB/s achieved
    23 / 360             =>  ~6% of peak memory bandwidth

The LM head alone is 80 MiB in 3.60 ms = **21.7 GiB/s**; the block stack is
~620 MiB in 26.4 ms = **~22.9 GiB/s**. The inefficiency is **uniform**, not
localised to one call site.

**The question: why is a memory-bound kernel achieving ~6% of memory
bandwidth, and what is the highest-leverage way to fix it?**

## WQ4 format

- 32 weights per block, **20 bytes per block**: 4 x u32 packed nibbles + 1 x
  i32 power-of-two exponent.
- Value = `(nibble - 8) * exp2(exponent)`.
- Weights are exposed to WGSL as a runtime-sized `array<u32>` (`weightRaw`).
- Rows are contiguous; `blocksPerRow = inputDim / 32`.

## Current kernel

Dispatch: `[rowCount, tokenCount, 1]` workgroups, **workgroup size 64**.
One workgroup computes one output row for one token.

For the LM head in decode this is **65536 workgroups**, each doing a
dot product of length 2048, i.e. `blocksPerRow = 64` blocks across 64 threads
= **exactly one block per thread**, then a 6-step shared-memory reduction.

```wgsl
// workgroupSize: [64, 1, 1]; workgroup_id -> wid, local_invocation_id -> lid
let localRow = wid.x;
let tokenIndex = wid.y;
if (localRow >= op.rowCount || tokenIndex >= op.tokenCount) { return; }

let inputBase = op.inputOffset + tokenIndex * op.inputDim;
let blocksPerRow = op.inputDim / 32u;
let rowBlockStart = localRow * blocksPerRow;

var sum: f32 = 0.0;
var b = lid.x;
loop {
  if (b >= blocksPerRow) { break; }

  let blockIdx = rowBlockStart + b;
  let baseU32 = blockIdx * 5u;                     // 4 packed words + exponent
  let expVal = bitcast<i32>(weightRaw[baseU32 + 4u]);
  let scale = exp2(f32(expVal));
  let kStart = b * 32u;

  for (var w = 0u; w < 4u; w++) {
    let packed = weightRaw[baseU32 + w];
    let kBase = kStart + w * 8u;
    sum += (f32((packed >>  0u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 0u];
    sum += (f32((packed >>  4u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 1u];
    // ... 6 more identical lines, nibbles 2..7
  }
  b += WG;
}

reduceF32[lid.x] = sum;
workgroupBarrier();
var width = WG >> 1u;
loop {                                              // 6-step tree reduction
  if (width == 0u) { break; }
  if (lid.x < width) { reduceF32[lid.x] += reduceF32[lid.x + width]; }
  workgroupBarrier();
  width >>= 1u;
}
if (lid.x == 0u) {
  arena[op.outputOffset + tokenIndex * op.outputDim + op.rowStart + localRow] = reduceF32[0];
}
```

Observations that may or may not be the dominant cost — do not assume, weigh
them:

- Scalar `u32` weight loads; no vector loads.
- The activation vector (`inputDim * 4 B` = 8 KiB) is re-read by every
  workgroup and is never staged in workgroup memory.
- `scale` is applied per element instead of factored out of the block sum.
- Per workgroup: 1280 B of weights vs 8 KiB of activation reads.
- One output value per 64-thread workgroup, with a full barrier tree per row.
- 20-byte blocks mean row starts are not 16-byte aligned.

## Constraints

- **WGSL / WebGPU only.** Must run in a browser as well as on Dawn native.
  No subgroup intrinsics unless gated behind feature detection with a
  fallback.
- The dequantization must stay numerically identical; inference results are
  covered by exact-output regression tests.
- `f16` shader support is not currently required by the device; if a proposal
  needs it, say so and give the fallback.
- The kernel is shared by every matmul in the model (attention projections,
  FFN, LM head). Shapes vary: `inputDim` 2048, `outputDim` in
  {2048, 8192, 65536}; `tokenCount` is 1 in decode and up to 1024 in prefill.
  A proposal may specialise per shape, but must say how the variants are
  selected.
- Changing the on-disk WQ4 layout is permitted **if** the win is quantified
  and a conversion path is described. State this explicitly when proposed.

## Deliverable

For each proposal:

1. What changes, concretely (sketch the WGSL if it is a kernel change).
2. Expected speedup and the reasoning behind that number.
3. What could make it *not* work here.
4. The measurement that settles it.

Prefer a small number of well-argued proposals over a long list.
