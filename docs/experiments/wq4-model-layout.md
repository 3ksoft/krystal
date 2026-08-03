# Experiment — WQ4 Whole-Model Memory and Runtime Layout

Status: **HIGH PRIORITY / ACTIVE**

## Goal

Explain and optimize the actual resident GPU memory footprint of the quantized model.

Raw quantization ratio is not the same thing as runtime model footprint.

## Current observation

The current loaded model occupies roughly **1.5 GB** of VRAM in the working runtime, which is larger than the comparison GGUF Q8 artifact.

This is a warning that the current WQ4 storage/repacking pipeline should not be treated as the final memory architecture even if individual matmul kernels perform well.

## Separate the accounting layers

Measure each category independently:

```text
on-disk model bytes
raw uploaded WQ4 tensor bytes
runtime/repacked matmul weight bytes
non-quantized tensor bytes
embeddings / norms / small tensors
scratch/intermediate buffers
pipeline/support buffers
attention KV state
convolution state
BlockStore/checkpoints
telemetry/readback staging
allocator/alignment overhead
peak duplicate buffers during load/repack
```

## Questions

- Are both raw and repacked copies retained after initialization?
- Which tensors remain F16/F32?
- Is the current i8/intermediate layout larger than the packed WQ4 representation?
- Can temporary conversion buffers be destroyed earlier?
- Are per-layer buffers over-allocated to simplify kernels?
- How much of the 1.5 GB is weights vs Context State vs scratch?
- Is the observed number peak memory or steady-state resident memory?
- Can decode and prefill share one packed layout without an expensive second permanent representation?
- If separate layouts are beneficial, is the throughput gain worth the memory duplication?

## Required telemetry/tooling

Produce a model-load memory report by category, ideally per tensor/layer class:

```text
logical tensor size
source format
GPU runtime format
GPU allocated bytes
lifetime
```

Also record adapter/device limits and total memory estimate where available.

## Decision output

The experiment should answer whether the final design should use:

- packed WQ4 directly,
- a different packed block layout,
- mixed precision by tensor class,
- one runtime layout,
- separate decode/prefill layouts,
- transient repacking only,
- a different quantization family.

Do not freeze the stable model container until this runtime memory picture is understood.
