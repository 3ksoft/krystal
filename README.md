# Chomato

Experimental local LFM2.5 inference runtime built on WebGPU with exact reusable checkpoints and GPU-resident typed structured generation.

```ts
const result = await engine.generate(
  type({ id: "number", name: "string < 64" }),
  { checkpoint, blocks },
);
```

The core API generates a value of the requested type from an ordered context. It is not centered on chat templates or text-only completion; a plain text result is simply a bounded string schema.

## Repository shape

The repository is split by implementation ownership:

```text
packages/
  engine-ts/    public engine API + transport/protocol
  lfm2/         LFM2 model definition, tokenizer, forward/checkpoint logic, shaders
  quant/        GGUF/WQ4 quantized model tooling/runtime support
  schema/       shared CPU/GPU ABI schemas and generated layouts
  webgpu/       WebGPU host/runtime support and tests
  finetune/     structured-generation datasets/tooling
  gui/          browser UI/harness

tests/          public/real-engine correctness and E2E tests
docs/           current architecture + implementation notes
misc/           historical one-off probes/benchmarks
```

The precise package graph continues to evolve, but the stable responsibility boundary is:

```text
schema/quant/model data
        ↓
LFM2 definition + Sandblaster AOT programs
        ↓
WebGPU runtime/backend
        ↓
engine-ts public API
```

## Current execution model

Build time:

```text
WGSL shader bodies/includes
→ Sandblaster link
→ serialized LFM2 artifact
```

Runtime:

```text
load model
→ deserialize/compile artifact for GPUDevice
→ prefill / decode / checkpoint / structured generation
```

Structured generation runs:

```text
model forward -> logits
→ exact GPU token mask
→ constrained argmax + decoder-state commit
→ strict JSON value
→ JSON.parse
→ T
```

For the current 65,536-token vocabulary the exact mask is 8 KiB (`2,048 × u32`).

## Checkpoints

`ContextCheckpoint` is a physical exact continuation snapshot, not prompt replay metadata.

For LFM2.5 it contains:

- populated attention KV prefix,
- fixed-size rolling short-convolution state,
- continuation metadata.

Restoring a checkpoint does not re-prefill its source prefix. Checkpoints can branch, chain, and survive dropping their source blocks.

## Current limits

The present implementation is intentionally focused:

- model: `LFM2.5-1.2B-Instruct`,
- vocabulary: 65,536,
- runtime context capacity: 1,024 tokens,
- decode allocation: up to 1,024 tokens subject to remaining context capacity,
- structured output: finite/bounded schema envelope rather than full JSON Schema,
- current structured sampler path: full-vocabulary masked dense, not sparse LM-head execution.

## Tests

Run the complete suite with:

```bash
bun test
```

The suite includes real-engine checkpoint tests, structured-generation E2E tests, transport tests, and WebGPU/Sandblaster regression coverage.

## Documentation

Start with [docs/README.md](docs/README.md) and [docs/architecture.md](docs/architecture.md).

The focused technical documents are:

- [runtime/WebGPU execution](docs/runtime.md)
- [context state/checkpoints](docs/checkpoints.md)
- [structured generation](docs/structured-generation.md)
