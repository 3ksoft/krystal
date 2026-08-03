# Chomato

Experimental local LFM2.5 inference runtime built directly on WebGPU.

```ts

chomato.generate<T>(
  schema: Type<T>,
  context: Context,
): Promise<T>

```

The repository is intentionally split by current implementation boundaries rather than by speculative future subsystems.

```text
apps/
  web/          browser harness

packages/
  schema/       interop / binary / GPU ABI contracts
  webgpu/       thin WebGPU bootstrap and utilities
  gguf/         GGUF reader and random-access sources
  lfm2/         LFM2 model loading, tokenizer, runtime and WGSL kernels

misc/           one-off probes, converters and benchmarks kept for reference
tests/          focused correctness tests
docs/           reserved for the current architecture specification
```

## Current dependency direction

```text
schema ──┐
         ├──> lfm2 ──┐
gguf ────┘           ├──> apps/web
webgpu ───────────────┘
```

`@chomato/webgpu` deliberately does not provide a resource graph, shader DSL or engine framework. Chomato talks to WebGPU directly.

`@chomato/schema` is the intended source of truth for data crossing subsystem / CPU / GPU / process boundaries. During this mechanical repository split the existing `LlmRuntime` codec remains explicit; the next schema step is to move that ABI to schema-pop-generated codec + WGSL layout.

BlockStore logic is still inside the LFM2 runtime. It should only become a separate package after its checkpoint / representation boundary is proven in code. The same applies to structured decoding and a generic core runtime: no empty abstraction packages are created yet.

## Browser harness

Put model files in:

```text
models/LFM2.5-1.2B-Instruct-F16.gguf
```

Then run:

```bash
bun install
bun run dev:web
```

or use the Chromium launcher required by the current NVIDIA/Dawn setup:

```bash
./apps/web/run-chromium.sh /
```

The Vite middleware serves `models/*.gguf` with HTTP Range support, so the browser does not need to fetch a whole GGUF into one JavaScript `ArrayBuffer`.

## Tests

```bash
deno test tests
```

## Notes

- LFM2.5-1.2B is the current reference model; generic model support is not a present goal.
- The runtime currently owns model tensors, activation state, attention KV, convolution state and experimental cached block representations on one `GPUDevice`.
- Normal greedy decode is GPU-resident; host readback is used for results/telemetry rather than a required per-token decision.
- One-off historical experiments are intentionally kept under `misc/` instead of at repository root.
- Architecture/design documentation is intentionally not duplicated here; put the current spec under `docs/`.
