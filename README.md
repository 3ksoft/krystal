# Krystal

Krystal is an experimental, record-based brain engine for simulations and interactive worlds. It is designed to learn semantic decisions—such as which record, action, or argument is relevant—while leaving exact logic, type checking, arithmetic, and world-state updates to the runtime.

The project is built around a compiled ABI for world data:

```text
encode records → encode query → select → soft gather → decide → emit a typed plan
```

## Packages

- `@krystal/krystal` — the model graph, CPU forward/backward implementation, masks, and host session API.
- `@krystal/schema` — schemas and generated layouts used to describe the engine ABI.
- `@krystal/webgpu` — WebGPU forward and training support, including WGSL shaders.


## Development

This is a TypeScript monorepo using Bun.

```bash
bun install
bun run build
bun test
```

The project is still under active development; the API and architecture may change.
