# Krystal

Krystal is an experimental, record-based brain engine for simulations and
interactive worlds. It learns one thing — **which record is relevant** — and
leaves exact logic, type checking, arithmetic and world state to whoever owns
the world.

```text
records in → one pointer per question out
```

A frame is a list of records, each up to eight tokens. Some are marked as
questions; the rest form the bank. For every question the model returns one bank
record and the distribution it came from. There is no generated text, no opcode,
no plan structure and no exact reference.

A relation is an ordinary record. Its roles are ordinary questions. The engine
has no special construct for either — which is the whole architecture in one
sentence.

```ts
import { BrainSession } from "@krystal/krystal/host";

const brain = new BrainSession({ tokenRows });

const { selections } = await brain.think([
  { schemaId: 1, band: 3, tokens: [APPLE, RED] },
  { schemaId: 1, band: 3, tokens: [STONE, GREY] },
  { schemaId: 9, query: true, tokens: [EAT, PATIENT] },
]);

selections[0].record;  // 0 or 1 — the slot it chose, as you numbered them
```

It can be shown what one does (`teach`), and it can live with what happened
(`learn`). It runs on the CPU, or on WebGPU behind the same API.

## Documentation

- **[docs/ENGINE.md](docs/ENGINE.md)** — how the engine works: the frame, the
  model graph, masks, learning, the CPU/GPU seam, invariants, and an honest
  inventory of what is in the tree but not wired.
- **[docs/API.md](docs/API.md)** — how to drive it from a simulation.
- [docs/archive/](docs/archive/README.md) — superseded designs and milestone
  notes. Nothing there is a contract.

## Packages

| Package | What it is |
|---|---|
| `@krystal/krystal` | The model graph, the CPU forward/backward oracle, the masks — and, under `/host`, the session a simulation actually calls |
| `@krystal/schema` | ABI constants, frame layout and the generated types |
| `@krystal/webgpu` | WGSL shaders, the compiled artifact, and a device backend that plugs into a session unchanged |

## Development

A TypeScript monorepo on Bun.

```bash
bun install
bun run build     # schema codegen, then the WebGPU artifact
bun test packages
```

Still under active development; the API and the architecture change.
