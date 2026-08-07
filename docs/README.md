# Chomato Architecture Pack v0.4

Status: Current technical architecture  
Revision: 0.4  
Date: 2026-08-07

Revision 0.4 records the implementation that is now running end to end on the real LFM2.5 WebGPU backend: exact physical checkpoints, Sandblaster AOT shader/program artifacts, typed structured generation, and GPU-resident constraint masking/sampling.

Start with [architecture.md](architecture.md). The implementation details are split into focused technical documents:

- [Runtime and WebGPU execution](runtime.md)
- [Context state and exact checkpoints](checkpoints.md)
- [Structured generation](structured-generation.md)
- [Changes in v0.4](changes-v0.4.md)

## Decisions

1. [ADA-0001 — Single-Owner Headless Runtime and Live Session](decisions/0001-single-owner-runtime-session.md)
2. [ADA-0002 — Reference Model: LFM2.5-1.2B-Instruct](decisions/0002-reference-model-lfm2.5.md)
3. [ADA-0003 — WebGPU Execution and Host/Device Boundary](decisions/0003-webgpu-host-device-boundary.md)
4. [ADA-0004 — Context State and Exact Checkpoints](decisions/0004-context-state-exact-checkpoints.md)
5. [ADA-0005 — Composable Context Representations](decisions/0005-composable-context-representations.md)
6. [ADA-0006 — Structured GPU Decoding](decisions/0006-structured-gpu-decoding.md)
7. [ADA-0007 — Host Runtime Language](decisions/0007-host-runtime-language.md)
8. [ADA-0008 — Correctness Classes and Optimization Boundaries](decisions/0008-correctness-classes.md)

## Historical experiments

The files under `docs/experiments/` are retained as experimental records. They are not the current implementation specification and are intentionally not expanded in v0.4. Future measured results should move toward benchmark reports rather than growing the historical experiment notes.

- [Exact checkpoints](experiments/exact-checkpoints.md)
- [Composable context blocks](experiments/composable-blocks.md)
- [Structured decoding](experiments/structured-decoding.md)
- [WQ4 model memory/layout](experiments/wq4-model-layout.md)
- [WQ4 matmul](experiments/wq4-matmul.md)

## Status vocabulary

- **ACCEPTED** — implemented/validated direction that current code may build against.
- **PROVISIONAL** — selected direction whose behavior or economics still need material validation.
- **EXPERIMENTAL** — implementation exists to discover behavior, not to define a stable contract.
- **OPEN** — requirements are known, solution not selected.
- **DEFERRED** — decision is intentionally postponed until named dependencies stabilize.
- **REJECTED** — investigated and intentionally not used.
- **SUPERSEDED** — replaced by a later decision.

## v0.4 in one page

```text
Engine.generate(Type<T>, Context) -> Promise<T>

Context
├── checkpoint?       exact model continuation state
└── blocks?           ordered token blocks appended after it

LFM2 forward
├── Sandblaster AOT programs
├── WQ4/F16 model execution
├── attention KV state
└── rolling short-conv state

Exact checkpoint
├── populated attention KV prefix only
└── fixed-size rolling conv state

Structured generation
Type / JSON Schema
    ↓ host compile
constraint byte program + tokenizer byte metadata
    ↓ GPU
constraint_mask (65,536 vocab -> 2,048 × u32 mask)
    ↓
constraint_argmax + decoder-state commit
    ↓
strict canonical JSON value
    ↓ host terminal readback
JSON.parse
    ↓
T
```

Current implementation limits are intentionally explicit:

- reference model: LFM2.5-1.2B-Instruct,
- vocabulary: 65,536,
- current runtime context capacity: 1,024 tokens,
- current maximum decode budget allocation: 1,024 tokens,
- structured generation requires a finite schema-derived output bound,
- sparse LM-head execution is not implemented; current constrained execution uses the full vocabulary mask.
