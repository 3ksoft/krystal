# Chomato Architecture Pack v0.3

Status: Draft / architecture candidate  
Revision: 0.3  
Date: 2026-08-03

This revision tightens the runtime model and separates stable capabilities from experimental accelerations.

Start with [architecture.md](architecture.md). The main changes from v0.2 are summarized in [changes-v0.3.md](changes-v0.3.md).

## Decisions

1. [ADA-0001 — Single-Owner Headless Runtime and Live Session](decisions/0001-single-owner-runtime-session.md)
2. [ADA-0002 — Reference Model: LFM2.5-1.2B-Instruct](decisions/0002-reference-model-lfm2.5.md)
3. [ADA-0003 — WebGPU Execution and Host/Device Boundary](decisions/0003-webgpu-host-device-boundary.md)
4. [ADA-0004 — Context State and Exact Checkpoints](decisions/0004-context-state-exact-checkpoints.md)
5. [ADA-0005 — Composable Context Representations](decisions/0005-composable-context-representations.md)
6. [ADA-0006 — Structured GPU Decoding](decisions/0006-structured-gpu-decoding.md)
7. [ADA-0007 — Host Runtime Language](decisions/0007-host-runtime-language.md)
8. [ADA-0008 — Correctness Classes and Optimization Boundaries](decisions/0008-correctness-classes.md)

## Experiments

- [Exact checkpoints](experiments/exact-checkpoints.md)
- [Composable context blocks](experiments/composable-blocks.md)
- [Structured decoding](experiments/structured-decoding.md)
- [WQ4 model memory/layout](experiments/wq4-model-layout.md)
- [WQ4 matmul](experiments/wq4-matmul.md)

## Status vocabulary

- **ACCEPTED** — architectural direction is stable enough to build against.
- **PROVISIONAL** — selected direction, but implementation/behavior still needs material validation.
- **EXPERIMENTAL** — implementation exists to discover behavior, not to define a stable contract.
- **OPEN** — requirements are known, solution not selected.
- **DEFERRED** — decision is intentionally postponed until named dependencies stabilize.
- **REJECTED** — investigated and intentionally not used.
- **SUPERSEDED** — replaced by a later decision.

## Core v0.3 framing

```text
single daemon process
    owns one model + one runtime + one BlockStore
    accepts one live client session at a time

ContextCheckpoint
    = exact reusable continuation state
    = fundamental capability

CachedRepresentation
    = optional acceleration for reusable context blocks
    = may be exact, prefix-dependent, or approximate

Structured GPU Decoding
    = schema/plan compiled before generation
    = per-token constraint execution designed to stay on GPU
    = DENSE / MASKED_DENSE / SPARSE are execution modes, not semantic modes
```

## Source basis

Revision 0.3 builds on:

- the v0.2 architecture pack,
- the current Chomato WebGPU/LFM2.5 implementation and BlockStore experiments,
- the current WQ4/matmul candidate work,
- the supplied v0.2 architecture reviews,
- the project discussion on GPU logit masking, tokenizer automata, and schema-driven structured decoding,
- the earlier Szczupak protocol pattern used as design precedent.
