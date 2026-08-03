# Changes in Architecture Revision 0.3

Revision 0.3 is not only an editorial cleanup of v0.2. It changes the hierarchy of several concepts so the spec better matches the intended runtime.

## 1. Daemon ownership is now explicit

v0.2 described a persistent daemon that could be read as a shared multi-client inference service.

v0.3 explicitly chooses the simpler model:

```text
one daemon
    owns one model/runtime/BlockStore
    accepts at most one live client session
    processes session work serially
```

Different applications are expected to launch or own independent daemon instances when they need independent model/cache state.

The initial disconnect rule is deliberately simple:

```text
client disconnect
→ cancel current request
→ discard session-scoped state
→ retain daemon-owned model + BlockStore
```

Persistent/reattachable sessions are a future protocol extension, not a baseline requirement.

## 2. Exact checkpoints are promoted above BlockStore composition

v0.2 grouped checkpointing and composable cached block representations under one BlockStore decision.

v0.3 separates them:

- **Exact ContextCheckpoint** — fundamental reusable continuation capability.
- **Composable CachedRepresentation** — optional acceleration built on top.

Even if deeper block composition proves unattractive, exact checkpoints remain valuable for branching and repeated continuation from expensive prefills.

## 3. Exactness claims require stronger evidence

Greedy-token agreement is no longer sufficient evidence for an `exact` cached representation.

v0.3 requires differential testing at state/logit level for exactness claims, with bitwise equality where restoration is a pure state copy and a defined numerical tolerance only where recomputation necessarily changes floating-point execution order.

## 4. Host/device responsibility is made explicit

The WebGPU decision now describes the intended CPU/GPU split and legal synchronization points.

The unconstrained decode baseline remains GPU-resident. Cancellation latency is explicitly coupled to how far ahead work is submitted to the GPU.

## 5. Structured generation moves from OPEN to PROVISIONAL direction

v0.2 listed several validator architectures without selecting a direction.

v0.3 adopts a target architecture based on:

```text
schema / constraint
→ host-compiled OutputPlan
→ tokenizer token-byte metadata / automata
→ GPU-resident structured decoder state
→ exact allowed-token set
→ GPU sampling
```

For the LFM2.5 vocabulary:

```text
65,536 tokens = 2^16
1 bit/token   = 8,192 bytes
              = 2,048 × u32
```

Three execution modes are defined:

- `DENSE`
- `MASKED_DENSE`
- `SPARSE`

They must preserve the same constraint semantics. `SPARSE` means evaluating the exact allowed token set directly; it does **not** mean top-K-before-validation.

## 6. JSON is no longer treated as the internal structured-decoding language

JSON may remain an API/rendering format.

Internally, structured generation may target a typed result representation driven by the plan. Syntax such as JSON punctuation is generated only when the selected output format actually requires it.

## 7. WQ4 is demoted as a settled representation

The current WQ4 quantizer and matmul remain useful experiments, but the whole loaded model currently occupies substantially more VRAM than the raw WQ4 compression ratio suggests.

v0.3 therefore separates:

- quantization math,
- matmul kernel performance,
- runtime repacking/layout,
- whole-model resident memory.

The final model-memory representation remains experimental/open.
