# WebGPU backward and training plan

Status: implementation plan for the first training vertical slice  
Scope: `packages/webgpu` and the smallest host-side training orchestration needed to execute it  
Non-goals: full LFM2 backward, dynamic autograd, simulation training, ALU, PPO, QAT

## 1. Goal

Add a minimal, deterministic, GPU-resident training path to the existing WebGPU runtime.

The first milestone must train this graph end to end:

```text
token ids
  -> embedding
  -> linear projection
  -> logits
  -> cross entropy
  -> dLogits
  -> linear backward
  -> embedding backward
  -> SGD update
```

The milestone is complete when a tiny fixed dataset can be overfit and all analytical gradients match a CPU oracle and numerical finite differences within the declared tolerance.

This is deliberately a small training system built on the existing runtime. It is not a second model implementation and it is not a general-purpose autograd framework.

## 2. Architectural rules

### 2.1 One shader per file

Keep the existing Sandblaster convention: one compute shader / entry point per WGSL file, even when the shader is only a few lines long.

The filename is the operation identity. Do not hide multiple unrelated training operations behind entry-point selection inside one file.

Suggested initial files:

```text
packages/webgpu/src/shaders/training/
  zero_f32.wgsl
  cross_entropy_forward_backward.wgsl
  loss_reduce.wgsl
  matmul_backward_input.wgsl
  matmul_backward_weight.wgsl
  embedding_backward.wgsl
  sgd_step.wgsl
```

Existing forward shaders should be reused where their layout and precision match:

```text
embedding.wgsl
matmul_f32.wgsl
```

Do not duplicate a forward shader merely to put it under `training/`.

### 2.2 Static backward plan

Backward is an explicitly compiled reverse dispatch plan:

```text
forward dispatches
saved tensor declarations
loss dispatch
reverse backward dispatches
optimizer dispatches
```

Do not record an imperative runtime graph. Do not build PyTorch-style dynamic autograd. The model graph is already known when the runtime is linked or compiled.

Each trainable operation eventually needs a declaration equivalent to:

```ts
interface TrainingOpSpec {
  name: string;
  forward: string;
  backward: readonly string[];
  savedTensors: readonly SavedTensorSpec[];
  parameterGradients: readonly GradientSpec[];
}
```

This registry does not need to be generalized in the first patch if doing so delays the vertical slice. A hard-coded first plan is acceptable, provided shader contracts and buffer ownership are explicit.

### 2.3 GPU residency

Between the start and end of `trainStep`, activations, gradients, parameters and optimizer state stay on the GPU.

Allowed readback:

- scalar or compact loss telemetry;
- explicit debug tensors in diagnostic tests;
- final parameter snapshots in tests.

Forbidden in the production path:

- readback between forward and backward;
- CPU gradient accumulation;
- CPU optimizer updates;
- rebuilding GPU buffers on every step.

### 2.4 Debug precision first

The first path uses:

- f32 parameters;
- f32 activations;
- f32 gradients;
- SGD without momentum;
- no shader fusion beyond fused cross-entropy plus `dLogits`;
- tiny deterministic shapes.

F16, mixed precision, AdamW, checkpoint recomputation and fusion come only after the f32 path is correct.

### 2.5 No portable `atomic<f32>` assumption

Do not depend on floating-point atomics.

Every gradient shader must use one of these patterns:

1. one invocation/workgroup owns one output element and performs its complete reduction;
2. workgroups write disjoint partial sums followed by a separate reduction shader;
3. a deterministic scan when the MVP dimensions are intentionally tiny.

Correctness and deterministic tests are more important than the first implementation's throughput.

## 3. Tensor conventions

Use the existing runtime's actual tensor conventions. For the formulas below, the expected LFM-style weight layout is:

```text
X:  [M, K]
W:  [N, K]  // output rows, input columns
Y:  [M, N]
Y[m,n] = sum_k X[m,k] * W[n,k]
```

If `matmul_f32.wgsl` uses another physical layout, keep that physical layout and adapt the indexing. Do not transpose or repack weights only for training.

All arena offsets must be expressed in the same unit already used by the WebGPU runtime. Do not mix byte offsets and element offsets in one ABI. Name fields or helpers so the unit is obvious.

## 4. First training graph

For the initial test:

```text
tokens:       [M]
embedding:    [V, H]
hidden:       [M, H]
classifier:   [V, H]
logits:       [M, V]
targets:      [M]
lossRows:     [M]
dLogits:      [M, V]
dHidden:      [M, H]
dClassifier:  [V, H]
dEmbedding:   [V, H]
loss:         [1]
```

Use deliberately small test dimensions, for example:

```text
M = 4
V = 8
H = 6
```

The implementation must not hard-code these exact values in WGSL. They are test fixtures, not ABI limits.

The first dataset may be a simple deterministic mapping such as:

```text
0 -> 3
1 -> 4
2 -> 5
3 -> 6
```

Repeat the examples for enough steps to prove that loss decreases and predictions converge.

## 5. Forward pass

### 5.1 Embedding

Reuse `embedding.wgsl` in f32 mode or add the smallest layout-compatible f32 path if the existing shader only supports another representation.

Forward definition:

```text
hidden[m,h] = embedding[tokens[m],h]
```

Save for backward:

- `tokens`;
- no internal embedding activation other than `dHidden` is required.

### 5.2 Linear projection

Reuse `matmul_f32.wgsl`:

```text
logits[m,v] = sum_h hidden[m,h] * classifier[v,h]
```

Save for backward:

- `hidden`;
- `classifier` or its stable parameter buffer;
- `dLogits` produced by the loss shader.

The output `logits` may be released after cross-entropy backward if no later consumer needs it.

## 6. Cross-entropy forward and backward

Implement numerically stable softmax cross-entropy.

For each row `m`:

```text
rowMax = max_v logits[m,v]
sumExp = sum_v exp(logits[m,v] - rowMax)
logZ = rowMax + log(sumExp)
lossRows[m] = logZ - logits[m,target[m]]
prob[m,v] = exp(logits[m,v] - logZ)
dLogits[m,v] = (prob[m,v] - one_hot(target[m],v)) / M
```

The division by `M` defines mean reduction and must happen exactly once.

`cross_entropy_forward_backward.wgsl` writes:

- one loss value per row;
- all `dLogits` values.

It does not need to persist the probability matrix.

`loss_reduce.wgsl` reduces `lossRows` to one mean scalar for telemetry. For the tiny debug path a deterministic single-workgroup reduction is acceptable. The gradient must not depend on CPU loss reduction.

Validation requirements:

- subtract the row maximum before `exp`;
- reject or report a target token outside `[0, V)`;
- no NaN/Inf for logits used by the test suite;
- use the same mean/sum convention in CPU and GPU implementations.

Future constraint masks will be applied to logits before softmax. Argmax is not part of the training graph.

## 7. Linear backward

Given:

```text
Y = X * W^T
```

compute:

```text
dX[m,k] = sum_n dY[m,n] * W[n,k]
dW[n,k] = sum_m dY[m,n] * X[m,k]
```

### 7.1 `matmul_backward_input.wgsl`

Inputs:

- `dY [M,N]`;
- `W [N,K]`.

Output:

- `dX [M,K]`.

Each output element has one owner. No atomics are needed.

### 7.2 `matmul_backward_weight.wgsl`

Inputs:

- `dY [M,N]`;
- `X [M,K]`.

Output:

- `dW [N,K]`.

Each output element has one owner and reduces over `M`. No atomics are needed.

Do not update parameters inside either backward shader. Gradient computation and optimizer mutation are separate dispatches.

Bias is outside the first milestone unless the existing forward classifier already requires it. If bias is present, add a separate `bias_backward.wgsl` with:

```text
dBias[n] = sum_m dY[m,n]
```

## 8. Embedding backward

Definition:

```text
dEmbedding[v,h] = sum_(m where tokens[m] == v) dHidden[m,h]
```

Repeated token ids must accumulate correctly.

For the deterministic MVP, use output ownership:

- one invocation owns `(v,h)`;
- it scans `m = 0..M-1`;
- it accumulates rows whose token equals `v`;
- it writes exactly one `dEmbedding[v,h]`.

This is intentionally not the fastest possible embedding backward, but it avoids floating-point atomics and gives a clean correctness oracle. Optimize only after profiling a real training workload.

The shader may overwrite every `dEmbedding` output, including writing zero for unused vocabulary rows. If it accumulates into an existing buffer instead, `zero_f32.wgsl` must run first and the behavior must be explicit in the operation contract.

## 9. Gradient buffers and accumulation

The first milestone performs one optimizer update per batch and does not accumulate gradients across multiple `trainStep` calls.

Rules:

- every gradient buffer has a declared producer;
- a producer either overwrites the complete buffer or explicitly requires zeroing;
- no buffer relies on uninitialized GPU contents;
- parameter gradients are distinct from parameters;
- activation gradients may reuse arena regions only after their previous values are dead;
- parameter and gradient aliasing is forbidden.

`zero_f32.wgsl` exists for buffers that truly require accumulation. Do not dispatch it for gradients that are fully overwritten by their backward shader.

## 10. SGD optimizer

Initial update:

```text
parameter[i] = parameter[i] - learningRate * gradient[i]
```

`sgd_step.wgsl` receives or resolves:

- parameter offset;
- gradient offset;
- element count;
- learning rate.

Each invocation owns one parameter element.

The first implementation has:

- no momentum;
- no weight decay;
- no loss scaling;
- no gradient clipping.

The optimizer must update only explicitly registered trainable parameters. Frozen buffers must remain byte-for-byte unchanged.

## 11. Dispatch order

The initial `trainStep` executes:

```text
1. embedding forward
2. classifier matmul forward
3. cross-entropy forward + dLogits
4. loss reduction
5. classifier backward input -> dHidden
6. classifier backward weight -> dClassifier
7. embedding backward -> dEmbedding
8. SGD classifier update
9. SGD embedding update
10. telemetry/readback only when requested
```

Steps 5 and 6 are independent once `dLogits` exists and may later be placed in the same command encoder without a CPU synchronization point. Do not add readback merely to sequence GPU work; command order already provides the required dependency.

## 12. Host API

The target shape is one host call:

```ts
const result = await trainer.trainStep({
  tokens,
  targets,
  learningRate,
  telemetry: true,
});
```

Expected result:

```ts
interface TrainStepResult {
  loss?: number;
  step: number;
}
```

Debug/test-only options may request selected tensor readbacks. They must be opt-in and absent from the normal path.

Do not force the final public API during the first shader patch. A narrow internal test harness is acceptable. The important boundary is that callers do not manually run individual backward dispatches.

## 13. Arena and lifetime plan

For the vertical slice, define explicit arena regions for:

- hidden;
- logits;
- loss rows;
- scalar loss;
- dLogits;
- dHidden;
- dClassifier;
- dEmbedding.

The first version may use non-overlapping regions for clarity. Record lifetimes so the planner can later reuse memory:

```text
hidden:      embedding forward -> classifier weight backward
logits:      classifier forward -> cross entropy
lossRows:    cross entropy -> loss reduction
dLogits:     cross entropy -> both classifier backward shaders
dHidden:     classifier input backward -> embedding backward
dClassifier: classifier weight backward -> optimizer
dEmbedding:  embedding backward -> optimizer
```

Do not retain every internal workgroup value from forward. When full LFM2 backward is added, save block inputs and small normalization/softmax statistics, then recompute cheap internal activations where memory savings justify it.

## 14. Test strategy

### 14.1 CPU oracle

Implement a tiny plain TypeScript f32 reference for each new operation. It is test code, not a fallback runtime.

Required references:

- stable cross-entropy and `dLogits`;
- matmul `dX`;
- matmul `dW`;
- embedding scatter/add semantics;
- SGD update.

### 14.2 Operator tests

For each shader:

1. initialize tiny deterministic tensors;
2. run the GPU shader;
3. read back its output;
4. compare with the CPU oracle;
5. include non-square dimensions and repeated embedding ids.

Suggested initial tolerances for small f32 tests:

```text
absolute error <= 1e-5
relative error <= 1e-4
```

Use a combined `atol + rtol * abs(expected)` comparison. Tighten or relax only with an explained numerical reason, never merely to make a test pass.

### 14.3 Finite-difference gradient checks

Check analytical gradients of the complete tiny graph with central differences:

```text
numericGrad[i] = (loss(p[i] + eps) - loss(p[i] - eps)) / (2 * eps)
```

Use a small selected subset of embedding and classifier parameters, with an `eps` suitable for f32, initially around `1e-3`.

The finite-difference path may repeatedly execute forward and read scalar loss because it is diagnostic test code.

### 14.4 Integration test

The integration test must demonstrate all of the following:

- initial loss is finite;
- loss decreases over training;
- the tiny mapping is overfit;
- expected predictions reach 100% on the tiny training set;
- at least one embedding parameter changes;
- at least one classifier parameter changes;
- registered frozen parameters do not change;
- no validation error is reported by WebGPU.

Use a fixed seed or fixed initial tensors. The test must be reproducible.

## 15. Error handling and telemetry

Reuse the runtime's existing status/error reporting conventions where possible.

At minimum detect on the host or device boundary:

- incompatible tensor dimensions;
- arena range overflow;
- target id outside vocabulary;
- missing trainable parameter registration;
- gradient/parameter length mismatch;
- non-finite loss in diagnostic mode.

Useful initial telemetry:

```ts
interface TrainingTelemetry {
  step: number;
  loss: number;
  forwardMs?: number;
  backwardMs?: number;
  optimizerMs?: number;
}
```

Timing must be optional and must not introduce synchronization when disabled.

## 16. Explicitly out of scope for milestone 1

Do not implement any of the following as part of the first vertical slice:

- dynamic autograd or runtime graph recording;
- full LFM2 backward;
- attention backward;
- ShortConv backward;
- RMSNorm backward;
- SiLU/gate backward;
- RoPE backward;
- WQ4 parameter updates;
- quantization-aware training;
- f16 gradient accumulation;
- Adam/AdamW;
- gradient clipping;
- checkpoint recomputation;
- distributed training;
- PPO, REINFORCE or Gumbel-Softmax;
- argmax backward;
- constrained-decoder backward;
- production shader fusion;
- simulation or Creator's Touch curriculum.

It is fine to leave narrow extension points. It is not fine to implement speculative subsystems before the first loss curve works.

## 17. Later operator order

After milestone 1 is green, add operators in this order, with a CPU oracle and gradient check for each:

1. residual add backward;
2. SiLU-multiply backward;
3. RMSNorm backward;
4. ShortConv backward;
5. QK normalization and RoPE backward;
6. attention backward;
7. record encoder and record mixer training path;
8. soft gather backward and pointer loss;
9. typed decision heads;
10. AdamW and mixed precision;
11. checkpoint/recompute planner;
12. safe fusion based on profiling.

For soft gather, the intended differentiable path is:

```text
p = softmax(masked selector logits)
gathered = sum_i p[i] * value[i]
pointerLoss = -log(p[gold])
```

Runtime argmax and exact handle resolution remain inference/execution behavior, not backward operators.

## 18. WQ4 policy

Do not update packed WQ4 tensors directly.

Initial supported policy:

- WQ4/frozen backbone may participate in forward;
- gradients may flow through a frozen operation to trainable inputs or heads when its backward-input shader exists;
- no gradient buffer or optimizer state is allocated for frozen WQ4 weights;
- new trainable heads use f32 first, then f16/master-f32 later.

If full-weight training becomes necessary, use unquantized master weights during training and quantize checkpoints separately. QAT is a separate milestone.

## 19. Expected deliverables from the first coding pass

The first coding pass should aim to produce:

- the seven WGSL files listed in section 2.1, or the minimal subset actually required by the existing buffer semantics;
- Sandblaster registrations/build wiring for those shaders;
- a narrow host-side `trainStep` orchestration;
- plain TypeScript CPU reference functions;
- per-operator GPU-vs-CPU tests;
- finite-difference checks for selected parameters;
- one deterministic overfit integration test;
- a short note listing deliberate shortcuts and the next blocker.

If time runs out, prefer a smaller fully tested vertical slice over unverified implementations of additional operators.

## 20. Definition of done

Milestone 1 is done only when:

```text
[ ] all new shaders compile through the normal Sandblaster build
[ ] all GPU operator outputs match the CPU oracle
[ ] analytical gradients pass finite-difference checks
[ ] repeated token ids accumulate embedding gradients correctly
[ ] trainStep performs no intermediate CPU readback
[ ] frozen parameters remain unchanged
[ ] the deterministic toy dataset reaches 100% accuracy
[ ] the loss curve decreases reproducibly
[ ] no WebGPU validation errors occur
```

Anything beyond this checklist is optional for the first pass.
