Here is a comprehensive, production-grade architectural specification document ready to be committed to your repository (e.g. as `docs/KRYSTAL_WGSL_ARCHITECTURE.md`).

---

# Krystal WebGPU / WGSL Architecture Specification

**Status:** Living Specification  
**Subsystem:** `@krystal/webgpu`  
**Execution Backends:** WebGPU (Browser via Chromium/Firefox), Native (via Deno / `wgpu` / Dawn)

---

## 1. Executive Summary & Design Principles

Krystal’s compute runtime is a **pure WebGPU Shading Language (WGSL) implementation** of both the forward inference engine and the complete backward backpropagation/training pipeline. It eliminates any reliance on external machine learning frameworks (PyTorch, TensorFlow, ONNX Runtime) and Python runtimes.

```
       Host Simulation Frame (SoA U32/F32 Payloads + Masks)
                              │
                              ▼  (Single submit)
   ┌─────────────────────────────────────────────────────────┐
   │            WebGPU Command Buffer Execution              │
   │                                                         │
   │  [Field Embed] ──► [Record Encoder x2] ──► [Pool]       │
   │                                              │          │
   │  [Decision Head] ◄── [Selector] ◄── [Mixer x2]          │
   │         │                                               │
   │         ▼ (Optional Training Step in same submit)       │
   │  [Loss Kernels] ──► [Analytical Backward Passes]        │
   │                           │                             │
   │                           ▼                             │
   │               [Interleaved In-Place SGD]                │
   └─────────────────────────────────────────────────────────┘
                              │
                              ▼  (Single readback)
              Action Selection / Evaluated Gradient
```

### Core Tenets

1. **Zero-Python & Zero-CUDA Vendor Independence:**  
   Every tensor operation, multi-head attention block, softmax backward pass, and weight optimizer step is written as an explicit compute shader in WGSL. The pipeline runs natively across Vulkan, Metal, and DirectX 12 without modification.
2. **One-Submit Execution Contract:**  
   Forward inference, loss evaluation, backward gradient propagation, and optimizer weight updates are queued into a **single GPU command buffer submission** (`engine.submit(...)`). CPU-GPU synchronization stalls within the execution loop are strictly zero.
3. **Monolithic Arena Memory Model:**  
   All intermediate activations, residual buffers, attention maps, and gradient tensors reside at statically compiled offsets inside a single, pre-allocated GPU storage buffer (`krystal.arena`). Dynamic allocations during inference and training are forbidden.
4. **Decoupled Architecture (AOT Linked Artifacts):**  
   Shaders contain body-only implementations. Sandblaster generates standard entry points and binding manifests at build time into `krystal.artifact.generated.ts`, allowing instant, static startup without schema parsing overhead.

---

## 2. Memory Model & Buffer Architecture

The WebGPU backend is structured around three primary resource abstractions:

```
+-------------------------------------------------------------------------+
|                              krystal.arena                              |
| [Training Arena (M1)] | [Forward Arena (M2b)] | [Backward Arena (M3)]   |
| (activations/losses)  | (SoA, enc, mix, pool) | (gradients, dW, dScores)|
+-------------------------------------------------------------------------+

+-----------------------------------+  +----------------------------------+
|            krystal.op             |  |      weight32 (Group 1)          |
| Ring of 256-byte aligned uniform  |  | Per-dispatch dynamic binding     |
| dispatch descriptors (OpParams)   |  | of dense parameter weight pages  |
+-----------------------------------+  +----------------------------------+
```

### 2.1 The Monolithic Arena (`krystal.arena`)

The arena is bound to Group 0, Binding 5 as `var<storage, read_write> arena: array<f32>;`. It is segmented into three continuous regions:

* **Training Base (`TRAINING_ARENA`):** Scratchpads for foundational linear layers, loss reduction buffers, and telemetry.
* **Forward Base (`KRYSTAL_FORWARD_ARENA`):** Host-uploaded Structure-of-Arrays (SoA) frame inputs, field embeddings ($T \times H$), multi-block encoder scratchpads, pooled record banks ($R \times H$), mixer outputs ($Q \times H$), and selector probabilities ($Q \times R$).
* **Backward Base (`KRYSTAL_BACKWARD_ARENA`):** Upstream gradient buffers ($d\text{Out}, d\text{Scores}, dQ, dK, dV$), intermediate activation checkpoints, parameter gradient buffers ($dW_1, dW_2, dW_q, dW_k, dW_v, dWh$), and selector pointer gradients.

### 2.2 Zero-Overhead U32 Bitcasting

The ABI frame format encodes categorical features as raw unsigned integers (`tokenIds`, `fieldRoles`, `schemaIds`, `bandIds`, `streamIds`, `activeTokens`). Rather than maintaining separate integer buffers, these arrays are written directly into the float arena by the host and reinterpreted inside WGSL via bitcast:

```wgsl
let frameTok = bitcast<u32>(arena[op.aux4Offset + t]);
let slot     = frameTok >> 3u;
let local    = frameTok & 7u;
let tok      = bitcast<u32>(arena[op.inputOffset + frameTok]);
```

### 2.3 Dispatch Metadata (`OpParams` & Dynamic Offsets)

Each compute pass is parameter-agnostic; its input, output, auxiliary offsets, and loop dimensions are supplied via a uniform struct (`OpParams`) bound with `hasDynamicOffset: true`:

```wgsl
struct OpParams {
  inputOffset:  u32, outputOffset: u32,
  auxOffset:    u32, aux2Offset:   u32,
  aux3Offset:   u32, aux4Offset:   u32,
  aux5Offset:   u32, aux6Offset:   u32,
  tokenCount:   u32, inputDim:     u32, outputDim: u32,
  rowStart:     u32, rowCount:     u32,
  layerIndex:   u32, attentionSlot:u32,
  mode:         u32,
  f0:           f32, f1:           f32,
  u0:           u32, u1:           u32, u2: u32, u3: u32, u4: u32, u5: u32,
};
```

`KrystalParamWriter` allocates and serializes a sequential stream of 96-byte parameter records (aligned to 256 bytes per device requirements) into a single buffer upload prior to command execution.

### 2.4 Weight Pages (`weight32`)

Tensor weights reside in dedicated `GPUBuffer` allocations bound to Bind Group 1:
* **Inference Mode:** Bound as `buffer: { type: "read-only-storage" }`.
* **Optimizer Mode (`sgd_step`, `krystal_field_embed_sgd`):** Bound as `buffer: { type: "storage" }` for in-place parameter mutation.

All shaders declare `var<storage, ...> weight32: array<f32>;` without a fixed element limit, allowing dynamically sized weight matrices without pipeline re-compilation.

---

## 3. Mathematical Contracts & Shader Catalog

### 3.1 Token & Field Representation

#### `krystal_field_embed` / `krystal_field_embed_sgd`
Each active token $t$ in record slot $s = \lfloor \text{frameTok} / 8 \rfloor$ at local position $p = \text{frameTok} \pmod 8$ is encoded via six additive embedding tables concatenated into a single weight page:

$$\mathbf{e}_t = \mathbf{E}_{\text{token}}[\text{tok}_t] + \mathbf{E}_{\text{field}}[\text{role}_t] + \mathbf{E}_{\text{schema}}[s] + \mathbf{E}_{\text{band}}[s] + \mathbf{E}_{\text{stream}}[s] + \mathbf{E}_{\text{pos}}[p]$$

* **Forward (`krystal_field_embed`):** Launched with workgroup size 256 over $T_{\text{active}} \times H$.
* **Fused Backward & SGD (`krystal_field_embed_sgd`):** Instead of computing dense gradients across all unused vocabulary rows (8,469+ rows), the host extracts active row indices. Invocations update only referenced embedding rows directly in VRAM:

  $$\mathbf{W}_{\text{emb}}[r, h] \leftarrow \mathbf{W}_{\text{emb}}[r, h] - \eta \sum_{t \in \text{active}(r)} d\text{FieldStates}[t, h]$$

---

### 3.2 Attention Mechanics

#### `krystal_attention_forward`
A masked multi-head attention kernel serving both the **Record Encoder** ($Q=K=V=T$) and the **Query-to-Record Mixer** ($Q=Q_{\text{queries}}, K=V=R_{\text{bank}}$).

* Scale factor: $\tau = \frac{1}{\sqrt{d_k}}$.
* Input shapes: $Q \in \mathbb{R}^{M \times H}$, $K \in \mathbb{R}^{N \times H}$, $V \in \mathbb{R}^{N \times H}$, $\text{mask} \in \mathbb{R}^{M \times N}$.
* Workgroup layout: One workgroup per `(head, row)` using 64 threads.
* Shared memory scratchpad: `var<workgroup> attentionScores: array<f32, 1024>;`.

```
Pass 1: Lane-strided dot product Q_i · K_j + mask[i, j] -> workgroup memory
Pass 2: Numerically stable reduction: rowMax -> sumExp -> Softmax normalization
Pass 3: Write probabilities P into arena (saved for backward) + Context vector accumulation
```

**Block-Diagonal Local Record Mode ($u_3 \ne 0$):**
When running record self-attention, cross-record attention is blocked by definition. The kernel reads precompiled `recordCompactOffset` and `recordCompactCount`, restricting key loops exclusively to valid tokens of the same record and skipping empty padding slots entirely.

#### `krystal_attention_backward_scores` & `krystal_attention_backward_qkv`
Analytical backpropagation through masked multi-head attention:

$$dP_{h,i,j} = \sum_{d=0}^{d_k-1} d\text{Out}_{i,h,d} \cdot V_{j,h,d}$$

$$\text{rowSum}_{h,i} = \sum_{j} P_{h,i,j} \cdot dP_{h,i,j}$$

$$d\text{Scores}_{h,i,j} = P_{h,i,j} \cdot (dP_{h,i,j} - \text{rowSum}_{h,i})$$

$$dQ_{i,h,d} = \tau \sum_{j} d\text{Scores}_{h,i,j} \cdot K_{j,h,d}, \quad dK_{j,h,d} = \tau \sum_{i} d\text{Scores}_{h,i,j} \cdot Q_{i,h,d}, \quad dV_{j,h,d} = \sum_{i} P_{h,i,j} \cdot d\text{Out}_{i,h,d}$$

The gradient routing kernel (`krystal_attention_backward_qkv`) splits linear workgroup threads into three independent chunks ($dQ$, $dK$, $dV$), guaranteeing atomic-free, single-owner writes.

---

### 3.3 Record Pooling

#### `krystal_pool` / `krystal_pool_backward` / `krystal_pool_dpool`
Aggregates field representations within a record into discrete Key and Value vectors via two learned queries ($q_k, q_v \in \mathbb{R}^{H}$):

$$\mathbf{k}_r = \sum_{j \in \text{tokens}(r)} \text{softmax}_j\left(\frac{q_k \cdot s_j}{\sqrt{H}}\right) s_j, \quad \mathbf{v}_r = \sum_{j \in \text{tokens}(r)} \text{softmax}_j\left(\frac{q_v \cdot s_j}{\sqrt{H}}\right) s_j$$

* Record width is fixed at ABI limit $W=8$. Score normalization runs completely within 8-slot shared workgroup arrays (`poolKeyScores`, `poolValueScores`).
* **Backward:** `krystal_pool_backward` writes per-record parameter partial gradients into `dPoolPartial[r, 2, H]`. `krystal_pool_dpool` reduces these partials deterministically across records into the master parameter gradient $d\text{Pool}$.

---

### 3.4 Selection & Soft Gather

#### `krystal_selector` / `krystal_selector_backward_scores` / `krystal_selector_backward_qkv`
Computes categorical choice distributions over bank records and performs soft content gathering:

$$S_{i,j} = \frac{\mathbf{qProj}_i \cdot \mathbf{kProj}_j}{\sqrt{H}} + \text{mask}_{i,j}$$

$$\mathbf{p}_i = \text{softmax}(S_{i, *}), \quad \mathbf{g}_i = \sum_{j=0}^{R-1} p_{i,j} \mathbf{v}_j, \quad \text{idx}_i = \arg\max_j p_{i,j}$$

**Supervised Pointer Loss Integration:**  
During training, the backward pass integrates an exact supervised pointer loss (or unlikelihood loss if the top bit of `gold` is set):

$$d\text{Score}_{i,j} = p_{i,j} \cdot (dP_{i,j} - \text{rowSum}_i) + \nabla_{\text{pointer}}$$

$$\nabla_{\text{pointer}} = \begin{cases} 
p_{i,j} - \mathbf{1}_{j = \text{gold}_i} & \text{for standard demonstration target} \\
\frac{p_{i,\text{gold}}}{1 - p_{i,\text{gold}}} (\mathbf{1}_{j = \text{gold}} - p_{i,j}) & \text{for negative / discouraged target (unlikelihood)} \\
0 & \text{if target is INVALID } (0\text{xFFFF\_FFFF})
\end{cases}$$

---

### 3.5 Decision & Critic Heads

#### `krystal_decision_head` / `krystal_decision_head_backward`
Computes linear logits over route kinds or state values from the concatenated gathered context:

$$\mathbf{ctx}_q = \big[\, \mathbf{y}_q \;\mathbin{\Vert}\; \mathbf{g}^{\text{intent}}_q \;\mathbin{\Vert}\; \mathbf{g}^{\text{arg / available}}_q \,\big] \in \mathbb{R}^{3H}$$

$$\text{logits}_q = \mathbf{ctx}_q \mathbf{W}_h^T$$

The backward kernel reads upstream cross-entropy $d\text{Logits}$ and maps the gradient back into the three constituent components of $\mathbf{ctx}$ while concurrently producing the parameter gradient $d\mathbf{W}_h$:

$$d\mathbf{y}_q = d\text{Logits}_q \mathbf{W}_{h, [0:H]}, \quad d\mathbf{g}^{\text{intent}}_q = d\text{Logits}_q \mathbf{W}_{h, [H:2H]}, \quad d\mathbf{g}^{\text{arg}}_q = d\text{Logits}_q \mathbf{W}_{h, [2H:3H]}$$

#### `krystal_value_head_loss`
Evaluates the squared-error critic loss against observed valence transitions:

$$\mathcal{L}_v = \frac{1}{2Q} \sum_{q=0}^{Q-1} (\hat{v}_q - v^*)^2, \quad d\hat{v}_q = \frac{\hat{v}_q - v^*}{Q}$$

The value head gradient feeds into the shared representation ($\mathbf{y}_q$ and $\mathbf{g}^{\text{intent}}_q$) via residual addition, conditioning the encoder and mixer representations on both action selection and expected future return.

---

## 4. Execution Flow & Pass Pipelines

### 4.1 Forward Pass Graph (`KrystalForward.dispatchForward`)

```
1. [krystal_field_embed] (table bases u0..u5, weight32: embeddingsPage)
2. Loop over Encoder Blocks (b = 0 .. encoderBlocks-1):
   a. [matmul_f32] x3 (projections Wq, Wk, Wv -> encQ, encK, encV)
   b. [krystal_attention_forward] (local record mode u3=1, encMask)
   c. [residual_add] (fieldStates += encOut)
   d. [matmul_f32] (W1 -> encH1)
   e. [relu] (encH1 in-place)
   f. [matmul_f32] (W2 -> encOut)
   g. [residual_add] (fieldStates += encOut)
3. [krystal_pool] (bank records -> bankKeys, bankValues)
   [krystal_pool] (query records -> queryKeys, queryValues)
4. Loop over Mixer Blocks (b = 0 .. mixerBlocks-1):
   a. [matmul_f32] (queryValues @ Wq -> mixerQ)
   b. [matmul_f32] (bankKeys @ Wk -> mixerK)
   c. [matmul_f32] (bankValues @ Wv -> mixerV)
   d. [krystal_attention_forward] (cross mode, mixerMask)
   e. [residual_add] (queryValues += mixed)
   f. [matmul_f32] -> [relu] -> [matmul_f32] -> [residual_add] (FFN)
5. Selection & Soft Gather:
   a. [matmul_f32] x2 (queryValues @ selectorWq, bankKeys @ selectorWk)
   b. [krystal_selector] (intent slot -> intentGather, intentP, intentIndices)
   c. Slot 2 ([krystal_selector] for argument OR [zero_f32] + mean selector for available)
6. Decision & Critic Heads:
   a. [krystal_decision_head] (ctx -> decisionLogits)
   b. [krystal_decision_head] (ctx_critic -> valuePrediction)
```

### 4.2 Backward Pass Graph (`KrystalBackward.trainStep`)

```
1. Route-Kind Loss:
   [cross_entropy_forward_backward] -> dDecisionLogits, lossRows
   [loss_reduce] -> scalar telemetry
2. Decision Head Backward:
   [krystal_decision_head_backward] -> dDecisionQuery, dDecisionIntent, dDecisionArg, dDecisionWh
   (opt) [sgd_step] on decisionHeadWh
3. Value Head Backward:
   [krystal_value_head_loss] -> dValuePrediction
   [krystal_decision_head_backward] -> dValueQuery, dValueIntent, dValueArg, dValueWv
   [residual_add] (dDecisionQuery += dValueQuery)
   [residual_add] (dDecisionIntent += dValueIntent)
   (opt) [sgd_step] on valueHeadWv
4. Selector Backward:
   [krystal_selector_backward_scores] (includes pointer loss gradient)
   [krystal_selector_backward_qkv] -> dSelectorQProj, dSelectorKProj, dSelectorValue
   [matmul_backward_weight] x2 -> dSelectorWq, dSelectorWk
   [matmul_backward_input] x2 -> route into residual streams (dQueryValues, dBankKeys, dBankValues)
   (opt) [sgd_step] on selectorWq, selectorWk
5. Reverse Mixer Loop (b = mixerBlocks-1 down to 0):
   [matmul_backward_weight] & [matmul_backward_input] (W2)
   [relu_backward]
   [matmul_backward_weight] & [matmul_backward_input] (W1)
   [residual_add]
   [krystal_attention_backward_scores]
   [krystal_attention_backward_qkv]
   [matmul_backward_weight] & [matmul_backward_input] (Wq, Wk, Wv)
   (opt) [sgd_step] on mixer block weights
6. Pooling Backward:
   [krystal_pool_backward] (bank) + [krystal_pool_dpool]
   [krystal_pool_backward] (query) + [krystal_pool_dpool]
   (opt) [sgd_step] on pool queries
7. Reverse Encoder Loop (b = encoderBlocks-1 down to 0):
   Backpropagate FFN and local record attention identically to mixer loop
   (opt) [sgd_step] on encoder block weights
8. Field Embeddings Backward:
   [krystal_field_embed_sgd] (fused sparse scatter-add + parameter update)
```

---

## 5. Host Infrastructure & Profiling

### 5.1 Host Backend Bridge (`GpuBrainBackend`)

`GpuBrainBackend` implements the Krystal host interface (`BrainBackend`), maintaining synchronization parity with the CPU reference oracle:

* **Serialization (`inTurn`):** Ensures that concurrent calls across web worker threads or multiple game pawns serialize GPU access over the shared arena.
* **Selective Weight Sync (`uploadWeights`):** Instead of uploading the entire parameter footprint (~7 MB) on every frame, delta manifests (`WeightChanges`) pinpoint modified tensors (e.g. only touched embedding rows or selector heads), reducing PCI-e/VRAM transfer bandwidth by >95%.

### 5.2 Zero-Copy Readback Strategy (`readMapped`)

Standard WebGPU readback copies staging buffers into CPU arrays via generic serializers, causing high garbage collection pressure. The runtime replaces this with `readMapped`:

1. Concatenates multiple region readback requests into a single continuous GPU-to-GPU copy pass (`copyBufferToBuffer`).
2. Dispatches copying inside the *same command buffer* as the compute workload.
3. Invokes `mapAsync(GPUMapMode.READ)` once per turn.
4. Returns non-allocating typed array views directly via `Float32Array(target.getMappedRange())`.

---

## 6. Verification, Parity & Testing Matrix

The implementation is verified against the CPU reference oracle (`packages/krystal/src/forward/oracle.ts` and `backward.ts`):

| Test Suite | File | Coverage |
|---|---|---|
| **Operator Parity** | `test/krystal-forward.test.ts` | Unit tests for `relu`, `attention_forward`, `pool`, `selector`, `field_embed` against CPU math. |
| **Backward Parity** | `test/krystal-backward.test.ts` | Exact gradient validation of backward kernels against CPU oracles and finite-difference checks. |
| **Composed End-to-End** | `test/krystal-composed.test.ts` | Full multi-block forward and backward gradient alignment on active game frames. |
| **Session Equivalence** | `test/session.test.ts` | Dual-session test running CPU vs GPU on identical seeds, verifying exact match in policy divergence during `learn` and `teach`. |

### Diagnostic Execution

```bash
# Validate and type-check WGSL kernels across WebGPU device adapters
deno run --allow-read --sloppy-imports scripts/validate-krystal-shaders.ts --full

# Execute unified GPU tests via Dawn / headless WebGPU
bun test webgpu/test/krystal-composed.test.ts
bun test webgpu/test/session.test.ts
```