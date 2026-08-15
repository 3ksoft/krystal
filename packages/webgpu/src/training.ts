// M1 tiny f32 training vertical slice: host orchestration for the GPU-resident
// trainStep path. See docs/WEBGPU_BACKWARD_PLAN.md §10–13.
//
// Design:
//   - One submit per trainStep; the whole step (forward, backward, optimizer)
//     stays on the GPU. The only readback in the normal path is the compact
//     scalar loss telemetry (4 bytes) when `telemetry: true`.
//   - Parameters are f32 pages (embedding [V,H], classifier [V,H], and — when
//     an encoder is configured — Wq/Wk/Wv [H,H]) owned by the trainer, written
//     once from host Float32Arrays and updated in place by sgd_step. Frozen
//     pages are simply never dispatched through sgd_step.
//   - Activations/gradients live in the shared f32 arena, in the training
//     regions appended after the LFM2 layout (TRAINING_ARENA_BASE).
//   - Dispatch order mirrors WEBGPU_BACKWARD_PLAN.md §11.
//
// The attention encoder block (§17 item 6 wiring) runs when `encoder` is set:
//
//   forward:  hidden = E[tokens]
//             q = hidden@Wq^T, k = hidden@Wk^T, v = hidden@Wv^T
//             out, P = attention(q, k, v, mask)
//             logits = out@classifier^T
//   backward: dOut = dLogits@classifier
//             dScores = dP·P gradient (attention_backward_scores)
//             dQ, dK, dV = attention_backward_qkv
//             dHidden = dQ@Wq + dK@Wk + dV@Wv  (3 matmul_backward_input,
//                       accumulated with residual_add)
//             dWq = q^T dQ? -> dQ^T@hidden, same for dWk/dWv
//   optimizer: sgd_step on classifier, Wq, Wk, Wv, embedding
//
// Without an encoder the graph is the M1 toy path (embedding -> classifier).
//
// Deliberate M1 shortcuts (documented in the plan):
//   - hard-coded static backward plan, no autograd framework;
//   - f32 everywhere, plain SGD without momentum;
//   - each gradient shader fully overwrites its output (no zeroing needed);
//   - debug readbacks (parameter snapshots) copy into the small
//     training-readback staging buffer instead of reading back pages directly.
import {
  LFM2_TRAINING_ARENA,
  TRAINING_ARENA_BASE,
  TRAINING_MAX_H,
  TRAINING_MAX_HEADS,
  TRAINING_MAX_M,
  TRAINING_MAX_V,
  TRAINING_READBACK_ELEMENTS,
} from "./lfm2-layout";
import { lfm2, type Lfm2Definition } from "./lfm2";
import { Lfm2Executor } from "./pass";

export interface EncoderConfig {
  /** Full attention heads; head h owns columns [h*headDim, (h+1)*headDim). */
  readonly headCount: number;
  readonly headDim: number;
  /** QKV projection pages, each [H,H] row-major (same layout as matmul W). */
  readonly wq: Float32Array;
  readonly wk: Float32Array;
  readonly wv: Float32Array;
}

export interface TrainingConfig {
  /** Vocabulary size V (embedding rows and logits width). */
  readonly vocabSize: number;
  /** Hidden size H. */
  readonly hiddenSize: number;
  /** Initial embedding table [V,H] f32, row-major. */
  readonly embedding: Float32Array;
  /** Initial classifier [V,H] f32, row-major (same layout as matmul_f32 W). */
  readonly classifier: Float32Array;
  /** Optional attention encoder block (§17 item 6). */
  readonly encoder?: EncoderConfig;
  /** Frozen pages stay byte-identical: they are never sent through sgd_step. */
  readonly frozen?: readonly (keyof TrainingParameters)[];
}

export interface TrainingParameters {
  embedding: Float32Array;
  classifier: Float32Array;
  wq?: Float32Array;
  wk?: Float32Array;
  wv?: Float32Array;
}

export interface TrainStepOptions {
  readonly tokens: readonly number[] | Uint32Array;
  readonly targets: readonly number[] | Uint32Array;
  readonly learningRate: number;
  /**
   * Host-compiled attention mask [M,M] f32 (0.0 = allowed, -1e30 = blocked).
   * Required when an encoder is configured; ignored otherwise.
   */
  readonly mask?: Float32Array;
  /** Read back the scalar loss (compact telemetry); off by default. */
  readonly telemetry?: boolean;
}

export interface TrainStepResult {
  readonly step: number;
  readonly loss?: number;
}

function validate(condition: boolean, message: string): void {
  if (!condition) throw new Error(`TrainingTrainer: ${message}`);
}

function assertDims(config: TrainingConfig): void {
  const { vocabSize: V, hiddenSize: H } = config;
  validate(V > 0, "vocabSize must be > 0");
  validate(H > 0, "hiddenSize must be > 0");
  validate(V <= TRAINING_MAX_V, `vocabSize ${V} exceeds capacity ${TRAINING_MAX_V}`);
  validate(H <= TRAINING_MAX_H, `hiddenSize ${H} exceeds capacity ${TRAINING_MAX_H}`);
  validate(
    config.embedding.length === V * H,
    `embedding length ${config.embedding.length} != V*H ${V * H}`,
  );
  validate(
    config.classifier.length === V * H,
    `classifier length ${config.classifier.length} != V*H ${V * H}`,
  );
  const encoder = config.encoder;
  if (encoder) {
    validate(encoder.headCount > 0, "encoder.headCount must be > 0");
    validate(encoder.headDim > 0, "encoder.headDim must be > 0");
    validate(
      encoder.headCount * encoder.headDim === H,
      `encoder heads*headDim ${encoder.headCount * encoder.headDim} != H ${H}`,
    );
    validate(encoder.headCount <= TRAINING_MAX_HEADS, `headCount ${encoder.headCount} exceeds capacity`);
    for (const [name, w] of [["wq", encoder.wq], ["wk", encoder.wk], ["wv", encoder.wv]] as const) {
      validate(w.length === H * H, `${name} length ${w.length} != H*H ${H * H}`);
    }
  }
}

/**
 * Host-facing training runner. Wraps the shared LFM2 definition and reuses the
 * existing OpParams/arena/pass.run orchestration; callers never run individual
 * backward dispatches themselves.
 */
export class TrainingTrainer {
  private readonly definition: Lfm2Definition;
  private readonly config: TrainingConfig;
  private readonly embeddingPage: GPUBuffer;
  private readonly classifierPage: GPUBuffer;
  private readonly wqPage: GPUBuffer | undefined;
  private readonly wkPage: GPUBuffer | undefined;
  private readonly wvPage: GPUBuffer | undefined;
  private readonly executor: Lfm2Executor;
  private step = 0;

  constructor(config: TrainingConfig, definition: Lfm2Definition = lfm2) {
    assertDims(config);
    this.config = config;
    this.definition = definition;
    this.executor = new Lfm2Executor(definition);

    const paramUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const { vocabSize: V, hiddenSize: H } = config;
    this.embeddingPage = this.createPage("krystal.train.embedding", V * H * 4, paramUsage);
    this.classifierPage = this.createPage("krystal.train.classifier", V * H * 4, paramUsage);
    const device = definition.engine.device;
    device.queue.writeBuffer(this.embeddingPage, 0, config.embedding);
    device.queue.writeBuffer(this.classifierPage, 0, config.classifier);

    if (config.encoder) {
      const { wq, wk, wv } = config.encoder;
      this.wqPage = this.createPage("krystal.train.wq", H * H * 4, paramUsage);
      this.wkPage = this.createPage("krystal.train.wk", H * H * 4, paramUsage);
      this.wvPage = this.createPage("krystal.train.wv", H * H * 4, paramUsage);
      device.queue.writeBuffer(this.wqPage, 0, wq);
      device.queue.writeBuffer(this.wkPage, 0, wk);
      device.queue.writeBuffer(this.wvPage, 0, wv);
    }
  }

  private createPage(label: string, bytes: number, usage: GPUBufferUsageFlags): GPUBuffer {
    return this.definition.engine.device.createBuffer({ label, size: bytes, usage });
  }

  private region(offset: number, elements: number): number {
    validate(
      offset + elements <= LFM2_TRAINING_ARENA.elements,
      "training arena region overflows the declared capacity",
    );
    return TRAINING_ARENA_BASE + offset;
  }

  /** Run one GPU-resident training step on the toy graph (with optional encoder). */
  async trainStep(options: TrainStepOptions): Promise<TrainStepResult> {
    const { vocabSize: V, hiddenSize: H } = this.config;
    const M = options.tokens.length;
    validate(options.targets.length === M, "tokens and targets must be same length");
    validate(M > 0, "batch must not be empty");
    validate(M <= TRAINING_MAX_M, `batch M ${M} exceeds capacity ${TRAINING_MAX_M}`);
    validate(Number.isFinite(options.learningRate) && options.learningRate > 0, "learningRate must be > 0");
    for (const token of options.tokens) {
      validate(Number.isInteger(token) && token >= 0 && token < V, `token ${token} outside [0, V)`);
    }
    for (const target of options.targets) {
      validate(Number.isInteger(target) && target >= 0 && target < V, `target ${target} outside [0, V)`);
    }

    const encoder = this.config.encoder;
    if (encoder) {
      validate(
        options.mask !== undefined,
        "a host-compiled mask [M,M] is required when an encoder is configured",
      );
      validate(options.mask!.length === M * M, `mask length ${options.mask!.length} != M*M ${M * M}`);
    }

    // Arena regions (WEBGPU_BACKWARD_PLAN.md §13), appended after the LFM2 ones.
    const A = LFM2_TRAINING_ARENA;
    const hidden = this.region(A.hidden, M * H);
    const logits = this.region(A.logits, M * V);
    const lossRows = this.region(A.lossRows, M);
    const scalarLoss = this.region(A.scalarLoss, 1);
    const dLogits = this.region(A.dLogits, M * V);
    const dHidden = this.region(A.dHidden, M * H);
    const dClassifier = this.region(A.dClassifier, V * H);
    const dEmbedding = this.region(A.dEmbedding, V * H);

    const tokenIds = options.tokens instanceof Uint32Array ? options.tokens : Uint32Array.from(options.tokens);
    const targetIds = options.targets instanceof Uint32Array ? options.targets : Uint32Array.from(options.targets);
    const device = this.definition.engine.device;
    device.queue.writeBuffer(this.definition.resources.tokens.gpu, 0, tokenIds);
    device.queue.writeBuffer(this.definition.resources.targets.gpu, 0, targetIds);
    if (encoder && options.mask) {
      device.queue.writeBuffer(this.definition.resources.arena.gpu, this.region(A.mask, M * M) * 4, options.mask);
    }

    const frozen = new Set(this.config.frozen ?? []);
    const step = ++this.step;

    this.executor.submit((encoderCb) => {
      // 1. embedding forward: hidden = E[tokens]
      encoderCb.compute((pass) => pass.run("embedding_f32", {
        tokenCount: M, inputDim: V, outputDim: H, outputOffset: hidden, u0: 0,
      }, this.embeddingPage));

      let logitsInput = hidden;
      let dHiddenGrad: number | undefined;

      if (encoder) {
        const { headCount, headDim } = encoder;
        const Aq = this.region(A.q, M * H);
        const Ak = this.region(A.k, M * H);
        const Av = this.region(A.v, M * H);
        const out = this.region(A.out, M * H);
        const p = this.region(A.p, headCount * M * M);
        const mask = this.region(A.mask, M * M);
        const dOut = this.region(A.dOut, M * H);
        const dScores = this.region(A.dScores, headCount * M * M);
        const dQ = this.region(A.dQ, M * H);
        const dK = this.region(A.dK, M * H);
        const dV = this.region(A.dV, M * H);
        const dHiddenQ = this.region(A.dHiddenQ, M * H);
        const dHiddenK = this.region(A.dHiddenK, M * H);
        const dHiddenV = this.region(A.dHiddenV, M * H);
        const dWq = this.region(A.dWq, H * H);
        const dWk = this.region(A.dWk, H * H);
        const dWv = this.region(A.dWv, H * H);

        // 2. QKV projections (matmul_f32: y = x @ W^T).
        for (const [page, off] of [[this.wqPage, Aq], [this.wkPage, Ak], [this.wvPage, Av]] as const) {
          encoderCb.compute((pass) => pass.run("matmul_f32", {
            inputOffset: hidden, outputOffset: off,
            tokenCount: M, inputDim: H, outputDim: H, rowStart: 0, rowCount: H,
          }, page));
        }

        // 3. attention forward (persists P for backward).
        encoderCb.compute((pass) => pass.run("attention_forward", {
          inputOffset: Aq, auxOffset: Ak, aux2Offset: Av, aux3Offset: mask,
          outputOffset: out, aux4Offset: p,
          tokenCount: M, inputDim: H, outputDim: headDim, u0: headCount,
        }));

        logitsInput = out;

        // 4. classifier forward: logits = out @ classifier^T (after attention).
        encoderCb.compute((pass) => pass.run("matmul_f32", {
          inputOffset: logitsInput, outputOffset: logits,
          tokenCount: M, inputDim: H, outputDim: V, rowStart: 0, rowCount: V,
        }, this.classifierPage));

        // 5. cross-entropy forward + dLogits (fused).
        encoderCb.compute((pass) => pass.run("cross_entropy_forward_backward", {
          inputOffset: logits, outputOffset: dLogits, auxOffset: lossRows,
          tokenCount: M, outputDim: V, u1: 0,
        }));

        // 6. loss reduction (telemetry scalar; gradient never depends on it).
        encoderCb.compute((pass) => pass.run("loss_reduce", {
          inputOffset: lossRows, outputOffset: scalarLoss, tokenCount: M,
        }));

        // 7. classifier backward input: dOut = dLogits @ classifier.
        encoderCb.compute((pass) => pass.run("matmul_backward_input", {
          inputOffset: dLogits, outputOffset: dOut,
          tokenCount: M, inputDim: V, outputDim: H,
        }, this.classifierPage));

        // 8. classifier backward weight: dClassifier = dLogits^T @ out.
        encoderCb.compute((pass) => pass.run("matmul_backward_weight", {
          inputOffset: dLogits, auxOffset: logitsInput, outputOffset: dClassifier,
          tokenCount: M, inputDim: V, outputDim: H,
        }));

        // 9. attention softmax-score gradient: dScores.
        encoderCb.compute((pass) => pass.run("attention_backward_scores", {
          inputOffset: dOut, auxOffset: Av, aux2Offset: p, outputOffset: dScores,
          tokenCount: M, inputDim: H, outputDim: headDim, u0: headCount,
        }));

        // 10. attention Q/K/V gradients.
        encoderCb.compute((pass) => pass.run("attention_backward_qkv", {
          inputOffset: dScores, auxOffset: Aq, aux2Offset: Ak,
          aux3Offset: p, aux4Offset: dOut,
          outputOffset: dQ, aux5Offset: dK, aux6Offset: dV,
          tokenCount: M, inputDim: H, outputDim: headDim, u0: headCount,
        }));

        // 11. projection-input gradients: dHiddenQ/K/V = dQ/dK/dV @ Wq/Wk/Wv.
        for (const [page, dy, dOut2] of [[this.wqPage, dQ, dHiddenQ], [this.wkPage, dK, dHiddenK], [this.wvPage, dV, dHiddenV]] as const) {
          encoderCb.compute((pass) => pass.run("matmul_backward_input", {
            inputOffset: dy, outputOffset: dOut2,
            tokenCount: M, inputDim: H, outputDim: H,
          }, page));
        }

        // 12. accumulate dHidden = dHiddenQ + dHiddenK + dHiddenV.
        encoderCb.compute((pass) => pass.run("residual_add", {
          inputOffset: dHiddenQ, auxOffset: dHiddenK, outputOffset: dHidden,
          tokenCount: M, inputDim: H,
        }));
        encoderCb.compute((pass) => pass.run("residual_add", {
          inputOffset: dHidden, auxOffset: dHiddenV, outputOffset: dHidden,
          tokenCount: M, inputDim: H,
        }));

        // 13. projection-weight gradients: dWq = dQ^T @ hidden.
        for (const [page, dy, dW] of [[this.wqPage, dQ, dWq], [this.wkPage, dK, dWk], [this.wvPage, dV, dWv]] as const) {
          encoderCb.compute((pass) => pass.run("matmul_backward_weight", {
            inputOffset: dy, auxOffset: hidden, outputOffset: dW,
            tokenCount: M, inputDim: H, outputDim: H,
          }, page));
        }

        dHiddenGrad = dHidden;

        // 14. embedding backward: scatter-add of dHidden onto token rows.
        encoderCb.compute((pass) => pass.run("embedding_backward", {
          inputOffset: dHiddenGrad, outputOffset: dEmbedding,
          tokenCount: M, inputDim: V, outputDim: H, u0: 0,
        }));

        // 15-18. SGD updates (only explicitly trainable pages).
        if (!frozen.has("classifier")) {
          encoderCb.compute((pass) => pass.run("sgd_step", {
            inputOffset: dClassifier, tokenCount: V * H, f0: options.learningRate,
          }, this.classifierPage));
        }
        for (const [key, page, dW] of [["wq", this.wqPage, dWq], ["wk", this.wkPage, dWk], ["wv", this.wvPage, dWv]] as const) {
          if (!frozen.has(key as keyof TrainingParameters)) {
            encoderCb.compute((pass) => pass.run("sgd_step", {
              inputOffset: dW, tokenCount: H * H, f0: options.learningRate,
            }, page));
          }
        }
        if (!frozen.has("embedding")) {
          encoderCb.compute((pass) => pass.run("sgd_step", {
            inputOffset: dEmbedding, tokenCount: V * H, f0: options.learningRate,
          }, this.embeddingPage));
        }
      } else {
        // M1 toy path (no encoder): embedding -> classifier -> CE.
        // 2. classifier forward: logits = hidden @ classifier^T
        encoderCb.compute((pass) => pass.run("matmul_f32", {
          inputOffset: logitsInput, outputOffset: logits,
          tokenCount: M, inputDim: H, outputDim: V, rowStart: 0, rowCount: V,
        }, this.classifierPage));

        // 3. cross-entropy forward + dLogits (fused)
        encoderCb.compute((pass) => pass.run("cross_entropy_forward_backward", {
          inputOffset: logits, outputOffset: dLogits, auxOffset: lossRows,
          tokenCount: M, outputDim: V, u1: 0,
        }));

        // 4. loss reduction (telemetry scalar; gradient never depends on it)
        encoderCb.compute((pass) => pass.run("loss_reduce", {
          inputOffset: lossRows, outputOffset: scalarLoss, tokenCount: M,
        }));

        // 5. classifier backward input: dHidden = dLogits @ classifier
        encoderCb.compute((pass) => pass.run("matmul_backward_input", {
          inputOffset: dLogits, outputOffset: dHidden,
          tokenCount: M, inputDim: V, outputDim: H,
        }, this.classifierPage));

        // 6. classifier backward weight: dClassifier = dLogits^T @ hidden
        encoderCb.compute((pass) => pass.run("matmul_backward_weight", {
          inputOffset: dLogits, auxOffset: hidden, outputOffset: dClassifier,
          tokenCount: M, inputDim: V, outputDim: H,
        }));

        // 7. embedding backward: scatter-add of dHidden onto token rows
        encoderCb.compute((pass) => pass.run("embedding_backward", {
          inputOffset: dHidden, outputOffset: dEmbedding,
          tokenCount: M, inputDim: V, outputDim: H, u0: 0,
        }));

        // 8-9. SGD updates (only explicitly trainable pages)
        if (!frozen.has("classifier")) {
          encoderCb.compute((pass) => pass.run("sgd_step", {
            inputOffset: dClassifier, tokenCount: V * H, f0: options.learningRate,
          }, this.classifierPage));
        }
        if (!frozen.has("embedding")) {
          encoderCb.compute((pass) => pass.run("sgd_step", {
            inputOffset: dEmbedding, tokenCount: V * H, f0: options.learningRate,
          }, this.embeddingPage));
        }
      }
    });

    if (!options.telemetry) return { step };

    // Compact scalar readback through the sandblaster-tested readback path.
    await device.queue.onSubmittedWorkDone();
    const loss = await this.readLoss();
    return { step, loss };
  }

  /**
   * Read the scalar mean loss produced by the last trainStep.
   * Debug/test-only; the production path never reads it back.
   */
  async readLoss(): Promise<number> {
    const value = await this.definition.resources.lossTelemetry.readback();
    if (typeof value === "number") return value;
    const array = value as unknown as Float32Array;
    return array[0]!;
  }

  /**
   * Copy one buffer into the small staging buffer and read it back.
   * Debug/test-only.
   */
  private async readbackInto(source: GPUBuffer, sourceOffset: number, elements: number): Promise<Float32Array> {
    validate(elements <= TRAINING_READBACK_ELEMENTS, `readback region ${elements} exceeds staging capacity`);
    const device = this.definition.engine.device;
    const staging = this.definition.resources.trainingReadback;
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, sourceOffset, staging.gpu, 0, elements * 4);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const raw = await staging.readback();
    // Sandblaster readback() returns a plain Array for f32 buffers; normalize
    // to a Float32Array so callers can rely on typed-array semantics (toEqual
    // against an initial Float32Array, `.byteLength`, etc.).
    const values = raw as unknown as ArrayLike<number>;
    return Float32Array.from(values).slice(0, elements);
  }

  /** Read back the current classifier page. Debug/test-only. */
  async readClassifier(): Promise<Float32Array> {
    const { vocabSize: V, hiddenSize: H } = this.config;
    return this.readbackInto(this.classifierPage, 0, V * H);
  }

  /** Read back the current embedding page. Debug/test-only. */
  async readEmbedding(): Promise<Float32Array> {
    const { vocabSize: V, hiddenSize: H } = this.config;
    return this.readbackInto(this.embeddingPage, 0, V * H);
  }

  /** Read back the current Wq/Wk/Wv page. Debug/test-only. */
  async readProjection(name: "wq" | "wk" | "wv"): Promise<Float32Array> {
    const { hiddenSize: H } = this.config;
    const page = name === "wq" ? this.wqPage : name === "wk" ? this.wkPage : this.wvPage;
    validate(page !== undefined, `encoder not configured; no ${name} page`);
    return this.readbackInto(page!, 0, H * H);
  }

  /** Read back the logits [M,V] produced by the last trainStep. Debug/test-only. */
  async readLogits(m: number, v: number): Promise<Float32Array> {
    const logits = this.region(LFM2_TRAINING_ARENA.logits, m * v);
    validate(m * v <= TRAINING_READBACK_ELEMENTS, `logits readback ${m * v} exceeds staging capacity`);
    return this.readbackInto(this.definition.resources.arena.gpu, logits * 4, m * v);
  }

  get currentStep(): number {
    return this.step;
  }

  /** Dispose the parameter pages owned by this trainer. */
  destroy(): void {
    this.embeddingPage.destroy();
    this.classifierPage.destroy();
    this.wqPage?.destroy();
    this.wkPage?.destroy();
    this.wvPage?.destroy();
  }
}
