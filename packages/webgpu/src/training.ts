// M1 tiny f32 training vertical slice: host orchestration for the GPU-resident
// trainStep path. See docs/WEBGPU_BACKWARD_PLAN.md §10–13.
//
// Design:
//   - One submit per trainStep; the whole step (forward, backward, optimizer)
//     stays on the GPU. The only readback in the normal path is the compact
//     scalar loss telemetry (4 bytes) when `telemetry: true`.
//   - Parameters are f32 pages (embedding [V,H], classifier [V,H]) owned by the
//     trainer, written once from host Float32Arrays and updated in place by
//     sgd_step. Frozen pages are simply never dispatched through sgd_step.
//   - Activations/gradients live in the shared f32 arena, in the training
//     regions appended after the LFM2 layout (TRAINING_ARENA_BASE).
//   - Dispatch order mirrors WEBGPU_BACKWARD_PLAN.md §11.
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
  TRAINING_MAX_M,
  TRAINING_MAX_V,
  TRAINING_READBACK_ELEMENTS,
} from "./lfm2-layout";
import { lfm2, type Lfm2Definition } from "./lfm2";
import { Lfm2Executor } from "./pass";

export interface TrainingConfig {
  /** Vocabulary size V (embedding rows and logits width). */
  readonly vocabSize: number;
  /** Hidden size H. */
  readonly hiddenSize: number;
  /** Initial embedding table [V,H] f32, row-major. */
  readonly embedding: Float32Array;
  /** Initial classifier [V,H] f32, row-major (same layout as matmul_f32 W). */
  readonly classifier: Float32Array;
  /** Frozen pages stay byte-identical: they are never sent through sgd_step. */
  readonly frozen?: readonly (keyof TrainingParameters)[];
}

export interface TrainingParameters {
  embedding: Float32Array;
  classifier: Float32Array;
}

export interface TrainStepOptions {
  readonly tokens: readonly number[] | Uint32Array;
  readonly targets: readonly number[] | Uint32Array;
  readonly learningRate: number;
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
  validate(config.vocabSize > 0, "vocabSize must be > 0");
  validate(config.hiddenSize > 0, "hiddenSize must be > 0");
  validate(config.vocabSize <= TRAINING_MAX_V, `vocabSize ${config.vocabSize} exceeds capacity ${TRAINING_MAX_V}`);
  validate(config.hiddenSize <= TRAINING_MAX_H, `hiddenSize ${config.hiddenSize} exceeds capacity ${TRAINING_MAX_H}`);
  validate(
    config.embedding.length === config.vocabSize * config.hiddenSize,
    `embedding length ${config.embedding.length} != V*H ${config.vocabSize * config.hiddenSize}`,
  );
  validate(
    config.classifier.length === config.vocabSize * config.hiddenSize,
    `classifier length ${config.classifier.length} != V*H ${config.vocabSize * config.hiddenSize}`,
  );
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
  private readonly executor: Lfm2Executor;
  private step = 0;

  constructor(config: TrainingConfig, definition: Lfm2Definition = lfm2) {
    assertDims(config);
    this.config = config;
    this.definition = definition;
    this.executor = new Lfm2Executor(definition);

    const paramBytes = config.vocabSize * config.hiddenSize * 4;
    const paramUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.embeddingPage = definition.engine.device.createBuffer({
      label: "krystal.train.embedding",
      size: paramBytes,
      usage: paramUsage,
    });
    this.classifierPage = definition.engine.device.createBuffer({
      label: "krystal.train.classifier",
      size: paramBytes,
      usage: paramUsage,
    });
    definition.engine.device.queue.writeBuffer(this.embeddingPage, 0, config.embedding);
    definition.engine.device.queue.writeBuffer(this.classifierPage, 0, config.classifier);
  }

  private region(offset: number, elements: number): number {
    validate(
      offset + elements <= LFM2_TRAINING_ARENA.elements,
      "training arena region overflows the declared capacity",
    );
    return TRAINING_ARENA_BASE + offset;
  }

  /** Run one GPU-resident training step on the toy graph. */
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

    // Arena regions (WEBGPU_BACKWARD_PLAN.md §13), appended after the LFM2 ones.
    const hidden = this.region(LFM2_TRAINING_ARENA.hidden, M * H);
    const logits = this.region(LFM2_TRAINING_ARENA.logits, M * V);
    const lossRows = this.region(LFM2_TRAINING_ARENA.lossRows, M);
    const scalarLoss = this.region(LFM2_TRAINING_ARENA.scalarLoss, 1);
    const dLogits = this.region(LFM2_TRAINING_ARENA.dLogits, M * V);
    const dHidden = this.region(LFM2_TRAINING_ARENA.dHidden, M * H);
    const dClassifier = this.region(LFM2_TRAINING_ARENA.dClassifier, V * H);
    const dEmbedding = this.region(LFM2_TRAINING_ARENA.dEmbedding, V * H);

    const tokenIds = options.tokens instanceof Uint32Array ? options.tokens : Uint32Array.from(options.tokens);
    const targetIds = options.targets instanceof Uint32Array ? options.targets : Uint32Array.from(options.targets);
    const device = this.definition.engine.device;
    device.queue.writeBuffer(this.definition.resources.tokens.gpu, 0, tokenIds);
    device.queue.writeBuffer(this.definition.resources.targets.gpu, 0, targetIds);

    const frozen = new Set(this.config.frozen ?? []);
    const step = ++this.step;

    this.executor.submit((encoder) => {
      // 1. embedding forward: hidden = E[tokens]
      encoder.compute((pass) => pass.run("embedding_f32", {
        tokenCount: M, inputDim: V, outputDim: H, outputOffset: hidden, u0: 0,
      }, this.embeddingPage));

      // 2. classifier forward: logits = hidden @ classifier^T
      encoder.compute((pass) => pass.run("matmul_f32", {
        inputOffset: hidden, outputOffset: logits,
        tokenCount: M, inputDim: H, outputDim: V, rowStart: 0, rowCount: V,
      }, this.classifierPage));

      // 3. cross-entropy forward + dLogits (fused)
      encoder.compute((pass) => pass.run("cross_entropy_forward_backward", {
        inputOffset: logits, outputOffset: dLogits, auxOffset: lossRows,
        tokenCount: M, outputDim: V, u1: 0,
      }));

      // 4. loss reduction (telemetry scalar; gradient never depends on it)
      encoder.compute((pass) => pass.run("loss_reduce", {
        inputOffset: lossRows, outputOffset: scalarLoss, tokenCount: M,
      }));

      // 5. classifier backward input: dHidden = dLogits @ classifier
      encoder.compute((pass) => pass.run("matmul_backward_input", {
        inputOffset: dLogits, outputOffset: dHidden,
        tokenCount: M, inputDim: V, outputDim: H,
      }, this.classifierPage));

      // 6. classifier backward weight: dClassifier = dLogits^T @ hidden
      encoder.compute((pass) => pass.run("matmul_backward_weight", {
        inputOffset: dLogits, auxOffset: hidden, outputOffset: dClassifier,
        tokenCount: M, inputDim: V, outputDim: H,
      }));

      // 7. embedding backward: scatter-add of dHidden onto token rows
      encoder.compute((pass) => pass.run("embedding_backward", {
        inputOffset: dHidden, outputOffset: dEmbedding,
        tokenCount: M, inputDim: V, outputDim: H, u0: 0,
      }));

      // 8-9. SGD updates (only explicitly trainable pages)
      if (!frozen.has("classifier")) {
        encoder.compute((pass) => pass.run("sgd_step", {
          inputOffset: dClassifier, tokenCount: V * H, f0: options.learningRate,
        }, this.classifierPage));
      }
      if (!frozen.has("embedding")) {
        encoder.compute((pass) => pass.run("sgd_step", {
          inputOffset: dEmbedding, tokenCount: V * H, f0: options.learningRate,
        }, this.embeddingPage));
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
    const array = raw as unknown as Float32Array;
    return array.slice(0, elements);
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
  }
}
