/**
 * Encoding a frame on the device — and differentiating one — behind the seam
 * `@krystal/krystal/host` declares.
 *
 * A host builds a `BrainSession` exactly as before and hands it one of these:
 *
 *   new BrainSession({ tokenRows, seed, backend: gpuBackend(device) })
 *
 * Everything else about the session is unchanged — `choose`, the update rules
 * of `learn` and `teach`, the checkpoints — all still run here, on the host's
 * own arrays, with the host's own closures. What moves is the arithmetic: the
 * encode (field embedding, record encoder, pooling, mixer) as one submit with
 * one readback of the three matrices every question reads; and the backward,
 * as one submit with one readback of the gradients the host's update consumes.
 *
 * The split is where the cost is. Measured in the simulation, `consider` took
 * 20 s over sixty ticks and all the `choose` calls together took 0.17 s; with
 * learning on, the CPU backward was 68% of a tick. A device round trip per
 * QUESTION would have spent the win on latency, while a round trip per FRAME
 * is one per two questions — and one per remembered turn when it learns.
 */
import type {
  BackwardRequest,
  BackwardResult,
  BrainBackend,
  BrainBackendFactory,
  EncodedFrame,
  WeightChanges,
} from "../../krystal/src/host/backend";
import type { HostFrame } from "../../krystal/src/host/frame";
import type { BrainForwardConfig, BrainForwardWeights } from "../../krystal/src/forward/model";
import { KrystalBackward } from "./krystal-backward";
import { KrystalForward } from "./krystal-forward";
import { krystal, type KrystalDefinition } from "./krystal";

export interface GpuBackendOptions {
  /**
   * The device to run on. A host that already has one — a browser that
   * acquired it for rendering — should pass it, so the creature and the picture
   * of it share a device instead of competing for two.
   */
  readonly device: GPUDevice;
  /** The compiled engine to run on; defaults to the module-level definition. */
  readonly definition?: KrystalDefinition;
}

/**
 * Build the pipelines once per engine.
 *
 * Two guards, because there are two ways to arrive already compiled: a second
 * session on the same definition (the map), and a host that compiled the engine
 * itself before constructing anything (the state). The engine refuses a second
 * compile by throwing, and a throw here would surface as a failed first frame
 * rather than as "someone already did this".
 */
const compiled = new WeakMap<object, Promise<void>>();

function ensureCompiled(definition: KrystalDefinition, device: GPUDevice): Promise<void> {
  let ready = compiled.get(definition);
  if (!ready) {
    ready = (async () => {
      if (definition.engine.state === "ready") return;
      const result = await definition.engine.compile({ device });
      if (result.failed) throw new Error(`gpuBackend: ${result.failed} shader(s) failed to compile`);
    })();
    compiled.set(definition, ready);
  }
  return ready;
}

/**
 * One thing in the arena at a time.
 *
 * Every session on an engine shares its arena: the frame, the activations, the
 * gradients all live at fixed offsets in one buffer. An encode is one submit
 * and reads back in that same submit, so two of them cannot see each other's
 * frames. A backward is a submit and then a readback, and anything submitted
 * between the two would overwrite the gradients before they were read — so
 * the device work on one engine is serialised here, across every backend
 * built on it, not merely within one.
 */
const turns = new WeakMap<object, Promise<unknown>>();

function inTurn<T>(engine: object, work: () => Promise<T>): Promise<T> {
  const previous = turns.get(engine) ?? Promise.resolve();
  const task = previous.then(work, work);
  turns.set(engine, task.catch(() => undefined));
  return task;
}

class GpuBrainBackend implements BrainBackend {
  private readonly definition: KrystalDefinition;
  private readonly ready: Promise<void>;
  private runner: KrystalForward | undefined;
  private trainer: KrystalBackward | undefined;
  /** Set while the runner does not exist yet, applied when it is created. */
  private pending: BrainForwardWeights;

  constructor(
    private readonly options: GpuBackendOptions,
    private readonly config: BrainForwardConfig,
    weights: BrainForwardWeights,
  ) {
    this.definition = options.definition ?? krystal;
    this.pending = weights;
    // Pipelines are built once per engine, and the first frame waits for them
    // rather than the constructor: a session is created synchronously, and a
    // constructor that could not finish its work would have to lie about it.
    this.ready = ensureCompiled(this.definition, options.device);
  }

  private async runnerFor(): Promise<KrystalForward> {
    await this.ready;
    if (!this.runner) this.runner = new KrystalForward(this.pending, this.config, this.definition);
    return this.runner;
  }

  encode(frame: HostFrame): Promise<EncodedFrame> {
    return inTurn(this.definition, async () => {
      const runner = await this.runnerFor();
      // Unconstrained mixer, exactly as the CPU path runs it: what a question may
      // attend to while it thinks is not what it may choose.
      const prepared = runner.prepare(frame.gpu, {});
      return runner.encodeAndRead(prepared);
    });
  }

  /**
   * The composed backward, with its optimizer off.
   *
   * `optimizer: "none"` is the mode written for exactly this caller: the
   * gradients are computed and left where they are, and the host's own rules —
   * an advantage standardised across the batch, only some parts unfrozen, the
   * whole thing one transaction that may be rolled back — apply them to its
   * arrays. The `learningRate` is required by the step and used by nothing.
   */
  backward(frame: HostFrame, request: BackwardRequest): Promise<BackwardResult> {
    return inTurn(this.definition, async () => {
      const runner = await this.runnerFor();
      this.trainer ??= new KrystalBackward(runner);
      const { active } = frame;
      await this.trainer.trainStep({
        frame: frame.gpu,
        masks: { selection: request.selection, context: "available" },
        ...(request.targets ? { selectionTargets: request.targets } : {}),
        ...(request.valenceTarget === undefined ? {} : { valenceTarget: request.valenceTarget }),
        learningRate: 1,
        optimizer: "none",
      });
      return this.trainer.readHostGradients({
        t: active.activeTokens.length,
        r: active.bankRecords.length,
        q: active.queryRecords.length,
      });
    });
  }

  sync(weights: BrainForwardWeights, changes?: WeightChanges): void {
    this.pending = weights;
    this.runner?.uploadWeights(weights, changes);
  }

  destroy(): void {
    this.runner?.destroy();
    this.runner = undefined;
    this.trainer = undefined;
  }
}

/**
 * A backend factory for `BrainSession`: give it a device, hand the result to
 * the session, and the encode — and the backward — run there instead of here.
 */
export function gpuBackend(
  deviceOrOptions: GPUDevice | GpuBackendOptions,
): BrainBackendFactory {
  const options: GpuBackendOptions = "device" in deviceOrOptions
    ? deviceOrOptions
    : { device: deviceOrOptions as GPUDevice };
  return (config, weights) => new GpuBrainBackend(options, config, weights);
}
