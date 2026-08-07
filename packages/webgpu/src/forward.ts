import {
  createGpuConstraintDecoderState,
  linkGpuConstraintTokenizer,
  type GpuConstraintTokenizer,
} from "../../engine-ts/src/gpu-constraint.ts";
import { gpuConstraintProgramFromBlob } from "../../engine-ts/src/structured.ts";
import { uploadGpuConstraint } from "./constraint.ts";
import { Lfm2Tokenizer } from "../../lfm2/src/tokenizer.ts";
import {
  LFM2_ARENA,
  lfm2,
  type Lfm2Mode,
  type Lfm2WorkLayout,
} from "./lfm2";
import {
  Lfm2ComputePass,
  Lfm2Executor,
  type Lfm2CommandEncoder,
} from "./pass";
import { GPU_SCHEMA_SENTINELS } from "../../schema/src/sparse";
import {
  Lfm2GpuModel,
  lfm2Block0TensorNames,
  lfm2BlockTensorNames,
  type Lfm2GpuTensor,
  type Lfm2GpuTensorPage,
} from "./model";

function product(values: readonly number[]): number {
  let result = 1;
  for (const value of values) result *= value;
  return result;
}

function pageBuffer(page: Lfm2GpuTensorPage): { buffer: GPUBuffer } {
  return page;
}

export const LFM2_GREEDY_SHADER_PATH = [
  "embedding_wq4",
  "rms_norm",
  "matmul_wq4",
  "residual_add",
  "silu_mul",
  "shortconv_prefill",
  "shortconv_decode",
  "qk_norm_rope",
  "kv_store",
  "attention",
  "argmax",
] as const;

export interface Lfm2BlockRunOptions {
  readonly mode?: Lfm2Mode;
  readonly work?: Lfm2WorkLayout;
  /** Absolute token position used by continuation mode. */
  readonly positionBase?: number;
}

export interface Lfm2CheckpointState {
  /** Number of context tokens materialized in this state. */
  readonly position: number;
  /** Physical bytes owned by this checkpoint snapshot. */
  readonly byteLength: number;
  /** Compact snapshot KV bytes for positions [0, position). */
  readonly kvBytes: number;
  /** Live KV capacity represented by the snapshot. */
  readonly kvCapacityBytes: number;
  readonly convBytes: number;
  readonly hiddenBytes: number;
  /** Whole CreateCheckpoint wall time (materialize + capture + GPU completion). */
  readonly createUs: number;
  /** Bytes copied from a base checkpoint while creating this checkpoint. */
  readonly creationRestoredBytes: number;
  readonly kv: GPUBuffer;
  readonly conv: GPUBuffer;
  readonly lastHidden: GPUBuffer;
  destroy(): void;
}

export interface Lfm2ExecutionFacts {
  /** Tokens that actually traversed the full prefill/continuation layer stack in this operation. */
  readonly prefillTokens: number;
  /** Bytes physically copied from a reusable checkpoint into live recurrent state. */
  readonly restoredCheckpointBytes: number;
  /** Restore-only time when the backend can measure it without perturbing execution; 0 otherwise. */
  readonly checkpointRestoreUs: number;
}

export interface Lfm2GenerationResult {
  readonly tokens: number[];
  readonly generatedCount: number;
  readonly status: string;
  readonly lastToken: number;
  readonly execution: Lfm2ExecutionFacts;
}

class Lfm2CheckpointStateImpl implements Lfm2CheckpointState {
  private destroyed = false;
  private _createUs = 0;
  private _creationRestoredBytes = 0;

  constructor(
    readonly position: number,
    readonly kv: GPUBuffer,
    readonly conv: GPUBuffer,
    readonly lastHidden: GPUBuffer,
    readonly kvCapacityBytes: number,
  ) {}

  get kvBytes(): number { return Number(this.kv.size); }
  get convBytes(): number { return Number(this.conv.size); }
  get hiddenBytes(): number { return Number(this.lastHidden.size); }
  get byteLength(): number { return this.kvBytes + this.convBytes + this.hiddenBytes; }
  get createUs(): number { return this._createUs; }
  get creationRestoredBytes(): number { return this._creationRestoredBytes; }

  markCreated(createUs: number, creationRestoredBytes: number): void {
    this._createUs = Math.max(0, Math.round(createUs));
    this._creationRestoredBytes = Math.max(0, Math.round(creationRestoredBytes));
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.kv.destroy();
    this.conv.destroy();
    this.lastHidden.destroy();
  }
}

/**
 * Target scheduler for the migrated Sandblaster runtime. The implementation is
 * intentionally expressed in semantic tensor names and Lfm2Executor passes;
 * it owns no bind groups, pipeline layouts or handwritten OpParams ABI.
 */
export class Lfm2Forward {
  private structuredTokenizer?: Lfm2Tokenizer;
  private structuredConstraintTokenizer?: GpuConstraintTokenizer;
  readonly executor: Lfm2Executor;

  constructor(
    readonly model: Lfm2GpuModel,
    executor = new Lfm2Executor(lfm2),
  ) {
    if (model.device !== lfm2.engine.device) {
      throw new Error("Lfm2GpuModel and Sandblaster LFM2 definition must use the same GPUDevice");
    }
    this.executor = executor;
    this.writeRuntime(0);
  }

  private writeRuntime(
    promptTokenCount: number,
    maxNewTokens: number = lfm2.capacities.maxNewTokens,
  ): void {
    if (!Number.isInteger(maxNewTokens) || maxNewTokens < 1 || maxNewTokens > lfm2.capacities.maxNewTokens) {
      throw new RangeError(`maxNewTokens must be 1..${lfm2.capacities.maxNewTokens}, got ${maxNewTokens}`);
    }
    lfm2.resources.runtime.write({
      contextCapacity: lfm2.capacities.context,
      maxNewTokens,
      eosToken: this.model.config.eosToken,
      promptTokenCount,
      position: promptTokenCount,
      generatedCount: 0,
      currentToken: 0,
      status: "running",
      telemetryRevision: 0,
      lastToken: 0,
      errorCode: 0,
      pad0: 0,
    });
  }

  /** Initialize request bookkeeping for lower-level orchestration/tests. */
  initializeRequest(promptTokenCount: number, maxNewTokens: number = lfm2.capacities.maxNewTokens): void {
    if (!Number.isInteger(promptTokenCount) || promptTokenCount < 1 || promptTokenCount > lfm2.capacities.context) {
      throw new RangeError(`promptTokenCount must be 1..${lfm2.capacities.context}, got ${promptTokenCount}`);
    }
    this.writeRuntime(promptTokenCount, maxNewTokens);
  }

  async prepareBlock(layer: number): Promise<void> {
    const kind = this.model.config.layers[layer];
    if (!kind) throw new RangeError(`LFM2 layer ${layer} does not exist`);
    await this.model.preload(lfm2BlockTensorNames(layer, kind));
  }

  async prepareBlock0(): Promise<void> {
    await this.model.preload(lfm2Block0TensorNames());
  }

  /** Load embedding + every block in [0, endLayerExclusive). */
  async preparePrefix(endLayerExclusive: number): Promise<void> {
    if (!Number.isInteger(endLayerExclusive) || endLayerExclusive < 0 || endLayerExclusive > this.model.config.blockCount) {
      throw new RangeError(`Invalid LFM2 prefix end ${endLayerExclusive}`);
    }
    const names = new Set<string>(["token_embd.weight"]);
    for (let layer = 0; layer < endLayerExclusive; layer++) {
      for (const name of lfm2BlockTensorNames(layer, this.model.config.layers[layer]!)) names.add(name);
    }
    await this.model.preload(names);
  }

  /** Load every tensor required by the normal full forward / greedy decode path. */
  async prepareAll(): Promise<void> {
    const names = new Set<string>(["token_embd.weight", "token_embd_norm.weight"]);
    for (let layer = 0; layer < this.model.config.blockCount; layer++) {
      for (const name of lfm2BlockTensorNames(layer, this.model.config.layers[layer]!)) names.add(name);
    }
    await this.model.preload(names);
  }

  writeTokens(tokens: Uint32Array | readonly number[], tokenOffset = 0): void {
    const values = tokens instanceof Uint32Array ? tokens : Uint32Array.from(tokens);
    if (tokenOffset < 0 || tokenOffset + values.length > lfm2.capacities.tokens) {
      throw new RangeError(
        `Token upload [${tokenOffset}, ${tokenOffset + values.length}) exceeds capacity ${lfm2.capacities.tokens}`,
      );
    }
    lfm2.engine.device.queue.writeBuffer(lfm2.resources.tokens.gpu, tokenOffset * 4, values);
  }

  /**
   * Upload a sparse set of token IDs accepted by the guide. Slots may contain
   * EMPTY_TOKEN (0xffff); argmax_candidates ignores them.
   */
  writeCandidateTokens(tokens: Uint32Array | readonly number[]): void {
    const values = tokens instanceof Uint32Array ? tokens : Uint32Array.from(tokens);
    if (values.length > this.model.config.vocabSize) {
      throw new RangeError(
        `Candidate token count ${values.length} exceeds vocabulary ${this.model.config.vocabSize}`,
      );
    }
    for (const token of values) {
      if (!Number.isInteger(token) || token < 0 || token > 0xffff) {
        throw new RangeError(`Candidate token ID must fit u16, got ${token}`);
      }
    }
    lfm2.engine.device.queue.writeBuffer(lfm2.resources.candidateTokens.gpu, 0, values);
  }

  /** Clear recurrent state without destroying/recreating long-lived buffers. */
  clearState(encoder: Lfm2CommandEncoder): void {
    encoder.gpu.clearBuffer(lfm2.resources.convCache.gpu);
    encoder.gpu.clearBuffer(lfm2.resources.kvCache.gpu);
  }

  /** Number of physical attention-cache slots in the specialized runtime. */
  private get attentionSlotCount(): number {
    let count = 0;
    for (const slot of this.model.config.attentionLayerSlots) {
      if (slot >= count) count = slot + 1;
    }
    return count;
  }

  /**
   * Layout of one K (or V) segment in the live KV cache. The shader stores
   * every attention slot as [K: contextCapacity * KV_DIM][V: ...].
   * Checkpoints pack only [0, position) from each segment, back-to-back.
   */
  private kvCheckpointLayout(position: number): {
    readonly slots: number;
    readonly capacitySegmentBytes: number;
    readonly usedSegmentBytes: number;
    readonly compactBytes: number;
    readonly capacityBytes: number;
  } {
    const slots = this.attentionSlotCount;
    const capacityBytes = lfm2.resources.kvCache.compiledInfo.byteSize;
    if (slots < 1) {
      if (capacityBytes !== 0) {
        throw new Error(`KV cache has ${capacityBytes} bytes but model has no attention slots`);
      }
      return { slots: 0, capacitySegmentBytes: 0, usedSegmentBytes: 0, compactBytes: 0, capacityBytes };
    }

    const segmentCount = slots * 2; // K + V per attention slot
    if (capacityBytes % segmentCount !== 0) {
      throw new Error(`KV cache byte size ${capacityBytes} is not divisible by ${segmentCount} K/V segments`);
    }
    const capacitySegmentBytes = capacityBytes / segmentCount;
    if (capacitySegmentBytes % lfm2.capacities.context !== 0) {
      throw new Error(
        `KV segment ${capacitySegmentBytes} is not divisible by context capacity ${lfm2.capacities.context}`,
      );
    }
    const bytesPerPosition = capacitySegmentBytes / lfm2.capacities.context;
    const usedSegmentBytes = position * bytesPerPosition;
    return {
      slots,
      capacitySegmentBytes,
      usedSegmentBytes,
      compactBytes: segmentCount * usedSegmentBytes,
      capacityBytes,
    };
  }

  private allocateCheckpoint(position: number): Lfm2CheckpointStateImpl {
    const device = lfm2.engine.device;
    const copyUsage = GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const kvLayout = this.kvCheckpointLayout(position);
    const kv = device.createBuffer({
      label: `lfm2.checkpoint.${position}.kv`,
      size: kvLayout.compactBytes,
      usage: copyUsage,
    });
    const conv = device.createBuffer({
      label: `lfm2.checkpoint.${position}.conv`,
      size: lfm2.resources.convCache.compiledInfo.byteSize,
      usage: copyUsage,
    });
    const lastHidden = device.createBuffer({
      label: `lfm2.checkpoint.${position}.hidden`,
      size: this.model.config.hiddenSize * 4,
      usage: copyUsage,
    });
    return new Lfm2CheckpointStateImpl(
      position, kv, conv, lastHidden, kvLayout.capacityBytes,
    );
  }

  private copyCheckpointKvToLive(
    encoder: Lfm2CommandEncoder,
    state: Lfm2CheckpointState,
  ): void {
    const layout = this.kvCheckpointLayout(state.position);
    if (state.kvBytes !== layout.compactBytes) {
      throw new Error(
        `Checkpoint KV size ${state.kvBytes} does not match position ${state.position} (${layout.compactBytes})`,
      );
    }
    for (let slot = 0; slot < layout.slots; slot++) {
      const liveK = (slot * 2) * layout.capacitySegmentBytes;
      const liveV = liveK + layout.capacitySegmentBytes;
      const compactK = (slot * 2) * layout.usedSegmentBytes;
      const compactV = compactK + layout.usedSegmentBytes;
      encoder.gpu.copyBufferToBuffer(state.kv, compactK, lfm2.resources.kvCache.gpu, liveK, layout.usedSegmentBytes);
      encoder.gpu.copyBufferToBuffer(state.kv, compactV, lfm2.resources.kvCache.gpu, liveV, layout.usedSegmentBytes);
    }
  }

  private copyLiveKvToCheckpoint(
    encoder: Lfm2CommandEncoder,
    state: Lfm2CheckpointState,
  ): void {
    const layout = this.kvCheckpointLayout(state.position);
    if (state.kvBytes !== layout.compactBytes) {
      throw new Error(
        `Checkpoint KV size ${state.kvBytes} does not match position ${state.position} (${layout.compactBytes})`,
      );
    }
    for (let slot = 0; slot < layout.slots; slot++) {
      const liveK = (slot * 2) * layout.capacitySegmentBytes;
      const liveV = liveK + layout.capacitySegmentBytes;
      const compactK = (slot * 2) * layout.usedSegmentBytes;
      const compactV = compactK + layout.usedSegmentBytes;
      encoder.gpu.copyBufferToBuffer(lfm2.resources.kvCache.gpu, liveK, state.kv, compactK, layout.usedSegmentBytes);
      encoder.gpu.copyBufferToBuffer(lfm2.resources.kvCache.gpu, liveV, state.kv, compactV, layout.usedSegmentBytes);
    }
  }

  private restoreCheckpoint(encoder: Lfm2CommandEncoder, state: Lfm2CheckpointState): void {
    this.copyCheckpointKvToLive(encoder, state);
    encoder.gpu.copyBufferToBuffer(
      state.conv, 0, lfm2.resources.convCache.gpu, 0, lfm2.resources.convCache.compiledInfo.byteSize,
    );
  }

  private captureCheckpoint(
    encoder: Lfm2CommandEncoder,
    state: Lfm2CheckpointState,
    hiddenElementOffset: number,
  ): void {
    this.copyLiveKvToCheckpoint(encoder, state);
    encoder.gpu.copyBufferToBuffer(
      lfm2.resources.convCache.gpu, 0, state.conv, 0, lfm2.resources.convCache.compiledInfo.byteSize,
    );
    encoder.gpu.copyBufferToBuffer(
      lfm2.resources.arena.gpu,
      hiddenElementOffset * 4,
      state.lastHidden,
      0,
      this.model.config.hiddenSize * 4,
    );
  }

  private requireTensor(name: string): Lfm2GpuTensor {
    return this.model.tensor(name);
  }

  private norm(
    pass: Lfm2ComputePass,
    tensorName: string,
    inputOffset: number,
    outputOffset: number,
    tokenCount: number,
    dim: number,
  ): void {
    const tensor = this.requireTensor(tensorName);
    if (tensor.format !== "f32" || tensor.pages.length !== 1) {
      throw new Error(`${tensorName}: RMSNorm requires one raw F32 page, got ${tensor.format}/${tensor.pages.length}`);
    }
    pass.run("rms_norm", {
      inputOffset,
      outputOffset,
      tokenCount,
      inputDim: dim,
      f0: this.model.config.normEpsilon,
    }, pageBuffer(tensor.pages[0]!));
  }

  private matrix(
    pass: Lfm2ComputePass,
    tensorName: string,
    inputOffset: number,
    outputOffset: number,
    tokenCount: number,
    inputDim: number,
    outputDim: number,
  ): void {
    const tensor = this.requireTensor(tensorName);
    if ((tensor.dimensions[0] ?? 0) !== inputDim) {
      throw new Error(`${tensorName}: input dim ${tensor.dimensions[0]} != expected ${inputDim}`);
    }
    const rows = tensor.dimensions.length > 1 ? product(tensor.dimensions.slice(1)) : 1;
    if (rows !== outputDim) {
      throw new Error(`${tensorName}: output rows ${rows} != expected ${outputDim}`);
    }
    if (tensor.format === "wq4" && inputDim % 32 !== 0) {
      throw new Error(`${tensorName}: WQ4 input dimension ${inputDim} is not divisible by 32`);
    }

    const kernel = tensor.format === "wq4"
      ? "matmul_wq4"
      : tensor.format === "f16"
        ? "matmul_f16"
        : "matmul_f32";

    for (const page of tensor.pages) {
      pass.run(kernel, {
        inputOffset,
        outputOffset,
        tokenCount,
        inputDim,
        outputDim,
        rowStart: page.rowStart,
        rowCount: page.rowCount,
      }, pageBuffer(page));
    }
  }

  private residual(
    pass: Lfm2ComputePass,
    left: number,
    right: number,
    output: number,
    tokenCount: number,
    dim: number,
  ): void {
    pass.run("residual_add", {
      inputOffset: left,
      auxOffset: right,
      outputOffset: output,
      tokenCount,
      inputDim: dim,
    });
  }

  embed(
    pass: Lfm2ComputePass,
    tokenCount: number,
    mode: Lfm2Mode = "prefill",
    work: Lfm2WorkLayout = LFM2_ARENA,
    tokenOffset = 0,
  ): void {
    const tensor = this.requireTensor("token_embd.weight");
    if (tensor.format !== "wq4" && tensor.format !== "f16") {
      throw new Error(`token_embd.weight: unsupported embedding format ${tensor.format}`);
    }
    const kernel = tensor.format === "wq4" ? "embedding_wq4" : "embedding";
    for (const page of tensor.pages) {
      pass.run(kernel, {
        outputOffset: work.hiddenA,
        tokenCount,
        outputDim: this.model.config.hiddenSize,
        rowStart: page.rowStart,
        rowCount: page.rowCount,
        mode,
        u0: tokenOffset,
      }, pageBuffer(page));
    }
  }

  private convOperator(
    pass: Lfm2ComputePass,
    layer: number,
    tokenCount: number,
    mode: Lfm2Mode,
    work: Lfm2WorkLayout,
    positionBase: number,
  ): void {
    const h = this.model.config.hiddenSize;
    this.norm(pass, `blk.${layer}.attn_norm.weight`, work.hiddenA, work.tmpH, tokenCount, h);
    this.matrix(pass, `blk.${layer}.shortconv.in_proj.weight`, work.tmpH, work.tmpA, tokenCount, h, 3 * h);

    const conv = this.requireTensor(`blk.${layer}.shortconv.conv.weight`);
    if (conv.format !== "f32" || conv.pages.length !== 1) {
      throw new Error(`blk.${layer}.shortconv.conv.weight must be one raw F32 page`);
    }
    const kernel = mode === "decode"
      ? "shortconv_decode"
      : mode === "continuation"
        ? "shortconv_continue"
        : "shortconv_prefill";
    pass.run(kernel, {
      inputOffset: work.tmpA,
      outputOffset: work.tmpH,
      tokenCount,
      inputDim: h,
      layerIndex: layer,
      mode,
      u1: positionBase,
    }, pageBuffer(conv.pages[0]!));

    this.matrix(pass, `blk.${layer}.shortconv.out_proj.weight`, work.tmpH, work.tmpA, tokenCount, h, h);
  }

  private attentionOperator(
    pass: Lfm2ComputePass,
    layer: number,
    tokenCount: number,
    mode: Lfm2Mode,
    work: Lfm2WorkLayout,
    positionBase: number,
  ): void {
    const h = this.model.config.hiddenSize;
    const kvHeads = this.model.config.kvHeadsByLayer[layer]!;
    const shaderKvHeads = Math.max(...this.model.config.kvHeadsByLayer);
    if (kvHeads !== shaderKvHeads) {
      throw new Error(`Layer ${layer}: current attention kernels require ${shaderKvHeads} KV heads, got ${kvHeads}`);
    }
    const kvDim = kvHeads * this.model.config.headDim;
    const vOffset = work.tmpB + lfm2.capacities.context * kvDim;

    this.norm(pass, `blk.${layer}.attn_norm.weight`, work.hiddenA, work.tmpH, tokenCount, h);
    this.matrix(pass, `blk.${layer}.attn_q.weight`, work.tmpH, work.tmpA, tokenCount, h, h);
    this.matrix(pass, `blk.${layer}.attn_k.weight`, work.tmpH, work.tmpB, tokenCount, h, kvDim);
    this.matrix(pass, `blk.${layer}.attn_v.weight`, work.tmpH, vOffset, tokenCount, h, kvDim);

    const qNorm = this.requireTensor(`blk.${layer}.attn_q_norm.weight`);
    const kNorm = this.requireTensor(`blk.${layer}.attn_k_norm.weight`);
    if (qNorm.format !== "f32" || qNorm.pages.length !== 1 || kNorm.format !== "f32" || kNorm.pages.length !== 1) {
      throw new Error(`blk.${layer}: Q/K norm weights must each be one raw F32 page`);
    }

    const common = {
      inputOffset: work.tmpA,
      auxOffset: work.tmpB,
      tokenCount,
      mode,
      f0: this.model.config.normEpsilon,
      f1: this.model.config.ropeTheta,
      ...(mode === "continuation" ? { u1: positionBase } : {}),
    } as const;
    pass.run("qk_norm_rope", { ...common, u0: 0 }, pageBuffer(qNorm.pages[0]!));
    pass.run("qk_norm_rope", { ...common, u0: 1 }, pageBuffer(kNorm.pages[0]!));

    const attentionSlot = this.model.config.attentionLayerSlots[layer]!;
    if (attentionSlot < 0) throw new Error(`Layer ${layer} has no attention cache slot`);
    pass.run("kv_store", {
      inputOffset: work.tmpB,
      auxOffset: vOffset,
      tokenCount,
      attentionSlot,
      mode,
      ...(mode === "continuation" ? { u1: positionBase } : {}),
    });
    pass.run("attention", {
      inputOffset: work.tmpA,
      outputOffset: work.tmpH,
      tokenCount,
      attentionSlot,
      mode,
      ...(mode === "continuation" ? { u1: positionBase } : {}),
    });
    this.matrix(pass, `blk.${layer}.attn_output.weight`, work.tmpH, work.tmpA, tokenCount, h, h);
  }

  private ffn(
    pass: Lfm2ComputePass,
    layer: number,
    tokenCount: number,
    work: Lfm2WorkLayout,
  ): void {
    const h = this.model.config.hiddenSize;
    const ff = this.model.config.feedForwardSize;
    this.norm(pass, `blk.${layer}.ffn_norm.weight`, work.hiddenB, work.tmpH, tokenCount, h);
    this.matrix(pass, `blk.${layer}.ffn_gate.weight`, work.tmpH, work.tmpA, tokenCount, h, ff);
    this.matrix(pass, `blk.${layer}.ffn_up.weight`, work.tmpH, work.tmpB, tokenCount, h, ff);
    pass.run("silu_mul", {
      inputOffset: work.tmpA,
      auxOffset: work.tmpB,
      outputOffset: work.tmpA,
      tokenCount,
      inputDim: ff,
    });
    this.matrix(pass, `blk.${layer}.ffn_down.weight`, work.tmpA, work.tmpH, tokenCount, ff, h);
    this.residual(pass, work.hiddenB, work.tmpH, work.hiddenA, tokenCount, h);
  }

  /** Full transformer block: operator + residual + FFN + residual. */
  block(
    pass: Lfm2ComputePass,
    layer: number,
    tokenCount: number,
    options: Lfm2BlockRunOptions = {},
  ): void {
    const kind = this.model.config.layers[layer];
    if (!kind) throw new RangeError(`LFM2 layer ${layer} does not exist`);
    const mode = options.mode ?? "prefill";
    const work = options.work ?? LFM2_ARENA;
    const h = this.model.config.hiddenSize;
    const positionBase = options.positionBase ?? 0;

    if (kind === "conv") {
      this.convOperator(pass, layer, tokenCount, mode, work, positionBase);
    } else {
      this.attentionOperator(pass, layer, tokenCount, mode, work, positionBase);
    }
    this.residual(pass, work.hiddenA, work.tmpA, work.hiddenB, tokenCount, h);
    this.ffn(pass, layer, tokenCount, work);
  }

  /** Execute a contiguous layer range over the current hiddenA frontier. */
  layers(
    pass: Lfm2ComputePass,
    startLayer: number,
    endLayerExclusive: number,
    tokenCount: number,
    options: Lfm2BlockRunOptions = {},
  ): void {
    if (startLayer < 0 || endLayerExclusive < startLayer || endLayerExclusive > this.model.config.blockCount) {
      throw new RangeError(`Invalid LFM2 layer range [${startLayer}, ${endLayerExclusive})`);
    }
    for (let layer = startLayer; layer < endLayerExclusive; layer++) {
      this.block(pass, layer, tokenCount, options);
    }
  }

  /** Final RMSNorm + tied token embedding projection into the logits arena. */
  projectLogits(
    pass: Lfm2ComputePass,
    tokenCount: number,
    mode: Lfm2Mode = "prefill",
    work: Lfm2WorkLayout = LFM2_ARENA,
  ): void {
    if (tokenCount < 1) throw new Error("projectLogits requires at least one token");
    const h = this.model.config.hiddenSize;
    const finalInput = mode === "decode"
      ? work.hiddenA
      : work.hiddenA + (tokenCount - 1) * h;
    this.norm(pass, "token_embd_norm.weight", finalInput, work.tmpH, 1, h);
    this.matrix(
      pass,
      "token_embd.weight",
      work.tmpH,
      LFM2_ARENA.logits,
      1,
      h,
      this.model.config.vocabSize,
    );
  }

  commitArgmax(pass: Lfm2ComputePass, mode: Lfm2Mode = "prefill"): void {
    pass.run("argmax", {
      inputOffset: LFM2_ARENA.logits,
      inputDim: this.model.config.vocabSize,
      // u0 is pass-local here: token 0xffff is a GPU sentinel and must never
      // escape through normal sampling.
      u0: GPU_SCHEMA_SENTINELS.emptyToken,
      mode,
    });
  }

  /** Exact GPU mask -> masked argmax -> transactional VM commit. */
  commitConstraintArgmax(pass: Lfm2ComputePass, mode: Lfm2Mode = "prefill"): void {
    pass.runStatic("constraint_mask", [lfm2.constraint.maskWorkgroups, 1, 1]);
    pass.run("constraint_argmax", {
      inputOffset: LFM2_ARENA.logits,
      inputDim: this.model.config.vocabSize,
      u0: GPU_SCHEMA_SENTINELS.emptyToken,
      mode,
    });
  }

  /**
   * Sample only from a sparse guide candidate table. This is the first WebGPU
   * guide primitive: the state machine may fill a fixed-size table with token
   * IDs and pad unused entries with EMPTY_TOKEN instead of materializing a
   * 65k-bit mask.
   */
  commitArgmaxCandidates(
    pass: Lfm2ComputePass,
    candidateCount: number,
    mode: Lfm2Mode = "prefill",
  ): void {
    if (!Number.isInteger(candidateCount) || candidateCount < 1 || candidateCount > this.model.config.vocabSize) {
      throw new RangeError(
        `candidateCount must be 1..${this.model.config.vocabSize}, got ${candidateCount}`,
      );
    }
    pass.run("argmax_candidates", {
      inputOffset: LFM2_ARENA.logits,
      inputDim: candidateCount,
      u0: GPU_SCHEMA_SENTINELS.emptyToken,
      mode,
    });
  }

  /** Embedding + every transformer block + final norm/logits. */
  forwardToLogits(
    pass: Lfm2ComputePass,
    tokenCount: number,
    mode: Lfm2Mode = "prefill",
    work: Lfm2WorkLayout = LFM2_ARENA,
    tokenOffset = 0,
  ): void {
    this.embed(pass, tokenCount, mode, work, tokenOffset);
    this.layers(pass, 0, this.model.config.blockCount, tokenCount, { mode, work });
    this.projectLogits(pass, tokenCount, mode, work);
  }

  forwardAndSample(
    pass: Lfm2ComputePass,
    tokenCount: number,
    mode: Lfm2Mode = "prefill",
    work: Lfm2WorkLayout = LFM2_ARENA,
    tokenOffset = 0,
  ): void {
    this.forwardToLogits(pass, tokenCount, mode, work, tokenOffset);
    this.commitArgmax(pass, mode);
  }

  /** Full forward followed by exact structured sampling entirely on GPU. */
  forwardAndSampleConstrained(
    pass: Lfm2ComputePass,
    tokenCount: number,
    mode: Lfm2Mode = "prefill",
    work: Lfm2WorkLayout = LFM2_ARENA,
    tokenOffset = 0,
  ): void {
    this.forwardToLogits(pass, tokenCount, mode, work, tokenOffset);
    this.commitConstraintArgmax(pass, mode);
  }

  /** Compatibility wrapper retained while block-0 callers migrate. */
  convBlock(
    pass: Lfm2ComputePass,
    layer: number,
    tokenCount: number,
    options: Lfm2BlockRunOptions = {},
  ): void {
    if (this.model.config.layers[layer] !== "conv") {
      throw new Error(`Layer ${layer} is ${this.model.config.layers[layer] ?? "missing"}, not conv`);
    }
    this.block(pass, layer, tokenCount, options);
  }

  /**
   * First production migration slice: token upload + embedding + complete block
   * 0 in one Sandblaster command buffer / queue submit.
   */
  prefillBlock0(
    tokens: Uint32Array | readonly number[],
    options: { readonly resetState?: boolean; readonly tokenOffset?: number } = {},
  ): void {
    const values = tokens instanceof Uint32Array ? tokens : Uint32Array.from(tokens);
    if (values.length === 0) throw new Error("prefillBlock0 requires at least one token");
    if (values.length > lfm2.capacities.context) {
      throw new Error(`prefillBlock0 token count ${values.length} exceeds context ${lfm2.capacities.context}`);
    }
    this.writeRuntime(values.length);
    this.writeTokens(values, options.tokenOffset ?? 0);
    this.executor.submit((encoder) => {
      if (options.resetState ?? true) this.clearState(encoder);
      encoder.compute((pass) => {
        this.embed(pass, values.length, "prefill", LFM2_ARENA, options.tokenOffset ?? 0);
        this.block(pass, 0, values.length, { mode: "prefill", work: LFM2_ARENA });
      }, { label: "lfm2.block0.prefill" });
    });
  }


  /**
   * Execute embedding plus layers [0, endLayerExclusive) in one command buffer.
   * This is the production prefill primitive used by the migrated scheduler.
   */
  prefillPrefix(
    tokens: Uint32Array | readonly number[],
    endLayerExclusive: number,
    options: { readonly resetState?: boolean; readonly tokenOffset?: number } = {},
  ): void {
    const values = tokens instanceof Uint32Array ? tokens : Uint32Array.from(tokens);
    if (values.length === 0) throw new Error("prefillPrefix requires at least one token");
    if (values.length > lfm2.capacities.context) {
      throw new Error(`prefillPrefix token count ${values.length} exceeds context ${lfm2.capacities.context}`);
    }
    if (!Number.isInteger(endLayerExclusive) || endLayerExclusive < 0 || endLayerExclusive > this.model.config.blockCount) {
      throw new RangeError(`Invalid LFM2 prefix end ${endLayerExclusive}`);
    }
    const tokenOffset = options.tokenOffset ?? 0;
    this.writeRuntime(values.length);
    this.writeTokens(values, tokenOffset);
    this.executor.submit((encoder) => {
      if (options.resetState ?? true) this.clearState(encoder);
      encoder.compute((pass) => {
        this.embed(pass, values.length, "prefill", LFM2_ARENA, tokenOffset);
        this.layers(pass, 0, endLayerExclusive, values.length, { mode: "prefill", work: LFM2_ARENA });
      }, { label: `lfm2.prefix.0-${endLayerExclusive}.prefill` });
    });
  }

  /**
   * Materialize an exact reusable context state. With base set, only tailTokens
   * are computed; the previous KV/conv snapshot is restored first and the tail
   * runs in continuation mode at absolute position base.position.
   */
  async createCheckpoint(
    tailTokens: Uint32Array | readonly number[],
    base?: Lfm2CheckpointState,
  ): Promise<Lfm2CheckpointState> {
    const tail = tailTokens instanceof Uint32Array ? tailTokens : Uint32Array.from(tailTokens);
    const basePosition = base?.position ?? 0;
    const position = basePosition + tail.length;
    if (position < 1) throw new Error("Checkpoint context must contain at least one token");
    if (position > lfm2.capacities.context) {
      throw new Error(`Checkpoint position ${position} exceeds context ${lfm2.capacities.context}`);
    }

    await this.prepareAll();
    const createStarted = performance.now();
    const checkpoint = this.allocateCheckpoint(position);
    this.writeRuntime(position, 1);
    if (tail.length) this.writeTokens(tail, 0);

    try {
      const creationRestoredBytes = base
        ? base.kvBytes + base.convBytes + (tail.length === 0 ? base.hiddenBytes : 0)
        : 0;
      this.executor.submit((encoder) => {
        if (base) this.restoreCheckpoint(encoder, base);
        else this.clearState(encoder);

        if (tail.length) {
          encoder.compute((pass) => {
            // New context blocks are ordinary token IDs, while only layer state
            // resumes from the checkpoint. Embedding therefore stays in prefill
            // mode; recurrent/attention layers use continuation positions.
            this.embed(pass, tail.length, "prefill", LFM2_ARENA, 0);
            this.layers(pass, 0, this.model.config.blockCount, tail.length, {
              mode: base ? "continuation" : "prefill",
              work: LFM2_ARENA,
              positionBase: basePosition,
            });
          }, { label: base ? "lfm2.checkpoint.extend" : "lfm2.checkpoint.prefill" });

          this.captureCheckpoint(
            encoder,
            checkpoint,
            LFM2_ARENA.hiddenA + (tail.length - 1) * this.model.config.hiddenSize,
          );
        } else {
          // Creating a new handle for exactly the same context stays independent
          // and immutable: clone the source snapshot without touching live state.
          encoder.gpu.copyBufferToBuffer(base!.kv, 0, checkpoint.kv, 0, base!.kv.size);
          encoder.gpu.copyBufferToBuffer(base!.conv, 0, checkpoint.conv, 0, base!.conv.size);
          encoder.gpu.copyBufferToBuffer(
            base!.lastHidden, 0, checkpoint.lastHidden, 0, base!.lastHidden.size,
          );
        }
      });
      await lfm2.engine.device.queue.onSubmittedWorkDone();
      checkpoint.markCreated((performance.now() - createStarted) * 1000, creationRestoredBytes);
      return checkpoint;
    } catch (error) {
      checkpoint.destroy();
      throw error;
    }
  }

  /** Generate from an immutable physical checkpoint, computing only tailTokens fresh. */
  async generateGreedyFromCheckpoint(
    checkpoint: Lfm2CheckpointState,
    tailTokens: Uint32Array | readonly number[],
    options: { readonly maxNewTokens?: number } = {},
  ): Promise<Lfm2GenerationResult> {
    const tail = tailTokens instanceof Uint32Array ? tailTokens : Uint32Array.from(tailTokens);
    const maxNewTokens = options.maxNewTokens ?? lfm2.capacities.maxNewTokens;
    const promptTokenCount = checkpoint.position + tail.length;
    if (promptTokenCount + maxNewTokens - 1 > lfm2.capacities.context) {
      throw new Error(
        `Checkpoint + tail + decode positions (${promptTokenCount + maxNewTokens - 1}) exceed context ${lfm2.capacities.context}`,
      );
    }

    await this.prepareAll();
    this.writeRuntime(promptTokenCount, maxNewTokens);
    if (tail.length) this.writeTokens(tail, 0);

    const restoredCheckpointBytes =
      checkpoint.kvBytes + checkpoint.convBytes + (tail.length === 0 ? checkpoint.hiddenBytes : 0);

    this.executor.submit((encoder) => {
      this.restoreCheckpoint(encoder, checkpoint);
      if (!tail.length) {
        encoder.gpu.copyBufferToBuffer(
          checkpoint.lastHidden, 0, lfm2.resources.arena.gpu, LFM2_ARENA.hiddenA * 4,
          this.model.config.hiddenSize * 4,
        );
      }

      encoder.compute((pass) => {
        if (tail.length) {
          this.embed(pass, tail.length, "prefill", LFM2_ARENA, 0);
          this.layers(pass, 0, this.model.config.blockCount, tail.length, {
            mode: "continuation",
            work: LFM2_ARENA,
            positionBase: checkpoint.position,
          });
          // Sampling this last context token is logically the same as ordinary
          // prefill: do not advance runtime.position until the first decode step.
          this.projectLogits(pass, tail.length, "prefill", LFM2_ARENA);
        } else {
          this.projectLogits(pass, 1, "prefill", LFM2_ARENA);
        }
        this.commitArgmax(pass, "prefill");
        for (let step = 1; step < maxNewTokens; step++) {
          this.forwardAndSample(pass, 1, "decode", LFM2_ARENA, 0);
        }
      }, { label: "lfm2.generate.checkpoint" });
    });

    return await this.readGenerationResult({
      prefillTokens: tail.length,
      restoredCheckpointBytes,
      // Measuring restore alone would require an extra submit/wait and would
      // perturb the path we are trying to optimize. Keep 0 until GPU/native
      // timestamp instrumentation can provide this without synchronization.
      checkpointRestoreUs: 0,
    });
  }

  private getStructuredTokenizer(): Lfm2Tokenizer {
    if (!this.structuredTokenizer) this.structuredTokenizer = new Lfm2Tokenizer(this.model as any);
    return this.structuredTokenizer;
  }

  private getStructuredConstraintTokenizer(): GpuConstraintTokenizer {
    if (this.structuredConstraintTokenizer) return this.structuredConstraintTokenizer;
    const tokenizer = this.getStructuredTokenizer();
    const entries = Array.from({ length: this.model.config.vocabSize }, (_, id) => {
      const bytes = tokenizer.tokenBytes(id);
      return {
        id,
        bytes,
        // No-byte ordinary tokens would allow zero-progress decode steps and
        // invalidate the schema-derived maxTokens bound. Treat them like other
        // control/special vocabulary entries; EOS remains explicitly handled.
        special: tokenizer.isSpecialToken(id) || bytes === null || bytes.length === 0,
      };
    });
    this.structuredConstraintTokenizer = linkGpuConstraintTokenizer(
      entries,
      this.model.config.eosToken,
    );
    return this.structuredConstraintTokenizer;
  }

  private prepareStructuredConstraint(constraintBlob: Uint32Array): void {
    const program = gpuConstraintProgramFromBlob(constraintBlob);
    const state = createGpuConstraintDecoderState(program);
    uploadGpuConstraint(
      lfm2,
      program,
      this.getStructuredConstraintTokenizer(),
      state,
    );
  }

  async generateStructured(
    promptTokens: Uint32Array | readonly number[],
    constraintBlob: Uint32Array,
    options: { readonly maxNewTokens: number },
  ) {
    const prompt = promptTokens instanceof Uint32Array ? promptTokens : Uint32Array.from(promptTokens);
    const maxNewTokens = options.maxNewTokens;
    if (maxNewTokens < 1 || maxNewTokens > lfm2.capacities.maxNewTokens) {
      throw new Error(`Structured schema requires decode budget ${maxNewTokens}, runtime capacity is ${lfm2.capacities.maxNewTokens}`);
    }
    if (prompt.length < 1) throw new Error("generateStructured requires at least one context token");
    if (prompt.length > lfm2.capacities.context) {
      throw new Error(`Prompt has ${prompt.length} tokens, context capacity is ${lfm2.capacities.context}`);
    }
    if (prompt.length + maxNewTokens - 1 > lfm2.capacities.context) {
      throw new Error(`Prompt + structured decode positions (${prompt.length + maxNewTokens - 1}) exceed context ${lfm2.capacities.context}`);
    }

    await this.prepareAll();
    this.prepareStructuredConstraint(constraintBlob);
    this.writeRuntime(prompt.length, maxNewTokens);
    this.writeTokens(prompt, 0);

    this.executor.submit((encoder) => {
      this.clearState(encoder);
      encoder.compute((pass) => {
        this.forwardAndSampleConstrained(pass, prompt.length, "prefill", LFM2_ARENA, 0);
        for (let step = 1; step < maxNewTokens; step++) {
          this.forwardAndSampleConstrained(pass, 1, "decode", LFM2_ARENA, 0);
        }
      }, { label: "lfm2.generate.structured" });
    });

    const result = await this.readGenerationResult({
      prefillTokens: prompt.length,
      restoredCheckpointBytes: 0,
      checkpointRestoreUs: 0,
    });
    return {
      ...result,
      text: this.getStructuredTokenizer().decode(result.tokens, { skipSpecial: true }),
    };
  }

  async generateStructuredFromCheckpoint(
    checkpoint: Lfm2CheckpointState,
    tailTokens: Uint32Array | readonly number[],
    constraintBlob: Uint32Array,
    options: { readonly maxNewTokens: number },
  ) {
    const tail = tailTokens instanceof Uint32Array ? tailTokens : Uint32Array.from(tailTokens);
    const maxNewTokens = options.maxNewTokens;
    if (maxNewTokens < 1 || maxNewTokens > lfm2.capacities.maxNewTokens) {
      throw new Error(`Structured schema requires decode budget ${maxNewTokens}, runtime capacity is ${lfm2.capacities.maxNewTokens}`);
    }
    const promptTokenCount = checkpoint.position + tail.length;
    if (promptTokenCount + maxNewTokens - 1 > lfm2.capacities.context) {
      throw new Error(`Checkpoint + tail + structured decode positions (${promptTokenCount + maxNewTokens - 1}) exceed context ${lfm2.capacities.context}`);
    }

    await this.prepareAll();
    this.prepareStructuredConstraint(constraintBlob);
    this.writeRuntime(promptTokenCount, maxNewTokens);
    if (tail.length) this.writeTokens(tail, 0);

    const restoredCheckpointBytes =
      checkpoint.kvBytes + checkpoint.convBytes + (tail.length === 0 ? checkpoint.hiddenBytes : 0);

    this.executor.submit((encoder) => {
      this.restoreCheckpoint(encoder, checkpoint);
      if (!tail.length) {
        encoder.gpu.copyBufferToBuffer(
          checkpoint.lastHidden,
          0,
          lfm2.resources.arena.gpu,
          LFM2_ARENA.hiddenA * 4,
          this.model.config.hiddenSize * 4,
        );
      }

      encoder.compute((pass) => {
        if (tail.length) {
          this.embed(pass, tail.length, "prefill", LFM2_ARENA, 0);
          this.layers(pass, 0, this.model.config.blockCount, tail.length, {
            mode: "continuation",
            work: LFM2_ARENA,
            positionBase: checkpoint.position,
          });
          this.projectLogits(pass, tail.length, "prefill", LFM2_ARENA);
        } else {
          this.projectLogits(pass, 1, "prefill", LFM2_ARENA);
        }
        this.commitConstraintArgmax(pass, "prefill");
        for (let step = 1; step < maxNewTokens; step++) {
          this.forwardAndSampleConstrained(pass, 1, "decode", LFM2_ARENA, 0);
        }
      }, { label: "lfm2.generate.structured-checkpoint" });
    });

    const result = await this.readGenerationResult({
      prefillTokens: tail.length,
      restoredCheckpointBytes,
      checkpointRestoreUs: 0,
    });
    return {
      ...result,
      text: this.getStructuredTokenizer().decode(result.tokens, { skipSpecial: true }),
    };
  }

  private async readGenerationResult(execution: Lfm2ExecutionFacts): Promise<Lfm2GenerationResult> {
    await lfm2.engine.device.queue.onSubmittedWorkDone();
    const [runtimeReadback, tokenReadback] = await Promise.all([
      lfm2.resources.runtime.readback({ dropIfBusy: false }),
      lfm2.resources.tokens.readback({ dropIfBusy: false }),
    ]);
    const runtime = runtimeReadback as any;
    const allTokens = tokenReadback as any;
    if (!runtime || Array.isArray(runtime)) throw new Error("Invalid LFM2 runtime readback");
    if (!Array.isArray(allTokens) || (allTokens.length > 0 && Array.isArray(allTokens[0]))) {
      throw new Error("Invalid LFM2 token readback");
    }
    const generatedCount = Number(runtime.generatedCount ?? 0);
    const start = lfm2.capacities.context;
    return {
      tokens: allTokens.slice(start, start + generatedCount).map(Number),
      generatedCount,
      status: String(runtime.status),
      lastToken: Number(runtime.lastToken ?? 0),
      execution,
    };
  }

  /**
   * Full greedy inference in one GPU submission: prompt prefill followed by up
   * to maxNewTokens-1 decode iterations. Runtime.status gates token commits
   * after EOS/done exactly like the legacy monolithic scheduler.
   */
  async generateGreedy(
    promptTokens: Uint32Array | readonly number[],
    options: { readonly maxNewTokens?: number; readonly resetState?: boolean } = {},
  ): Promise<Lfm2GenerationResult> {
    const prompt = promptTokens instanceof Uint32Array ? promptTokens : Uint32Array.from(promptTokens);
    const maxNewTokens = options.maxNewTokens ?? lfm2.capacities.maxNewTokens;
    if (prompt.length < 1) throw new Error("generateGreedy requires at least one prompt token");
    if (prompt.length > lfm2.capacities.context) {
      throw new Error(`Prompt has ${prompt.length} tokens, context capacity is ${lfm2.capacities.context}`);
    }
    if (prompt.length + maxNewTokens - 1 > lfm2.capacities.context) {
      throw new Error(
        `Prompt + decode positions (${prompt.length + maxNewTokens - 1}) exceed context ${lfm2.capacities.context}`,
      );
    }

    await this.prepareAll();
    this.writeRuntime(prompt.length, maxNewTokens);
    this.writeTokens(prompt, 0);
    this.executor.submit((encoder) => {
      if (options.resetState ?? true) this.clearState(encoder);
      encoder.compute((pass) => {
        this.forwardAndSample(pass, prompt.length, "prefill", LFM2_ARENA, 0);
        for (let step = 1; step < maxNewTokens; step++) {
          this.forwardAndSample(pass, 1, "decode", LFM2_ARENA, 0);
        }
      }, { label: "lfm2.generate.greedy" });
    });

    return await this.readGenerationResult({
      prefillTokens: prompt.length,
      restoredCheckpointBytes: 0,
      checkpointRestoreUs: 0,
    });
  }

}
