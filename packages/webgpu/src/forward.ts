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

/**
 * Target scheduler for the migrated Sandblaster runtime. The implementation is
 * intentionally expressed in semantic tensor names and Lfm2Executor passes;
 * it owns no bind groups, pipeline layouts or handwritten OpParams ABI.
 */
export class Lfm2Forward {
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

  /** Clear recurrent state without destroying/recreating long-lived buffers. */
  clearState(encoder: Lfm2CommandEncoder): void {
    encoder.gpu.clearBuffer(lfm2.resources.convCache.gpu);
    encoder.gpu.clearBuffer(lfm2.resources.kvCache.gpu);
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
   * Full greedy inference in one GPU submission: prompt prefill followed by up
   * to maxNewTokens-1 decode iterations. Runtime.status gates token commits
   * after EOS/done exactly like the legacy monolithic scheduler.
   */
  async generateGreedy(
    promptTokens: Uint32Array | readonly number[],
    options: { readonly maxNewTokens?: number; readonly resetState?: boolean } = {},
  ): Promise<{
    readonly tokens: number[];
    readonly generatedCount: number;
    readonly status: string;
    readonly lastToken: number;
  }> {
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
    };
  }

}
