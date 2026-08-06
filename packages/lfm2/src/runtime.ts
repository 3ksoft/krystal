import type { Sandblaster } from "@sandblaster/core";
import { Lfm2Model } from "./model.ts";
import {
  createInitialRuntimeState,
  deserializeLlmRuntime,
  type LlmRuntimeState,
} from "../../schema/src/schema.ts";
import {
  DECODE_TELEMETRY_ENTRIES_LEN,
  SIZEOF_DecodeTelemetry,
  deserializeDecodeTelemetry,
} from "../../schema/dist/util/codec.ts";
import { Lfm2BlockCache } from "./runtime/block-cache.ts";
import { Lfm2CachedBlock } from "./runtime/cache.ts";
import { Lfm2GpuRuntime } from "./runtime/gpu.ts";
import type {
  CacheBlockOptions,
  CpuCandidateProcessor,
  CpuLogitProcessor,
  GenerateOptions,
  GenerateResult,
  GenerateTimings,
  DecodeTelemetryResult,
  Lfm2RuntimeOptions,
} from "./runtime/types.ts";

export { Lfm2CachedBlock } from "./runtime/cache.ts";
export type {
  CacheBlockOptions,
  CpuCandidateProcessor,
  CpuLogitProcessor,
  DecodeTelemetryResult,
  GenerateOptions,
  GenerateResult,
  GenerateTimings,
  Lfm2RuntimeOptions,
  MatmulDispatchArgs,
  MatmulKernelSpec,
} from "./runtime/types.ts";

function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

function gpuBufferSize(bytes: number): number {
  return Math.max(4, align(bytes, 4));
}

function cpuArgmax(logits: Float32Array): number {
  let bestToken = 0;
  let bestValue = -Number.MAX_VALUE;
  for (let i = 0; i < logits.length; i++) {
    const value = logits[i]!;
    if (value > bestValue || (value === bestValue && i < bestToken)) {
      bestValue = value;
      bestToken = i;
    }
  }
  return bestToken;
}

/**
 * Public orchestration layer for LFM2 inference.
 *
 * - runtime/gpu.ts owns bind groups, buffers, pipelines and forward kernels.
 * - runtime/block-cache.ts owns immutable block checkpoints/composition.
 * - this file owns request scheduling, diagnostic CPU paths and final readback.
 */
export class Lfm2Runtime {
  readonly device: GPUDevice;
  readonly model: Lfm2Model;
  readonly contextCapacity: number;
  readonly defaultMaxNewTokens: number;
  readonly blockCacheDepth: number;
  readonly blockCacheMaxTokens: number;
  readonly blockCacheDepths: readonly number[];

  private readonly blockCache: Lfm2BlockCache;
  private telemetryInFlight = false;

  private constructor(private readonly gpu: Lfm2GpuRuntime) {
    this.device = gpu.device;
    this.model = gpu.model;
    this.contextCapacity = gpu.contextCapacity;
    this.defaultMaxNewTokens = gpu.defaultMaxNewTokens;
    this.blockCacheDepth = gpu.blockCacheDepth;
    this.blockCacheMaxTokens = gpu.blockCacheMaxTokens;
    this.blockCacheDepths = gpu.blockCacheDepths;
    this.blockCache = new Lfm2BlockCache(gpu, gpu.model);
  }

  static async create(
    engine: Sandblaster<any>,
    model: Lfm2Model,
    options: Lfm2RuntimeOptions = {},
  ): Promise<Lfm2Runtime> {
    return new Lfm2Runtime(await Lfm2GpuRuntime.create(engine, model, options));
  }

  // Compatibility surface retained for the current bring-up harness.
  get runtimeBuffer(): GPUBuffer { return this.gpu.runtimeBuffer; }
  get tokenBuffer(): GPUBuffer { return this.gpu.tokenBuffer; }
  get arenaBuffer(): GPUBuffer { return this.gpu.arenaBuffer; }
  get kvBuffer(): GPUBuffer { return this.gpu.kvBuffer; }
  get convBuffer(): GPUBuffer { return this.gpu.convBuffer; }
  get decodeTelemetryBuffer(): GPUBuffer { return this.gpu.decodeTelemetryBuffer; }

  private get arena() { return this.gpu.arena; }
  private get runtimeByteSize() { return this.gpu.runtimeByteSize; }
  private get runtimeCpu() { return this.gpu.runtimeCpu; }
  private get runtimeView() { return this.gpu.runtimeView; }
  private get paramBuffer() { return this.gpu.paramBuffer; }
  private get paramWriter() { return this.gpu.paramWriter; }
  private get telemetryStaging() { return this.gpu.telemetryStaging; }
  private get decodeTelemetryStaging() { return this.gpu.decodeTelemetryStaging; }
  private get logitsStaging() { return this.gpu.logitsStaging; }
  private get candidateTokenBuffer() { return this.gpu.candidateTokenBuffer; }
  private get selectedTokenStaging() { return this.gpu.selectedTokenStaging; }

  cacheBlock(tokenIds: readonly number[], options: CacheBlockOptions = {}): Promise<Lfm2CachedBlock> {
    return this.blockCache.cacheBlock(tokenIds, options);
  }

  destroyCachedBlock(block: Lfm2CachedBlock): void {
    this.blockCache.destroyCachedBlock(block);
  }

  destroyAllCachedBlocks(): void {
    this.blockCache.destroyAllCachedBlocks();
  }

  async generateFromBlocks(
    blocks: readonly Lfm2CachedBlock[],
    options: GenerateOptions = {},
  ): Promise<GenerateResult> {
    const validated = this.blockCache.validateCachedBlocks(blocks);
    const promptTokenCount = validated.tokenCount;
    const cacheDepth = validated.cacheDepth;
    const maxNewTokens = options.maxNewTokens ?? this.defaultMaxNewTokens;
    if (maxNewTokens < 1 || maxNewTokens > this.defaultMaxNewTokens) {
      throw new Error(`maxNewTokens must be 1..${this.defaultMaxNewTokens} for this allocated runtime`);
    }
    if (promptTokenCount + maxNewTokens - 1 > this.contextCapacity) {
      throw new Error(`Cached prompt + decode positions exceed contextCapacity ${this.contextCapacity}`);
    }

    const state = createInitialRuntimeState({
      contextCapacity: this.contextCapacity,
      maxNewTokens,
      eosToken: this.model.config.eosToken,
      promptTokenCount,
    });
    this.gpu.writeRuntime(state);

    if (options.profile) {
      const totalStarted = performance.now();
      this.paramWriter.reset();
      const prefillEncoder = this.device.createCommandEncoder({ label: "lfm2.block-cache.profile.prefill" });
      const cache = this.blockCache.recordCachedPrefill(prefillEncoder, blocks, promptTokenCount, cacheDepth);
      if (this.paramWriter.usedBytes > 0) {
        this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
      }
      const prefillStarted = performance.now();
      this.device.queue.submit([prefillEncoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const prefillMs = performance.now() - prefillStarted;

      let decodeMs = 0;
      const scheduledDecodeSteps = Math.max(0, maxNewTokens - 1);
      if (scheduledDecodeSteps > 0) {
        this.paramWriter.reset();
        const decodeEncoder = this.device.createCommandEncoder({ label: "lfm2.block-cache.profile.decode" });
        const decodePass = decodeEncoder.beginComputePass({ label: "lfm2.block-cache.decode" });
        for (let step = 0; step < scheduledDecodeSteps; step++) this.gpu.forwardAndSample(decodePass, 1, true);
        decodePass.end();
        if (this.paramWriter.usedBytes > 0) {
          this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
        }
        const decodeStarted = performance.now();
        this.device.queue.submit([decodeEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        decodeMs = performance.now() - decodeStarted;
      }

      const readbackStarted = performance.now();
      const result = await this.readResult(maxNewTokens);
      const readbackMs = performance.now() - readbackStarted;
      return {
        ...result,
        timings: {
          prefillMs,
          decodeMs,
          readbackMs,
          totalMs: performance.now() - totalStarted,
          promptTokens: promptTokenCount,
          scheduledDecodeSteps,
          cacheDepth: cache.cacheDepth,
          cachedBlocks: blocks.length,
          cachedTokens: promptTokenCount,
          repairedTokens: cache.repairedTokens,
        },
      };
    }

    this.paramWriter.reset();
    const encoder = this.device.createCommandEncoder({ label: "lfm2.generate-from-blocks" });
    this.blockCache.recordCachedPrefill(encoder, blocks, promptTokenCount, cacheDepth);
    const decodePass = encoder.beginComputePass({ label: "lfm2.block-cache.decode" });
    for (let step = 1; step < maxNewTokens; step++) this.gpu.forwardAndSample(decodePass, 1, true);
    decodePass.end();
    if (this.paramWriter.usedBytes > 0) {
      this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
    }
    this.device.queue.submit([encoder.finish()]);
    return await this.readResult(maxNewTokens);
  }

  /**
   * Cached blocks are persistent context; queryTokenIds are a live causal
   * frontier evaluated after them. The live query sees cached context already
   * at the first attention layer instead of being precached standalone.
   */
  async generateFromBlocksWithQuery(
    blocks: readonly Lfm2CachedBlock[],
    queryTokenIds: readonly number[],
    options: GenerateOptions = {},
  ): Promise<GenerateResult> {
    const validated = this.blockCache.validateCachedBlocks(blocks);
    const contextTokenCount = validated.tokenCount;
    const cacheDepth = validated.cacheDepth;
    if (queryTokenIds.length < 1) throw new Error("Live query must contain at least one token");
    const promptTokenCount = contextTokenCount + queryTokenIds.length;
    const maxNewTokens = options.maxNewTokens ?? this.defaultMaxNewTokens;
    if (maxNewTokens < 1 || maxNewTokens > this.defaultMaxNewTokens) {
      throw new Error(`maxNewTokens must be 1..${this.defaultMaxNewTokens} for this allocated runtime`);
    }
    if (promptTokenCount + maxNewTokens - 1 > this.contextCapacity) {
      throw new Error(`Cached context + live query + decode positions exceed contextCapacity ${this.contextCapacity}`);
    }

    const state = createInitialRuntimeState({
      contextCapacity: this.contextCapacity,
      maxNewTokens,
      eosToken: this.model.config.eosToken,
      promptTokenCount,
    });
    this.gpu.writeRuntime(state);
    this.device.queue.writeBuffer(
      this.tokenBuffer,
      contextTokenCount * 4,
      new Uint32Array(queryTokenIds),
    );

    if (options.profile) {
      const totalStarted = performance.now();
      this.paramWriter.reset();
      const prefillEncoder = this.device.createCommandEncoder({ label: "lfm2.block-cache-query.profile.prefill" });
      const cache = this.blockCache.recordCachedQueryPrefill(
        prefillEncoder, blocks, contextTokenCount, queryTokenIds.length, cacheDepth,
      );
      if (this.paramWriter.usedBytes > 0) {
        this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
      }
      const prefillStarted = performance.now();
      this.device.queue.submit([prefillEncoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const prefillMs = performance.now() - prefillStarted;

      let decodeMs = 0;
      const scheduledDecodeSteps = Math.max(0, maxNewTokens - 1);
      if (scheduledDecodeSteps > 0) {
        this.paramWriter.reset();
        const decodeEncoder = this.device.createCommandEncoder({ label: "lfm2.block-cache-query.profile.decode" });
        const decodePass = decodeEncoder.beginComputePass({ label: "lfm2.block-cache-query.decode" });
        for (let step = 0; step < scheduledDecodeSteps; step++) this.gpu.forwardAndSample(decodePass, 1, true);
        decodePass.end();
        if (this.paramWriter.usedBytes > 0) {
          this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
        }
        const decodeStarted = performance.now();
        this.device.queue.submit([decodeEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        decodeMs = performance.now() - decodeStarted;
      }

      const readbackStarted = performance.now();
      const result = await this.readResult(maxNewTokens);
      const readbackMs = performance.now() - readbackStarted;
      return {
        ...result,
        timings: {
          prefillMs,
          decodeMs,
          readbackMs,
          totalMs: performance.now() - totalStarted,
          promptTokens: promptTokenCount,
          scheduledDecodeSteps,
          cacheDepth: cache.cacheDepth,
          cachedBlocks: blocks.length,
          cachedTokens: contextTokenCount,
          liveQueryTokens: queryTokenIds.length,
          repairedTokens: cache.repairedTokens,
        },
      };
    }

    this.paramWriter.reset();
    const encoder = this.device.createCommandEncoder({ label: "lfm2.generate-from-blocks-with-query" });
    this.blockCache.recordCachedQueryPrefill(encoder, blocks, contextTokenCount, queryTokenIds.length, cacheDepth);
    const decodePass = encoder.beginComputePass({ label: "lfm2.block-cache-query.decode" });
    for (let step = 1; step < maxNewTokens; step++) this.gpu.forwardAndSample(decodePass, 1, true);
    decodePass.end();
    if (this.paramWriter.usedBytes > 0) {
      this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
    }
    this.device.queue.submit([encoder.finish()]);
    return await this.readResult(maxNewTokens);
  }

  async generateTokens(promptTokenIds: readonly number[], options: GenerateOptions = {}): Promise<GenerateResult> {
    const maxNewTokens = options.maxNewTokens ?? this.defaultMaxNewTokens;
    if (promptTokenIds.length < 1) throw new Error("Prompt must contain at least one token");
    if (promptTokenIds.length > this.contextCapacity) {
      throw new Error(`Prompt has ${promptTokenIds.length} tokens, contextCapacity is ${this.contextCapacity}`);
    }
    if (maxNewTokens < 1 || maxNewTokens > this.defaultMaxNewTokens) {
      throw new Error(`maxNewTokens must be 1..${this.defaultMaxNewTokens} for this allocated runtime`);
    }
    if (promptTokenIds.length + maxNewTokens - 1 > this.contextCapacity) {
      throw new Error(`Prompt + decode positions exceed bring-up contextCapacity ${this.contextCapacity}`);
    }

    const state = createInitialRuntimeState({
      contextCapacity: this.contextCapacity,
      maxNewTokens,
      eosToken: this.model.config.eosToken,
      promptTokenCount: promptTokenIds.length,
    });
    this.gpu.writeRuntime(state);
    this.device.queue.writeBuffer(this.tokenBuffer, 0, new Uint32Array(promptTokenIds));

    if (options.profile) {
      return await this.generateProfiled(promptTokenIds.length, maxNewTokens);
    }

    this.paramWriter.reset();
    const encoder = this.device.createCommandEncoder({ label: "lfm2.generate" });
    encoder.clearBuffer(this.kvBuffer);
    encoder.clearBuffer(this.convBuffer);
    const pass = encoder.beginComputePass({ label: "lfm2.inference" });

    this.gpu.forwardAndSample(pass, promptTokenIds.length, false);
    for (let step = 1; step < maxNewTokens; step++) {
      this.gpu.forwardAndSample(pass, 1, true);
    }
    pass.end();

    if (this.paramWriter.usedBytes > 0) {
      this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
    }
    this.device.queue.submit([encoder.finish()]);
    return await this.readResult(maxNewTokens);
  }

  /**
   * Faster diagnostic constrained-decoding path. The CPU still computes the
   * legal token ids, but logits never leave the GPU. Each step uploads only
   * the sparse candidate list, runs forward + candidate argmax in one submit,
   * then reads back the selected u32 so the CPU oracle can advance.
   */
  async generateTokensWithCpuCandidates(
    promptTokenIds: readonly number[],
    processor: CpuCandidateProcessor,
    options: Pick<GenerateOptions, "maxNewTokens"> = {},
  ): Promise<GenerateResult> {
    const maxNewTokens = options.maxNewTokens ?? this.defaultMaxNewTokens;
    if (promptTokenIds.length < 1) throw new Error("Prompt must contain at least one token");
    if (promptTokenIds.length > this.contextCapacity) {
      throw new Error(`Prompt has ${promptTokenIds.length} tokens, contextCapacity is ${this.contextCapacity}`);
    }
    if (maxNewTokens < 1 || maxNewTokens > this.defaultMaxNewTokens) {
      throw new Error(`maxNewTokens must be 1..${this.defaultMaxNewTokens} for this allocated runtime`);
    }
    if (promptTokenIds.length + maxNewTokens - 1 > this.contextCapacity) {
      throw new Error(`Prompt + decode positions exceed bring-up contextCapacity ${this.contextCapacity}`);
    }

    const state = createInitialRuntimeState({
      contextCapacity: this.contextCapacity,
      maxNewTokens,
      eosToken: this.model.config.eosToken,
      promptTokenCount: promptTokenIds.length,
    });
    this.gpu.writeRuntime(state);
    this.device.queue.writeBuffer(this.tokenBuffer, 0, new Uint32Array(promptTokenIds));

    const clearEncoder = this.device.createCommandEncoder({ label: "lfm2.cpu-candidates.clear" });
    clearEncoder.clearBuffer(this.kvBuffer);
    clearEncoder.clearBuffer(this.convBuffer);
    this.device.queue.submit([clearEncoder.finish()]);

    for (let step = 0; step < maxNewTokens; step++) {
      const decode = step > 0;
      const tokenCount = decode ? 1 : promptTokenIds.length;
      const context = { step, decode };
      const candidates = await processor.candidates(context);
      if (candidates.length < 1) throw new Error(`CPU candidate processor returned an empty set at step ${step}`);
      if (candidates.length > this.model.config.vocabSize) {
        throw new Error(`CPU candidate processor returned ${candidates.length} ids for vocab ${this.model.config.vocabSize}`);
      }
      this.device.queue.writeBuffer(this.candidateTokenBuffer, 0, candidates);

      this.paramWriter.reset();
      const encoder = this.device.createCommandEncoder({ label: `lfm2.cpu-candidates.step.${step}` });
      const pass = encoder.beginComputePass({ label: `lfm2.cpu-candidates.pass.${step}` });
      this.gpu.forwardToLogits(pass, tokenCount, decode);
      this.gpu.dispatch(pass, "argmax_candidates", {
        inputOffset: this.arena.logits,
        inputDim: candidates.length,
        mode: decode ? 1 : 0,
      }, [1]);
      pass.end();

      const selectedOffset = (this.contextCapacity + step) * 4;
      encoder.copyBufferToBuffer(this.tokenBuffer, selectedOffset, this.selectedTokenStaging, 0, 4);
      if (this.paramWriter.usedBytes > 0) {
        this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
      }
      this.device.queue.submit([encoder.finish()]);

      await this.selectedTokenStaging.mapAsync(GPUMapMode.READ);
      let selectedToken: number;
      try {
        selectedToken = new Uint32Array(this.selectedTokenStaging.getMappedRange())[0]!;
      } finally {
        this.selectedTokenStaging.unmap();
      }
      await processor.accept?.(selectedToken, context);
      if (selectedToken === this.model.config.eosToken || processor.shouldStop?.()) break;
    }

    return await this.readResult(maxNewTokens);
  }

  /**
   * Diagnostic constrained-decoding path. Each token deliberately performs a
   * logits readback to CPU, lets `processor` mutate the values, writes them
   * back, and then uses the normal GPU argmax/runtime bookkeeping.
   *
   * This is intentionally synchronization-heavy and exists as a correctness
   * oracle before moving masking/state transitions into WGSL.
   */
  async generateTokensWithCpuLogits(
    promptTokenIds: readonly number[],
    processor: CpuLogitProcessor,
    options: Pick<GenerateOptions, "maxNewTokens"> = {},
  ): Promise<GenerateResult> {
    const maxNewTokens = options.maxNewTokens ?? this.defaultMaxNewTokens;
    if (promptTokenIds.length < 1) throw new Error("Prompt must contain at least one token");
    if (promptTokenIds.length > this.contextCapacity) {
      throw new Error(`Prompt has ${promptTokenIds.length} tokens, contextCapacity is ${this.contextCapacity}`);
    }
    if (maxNewTokens < 1 || maxNewTokens > this.defaultMaxNewTokens) {
      throw new Error(`maxNewTokens must be 1..${this.defaultMaxNewTokens} for this allocated runtime`);
    }
    if (promptTokenIds.length + maxNewTokens - 1 > this.contextCapacity) {
      throw new Error(`Prompt + decode positions exceed bring-up contextCapacity ${this.contextCapacity}`);
    }

    const state = createInitialRuntimeState({
      contextCapacity: this.contextCapacity,
      maxNewTokens,
      eosToken: this.model.config.eosToken,
      promptTokenCount: promptTokenIds.length,
    });
    this.gpu.writeRuntime(state);
    this.device.queue.writeBuffer(this.tokenBuffer, 0, new Uint32Array(promptTokenIds));

    // The normal path clears these once before the monolithic pass. Do the same
    // here, then preserve them across token-level submissions.
    const clearEncoder = this.device.createCommandEncoder({ label: "lfm2.cpu-logits.clear" });
    clearEncoder.clearBuffer(this.kvBuffer);
    clearEncoder.clearBuffer(this.convBuffer);
    this.device.queue.submit([clearEncoder.finish()]);

    for (let step = 0; step < maxNewTokens; step++) {
      const decode = step > 0;
      const tokenCount = decode ? 1 : promptTokenIds.length;

      this.paramWriter.reset();
      const forwardEncoder = this.device.createCommandEncoder({ label: `lfm2.cpu-logits.forward.${step}` });
      const forwardPass = forwardEncoder.beginComputePass({ label: `lfm2.cpu-logits.forward-pass.${step}` });
      this.gpu.forwardToLogits(forwardPass, tokenCount, decode);
      forwardPass.end();
      forwardEncoder.copyBufferToBuffer(
        this.arenaBuffer,
        this.arena.logits * 4,
        this.logitsStaging,
        0,
        this.model.config.vocabSize * 4,
      );
      if (this.paramWriter.usedBytes > 0) {
        this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
      }
      this.device.queue.submit([forwardEncoder.finish()]);

      await this.logitsStaging.mapAsync(GPUMapMode.READ);
      let logits: Float32Array;
      try {
        // Copy out before unmapping; mapped ranges become invalid immediately.
        logits = new Float32Array(
          new Float32Array(this.logitsStaging.getMappedRange()).slice().buffer,
        );
      } finally {
        this.logitsStaging.unmap();
      }

      const context = { step, decode };
      await processor.process(logits, context);
      const selectedToken = cpuArgmax(logits);
      await processor.accept?.(selectedToken, context);

      this.device.queue.writeBuffer(this.arenaBuffer, this.arena.logits * 4, logits);
      this.paramWriter.reset();
      const commitEncoder = this.device.createCommandEncoder({ label: `lfm2.cpu-logits.commit.${step}` });
      const commitPass = commitEncoder.beginComputePass({ label: `lfm2.cpu-logits.argmax.${step}` });
      this.gpu.commitLogitsArgmax(commitPass, decode);
      commitPass.end();
      if (this.paramWriter.usedBytes > 0) {
        this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
      }
      this.device.queue.submit([commitEncoder.finish()]);

      if (selectedToken === this.model.config.eosToken || processor.shouldStop?.()) break;
    }

    return await this.readResult(maxNewTokens);
  }

  private async generateProfiled(promptTokenCount: number, maxNewTokens: number): Promise<GenerateResult> {
    const totalStarted = performance.now();

    // Profiling intentionally introduces one synchronization point between
    // prefill and decode. It is diagnostic-only; the normal path above remains
    // one monolithic GPU submission with no CPU dependency between phases.
    this.paramWriter.reset();
    const prefillEncoder = this.device.createCommandEncoder({ label: "lfm2.profile.prefill" });
    prefillEncoder.clearBuffer(this.kvBuffer);
    prefillEncoder.clearBuffer(this.convBuffer);
    const prefillPass = prefillEncoder.beginComputePass({ label: "lfm2.prefill" });
    this.gpu.forwardAndSample(prefillPass, promptTokenCount, false);
    prefillPass.end();
    if (this.paramWriter.usedBytes > 0) {
      this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
    }

    const prefillStarted = performance.now();
    this.device.queue.submit([prefillEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const prefillMs = performance.now() - prefillStarted;

    let decodeMs = 0;
    const scheduledDecodeSteps = Math.max(0, maxNewTokens - 1);
    if (scheduledDecodeSteps > 0) {
      this.paramWriter.reset();
      const decodeEncoder = this.device.createCommandEncoder({ label: "lfm2.profile.decode" });
      const decodePass = decodeEncoder.beginComputePass({ label: "lfm2.decode" });
      for (let step = 0; step < scheduledDecodeSteps; step++) {
        this.gpu.forwardAndSample(decodePass, 1, true);
      }
      decodePass.end();
      if (this.paramWriter.usedBytes > 0) {
        this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
      }

      const decodeStarted = performance.now();
      this.device.queue.submit([decodeEncoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      decodeMs = performance.now() - decodeStarted;
    }

    const readbackStarted = performance.now();
    const result = await this.readResult(maxNewTokens);
    const readbackMs = performance.now() - readbackStarted;
    return {
      ...result,
      timings: {
        prefillMs,
        decodeMs,
        readbackMs,
        totalMs: performance.now() - totalStarted,
        promptTokens: promptTokenCount,
        scheduledDecodeSteps,
      },
    };
  }

  private async readResult(maxNewTokens: number): Promise<GenerateResult> {
    const runtimeStaging = this.device.createBuffer({
      label: "lfm2.result-runtime",
      size: gpuBufferSize(this.runtimeByteSize),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const tokenStaging = this.device.createBuffer({
      label: "lfm2.result-tokens",
      size: gpuBufferSize(maxNewTokens * 4),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "lfm2.result-readback" });
    encoder.copyBufferToBuffer(this.runtimeBuffer, 0, runtimeStaging, 0, this.runtimeByteSize);
    encoder.copyBufferToBuffer(this.tokenBuffer, this.contextCapacity * 4, tokenStaging, 0, maxNewTokens * 4);
    encoder.copyBufferToBuffer(
      this.decodeTelemetryBuffer, 0, this.decodeTelemetryStaging, 0, SIZEOF_DecodeTelemetry,
    );
    this.device.queue.submit([encoder.finish()]);

    try {
      await Promise.all([
        runtimeStaging.mapAsync(GPUMapMode.READ),
        tokenStaging.mapAsync(GPUMapMode.READ),
        this.decodeTelemetryStaging.mapAsync(GPUMapMode.READ),
      ]);
      const runtimeBytes = new Uint8Array(runtimeStaging.getMappedRange());
      new Uint8Array(this.runtimeCpu).set(runtimeBytes.subarray(0, this.runtimeByteSize));
      const state = deserializeLlmRuntime(this.runtimeView);
      const mappedTokens = new Uint32Array(tokenStaging.getMappedRange());
      const tokenIds = Array.from(mappedTokens.subarray(0, state.generatedCount));

      const telemetryBytes = new Uint8Array(this.decodeTelemetryStaging.getMappedRange());
      const telemetryView = new DataView(
        telemetryBytes.buffer, telemetryBytes.byteOffset, telemetryBytes.byteLength,
      );
      const decodedTelemetry = deserializeDecodeTelemetry(telemetryView, 0);
      const telemetryCount = Math.min(decodedTelemetry.cursor, DECODE_TELEMETRY_ENTRIES_LEN);
      const decodeTelemetry: DecodeTelemetryResult = {
        cursor: decodedTelemetry.cursor,
        entries: decodedTelemetry.entries.slice(0, telemetryCount),
      };
      return { tokenIds, state, decodeTelemetry };
    } finally {
      if (runtimeStaging.mapState === "mapped") runtimeStaging.unmap();
      if (tokenStaging.mapState === "mapped") tokenStaging.unmap();
      if (this.decodeTelemetryStaging.mapState === "mapped") this.decodeTelemetryStaging.unmap();
      runtimeStaging.destroy();
      tokenStaging.destroy();
    }
  }

  /**
   * Drop-if-busy telemetry readback. With the v0 monolithic command buffer this
   * observes request completion; later token-level command chaining can reuse
   * the same API for live telemetry without changing callers.
   */
  async readTelemetry(): Promise<LlmRuntimeState | null> {
    if (this.telemetryInFlight) return null;
    this.telemetryInFlight = true;
    const encoder = this.device.createCommandEncoder({ label: "lfm2.telemetry" });
    encoder.copyBufferToBuffer(this.runtimeBuffer, 0, this.telemetryStaging, 0, this.runtimeByteSize);
    this.device.queue.submit([encoder.finish()]);
    try {
      await this.telemetryStaging.mapAsync(GPUMapMode.READ);
      new Uint8Array(this.runtimeCpu).set(
        new Uint8Array(this.telemetryStaging.getMappedRange()).subarray(0, this.runtimeByteSize),
      );
      return deserializeLlmRuntime(this.runtimeView);
    } finally {
      if (this.telemetryStaging.mapState === "mapped") this.telemetryStaging.unmap();
      this.telemetryInFlight = false;
    }
  }


  destroy(): void {
    this.blockCache.destroyAllCachedBlocks();
    this.gpu.destroy();
  }
}
