import { Lfm2Model } from "../model.ts";
import { Lfm2CachedBlock } from "./cache.ts";
import { BLOCK_CACHE_MAX_TOKENS, BLOCK_EXACT_DEPTH } from "./constants.ts";
import { Lfm2GpuRuntime } from "./gpu.ts";
import type { CacheBlockOptions } from "./types.ts";

function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

function gpuBufferSize(bytes: number): number {
  return Math.max(4, align(bytes, 4));
}

/** Owns immutable block creation/composition and its GPU-side checkpoints. */
export class Lfm2BlockCache {
  private nextCachedBlockId = 0;
  private readonly cachedBlocks = new Set<Lfm2CachedBlock>();

  constructor(
    private readonly gpu: Lfm2GpuRuntime,
    private readonly model: Lfm2Model,
  ) {}

  private get device() { return this.gpu.device; }
  private get contextCapacity() { return this.gpu.contextCapacity; }
  private get blockCacheDepths() { return this.gpu.blockCacheDepths; }
  private get arena() { return this.gpu.arena; }
  private get arenaBuffer() { return this.gpu.arenaBuffer; }
  private get tokenBuffer() { return this.gpu.tokenBuffer; }
  private get kvBuffer() { return this.gpu.kvBuffer; }
  private get convBuffer() { return this.gpu.convBuffer; }
  private get paramWriter() { return this.gpu.paramWriter; }
  private get paramBuffer() { return this.gpu.paramBuffer; }

  async cacheBlock(
    tokenIds: readonly number[],
    options: CacheBlockOptions = {},
  ): Promise<Lfm2CachedBlock> {
    if (tokenIds.length < 1) throw new Error("Cached block must contain at least one token");
    if (tokenIds.length > BLOCK_CACHE_MAX_TOKENS) {
      throw new Error(`Cached block has ${tokenIds.length} tokens; PoC limit is ${BLOCK_CACHE_MAX_TOKENS}`);
    }

    const cacheDepth = this.gpu.resolveBlockCacheDepth(options.depth);
    const checkpoints = this.blockCacheDepths.filter((depth) => depth <= cacheDepth);
    const convLayers = this.model.config.layers
      .map((kind, layer) => ({ kind, layer }))
      .filter(({ kind, layer }) => kind === "conv" && layer >= BLOCK_EXACT_DEPTH && layer < cacheDepth)
      .map(({ layer }) => layer);

    const h = this.model.config.hiddenSize;
    const id = this.nextCachedBlockId++;
    const hiddenBytes = tokenIds.length * h * 4;
    const convLayerBytes = h * this.model.config.convCacheLength * 4;
    const tokenBuffer = this.device.createBuffer({
      label: `lfm2.block.${id}.tokens`,
      size: gpuBufferSize(tokenIds.length * 4),
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const stateBuffers = new Map<number, GPUBuffer>();
    for (const depth of checkpoints) {
      stateBuffers.set(depth, this.device.createBuffer({
        label: `lfm2.block.${id}.depth${depth}.hidden.f32`,
        size: gpuBufferSize(hiddenBytes),
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }));
    }
    const convStateBuffer = this.device.createBuffer({
      label: `lfm2.block.${id}.depth${cacheDepth}.conv-tails`,
      size: gpuBufferSize(convLayers.length * convLayerBytes),
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(tokenBuffer, 0, new Uint32Array(tokenIds));

    this.paramWriter.reset();
    const encoder = this.device.createCommandEncoder({ label: `lfm2.cache-block.${id}.depth${cacheDepth}` });
    encoder.clearBuffer(this.kvBuffer);
    encoder.clearBuffer(this.convBuffer);
    encoder.copyBufferToBuffer(tokenBuffer, 0, this.tokenBuffer, 0, tokenIds.length * 4);

    const exactPass = encoder.beginComputePass({ label: `lfm2.cache-block.${id}.depth${BLOCK_EXACT_DEPTH}` });
    this.gpu.forwardBlockExactPrefix(exactPass, tokenIds.length);
    exactPass.end();
    encoder.copyBufferToBuffer(
      this.arenaBuffer,
      this.arena.hiddenA * 4,
      stateBuffers.get(BLOCK_EXACT_DEPTH)!,
      0,
      hiddenBytes,
    );

    let currentDepth = BLOCK_EXACT_DEPTH;
    for (const targetDepth of checkpoints) {
      if (targetDepth <= currentDepth) continue;
      const pass = encoder.beginComputePass({ label: `lfm2.cache-block.${id}.${currentDepth}-to-${targetDepth}` });
      this.gpu.forwardLayers(pass, currentDepth, targetDepth, tokenIds.length, false);
      pass.end();
      encoder.copyBufferToBuffer(
        this.arenaBuffer,
        this.arena.hiddenA * 4,
        stateBuffers.get(targetDepth)!,
        0,
        hiddenBytes,
      );
      currentDepth = targetDepth;
    }

    for (let i = 0; i < convLayers.length; i++) {
      const layer = convLayers[i]!;
      encoder.copyBufferToBuffer(
        this.convBuffer,
        layer * convLayerBytes,
        convStateBuffer,
        i * convLayerBytes,
        convLayerBytes,
      );
    }

    if (this.paramWriter.usedBytes > 0) {
      this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
    }
    const started = performance.now();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const cacheMs = performance.now() - started;

    const block = new Lfm2CachedBlock(
      id,
      [...tokenIds],
      tokenBuffer,
      cacheDepth,
      stateBuffers,
      convLayers,
      convStateBuffer,
      convLayerBytes,
      cacheMs,
    );
    this.cachedBlocks.add(block);
    return block;
  }

  destroyCachedBlock(block: Lfm2CachedBlock): void {
    if (!this.cachedBlocks.delete(block)) return;
    block.destroy();
  }

  destroyAllCachedBlocks(): void {
    for (const block of this.cachedBlocks) block.destroy();
    this.cachedBlocks.clear();
  }

  validateCachedBlocks(blocks: readonly Lfm2CachedBlock[]): { tokenCount: number; cacheDepth: number } {
    if (blocks.length < 1) throw new Error("generateFromBlocks requires at least one cached block");
    let total = 0;
    let cacheDepth = this.model.config.blockCount;
    for (const block of blocks) {
      if (!this.cachedBlocks.has(block)) throw new Error(`Cached block ${block.id} does not belong to this runtime or was destroyed`);
      total += block.tokenCount;
      cacheDepth = Math.min(cacheDepth, block.cacheDepth);
    }
    if (total > this.contextCapacity) {
      throw new Error(`Cached blocks contain ${total} tokens, contextCapacity is ${this.contextCapacity}`);
    }
    this.gpu.resolveBlockCacheDepth(cacheDepth);
    for (const block of blocks) block.stateAtDepth(cacheDepth);
    return { tokenCount: total, cacheDepth };
  }

  private copyCachedStatesAtDepth(
    encoder: GPUCommandEncoder,
    blocks: readonly Lfm2CachedBlock[],
    depth: number,
  ): void {
    const h = this.model.config.hiddenSize;
    let tokenCursor = 0;
    for (const block of blocks) {
      encoder.copyBufferToBuffer(
        block.stateAtDepth(depth),
        0,
        this.arenaBuffer,
        (this.arena.hiddenA + tokenCursor * h) * 4,
        block.tokenCount * h * 4,
      );
      tokenCursor += block.tokenCount;
    }
  }

  /**
   * Materialize cached context to the final layer. Mixed-depth blocks compose
   * at the shallowest selected frontier: a block cached to depth 8 can be used
   * at depth 5 because it retained its depth-5 checkpoint, while a depth-5
   * block intentionally prevents the request from skipping layers 5..7.
   */
  private recordCachedContext(
    encoder: GPUCommandEncoder,
    blocks: readonly Lfm2CachedBlock[],
    contextTokenCount: number,
    cacheDepth: number,
  ): { repairedTokens: number; cacheDepth: number } {
    const h = this.model.config.hiddenSize;
    const convLayerBytes = h * this.model.config.convCacheLength * 4;
    encoder.clearBuffer(this.kvBuffer);
    encoder.clearBuffer(this.convBuffer);

    let tokenCursor = 0;
    for (const block of blocks) {
      encoder.copyBufferToBuffer(block.tokenBuffer, 0, this.tokenBuffer, tokenCursor * 4, block.tokenCount * 4);
      tokenCursor += block.tokenCount;
    }

    // Depth 2 is the exact composable prefix. Load it in final token order and
    // repair the finite conv boundary before using it as the first attention
    // frontier.
    this.copyCachedStatesAtDepth(encoder, blocks, BLOCK_EXACT_DEPTH);
    const exactPass = encoder.beginComputePass({ label: `lfm2.block-cache.depth${cacheDepth}.exact-hydrate` });
    let boundary = blocks[0]!.tokenCount;
    let repairedTokens = 0;
    for (let i = 1; i < blocks.length; i++) {
      const block = blocks[i]!;
      repairedTokens += this.gpu.repairBlockBoundary(exactPass, boundary, block.tokenCount);
      boundary += block.tokenCount;
    }
    this.gpu.rebuildCachedPrefixConvTail(exactPass, contextTokenCount);
    if (cacheDepth > BLOCK_EXACT_DEPTH) {
      this.gpu.bootstrapAttentionKv(exactPass, BLOCK_EXACT_DEPTH, contextTokenCount);
    }
    exactPass.end();

    // Every deeper cache frontier starts immediately before an attention layer.
    // Restore that standalone checkpoint in composed order and rebuild K/V with
    // the request's actual global token positions. No Q/attention/output work is
    // needed for the frozen prefix itself.
    for (const layer of this.blockCacheDepths) {
      if (layer <= BLOCK_EXACT_DEPTH || layer >= cacheDepth || layer >= this.model.config.blockCount) continue;
      if (this.model.config.layers[layer] !== "attention") continue;
      this.copyCachedStatesAtDepth(encoder, blocks, layer);
      const pass = encoder.beginComputePass({ label: `lfm2.block-cache.bootstrap-attn-${layer}` });
      this.gpu.bootstrapAttentionKv(pass, layer, contextTokenCount);
      pass.end();
    }

    // Conv layers skipped by the cached prefix need the standalone tail of the
    // final block so a live query can continue causally from that block.
    const last = blocks[blocks.length - 1]!;
    for (const layer of last.convLayers) {
      if (layer >= cacheDepth) continue;
      encoder.copyBufferToBuffer(
        last.convStateBuffer,
        last.convStateOffset(layer),
        this.convBuffer,
        layer * convLayerBytes,
        convLayerBytes,
      );
    }

    // At exact depth 2 keep the repaired state already resident. Deeper
    // frontiers deliberately replace it with each block's standalone cached
    // representation, then global interaction resumes from effective depth.
    if (cacheDepth > BLOCK_EXACT_DEPTH) {
      this.copyCachedStatesAtDepth(encoder, blocks, cacheDepth);
    }

    if (cacheDepth < this.model.config.blockCount) {
      const contextPass = encoder.beginComputePass({ label: `lfm2.block-cache.context-from-${cacheDepth}` });
      this.gpu.forwardLayers(contextPass, cacheDepth, this.model.config.blockCount, contextTokenCount, false);
      contextPass.end();
    }
    return { repairedTokens, cacheDepth };
  }

  recordCachedPrefill(
    encoder: GPUCommandEncoder,
    blocks: readonly Lfm2CachedBlock[],
    promptTokenCount: number,
    cacheDepth: number,
  ): { repairedTokens: number; cacheDepth: number } {
    const cache = this.recordCachedContext(encoder, blocks, promptTokenCount, cacheDepth);
    const samplePass = encoder.beginComputePass({ label: "lfm2.block-cache.sample-context" });
    this.gpu.sampleFromHidden(samplePass, promptTokenCount, false);
    samplePass.end();
    return cache;
  }

  recordCachedQueryPrefill(
    encoder: GPUCommandEncoder,
    blocks: readonly Lfm2CachedBlock[],
    contextTokenCount: number,
    queryTokenCount: number,
    cacheDepth: number,
  ): { repairedTokens: number; cacheDepth: number } {
    const cache = this.recordCachedContext(encoder, blocks, contextTokenCount, cacheDepth);

    // Reuse the main arena from token zero. The cached context has already been
    // distilled into KV/conv state, so its final hidden vectors are no longer
    // needed while the live query advances through all layers.
    const queryPass = encoder.beginComputePass({ label: "lfm2.block-cache.live-query" });
    this.gpu.embed(queryPass, queryTokenCount, false, this.arena, contextTokenCount);
    this.gpu.forwardLayersContinuation(
      queryPass,
      0,
      this.model.config.blockCount,
      queryTokenCount,
      contextTokenCount,
    );
    this.gpu.sampleFromHidden(queryPass, queryTokenCount, false);
    queryPass.end();
    return cache;
  }

}
