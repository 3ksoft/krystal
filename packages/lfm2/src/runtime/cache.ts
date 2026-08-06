import { BLOCK_EXACT_DEPTH } from "./constants.ts";

export class Lfm2CachedBlock {
  readonly tokenBytes: number;

  constructor(
    readonly id: number,
    readonly tokenIds: readonly number[],
    readonly tokenBuffer: GPUBuffer,
    readonly cacheDepth: number,
    /** Hidden-state checkpoints at every reusable cache frontier up to cacheDepth. */
    readonly stateBuffers: ReadonlyMap<number, GPUBuffer>,
    /** Conv layers whose standalone decode tail is retained in convStateBuffer. */
    readonly convLayers: readonly number[],
    readonly convStateBuffer: GPUBuffer,
    readonly convLayerBytes: number,
    readonly cacheMs: number,
  ) {
    this.tokenBytes = tokenIds.length * 4;
  }

  get tokenCount(): number {
    return this.tokenIds.length;
  }

  stateAtDepth(depth: number): GPUBuffer {
    const buffer = this.stateBuffers.get(depth);
    if (!buffer) {
      throw new Error(`Cached block ${this.id} has no checkpoint at depth ${depth} (max ${this.cacheDepth})`);
    }
    return buffer;
  }

  /** Compatibility aliases for the original depth-5 experiment. */
  get exactHiddenBuffer(): GPUBuffer { return this.stateAtDepth(BLOCK_EXACT_DEPTH); }
  get hiddenBuffer(): GPUBuffer { return this.stateAtDepth(this.cacheDepth); }

  convStateOffset(layer: number): number {
    const index = this.convLayers.indexOf(layer);
    if (index < 0) throw new Error(`Cached block ${this.id} has no conv tail for layer ${layer}`);
    return index * this.convLayerBytes;
  }

  get gpuBytes(): number {
    let bytes = this.tokenBuffer.size + this.convStateBuffer.size;
    for (const buffer of this.stateBuffers.values()) bytes += buffer.size;
    return bytes;
  }

  destroy(): void {
    this.tokenBuffer.destroy();
    for (const buffer of this.stateBuffers.values()) buffer.destroy();
    this.convStateBuffer.destroy();
  }
}
