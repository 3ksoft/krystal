import { GgmlType, type GgufTensorInfo } from "../gguf/types.ts";
import type { RandomAccessSource } from "../gguf/source.ts";

// Legacy v1 magic already emitted by the original converter. Keep it stable
// so existing sidecars remain readable even though the old ASCII comment was
// inaccurate.
export const WQ4_MAGIC = 0x34515157;
export const WQ4_VERSION = 1;
export const WQ4_BLOCK_SIZE = 32;
export const WQ4_WORDS_PER_BLOCK = 5; // 4 x packed q4 + 1 x i32 exponent
export const WQ4_I8_WORDS_PER_BLOCK = 9; // 8 x packed i8 + 1 x i32 exponent
export const WQ4_BYTES_PER_BLOCK = WQ4_WORDS_PER_BLOCK * 4;
export const WQ4_I8_BYTES_PER_BLOCK = WQ4_I8_WORDS_PER_BLOCK * 4;
export const WQ4_ZERO_WORD = 0x88888888;
export const WQ4_MIN_EXP = -24;
export const WQ4_MAX_EXP = 8;

export interface Wq4TensorInfo {
  name: string;
  offset: number;
  byteLength: number;
  blockCount: number;
}

function clampExp(exp: number): number {
  return Math.max(WQ4_MIN_EXP, Math.min(WQ4_MAX_EXP, exp));
}

function quantizedValue(value: number, scale: number): number {
  return Math.max(-8, Math.min(7, Math.round(value / scale)));
}

/**
 * Quantizer used by the v13b benchmark path. The on-disk representation stays
 * compact q4 (5 u32 / 32 weights); runtime repacks it to the i8 intermediate
 * consumed by the matmul kernel.
 */
export function quantizeBlockWq4(
  data: Float32Array,
  start: number,
  out: Uint32Array,
  outOffset: number,
): void {
  const end = Math.min(start + WQ4_BLOCK_SIZE, data.length);
  let maxPositive = 0;
  let minNegative = 0;
  for (let i = start; i < end; i++) {
    const value = data[i]!;
    if (value > maxPositive) maxPositive = value;
    if (value < minNegative) minNegative = value;
  }

  if (maxPositive === 0 && minNegative === 0) {
    out[outOffset] = WQ4_ZERO_WORD;
    out[outOffset + 1] = WQ4_ZERO_WORD;
    out[outOffset + 2] = WQ4_ZERO_WORD;
    out[outOffset + 3] = WQ4_ZERO_WORD;
    out[outOffset + 4] = WQ4_MIN_EXP >>> 0;
    return;
  }

  const maxAbs = Math.max(maxPositive, -minNegative);
  const legacyExp = clampExp(Math.floor(Math.log2(maxAbs / 7)));
  const noClipScale = Math.max(maxPositive / 7, (-minNegative) / 8, 2 ** WQ4_MIN_EXP);
  const exactExp = Math.log2(noClipScale);
  const expLo = clampExp(Math.floor(exactExp));
  const expHi = clampExp(Math.ceil(exactExp));

  const candidates = [legacyExp, expLo, expHi];
  let bestExp = candidates[0]!;
  let bestSse = Number.POSITIVE_INFINITY;
  for (const exp of candidates) {
    const scale = 2 ** exp;
    let sse = 0;
    for (let i = start; i < start + WQ4_BLOCK_SIZE; i++) {
      const value = i < data.length ? data[i]! : 0;
      const err = value - quantizedValue(value, scale) * scale;
      sse += err * err;
    }
    if (sse < bestSse) {
      bestSse = sse;
      bestExp = exp;
    }
  }

  const scale = 2 ** bestExp;
  for (let wordIndex = 0; wordIndex < 4; wordIndex++) {
    let word = 0;
    for (let nibbleIndex = 0; nibbleIndex < 8; nibbleIndex++) {
      const idx = start + wordIndex * 8 + nibbleIndex;
      const value = idx < data.length ? data[idx]! : 0;
      const q = quantizedValue(value, scale);
      word |= ((q + 8) & 0x0f) << (nibbleIndex * 4);
    }
    out[outOffset + wordIndex] = word >>> 0;
  }
  out[outOffset + 4] = bestExp >>> 0;
}

/** Repack compact WQ4 into the exact 9-u32/block layout used by candidate v13b. */
export function repackWq4ToI8(packedBytes: Uint8Array): Uint32Array {
  if (packedBytes.byteLength % WQ4_BYTES_PER_BLOCK !== 0) {
    throw new Error(`WQ4 byte length ${packedBytes.byteLength} is not a multiple of ${WQ4_BYTES_PER_BLOCK}`);
  }

  const aligned = packedBytes.byteOffset % 4 === 0
    ? packedBytes
    : new Uint8Array(packedBytes);
  const src = new Uint32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4);
  const blockCount = packedBytes.byteLength / WQ4_BYTES_PER_BLOCK;
  const out = new Uint32Array(blockCount * WQ4_I8_WORDS_PER_BLOCK);

  for (let b = 0; b < blockCount; b++) {
    const srcBase = b * WQ4_WORDS_PER_BLOCK;
    const dstBase = b * WQ4_I8_WORDS_PER_BLOCK;
    for (let i = 0; i < WQ4_BLOCK_SIZE; i++) {
      const nibble = (src[srcBase + (i >> 3)]! >> ((i & 7) * 4)) & 0x0f;
      const signed = nibble - 8;
      const dstWord = dstBase + (i >> 2);
      out[dstWord] = (out[dstWord]! | ((signed & 0xff) << ((i & 3) * 8))) >>> 0;
    }
    out[dstBase + 8] = src[srcBase + 4]!;
  }

  return out;
}

export class Wq4Sidecar {
  private constructor(
    readonly source: RandomAccessSource,
    readonly tensors: Map<string, Wq4TensorInfo>,
  ) {}

  static async open(
    source: RandomAccessSource,
    ggufTensors: Iterable<GgufTensorInfo>,
  ): Promise<Wq4Sidecar> {
    if (source.size < 16) throw new Error(`WQ4 sidecar is too small: ${source.size} bytes`);
    const header = await source.read(0, 16);
    const view = new DataView(header.buffer, header.byteOffset, header.byteLength);
    const magic = view.getUint32(0, true);
    const version = view.getUint32(4, true);
    const tensorCount = view.getUint32(8, true);
    if (magic !== WQ4_MAGIC) throw new Error(`Invalid WQ4 magic 0x${magic.toString(16)}`);
    if (version !== WQ4_VERSION) throw new Error(`Unsupported WQ4 version ${version}`);

    const matrices = [...ggufTensors].filter((tensor) =>
      tensor.dimensions.length >= 2 && (tensor.type === GgmlType.F16 || tensor.type === GgmlType.F32)
    );
    if (tensorCount !== matrices.length) {
      throw new Error(`WQ4 tensor count ${tensorCount} does not match GGUF matrix count ${matrices.length}`);
    }

    const tensors = new Map<string, Wq4TensorInfo>();
    let offset = 16;
    for (const tensor of matrices) {
      const elementCount = tensor.dimensions.reduce((a, b) => a * b, 1);
      const blockCount = Math.ceil(elementCount / WQ4_BLOCK_SIZE);
      const byteLength = blockCount * WQ4_BYTES_PER_BLOCK;
      if (offset + byteLength > source.size) {
        throw new Error(`${tensor.name}: WQ4 payload ends beyond sidecar size`);
      }
      tensors.set(tensor.name, { name: tensor.name, offset, byteLength, blockCount });
      offset += byteLength;
    }
    if (offset !== source.size) {
      throw new Error(`WQ4 sidecar has ${source.size - offset} unexpected trailing bytes`);
    }

    return new Wq4Sidecar(source, tensors);
  }

  tensor(name: string): Wq4TensorInfo | undefined {
    return this.tensors.get(name);
  }

  async readBlocks(tensor: Wq4TensorInfo, blockStart: number, blockCount: number): Promise<Uint8Array> {
    if (blockStart < 0 || blockCount < 0 || blockStart + blockCount > tensor.blockCount) {
      throw new RangeError(`${tensor.name}: WQ4 block range ${blockStart}+${blockCount} outside ${tensor.blockCount}`);
    }
    return await this.source.read(
      tensor.offset + blockStart * WQ4_BYTES_PER_BLOCK,
      blockCount * WQ4_BYTES_PER_BLOCK,
    );
  }
}
