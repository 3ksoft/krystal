/**
 * Typed, lazily cached weight access for the vision tower + projector
 * (mmproj). Works over either a GGUF reader (exact F16 source, used by the
 * CPU reference oracle) or a WQ4 v3 reader (runtime sidecar). The two readers
 * expose the same tensor/read contract, so this module only depends on a
 * minimal structural type.
 *
 * Tensor layout follows the GGUF convention used everywhere else in the repo:
 * `dimensions[0]` is the row width (matmul input dim), the product of the
 * remaining dims is the row count (output dim), data is row-major.
 *
 * Sources: an exact F16/F32 GGUF reader (differential oracle + reference
 * path) or the WQ4 v3 sidecar (runtime path). Sidecar tensors carry an
 * `encoding`: WQ4-packed matrices are dequantized on the host here (same
 * block format and math as the text runtime's WGSL `matmul_wq4` kernel, so
 * host-dequant and in-shader dequant agree), while non-matrix tensors
 * (`v.position_embd`, norms, biases, `v.patch_embd`, and the raw-F16
 * `ffn_down` whose width 4304 is not a WQ4 block multiple) pass through as
 * raw source bytes. The decoded layout is byte-identical to the F16 GGUF
 * path, so the tower, CPU oracle, and differential harness are agnostic to
 * which source the weights came from.
 */

import { GgmlType } from "../gguf/types.ts";
import { WQ4_BLOCK_SIZE, WQ4_BYTES_PER_BLOCK } from "../wq4/reader.ts";

export interface VisionTensorInfo {
  readonly dimensions: readonly number[];
  readonly sourceType?: number;
  /** WQ4 v3 storage encoding ("wq4" | "raw") when the source is a sidecar. */
  readonly encoding?: string;
}

/** Minimal structural contract shared by GgufReader and Wq4Reader. */
export interface VisionTensorSource {
  tensor(name: string): VisionTensorInfo | undefined;
  readTensor(name: string, offset?: number, length?: number): Promise<Uint8Array>;
}

/** IEEE-754 half → double conversion (same math as the WQ4 converter). */
export function decodeF16(u16: number): number {
  const exponent = (u16 >> 10) & 0x1f;
  const fraction = u16 & 0x03ff;
  const sign = u16 & 0x8000 ? -1 : 1;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

/**
 * Host-side dequant of one WQ4 v3 tensor into exact F32.
 *
 * The sidecar stores each matrix row-major in blocks of 32 elements, 5 u32
 * words per block: [0..3] four 8×4-bit signed nibble words (encoded -8..+7
 * as 0..15, element `w*8+n` in nibble `n` of word `w`), [4] a signed
 * power-of-two exponent `e` with scale 2^e. This is the inverse of
 * `convert_gguf_to_wq4.ts quantizeBlockWq4` and is bit-identical to the text
 * runtime's WGSL dequant (`scale = exp2(bitcast<i32>(word))`, `q - 8` per
 * nibble) — the packed block layout is shared, so host-dequant and in-shader
 * dequant cannot drift.
 */
export function dequantizeWq4Tensor(bytes: Uint8Array, count: number): Float32Array {
  if (count <= 0 || count % WQ4_BLOCK_SIZE !== 0) {
    throw new Error(`WQ4 tensor: element count ${count} must be a positive multiple of ${WQ4_BLOCK_SIZE}`);
  }
  if (bytes.byteLength % WQ4_BYTES_PER_BLOCK !== 0) {
    throw new Error(`WQ4 tensor: ${bytes.byteLength} bytes is not a whole number of ${WQ4_BYTES_PER_BLOCK}-byte blocks`);
  }
  const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  const blockCount = count / WQ4_BLOCK_SIZE;
  if (words.length !== blockCount * 5) {
    throw new Error(`WQ4 tensor: ${words.length} words != ${count} elements (${blockCount} blocks x 5 words)`);
  }
  const out = new Float32Array(count);
  for (let b = 0; b < blockCount; b++) {
    const wordBase = b * 5;
    // The exponent is stored as the u32 bit pattern of a signed i32.
    const scale = 2 ** (words[wordBase + 4]! | 0);
    const elemBase = b * WQ4_BLOCK_SIZE;
    for (let e = 0; e < WQ4_BLOCK_SIZE; e++) {
      const q = ((words[wordBase + (e >> 3)]! >>> ((e & 7) * 4)) & 0x0f) - 8;
      out[elemBase + e] = q * scale;
    }
  }
  return out;
}

/** Adapt a GgufReader or Wq4Reader to the minimal VisionTensorSource. */
export function visionTensorSource(reader: {
  tensor(name: string): {
    dimensions: readonly number[];
    type?: number;
    sourceType?: number;
    encoding?: string;
  } | undefined;
  readTensor(name: string, offset?: number, length?: number): Promise<Uint8Array>;
}): VisionTensorSource {
  return {
    tensor(name) {
      try {
        const info = reader.tensor(name);
        if (!info) return undefined;
        const sourceType = info.type !== undefined ? info.type : info.sourceType;
        return {
          dimensions: info.dimensions,
          ...(sourceType !== undefined ? { sourceType } : {}),
          ...(info.encoding !== undefined ? { encoding: info.encoding } : {}),
        };
      } catch {
        return undefined;
      }
    },
    readTensor: (name, offset, length) => reader.readTensor(name, offset, length),
  };
}

export class VisionWeights {
  private readonly cache = new Map<string, Float32Array>();

  constructor(
    private readonly source: VisionTensorSource,
    private readonly hiddenSize: number,
    private readonly feedForwardSize: number,
  ) {}

  private require(name: string, expectedDims?: readonly number[]): VisionTensorInfo {
    const info = this.source.tensor(name);
    if (!info) throw new Error(`Vision weight not found: ${name}`);
    if (expectedDims && info.dimensions.length !== expectedDims.length) {
      throw new Error(`${name}: expected ${expectedDims.length} dims, got [${info.dimensions.join("x")}]`);
    }
    if (expectedDims && expectedDims.some((d, i) => d !== info.dimensions[i])) {
      throw new Error(`${name}: expected [${expectedDims.join("x")}], got [${info.dimensions.join("x")}]`);
    }
    return info;
  }

  /** Decode a tensor into Float32Array (WQ4 dequant, F16 → F32, F32 pass-through). */
  private async loadF32(name: string, expectedDims?: readonly number[]): Promise<Float32Array> {
    const cached = this.cache.get(name);
    if (cached) return cached;
    const info = this.require(name, expectedDims);
    const bytes = await this.source.readTensor(name);
    let out: Float32Array;
    if (info.encoding === "wq4") {
      const count = info.dimensions.reduce((a, d) => a * d, 1);
      out = dequantizeWq4Tensor(bytes, count);
    } else {
      const type = info.sourceType;
      if (type === GgmlType.F32) {
        if (bytes.byteLength % 4 !== 0) throw new Error(`${name}: F32 tensor has ${bytes.byteLength} bytes`);
        out = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
      } else if (type === GgmlType.F16) {
        if (bytes.byteLength % 2 !== 0) throw new Error(`${name}: F16 tensor has ${bytes.byteLength} bytes`);
        const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
        out = new Float32Array(u16.length);
        for (let i = 0; i < u16.length; i++) out[i] = decodeF16(u16[i]!);
      } else {
        const label = type !== undefined ? (GgmlType[type] ?? String(type)) : "unknown";
        throw new Error(`${name}: unsupported vision source type ${label}`);
      }
    }
    this.cache.set(name, out);
    return out;
  }

  /** 1-D parameter vector (norms, biases) with shape validation. */
  async vector(name: string, length: number): Promise<Float32Array> {
    return this.loadF32(name, [length]);
  }

  /**
   * Linear weight matrix. GGUF dims [input, output] → returned row-major
   * `[output * input]` (row = output neuron, column = input).
   */
  async linear(name: string, inputDim: number, outputDim: number): Promise<Float32Array> {
    const data = await this.loadF32(name, [inputDim, outputDim]);
    return data;
  }

  /**
   * Patch embedding: the GGUF conv kernel [16,16,3,hidden] is stored with the
   * output channel as the slowest dim, so its flat F16 data is exactly the
   * `[hidden, 3*16*16]` matmul weight (input order (c,h,w), w fastest).
   */
  async patchEmbedding(patchSize: number): Promise<Float32Array> {
    return this.loadF32("v.patch_embd.weight", [patchSize, patchSize, 3, this.hiddenSize]);
  }

  /**
   * Patch embedding bias: [hidden]. The HF tower uses nn.Linear (bias=True),
   * and the mmproj carries it as an F32 tensor — the reference and the tower
   * MUST add it or every image is off by a per-channel constant (a bug that
   * kept CPU==GPU consistent while degrading perception).
   */
  async patchEmbeddingBias(): Promise<Float32Array> {
    return this.vector("v.patch_embd.bias", this.hiddenSize);
  }

  /**
   * Learned position embedding grid: [grid*grid, hidden] row-major over the
   * grid (position slowest, hidden dim fastest), exactly what the processor's
   * resizePositionEmbedding consumes.
   *
   * GGUF stores the tensor with dims [hidden, grid*grid] and row-major data,
   * i.e. element (p, d) at offset `p * hidden + d` (hidden is the fastest
   * index — same as torch's [grid*grid, hidden] row-major). NO transpose is
   * needed: reading it as [d, p] (data[d*n + p]) swaps grid positions for
   * hidden dims and yields a degenerate grid, a bug that kept the tower
   * self-consistent (CPU oracle == WGSL, differential tests green) while the
   * model could not localize any patch content.
   */
  async positionEmbedding(gridSide: number): Promise<Float32Array> {
    const n = gridSide * gridSide;
    return this.loadF32("v.position_embd.weight", [this.hiddenSize, n]);
  }

  /** LayerNorm weight/bias pair. */
  async layerNorm(prefix: string): Promise<{ weight: Float32Array; bias: Float32Array }> {
    const [weight, bias] = await Promise.all([
      this.vector(`${prefix}.weight`, this.hiddenSize),
      this.vector(`${prefix}.bias`, this.hiddenSize),
    ]);
    return { weight, bias };
  }

  /** Load every tensor of one vision block, validated against the config. */
  async block(layer: number): Promise<VisionBlockWeights> {
    const base = `v.blk.${layer}`;
    const [ln1, ln2] = await Promise.all([this.layerNorm(`${base}.ln1`), this.layerNorm(`${base}.ln2`)]);
    const [q, k, v, o] = await Promise.all([
      this.linear(`${base}.attn_q.weight`, this.hiddenSize, this.hiddenSize),
      this.linear(`${base}.attn_k.weight`, this.hiddenSize, this.hiddenSize),
      this.linear(`${base}.attn_v.weight`, this.hiddenSize, this.hiddenSize),
      this.linear(`${base}.attn_out.weight`, this.hiddenSize, this.hiddenSize),
    ]);
    const [qBias, kBias, vBias, oBias] = await Promise.all([
      this.vector(`${base}.attn_q.bias`, this.hiddenSize),
      this.vector(`${base}.attn_k.bias`, this.hiddenSize),
      this.vector(`${base}.attn_v.bias`, this.hiddenSize),
      this.vector(`${base}.attn_out.bias`, this.hiddenSize),
    ]);
    const [up, down] = await Promise.all([
      this.linear(`${base}.ffn_up.weight`, this.hiddenSize, this.feedForwardSize),
      this.linear(`${base}.ffn_down.weight`, this.feedForwardSize, this.hiddenSize),
    ]);
    const [upBias, downBias] = await Promise.all([
      this.vector(`${base}.ffn_up.bias`, this.feedForwardSize),
      this.vector(`${base}.ffn_down.bias`, this.hiddenSize),
    ]);
    return {
      ln1,
      ln2,
      q, k, v, o,
      qBias, kBias, vBias, oBias,
      up, down,
      upBias, downBias,
    };
  }

  /** Projector: mm.1 (unshuffled → projector hidden), mm.2 (projector hidden → projector hidden). */
  async projector(
    projectorHiddenSize: number,
    scaleFactor: number,
  ): Promise<{ mm1: Float32Array; mm1Bias: Float32Array; mm2: Float32Array; mm2Bias: Float32Array }> {
    const unshuffled = this.hiddenSize * scaleFactor * scaleFactor;
    const [mm1, mm1Bias, mm2, mm2Bias] = await Promise.all([
      this.linear("mm.1.weight", unshuffled, projectorHiddenSize),
      this.vector("mm.1.bias", projectorHiddenSize),
      this.linear("mm.2.weight", projectorHiddenSize, projectorHiddenSize),
      this.vector("mm.2.bias", projectorHiddenSize),
    ]);
    return { mm1, mm1Bias, mm2, mm2Bias };
  }
}

export interface VisionBlockWeights {
  readonly ln1: { weight: Float32Array; bias: Float32Array };
  readonly ln2: { weight: Float32Array; bias: Float32Array };
  readonly q: Float32Array;
  readonly k: Float32Array;
  readonly v: Float32Array;
  readonly o: Float32Array;
  readonly qBias: Float32Array;
  readonly kBias: Float32Array;
  readonly vBias: Float32Array;
  readonly oBias: Float32Array;
  readonly up: Float32Array;
  readonly down: Float32Array;
  readonly upBias: Float32Array;
  readonly downBias: Float32Array;
}

/** Load the full vision weight set (patch emb, pos emb, blocks, post ln, projector). */
export async function loadVisionWeights(
  source: VisionTensorSource,
  config: {
    hiddenSize: number;
    feedForwardSize: number;
    blockCount: number;
    positionEmbeddingGrid: number;
    patchSize: number;
    projectorHiddenSize: number;
    projectorScaleFactor: number;
  },
): Promise<{
  patchEmb: Float32Array;
  patchEmbBias: Float32Array;
  posEmb: Float32Array;
  postLn: { weight: Float32Array; bias: Float32Array };
  blocks: VisionBlockWeights[];
  projector: { mm1: Float32Array; mm1Bias: Float32Array; mm2: Float32Array; mm2Bias: Float32Array };
}> {
  const weights = new VisionWeights(source, config.hiddenSize, config.feedForwardSize);
  const [patchEmb, patchEmbBias, posEmb, postLn, projector] = await Promise.all([
    weights.patchEmbedding(config.patchSize),
    weights.patchEmbeddingBias(),
    weights.positionEmbedding(config.positionEmbeddingGrid),
    weights.layerNorm("v.post_ln"),
    weights.projector(config.projectorHiddenSize, config.projectorScaleFactor),
  ]);
  const blocks: VisionBlockWeights[] = [];
  for (let layer = 0; layer < config.blockCount; layer++) {
    blocks.push(await weights.block(layer));
  }
  return { patchEmb, patchEmbBias, posEmb, postLn, blocks, projector };
}

