/**
 * Exact CPU reference implementation of the LFM2.5-VL vision tower + projector.
 *
 * This is the differential-test oracle for the future WGSL implementation
 * (ADA-0009): it reproduces the HF `Siglip2VisionModel` + `Lfm2Vl` projector
 * math in plain TypeScript over F16 source weights, so quantization/kernel
 * error can be measured separately from semantic correctness.
 *
 * Math notes (all verified against `modeling_siglip2.py` and the mmproj
 * layout in docs/decisions/0009-vision-language-integration.md):
 * - patch embedding: conv kernel read as a [hidden, 3*patch^2] matmul,
 * - learned 16×16 position embeddings resized bilinearly (align_corners=False)
 *   to the image grid (host-side, in the processor),
 * - pre-LayerNorm blocks with biases everywhere, bidirectional attention
 *   (padding mask only, no RoPE, scale = head_dim^-0.5),
 * - MLP activation: tanh-approximated GELU (`gelu_pytorch_tanh`),
 * - post LayerNorm, pixel-unshuffle (torch `PixelUnshuffle` channel order),
 * - projector: Linear → GELU (exact erf) → Linear (biases).
 */

import type { VisionConfig } from "./config.ts";
import type { VisionBlockWeights } from "./weights.ts";

export interface VisionReferenceInput {
  /**
   * Patch tokens [patchCount, 3*patchSize^2] in (c,h,w) w-fastest order.
   *
   * patchCount is the FULL grid size (gridH*gridW): masked padding rows are
   * present with zero patches (and zero posEmb) — exactly what the processor
   * emits — so the pixel-unshuffle tail can gather every grid position from a
   * fully computed hidden buffer.
   */
  readonly patches: Float32Array;
  /** Resized position embeddings [patchCount, hiddenSize]; zero for masked rows. */
  readonly posEmb: Float32Array;
  /** Full grid row count (must equal gridH*gridW). */
  readonly patchCount: number;
  readonly gridH: number;
  readonly gridW: number;
  /** [patchCount], 1 = valid. All-valid by default. */
  readonly paddingMask?: Uint8Array;
}

export interface VisionReferenceResult {
  /** Image embeddings [tokens, projectorHiddenSize]. */
  readonly embeddings: Float32Array;
  /** Number of output image tokens. */
  readonly tokens: number;
}

export interface VisionReferenceWeights {
  readonly patchEmb: Float32Array;
  /** Patch embedding bias [hidden] (HF nn.Linear bias; the mmproj carries it as F32). */
  readonly patchEmbBias: Float32Array;
  readonly posEmb: Float32Array;
  readonly postLn: { weight: Float32Array; bias: Float32Array };
  readonly blocks: readonly VisionBlockWeights[];
  readonly projector: {
    readonly mm1: Float32Array;
    readonly mm1Bias: Float32Array;
    readonly mm2: Float32Array;
    readonly mm2Bias: Float32Array;
  };
}

export function geluTanh(x: number): number {
  return 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

/** Abramowitz–Stegun 7.1.26 erf approximation (max abs error ~1.5e-7). */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t *
      Math.exp(-ax * ax);
  return sign * y;
}

export function geluErf(x: number): number {
  return 0.5 * x * (1 + erf(x * 0.7071067811865476));
}

/** y[i] = (x[i] - mean)/sqrt(var + eps) * w[i] + b[i], row at `offset`. */
export function layerNormInto(
  x: Float32Array,
  offset: number,
  w: Float32Array,
  b: Float32Array,
  eps: number,
  dim: number,
  out: Float32Array,
  outOffset: number,
): void {
  let mean = 0;
  for (let i = 0; i < dim; i++) mean += x[offset + i]!;
  mean /= dim;
  let variance = 0;
  for (let i = 0; i < dim; i++) {
    const d = x[offset + i]! - mean;
    variance += d * d;
  }
  variance /= dim;
  const inv = 1 / Math.sqrt(variance + eps);
  for (let i = 0; i < dim; i++) {
    out[outOffset + i] = (x[offset + i]! - mean) * inv * w[i]! + b[i]!;
  }
}

/** out[offset+o] = bias[o] + sum_i x[xOffset+i] * w[o*inputDim + i]. */
export function linearInto(
  x: Float32Array,
  xOffset: number,
  w: Float32Array,
  bias: Float32Array,
  inputDim: number,
  outputDim: number,
  out: Float32Array,
  outOffset: number,
): void {
  for (let o = 0; o < outputDim; o++) {
    let acc = bias[o]!;
    const row = o * inputDim;
    for (let i = 0; i < inputDim; i++) acc += x[xOffset + i]! * w[row + i]!;
    out[outOffset + o] = acc;
  }
}

/**
 * Bidirectional multi-head attention. `qkv` holds per-token [dim] rows; the
 * per-head attention output accumulates into `attn`, then the output
 * projection is written into `h` (which the caller adds to the residual).
 */
export function attentionInto(
  qkv: Float32Array,
  block: VisionBlockWeights,
  attn: Float32Array,
  h: Float32Array,
  n: number,
  heads: number,
  headDim: number,
  scale: number,
  paddingMask: Uint8Array | undefined,
): void {
  const dim = heads * headDim;
  const qOff = 0;
  const kOff = n * dim;
  const vOff = 2 * n * dim;
  const scores = new Float32Array(n);
  attn.fill(0);

  for (let head = 0; head < heads; head++) {
    const hOff = head * headDim;
    for (let a = 0; a < n; a++) {
      if (paddingMask && paddingMask[a] === 0) continue;
      let max = Number.NEGATIVE_INFINITY;
      for (let b = 0; b < n; b++) {
        let dot = 0;
        for (let d = 0; d < headDim; d++) {
          dot += qkv[a * dim + hOff + d]! * qkv[kOff + b * dim + hOff + d]!;
        }
        const s = dot * scale;
        scores[b] = paddingMask && paddingMask[b] === 0 ? Number.NEGATIVE_INFINITY : s;
        if (scores[b]! > max) max = scores[b]!;
      }
      // Guard: a query whose keys are all padded must not produce NaN.
      // (v0 has no padding, but the GPU mask path will be tested against this
      // oracle, so keep it well-defined.)
      if (max === Number.NEGATIVE_INFINITY) continue;
      let sum = 0;
      for (let b = 0; b < n; b++) {
        const e = Math.exp(scores[b]! - max);
        scores[b] = e;
        sum += e;
      }
      const inv = 1 / sum;
      for (let d = 0; d < headDim; d++) {
        let acc = 0;
        for (let b = 0; b < n; b++) {
          if (paddingMask && paddingMask[b] === 0) continue;
          acc += scores[b]! * qkv[vOff + b * dim + hOff + d]!;
        }
        attn[a * dim + hOff + d] = acc * inv;
      }
    }
  }

  // output projection: h[a] = oBias + attn[a] @ Wo^T
  for (let a = 0; a < n; a++) {
    const row = a * dim;
    for (let o = 0; o < dim; o++) {
      let acc = block.oBias[o]!;
      const wRow = o * dim;
      for (let d = 0; d < dim; d++) acc += attn[row + d]! * block.o[wRow + d]!;
      h[row + o] = acc;
    }
  }
}

export function forwardVision(
  w: VisionReferenceWeights,
  config: VisionConfig,
  input: VisionReferenceInput,
): VisionReferenceResult {
  const { patchCount, gridH, gridW } = input;
  const dim = config.hiddenSize;
  const patchDim = 3 * config.patchSize * config.patchSize;
  const heads = config.attentionHeads;
  const headDim = config.headDim;
  const scale = Math.pow(headDim, -0.5);
  const ff = config.feedForwardSize;
  if (patchCount !== gridH * gridW) {
    throw new Error(`Vision reference: patchCount ${patchCount} != grid ${gridH}x${gridW}`);
  }
  if (input.patches.length < patchCount * patchDim) throw new Error("Vision reference: patches buffer too small");
  if (input.posEmb.length < patchCount * dim) throw new Error("Vision reference: posEmb buffer too small");

  // --- patch embedding + position embeddings -------------------------------
  const hidden = new Float32Array(patchCount * dim);
  const patchEmb = w.patchEmb;
  for (let p = 0; p < patchCount; p++) {
    const pin = p * patchDim;
    const pout = p * dim;
    for (let o = 0; o < dim; o++) {
      let acc = w.patchEmbBias[o]!;
      const row = o * patchDim;
      for (let i = 0; i < patchDim; i++) acc += input.patches[pin + i]! * patchEmb[row + i]!;
      hidden[pout + o] = acc + input.posEmb[pout + o]!;
    }
  }

  // --- transformer blocks --------------------------------------------------
  // qkv packs [q | k | v] rows per token; attn accumulates per-head output;
  // h is reused as the layerNorm scratch and the attention output row.
  const qkv = new Float32Array(3 * patchCount * dim);
  const attn = new Float32Array(patchCount * dim);
  const h = new Float32Array(patchCount * dim);
  const mlpScratch = new Float32Array(Math.max(dim, ff));

  for (let layer = 0; layer < config.blockCount; layer++) {
    const block = w.blocks[layer]!;
    const inputHidden = hidden;

    for (let p = 0; p < patchCount; p++) {
      const base = p * dim;
      layerNormInto(inputHidden, base, block.ln1.weight, block.ln1.bias, config.layerNormEpsilon, dim, h, base);
      linearInto(h, base, block.q, block.qBias, dim, dim, qkv, base);
      linearInto(h, base, block.k, block.kBias, dim, dim, qkv, patchCount * dim + base);
      linearInto(h, base, block.v, block.vBias, dim, dim, qkv, 2 * patchCount * dim + base);
    }

    attentionInto(qkv, block, attn, h, patchCount, heads, headDim, scale, input.paddingMask);

    // residual after attention
    for (let i = 0; i < patchCount * dim; i++) hidden[i] = inputHidden[i]! + h[i]!;

    // ln2 → MLP (up → gelu → down) → residual
    for (let p = 0; p < patchCount; p++) {
      const base = p * dim;
      layerNormInto(hidden, base, block.ln2.weight, block.ln2.bias, config.layerNormEpsilon, dim, h, base);
      linearInto(h, base, block.up, block.upBias, dim, ff, mlpScratch, 0);
      for (let i = 0; i < ff; i++) mlpScratch[i] = geluTanh(mlpScratch[i]!);
      linearInto(mlpScratch, 0, block.down, block.downBias, ff, dim, h, base);
      for (let i = 0; i < dim; i++) {
        hidden[base + i] = hidden[base + i]! + h[base + i]!;
      }
    }
  }

  // --- post LayerNorm ------------------------------------------------------
  for (let p = 0; p < patchCount; p++) {
    const base = p * dim;
    layerNormInto(hidden, base, w.postLn.weight, w.postLn.bias, config.layerNormEpsilon, dim, h, base);
    for (let i = 0; i < dim; i++) hidden[base + i] = h[base + i]!;
  }

  // --- pixel unshuffle ---
  // Channel order follows the mmproj/llama.cpp convention (verified against
  // llama.cpp ground truth, 2026-08-09): for factor 2 the unshuffled channel is
  //   c + dim*i + dim*factor*j   (i = vertical sub-pixel, j = horizontal)
  // — the sub-pixel bits are BLOCKED at the top of the channel space, NOT
  // interleaved like torch's PixelUnshuffle (c*4 + i*2 + j). The GGUF mm.1
  // columns are stored in this order, so torch's order produces garbage
  // embeddings that still differ per image (the model sees "something" but
  // never the actual content).
  const factor = config.projectorScaleFactor;
  const outH = gridH / factor;
  const outW = gridW / factor;
  if (!Number.isInteger(outH) || !Number.isInteger(outW)) {
    throw new Error(`Vision reference: grid ${gridH}x${gridW} not divisible by unshuffle factor ${factor}`);
  }
  const tokens = outH * outW;
  const unshuffledDim = dim * factor * factor;
  const unshuffled = new Float32Array(tokens * unshuffledDim);
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      const outBase = (oy * outW + ox) * unshuffledDim;
      for (let c = 0; c < dim; c++) {
        for (let i = 0; i < factor; i++) {
          for (let j = 0; j < factor; j++) {
            const srcIndex = ((oy * factor + i) * gridW + (ox * factor + j)) * dim + c;
            const dstIndex = outBase + c + dim * i + dim * factor * j;
            unshuffled[dstIndex] = hidden[srcIndex]!;
          }
        }
      }
    }
  }

  // --- projector: mm.1 → GELU → mm.2 ---------------------------------------
  const { mm1, mm1Bias, mm2, mm2Bias } = w.projector;
  const projectorHidden = config.projectorHiddenSize;
  const projected = new Float32Array(tokens * projectorHidden);
  const mid = new Float32Array(projectorHidden);
  for (let t = 0; t < tokens; t++) {
    const srcBase = t * unshuffledDim;
    const outBase = t * projectorHidden;
    for (let o = 0; o < projectorHidden; o++) {
      let acc = mm1Bias[o]!;
      const row = o * unshuffledDim;
      for (let i = 0; i < unshuffledDim; i++) acc += unshuffled[srcBase + i]! * mm1[row + i]!;
      mid[o] = geluErf(acc);
    }
    for (let o = 0; o < projectorHidden; o++) {
      let acc = mm2Bias[o]!;
      const row = o * projectorHidden;
      for (let i = 0; i < projectorHidden; i++) acc += mid[i]! * mm2[row + i]!;
      projected[outBase + o] = acc;
    }
  }

  return { embeddings: projected, tokens };
}
