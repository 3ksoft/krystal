/**
 * Composed CPU forward reference for the Krystal record/query encoder and
 * mixer (M2b, concerns answer 22: CPU references per operator plus one small
 * composed CPU forward for parity). This is the oracle the WebGPU path is
 * compared against; it implements exactly the same math as the shaders.
 *
 *   field embed (5 additive + record-local position)
 *   -> 2 local encoder blocks (block-diagonal masked self-attention + ReLU FFN)
 *   -> learned-query pooling -> record bank (keys/values) and query states
 *   -> 2 mixer blocks (query -> bank cross-attention + ReLU FFN)
 */
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import type {
  ActiveFrame,
} from "./masks.ts";
import {
  BRAIN_FORWARD_CONFIG,
  embeddingTableBases,
  type BlockWeights,
  type BrainForwardConfig,
  type BrainForwardWeights,
} from "./model.ts";

export interface BrainForwardResult {
  /** [T_active, H] encoded field states after the record encoder. */
  readonly fieldStates: Float32Array;
  /** [R, H] record bank keys and values (pooled, post-encoder). */
  readonly bankKeys: Float32Array;
  readonly bankValues: Float32Array;
  /** [Q, H] pooled query states (before the mixer). */
  readonly queryKeys: Float32Array;
  readonly queryValues: Float32Array;
  /** [Q, H] mixed query output after the mixer blocks. */
  readonly queryOutput: Float32Array;
}

const NEG_INF = -1e30;

export function softmaxRow(values: Float32Array | number[], start: number, count: number): void {
  let max = -Infinity;
  for (let i = 0; i < count; i++) max = Math.max(max, values[start + i]!);
  let sum = 0;
  for (let i = 0; i < count; i++) {
    const e = Math.exp(values[start + i]! - max);
    values[start + i] = e;
    sum += e;
  }
  const inv = 1 / Math.max(sum, 1e-20);
  for (let i = 0; i < count; i++) values[start + i] = values[start + i]! * inv;
}

/** Multi-head masked attention; Q [q,H], K/V [k,H], out [q,H]. */
export function attentionOracle(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  mask: Float32Array,
  qRows: number,
  kRows: number,
  h: number,
  heads: number,
  headDim: number,
): Float32Array {
  const out = new Float32Array(qRows * h);
  const scores = new Float32Array(kRows);
  const scale = 1 / Math.sqrt(headDim);
  for (let head = 0; head < heads; head++) {
    const hb = head * headDim;
    for (let i = 0; i < qRows; i++) {
      for (let j = 0; j < kRows; j++) {
        let s = 0;
        for (let d = 0; d < headDim; d++) {
          s += q[i * h + hb + d]! * k[j * h + hb + d]!;
        }
        scores[j] = s * scale + mask[i * kRows + j]!;
      }
      softmaxRow(scores, 0, kRows);
      for (let d = 0; d < headDim; d++) {
        let value = 0;
        for (let j = 0; j < kRows; j++) {
          value += scores[j]! * v[j * h + hb + d]!;
        }
        out[i * h + hb + d] = value;
      }
    }
  }
  return out;
}

export function reluOracle(values: Float32Array): Float32Array {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = Math.max(0, values[i]!);
  return out;
}

/** y = x @ W^T with W [outDim, inDim] (matmul_f32 layout). */
export function matmulOracle(x: Float32Array, w: Float32Array, outDim: number, inDim: number): Float32Array {
  const rows = x.length / inDim;
  const out = new Float32Array(rows * outDim);
  for (let m = 0; m < rows; m++) {
    for (let o = 0; o < outDim; o++) {
      let value = 0;
      for (let d = 0; d < inDim; d++) value += x[m * inDim + d]! * w[o * inDim + d]!;
      out[m * outDim + o] = value;
    }
  }
  return out;
}

export function ffnOracle(
  x: Float32Array,
  block: BlockWeights,
  h: number,
  ffn: number,
): Float32Array {
  const h1 = reluOracle(matmulOracle(x, block.w1, ffn, h));
  return matmulOracle(h1, block.w2, h, ffn);
}

export function addInPlace(a: Float32Array, b: Float32Array): void {
  for (let i = 0; i < a.length; i++) a[i] = a[i]! + b[i]!;
}

function fieldEmbed(
  frame: v1_0_0.BrainFrameGpu,
  active: ActiveFrame,
  config: BrainForwardConfig,
  weights: BrainForwardWeights,
): Float32Array {
  const { hiddenSize: h } = config;
  const bases = embeddingTableBases(config);
  const states = new Float32Array(active.activeTokens.length * h);
  const table = weights.embeddings;
  for (let t = 0; t < active.activeTokens.length; t++) {
    const frameTok = active.activeTokens[t]!;
    const slot = frameTok >> 3; // record width is the frozen ABI value 8
    const local = frameTok & 7;
    const tok = frame.tokenIds[frameTok]!;
    const role = frame.fieldRoles[frameTok]!;
    const schema = frame.schemaIds[slot]!;
    const band = frame.bandIds[slot]!;
    const stream = active.streamIds[slot]!;
    for (let d = 0; d < h; d++) {
      let value = 0;
      value += table[bases.token + tok * h + d]!;
      value += table[bases.field + role * h + d]!;
      value += table[bases.schema + schema * h + d]!;
      value += table[bases.band + band * h + d]!;
      value += table[bases.stream + stream * h + d]!;
      value += table[bases.pos + local * h + d]!;
      states[t * h + d] = value;
    }
  }
  return states;
}

function poolRecord(
  fieldStates: Float32Array,
  start: number,
  count: number,
  pool: Float32Array,
  h: number,
): { key: Float32Array; value: Float32Array } {
  const key = new Float32Array(h);
  const value = new Float32Array(h);
  if (count === 0) return { key, value };
  const scale = 1 / Math.sqrt(h);
  const keyScores = new Float32Array(count);
  const valueScores = new Float32Array(count);
  for (let j = 0; j < count; j++) {
    let ks = 0;
    let vs = 0;
    for (let d = 0; d < h; d++) {
      const s = fieldStates[(start + j) * h + d]!;
      ks += pool[d]! * s;
      vs += pool[h + d]! * s;
    }
    keyScores[j] = ks * scale;
    valueScores[j] = vs * scale;
  }
  softmaxRow(keyScores, 0, count);
  softmaxRow(valueScores, 0, count);
  for (let d = 0; d < h; d++) {
    let kAcc = 0;
    let vAcc = 0;
    for (let j = 0; j < count; j++) {
      const s = fieldStates[(start + j) * h + d]!;
      kAcc += keyScores[j]! * s;
      vAcc += valueScores[j]! * s;
    }
    key[d] = kAcc;
    value[d] = vAcc;
  }
  return { key, value };
}

/**
 * Composed CPU forward. Takes the packed SoA frame and its compiled active
 * lists; the GPU runner consumes the same inputs for a like-for-like parity
 * comparison.
 */
export function brainForwardOracle(
  frame: v1_0_0.BrainFrameGpu,
  active: ActiveFrame,
  weights: BrainForwardWeights,
  config: BrainForwardConfig = BRAIN_FORWARD_CONFIG,
  recordMask: Float32Array,
  mixerMask: Float32Array,
): BrainForwardResult {
  const { hiddenSize: h, ffnSize: ffn, headCount: heads, headDim, encoderBlocks, mixerBlocks } = config;
  const t = active.activeTokens.length;

  // 1. Field embedding.
  let x = fieldEmbed(frame, active, config, weights);

  // 2. Encoder blocks (QKV projections, record-local masked self-attention,
  // ReLU FFN). Mirrors the GPU dispatches exactly.
  for (let b = 0; b < encoderBlocks; b++) {
    const block = weights.enc[b]!;
    const q = matmulOracle(x, block.wq, h, h);
    const k = matmulOracle(x, block.wk, h, h);
    const v = matmulOracle(x, block.wv, h, h);
    const attn = attentionOracle(q, k, v, recordMask, t, t, h, heads, headDim);
    addInPlace(x, attn);
    const ff = ffnOracle(x, block, h, ffn);
    addInPlace(x, ff);
  }

  // 3. Learned-query pooling -> bank + query states.
  const r = active.bankRecords.length;
  const q = active.queryRecords.length;
  const bankKeys = new Float32Array(r * h);
  const bankValues = new Float32Array(r * h);
  for (let i = 0; i < r; i++) {
    const slot = active.bankRecords[i]!;
    const start = active.recordCompactOffset[slot]!;
    const count = active.recordCompactCount[slot]!;
    const { key, value } = poolRecord(x, start, count, weights.pool, h);
    bankKeys.set(key, i * h);
    bankValues.set(value, i * h);
  }
  const queryKeys = new Float32Array(q * h);
  const queryValues = new Float32Array(q * h);
  for (let i = 0; i < q; i++) {
    const slot = active.queryRecords[i]!;
    const start = active.recordCompactOffset[slot]!;
    const count = active.recordCompactCount[slot]!;
    const { key, value } = poolRecord(x, start, count, weights.pool, h);
    queryKeys.set(key, i * h);
    queryValues.set(value, i * h);
  }

  // 4. Mixer blocks (query -> bank cross-attention + ReLU FFN).
  let query = queryValues;
  for (let b = 0; b < mixerBlocks; b++) {
    const block = weights.mixer[b]!;
    const qProj = matmulOracle(query, block.wq, h, h);
    const kProj = matmulOracle(bankKeys, block.wk, h, h);
    const vProj = matmulOracle(bankValues, block.wv, h, h);
    const attn = attentionOracle(qProj, kProj, vProj, mixerMask, q, r, h, heads, headDim);
    addInPlace(query, attn);
    const ff = ffnOracle(query, block, h, ffn);
    addInPlace(query, ff);
  }

  return { fieldStates: x, bankKeys, bankValues, queryKeys, queryValues, queryOutput: query };
}

export { NEG_INF };
