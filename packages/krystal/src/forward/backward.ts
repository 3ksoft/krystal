/**
 * CPU backward references for the M3 Krystal backward operators (plan §17
 * order). Each function mirrors one GPU shader's contract exactly, so the GPU
 * output can be compared within the declared f32 tolerance and gradients can
 * be finite-difference checked against a forward-only loss.
 *
 *   relu_backward                      -> reluBackward
 *   krystal_attention_backward_scores  -> attentionBackwardScores
 *   krystal_attention_backward_qkv     -> attentionBackwardQkv
 *   krystal_field_embed_backward       -> fieldEmbedBackward
 */
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import { embeddingTableBases, type BrainForwardConfig } from "./model.ts";
import type { ActiveFrame } from "./masks.ts";

/** dIn[i] = (out[i] > 0) ? dOut[i] : 0. Mirrors relu_backward.wgsl. */
export function reluBackward(out: Float32Array, dOut: Float32Array): Float32Array {
  if (out.length !== dOut.length) throw new Error("relu backward length mismatch");
  const dIn = new Float32Array(out.length);
  for (let i = 0; i < out.length; i++) dIn[i] = out[i]! > 0 ? dOut[i]! : 0;
  return dIn;
}

/**
 * Softmax-score gradient of masked multi-head attention (cross-capable).
 * Mirrors krystal_attention_backward_scores.wgsl:
 *   dScores[h][i][j] = P[h][i][j] * (dP[h][i][j] - rowSum[h][i])
 * with dP = dot(dOut[i,h], V[j,h]) and rowSum = sum_j P*dP.
 * Layouts: dOut [qRows,H], V [kRows,H], P/dScores [heads,qRows,kRows].
 */
export function attentionBackwardScores(
  dOut: Float32Array,
  v: Float32Array,
  p: Float32Array,
  qRows: number,
  kRows: number,
  heads: number,
  headDim: number,
): Float32Array {
  const h = heads * headDim;
  const dScores = new Float32Array(heads * qRows * kRows);
  for (let head = 0; head < heads; head++) {
    for (let i = 0; i < qRows; i++) {
      const dP = new Float32Array(kRows);
      let rowSum = 0;
      for (let j = 0; j < kRows; j++) {
        let dp = 0;
        for (let d = 0; d < headDim; d++) {
          dp += dOut[i * h + head * headDim + d]! * v[j * h + head * headDim + d]!;
        }
        dP[j] = dp;
        rowSum += p[(head * qRows + i) * kRows + j]! * dp;
      }
      for (let j = 0; j < kRows; j++) {
        dScores[(head * qRows + i) * kRows + j] =
          p[(head * qRows + i) * kRows + j]! * (dP[j]! - rowSum);
      }
    }
  }
  return dScores;
}

/**
 * Q/K/V gradients of masked multi-head attention (cross-capable). Mirrors
 * krystal_attention_backward_qkv.wgsl:
 *   dQ[i,h,d] = scale * sum_j dScores[h,i,j] * K[j,h,d]
 *   dK[j,h,d] = scale * sum_i dScores[h,i,j] * Q[i,h,d]
 *   dV[j,h,d] =        sum_i P[h,i,j] * dOut[i,h,d]
 * Layouts: Q/dOut [qRows,H], K/V [kRows,H], dScores/P [heads,qRows,kRows],
 * dQ [qRows,H], dK/dV [kRows,H]. scale = 1/sqrt(headDim).
 */
export function attentionBackwardQkv(
  dScores: Float32Array,
  q: Float32Array,
  k: Float32Array,
  p: Float32Array,
  dOut: Float32Array,
  qRows: number,
  kRows: number,
  heads: number,
  headDim: number,
): { dQ: Float32Array; dK: Float32Array; dV: Float32Array } {
  const h = heads * headDim;
  const scale = 1 / Math.sqrt(headDim);
  const dQ = new Float32Array(qRows * h);
  const dK = new Float32Array(kRows * h);
  const dV = new Float32Array(kRows * h);
  for (let i = 0; i < qRows; i++) {
    for (let col = 0; col < h; col++) {
      const head = Math.floor(col / headDim);
      let acc = 0;
      for (let j = 0; j < kRows; j++) {
        acc += dScores[(head * qRows + i) * kRows + j]! * k[j * h + col]!;
      }
      dQ[i * h + col] = acc * scale;
    }
  }
  for (let j = 0; j < kRows; j++) {
    for (let col = 0; col < h; col++) {
      const head = Math.floor(col / headDim);
      let acc = 0;
      for (let i = 0; i < qRows; i++) {
        acc += dScores[(head * qRows + i) * kRows + j]! * q[i * h + col]!;
      }
      dK[j * h + col] = acc * scale;
    }
  }
  for (let j = 0; j < kRows; j++) {
    for (let col = 0; col < h; col++) {
      const head = Math.floor(col / headDim);
      let acc = 0;
      for (let i = 0; i < qRows; i++) {
        acc += p[(head * qRows + i) * kRows + j]! * dOut[i * h + col]!;
      }
      dV[j * h + col] = acc;
    }
  }
  return { dQ, dK, dV };
}

/**
 * Scatter-add gradient of the Krystal field embedding. Mirrors
 * krystal_field_embed_backward.wgsl: dEmbedding[table,row,h] = sum over active
 * tokens whose per-table index == row of dFieldStates[t,h]. The result is the
 * full concatenated embeddings page layout (see embeddingTableBases).
 */
export function fieldEmbedBackward(
  frame: v1_0_0.BrainFrameGpu,
  active: ActiveFrame,
  dFieldStates: Float32Array,
  config: BrainForwardConfig,
): Float32Array {
  const { hiddenSize: h, tokenSpace, fieldSpace, schemaSpace, bandSpace, streamSpace, posSpace } = config;
  const bases = embeddingTableBases(config);
  const tableRows = [tokenSpace, fieldSpace, schemaSpace, bandSpace, streamSpace, posSpace];
  const tableBases = [bases.token, bases.field, bases.schema, bases.band, bases.stream, bases.pos];
  const totalRows = tableRows.reduce((a, b) => a + b, 0);
  const dEmbedding = new Float32Array(totalRows * h);
  const indexOf = (t: number, tableId: number): number => {
    const frameTok = active.activeTokens[t]!;
    const slot = frameTok >> 3; // record width is the frozen ABI value 8
    switch (tableId) {
      case 0: return frame.tokenIds[frameTok]!;
      case 1: return frame.fieldRoles[frameTok]!;
      case 2: return frame.schemaIds[slot]!;
      case 3: return frame.bandIds[slot]!;
      case 4: return active.streamIds[slot]!;
      default: return frameTok & 7;
    }
  };
  for (let tableId = 0; tableId < 6; tableId++) {
    const base = tableBases[tableId]!;
    for (let row = 0; row < tableRows[tableId]!; row++) {
      for (let d = 0; d < h; d++) {
        let sum = 0;
        for (let t = 0; t < active.activeTokens.length; t++) {
          if (indexOf(t, tableId) === row) sum += dFieldStates[t * h + d]!;
        }
        dEmbedding[base + row * h + d] = sum;
      }
    }
  }
  return dEmbedding;
}
