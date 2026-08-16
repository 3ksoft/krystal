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
import { softmaxRow } from "./oracle.ts";

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
 * Learned-query pooling backward (record mixer training path, §17 item 7
 * second half). Mirrors krystal_pool_backward.wgsl + krystal_pool_dpool.wgsl:
 * given upstream dKeys/dValues per record, compute the fieldStates gradient
 * and the shared pool-query gradient.
 *
 *   key[r,h]   = sum_j pk[j] * s[j,h],   pk = softmax(qk . s_j / sqrt(H))
 *   value[r,h] = sum_j pv[j] * s[j,h],   pv = softmax(qv . s_j / sqrt(H))
 *
 *   dPk[j]     = dot(dKeys[r], s_j)
 *   dScoreK[j] = pk[j] * (dPk[j] - rowSumK), rowSumK = sum pk*dPk
 *   dQk[d]     = scale * sum_j dScoreK[j] * s[j,d]
 *   dS[j,d]   += pk[j]*dKeys[r,d] + scale*dScoreK[j]*qk[d]
 *   (same for the value side with qv/pv/dValues)
 *
 * recordIndices holds the record slot per active record; compact offsets/counts
 * locate each record's token range inside the compacted fieldStates [T, H].
 */
export function poolBackward(
  fieldStates: Float32Array, // [T, H]
  recordIndices: readonly number[], // [R] record slots
  recordCompactOffset: readonly number[], // [maxRecords]
  recordCompactCount: readonly number[], // [maxRecords]
  pool: Float32Array, // [2, H] qk row 0, qv row 1
  dKeys: Float32Array, // [R, H]
  dValues: Float32Array, // [R, H]
  h: number,
): { dFieldStates: Float32Array; dPool: Float32Array } {
  const r = recordIndices.length;
  const dFieldStates = new Float32Array(fieldStates.length);
  const dPool = new Float32Array(2 * h);
  const scale = 1 / Math.sqrt(h);
  for (let rec = 0; rec < r; rec++) {
    const slot = recordIndices[rec]!;
    const start = recordCompactOffset[slot]!;
    const count = recordCompactCount[slot]!;
    if (count === 0) continue;
    const pk = new Float32Array(count);
    const pv = new Float32Array(count);
    const dPk = new Float32Array(count);
    const dPv = new Float32Array(count);
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
    for (let j = 0; j < count; j++) {
      pk[j] = keyScores[j]!;
      pv[j] = valueScores[j]!;
      let dk = 0;
      let dv = 0;
      for (let d = 0; d < h; d++) {
        const s = fieldStates[(start + j) * h + d]!;
        dk += dKeys[rec * h + d]! * s;
        dv += dValues[rec * h + d]! * s;
      }
      dPk[j] = dk;
      dPv[j] = dv;
    }
    let kRowSum = 0;
    let vRowSum = 0;
    for (let j = 0; j < count; j++) {
      kRowSum += pk[j]! * dPk[j]!;
      vRowSum += pv[j]! * dPv[j]!;
    }
    for (let j = 0; j < count; j++) {
      const dScoreK = pk[j]! * (dPk[j]! - kRowSum);
      const dScoreV = pv[j]! * (dPv[j]! - vRowSum);
      for (let d = 0; d < h; d++) {
        const s = fieldStates[(start + j) * h + d]!;
        const ds = pk[j]! * dKeys[rec * h + d]! +
          pv[j]! * dValues[rec * h + d]! +
          scale * (dScoreK * pool[d]! + dScoreV * pool[h + d]!);
        dFieldStates[(start + j) * h + d] = (dFieldStates[(start + j) * h + d] ?? 0) + ds;
        dPool[d] = (dPool[d] ?? 0) + scale * dScoreK * s;
        dPool[h + d] = (dPool[h + d] ?? 0) + scale * dScoreV * s;
      }
    }
  }
  return { dFieldStates, dPool };
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
