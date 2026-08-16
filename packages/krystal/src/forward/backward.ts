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
 *   krystal_pool_backward              -> poolBackward
 *   krystal_selector_backward_scores   -> selectorBackwardScores
 *   krystal_selector_backward_qkv      -> selectorBackwardQkv
 *   krystal_decision_head_backward     -> decisionHeadBackward
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
/**
 * Selector soft-gather score gradient (one typed slot, §17 item 8). Mirrors
 * krystal_selector_backward_scores.wgsl:
 *
 *   dP[i,j]     = dot(dGather[i], value[j])
 *   rowSum[i]   = sum_j p[i,j] * dP[i,j]
 *   dScore[i,j] = p[i,j] * (dP[i,j] - rowSum[i]) + pointerLossGrad[i,j]
 *   pointerLossGrad[i,j] = p[i,j] - onehot(j == gold[i])  (gold valid only)
 *
 * Layouts: dGather [Q,H], value [R,H], p/dScore [Q,R], gold [Q] u32 payloads
 * (0xffffffff = no pointer loss for that row).
 */
export function selectorBackwardScores(
  dGather: Float32Array, // [Q, H]
  value: Float32Array, // [R, H]
  p: Float32Array, // [Q, R]
  gold: readonly number[], // [Q] u32 payloads; 0xffffffff = none
  q: number,
  r: number,
  h: number,
): Float32Array {
  const dScore = new Float32Array(q * r);
  for (let i = 0; i < q; i++) {
    const dP = new Float32Array(r);
    let rowSum = 0;
    for (let j = 0; j < r; j++) {
      let dp = 0;
      for (let d = 0; d < h; d++) {
        dp += dGather[i * h + d]! * value[j * h + d]!;
      }
      dP[j] = dp;
      rowSum += p[i * r + j]! * dp;
    }
    const goldValid = gold[i] !== 0xffff_ffff;
    for (let j = 0; j < r; j++) {
      const pointerGrad = goldValid ? p[i * r + j]! - (j === gold[i] ? 1 : 0) : 0;
      dScore[i * r + j] = p[i * r + j]! * (dP[j]! - rowSum) + pointerGrad;
    }
  }
  return dScore;
}

/**
 * Selector projection/value gradients (one typed slot, §17 item 8). Mirrors
 * krystal_selector_backward_qkv.wgsl:
 *
 *   dQProj[i,d] = scale * sum_j dScore[i,j] * kProj[j,d]
 *   dKProj[j,d] = scale * sum_i dScore[i,j] * qProj[i,d]
 *   dValue[j,d] =        sum_i p[i,j]       * dGather[i,d]
 *
 * Layouts: dScore/p [Q,R], qProj/dGather [Q,H], kProj [R,H], dQProj [Q,H],
 * dKProj/dValue [R,H]. scale = 1/sqrt(H).
 */
export function selectorBackwardQkv(
  dScore: Float32Array, // [Q, R]
  qProj: Float32Array, // [Q, H]
  kProj: Float32Array, // [R, H]
  p: Float32Array, // [Q, R]
  dGather: Float32Array, // [Q, H]
  q: number,
  r: number,
  h: number,
): { dQProj: Float32Array; dKProj: Float32Array; dValue: Float32Array } {
  const scale = 1 / Math.sqrt(h);
  const dQProj = new Float32Array(q * h);
  const dKProj = new Float32Array(r * h);
  const dValue = new Float32Array(r * h);
  for (let i = 0; i < q; i++) {
    for (let col = 0; col < h; col++) {
      let acc = 0;
      for (let j = 0; j < r; j++) {
        acc += dScore[i * r + j]! * kProj[j * h + col]!;
      }
      dQProj[i * h + col] = acc * scale;
    }
  }
  for (let j = 0; j < r; j++) {
    for (let col = 0; col < h; col++) {
      let accK = 0;
      let accV = 0;
      for (let i = 0; i < q; i++) {
        accK += dScore[i * r + j]! * qProj[i * h + col]!;
        accV += p[i * r + j]! * dGather[i * h + col]!;
      }
      dKProj[j * h + col] = accK * scale;
      dValue[j * h + col] = accV;
    }
  }
  return { dQProj, dKProj, dValue };
}

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

/**
 * Typed decision head backward (§17 item 9): the final linear head over the
 * gathered context (architecture v2 §12.9, TypedPlan.routeKind). Forward:
 *
 *   logits[q,c] = sum_{d in [0,3H)} ctx[q,d] * Wh[c,d]
 *
 * with ctx[q] = concat(queryOutput[q], intentGather[q], argGather[q]). Given
 * upstream dLogits (e.g. cross-entropy gradient over route kinds), returns
 * the three gathered-context gradient parts and the head-weight gradient:
 *
 *   dQueryOutput[q,d]  = sum_c dLogits[q,c] * Wh[c, d]
 *   dIntentGather[q,d] = sum_c dLogits[q,c] * Wh[c, H + d]
 *   dArgGather[q,d]    = sum_c dLogits[q,c] * Wh[c, 2H + d]
 *   dWh[c,d']          = sum_q dLogits[q,c] * ctx[q, d']
 *
 * Mirrors krystal_decision_head_backward.wgsl exactly.
 */
export function decisionHeadBackward(
  dLogits: Float32Array, // [Q, C]
  queryOutput: Float32Array, // [Q, H]
  intentGather: Float32Array, // [Q, H]
  argGather: Float32Array, // [Q, H]
  wh: Float32Array, // [C, 3H] row-major
  q: number,
  h: number,
  c: number,
): {
  dQueryOutput: Float32Array;
  dIntentGather: Float32Array;
  dArgGather: Float32Array;
  dWh: Float32Array;
} {
  const hin = 3 * h;
  const dQueryOutput = new Float32Array(q * h);
  const dIntentGather = new Float32Array(q * h);
  const dArgGather = new Float32Array(q * h);
  const dWh = new Float32Array(c * hin);
  for (let qi = 0; qi < q; qi++) {
    for (let d = 0; d < h; d++) {
      let dq = 0;
      let di = 0;
      let da = 0;
      for (let cl = 0; cl < c; cl++) {
        const dl = dLogits[qi * c + cl]!;
        dq += dl * wh[cl * hin + d]!;
        di += dl * wh[cl * hin + h + d]!;
        da += dl * wh[cl * hin + 2 * h + d]!;
      }
      dQueryOutput[qi * h + d] = dq;
      dIntentGather[qi * h + d] = di;
      dArgGather[qi * h + d] = da;
    }
  }
  for (let cl = 0; cl < c; cl++) {
    for (let d = 0; d < hin; d++) {
      let sum = 0;
      for (let qi = 0; qi < q; qi++) {
        const ctx = d < h
          ? queryOutput[qi * h + d]!
          : d < 2 * h
            ? intentGather[qi * h + (d - h)]!
            : argGather[qi * h + (d - 2 * h)]!;
        sum += dLogits[qi * c + cl]! * ctx;
      }
      dWh[cl * hin + d] = sum;
    }
  }
  return { dQueryOutput, dIntentGather, dArgGather, dWh };
}
