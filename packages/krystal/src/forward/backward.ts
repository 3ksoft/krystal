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
import {
  BRAIN_FORWARD_CONFIG,
  embeddingTableBases,
  type BlockWeights,
  type BrainForwardConfig,
  type BrainForwardWeights,
} from "./model.ts";
import type { ActiveFrame } from "./masks.ts";
import {
  addInPlace,
  decisionHeadOracle,
  matmulOracle,
  reluOracle,
  selectorOracle,
  softmaxRow,
} from "./oracle.ts";

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

// ---------------------------------------------------------------------------
// Composed CPU backward reference (M3 close, §17 item 10)
// ---------------------------------------------------------------------------

export interface BrainBackwardResult {
  readonly dFieldStates: Float32Array; // [T, H]
  readonly dQueryValues: Float32Array; // [Q, H] (mixed query gradient)
  readonly dBankKeys: Float32Array; // [R, H]
  readonly dBankValues: Float32Array; // [R, H]
  readonly dPool: Float32Array; // [2, H]
  readonly dSelectorWq: Float32Array; // [H, H]
  readonly dSelectorWk: Float32Array; // [H, H]
  readonly dDecisionWh: Float32Array; // [C, 3H]
  readonly dQueryOutput: Float32Array; // [Q, H]
  readonly dIntentGather: Float32Array; // [Q, H]
  readonly dArgGather: Float32Array; // [Q, H]
}

/** dX = dY @ W with W [N, K] row-major (mirrors matmul_backward_input). */
function matmulBackInput(dY: Float32Array, w: Float32Array, m: number, n: number, k: number): Float32Array {
  const dX = new Float32Array(m * k);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let l = 0; l < n; l++) s += dY[i * n + l]! * w[l * k + j]!;
      dX[i * k + j] = s;
    }
  }
  return dX;
}

/** dW = dY^T @ X with W [N, K] (mirrors matmul_backward_weight). */
function matmulBackWeight(dY: Float32Array, x: Float32Array, m: number, n: number, k: number): Float32Array {
  const dW = new Float32Array(n * k);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let l = 0; l < m; l++) s += dY[l * n + i]! * x[l * k + j]!;
      dW[i * k + j] = s;
    }
  }
  return dW;
}

/** Multi-head masked attention returning both the output and P [heads,q,k]. */
function attentionWithP(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  mask: Float32Array,
  qRows: number,
  kRows: number,
  h: number,
  heads: number,
  headDim: number,
): { out: Float32Array; p: Float32Array } {
  const out = new Float32Array(qRows * h);
  const p = new Float32Array(heads * qRows * kRows);
  const scores = new Float32Array(kRows);
  const scale = 1 / Math.sqrt(headDim);
  for (let head = 0; head < heads; head++) {
    const hb = head * headDim;
    for (let i = 0; i < qRows; i++) {
      for (let j = 0; j < kRows; j++) {
        let s = 0;
        for (let d = 0; d < headDim; d++) s += q[i * h + hb + d]! * k[j * h + hb + d]!;
        scores[j] = s * scale + mask[i * kRows + j]!;
      }
      softmaxRow(scores, 0, kRows);
      for (let j = 0; j < kRows; j++) p[(head * qRows + i) * kRows + j] = scores[j]!;
      for (let d = 0; d < headDim; d++) {
        let value = 0;
        for (let j = 0; j < kRows; j++) value += scores[j]! * v[j * h + hb + d]!;
        out[i * h + hb + d] = value;
      }
    }
  }
  return { out, p };
}

/** Record pooling returning {key, value} [H] each (mirrors the forward). */
function poolKeyValue(
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

/** The six additive embeddings for the active token range (mirrors forward). */
function fieldEmbedCpu(
  frame: v1_0_0.BrainFrameGpu,
  active: ActiveFrame,
  config: BrainForwardConfig,
  weights: BrainForwardWeights,
): Float32Array {
  const { hiddenSize: h } = config;
  const bases = embeddingTableBases(config);
  const table = weights.embeddings;
  const states = new Float32Array(active.activeTokens.length * h);
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

interface SavedBlock {
  in: Float32Array;
  ffnIn: Float32Array;
  q: Float32Array;
  k: Float32Array;
  v: Float32Array;
  p: Float32Array;
  h1: Float32Array;
}

export interface BrainBackwardOracleInput {
  readonly frame: v1_0_0.BrainFrameGpu;
  readonly active: ActiveFrame;
  readonly weights: BrainForwardWeights;
  readonly config?: BrainForwardConfig;
  readonly recordMask: Float32Array;
  readonly mixerMask: Float32Array;
  readonly intentMask: Float32Array;
  readonly argMask: Float32Array;
  readonly routeKinds: readonly number[];
  /** Pointer-loss targets [Q] for the argument selector; default none. */
  readonly argGold?: readonly number[];
}

/**
 * Composed CPU backward for the full Krystal graph (M3 close, §17 item 10).
 * Mirrors the KrystalBackward runner's dispatch order exactly: forward with
 * per-block saves -> cross-entropy -> decision head -> both selectors -> mixer
 * reverse -> pool -> encoder reverse. Only the gradients whose regions the
 * composed runner exposes for parity are returned (per-block dW regions are
 * consumed by SGD in the runner and reused across blocks).
 */
export function brainBackwardOracle(
  input: BrainBackwardOracleInput,
): BrainBackwardResult {
  const {
    frame, active, weights, recordMask, mixerMask, intentMask, argMask, routeKinds, argGold,
  } = input;
  const config = input.config ?? BRAIN_FORWARD_CONFIG;
  const { hiddenSize: h, ffnSize: ffn, headCount: heads, headDim, encoderBlocks, mixerBlocks, routeKindCount: C } = config;
  const t = active.activeTokens.length;
  const r = active.bankRecords.length;
  const q = active.queryRecords.length;
  const zeroQ = new Float32Array(q * h);

  // --- forward with per-block saves ---
  let x = fieldEmbedCpu(frame, active, config, weights);
  const enc: SavedBlock[] = [];
  for (let b = 0; b < encoderBlocks; b++) {
    const block = weights.enc[b]!;
    const sIn = Float32Array.from(x);
    const qq = matmulOracle(x, block.wq, h, h);
    const kk = matmulOracle(x, block.wk, h, h);
    const vv = matmulOracle(x, block.wv, h, h);
    const attn = attentionWithP(qq, kk, vv, recordMask, t, t, h, heads, headDim);
    addInPlace(x, attn.out);
    const sFfnIn = Float32Array.from(x);
    const h1 = reluOracle(matmulOracle(x, block.w1, ffn, h));
    const ff = matmulOracle(h1, block.w2, h, ffn);
    addInPlace(x, ff);
    enc.push({ in: sIn, ffnIn: sFfnIn, q: qq, k: kk, v: vv, p: attn.p, h1 });
  }

  const bankKeys = new Float32Array(r * h);
  const bankValues = new Float32Array(r * h);
  for (let i = 0; i < r; i++) {
    const slot = active.bankRecords[i]!;
    const start = active.recordCompactOffset[slot]!;
    const count = active.recordCompactCount[slot]!;
    const kv = poolKeyValue(x, start, count, weights.pool, h);
    bankKeys.set(kv.key, i * h);
    bankValues.set(kv.value, i * h);
  }
  const queryKeys = new Float32Array(q * h);
  const queryValues = new Float32Array(q * h);
  for (let i = 0; i < q; i++) {
    const slot = active.queryRecords[i]!;
    const start = active.recordCompactOffset[slot]!;
    const count = active.recordCompactCount[slot]!;
    const kv = poolKeyValue(x, start, count, weights.pool, h);
    queryKeys.set(kv.key, i * h);
    queryValues.set(kv.value, i * h);
  }

  const mixer: SavedBlock[] = [];
  const query = queryValues;
  for (let b = 0; b < mixerBlocks; b++) {
    const block = weights.mixer[b]!;
    const sIn = Float32Array.from(query);
    const qq = matmulOracle(query, block.wq, h, h);
    const kk = matmulOracle(bankKeys, block.wk, h, h);
    const vv = matmulOracle(bankValues, block.wv, h, h);
    const attn = attentionWithP(qq, kk, vv, mixerMask, q, r, h, heads, headDim);
    addInPlace(query, attn.out);
    const sFfnIn = Float32Array.from(query);
    const h1 = reluOracle(matmulOracle(query, block.w1, ffn, h));
    const ff = matmulOracle(h1, block.w2, h, ffn);
    addInPlace(query, ff);
    mixer.push({ in: sIn, ffnIn: sFfnIn, q: qq, k: kk, v: vv, p: attn.p, h1 });
  }
  const queryOutput = query;

  const selector = weights.selector;
  const intent = selectorOracle(queryOutput, bankKeys, bankValues, intentMask, selector, h);
  const argument = selectorOracle(queryOutput, bankKeys, bankValues, argMask, selector, h);
  const logits = decisionHeadOracle(queryOutput, intent.gather, argument.gather, weights.decisionHeadWh, q, h, C);

  // --- loss: mean CE, dLogits = (softmax - onehot) / Q ---
  const dLogits = new Float32Array(q * C);
  for (let i = 0; i < q; i++) {
    const probs = Float32Array.from(logits.subarray(i * C, i * C + C));
    softmaxRow(probs, 0, C);
    const gold = routeKinds[i]!;
    for (let c = 0; c < C; c++) dLogits[i * C + c] = (probs[c]! - (c === gold ? 1 : 0)) / q;
  }

  // --- decision head backward ---
  const dh = decisionHeadBackward(dLogits, queryOutput, intent.gather, argument.gather, weights.decisionHeadWh, q, h, C);

  // --- selectors (both slots accumulate into shared dQProj/dKProj/dValue) ---
  const argTargets = argGold ?? new Array<number>(q).fill(0xffff_ffff);
  const noTargets = new Array<number>(q).fill(0xffff_ffff);
  const dQProj = new Float32Array(q * h);
  const dKProj = new Float32Array(r * h);
  const dValue = new Float32Array(r * h);
  const selQProj = matmulOracle(queryOutput, selector.wq, h, h); // [Q, H]
  const selKProj = matmulOracle(bankKeys, selector.wk, h, h); // [R, H]
  for (const [dGather, p, gold] of [
    [dh.dIntentGather, intent.p, noTargets],
    [dh.dArgGather, argument.p, argTargets],
  ] as const) {
    const dScore = selectorBackwardScores(dGather, bankValues, p, gold as number[], q, r, h);
    const g = selectorBackwardQkv(dScore, selQProj, selKProj, p, dGather, q, r, h);
    addInPlace(dQProj, g.dQProj);
    addInPlace(dKProj, g.dKProj);
    addInPlace(dValue, g.dValue);
  }

  let dQueryValues = Float32Array.from(dh.dQueryOutput);
  addInPlace(dQueryValues, matmulBackInput(dQProj, selector.wq, q, h, h));
  const dBankKeys = matmulBackInput(dKProj, selector.wk, r, h, h);
  const dBankValues = Float32Array.from(dValue);
  const dSelectorWq = matmulBackWeight(dQProj, queryOutput, q, h, h);
  const dSelectorWk = matmulBackWeight(dKProj, bankKeys, r, h, h);

  // --- mixer blocks reverse ---
  for (let b = mixerBlocks - 1; b >= 0; b--) {
    const block = weights.mixer[b]!;
    const s = mixer[b]!;
    const dH1r = matmulBackInput(dQueryValues, block.w2, q, h, ffn); // dFfnOut @ W2
    const dH1p = reluBackward(s.h1, dH1r);
    const dFfnIn = matmulBackInput(dH1p, block.w1, q, ffn, h); // dH1p @ W1
    const dAttnOut = new Float32Array(q * h);
    for (let i = 0; i < q * h; i++) dAttnOut[i] = dQueryValues[i]! + dFfnIn[i]!;
    const dScores = attentionBackwardScores(dAttnOut, s.v, s.p, q, r, heads, headDim);
    const { dQ, dK, dV } = attentionBackwardQkv(dScores, s.q, s.k, s.p, dAttnOut, q, r, heads, headDim);
    dQueryValues = Float32Array.from(dAttnOut);
    addInPlace(dQueryValues, matmulBackInput(dQ, block.wq, q, h, h));
    addInPlace(dBankKeys, matmulBackInput(dK, block.wk, r, h, h));
    addInPlace(dBankValues, matmulBackInput(dV, block.wv, r, h, h));
  }

  // --- pool backward (bank + query), then encoder blocks reverse ---
  const dFieldStates = new Float32Array(t * h);
  const dPool = new Float32Array(2 * h);
  const bank = poolBackward(x, Array.from(active.bankRecords), Array.from(active.recordCompactOffset), Array.from(active.recordCompactCount), weights.pool, dBankKeys, dBankValues, h);
  const queryP = poolBackward(x, Array.from(active.queryRecords), Array.from(active.recordCompactOffset), Array.from(active.recordCompactCount), weights.pool, zeroQ, dQueryValues, h);
  addInPlace(dFieldStates, bank.dFieldStates);
  addInPlace(dFieldStates, queryP.dFieldStates);
  addInPlace(dPool, bank.dPool);
  addInPlace(dPool, queryP.dPool);

  for (let b = encoderBlocks - 1; b >= 0; b--) {
    const block = weights.enc[b]!;
    const s = enc[b]!;
    const dH1r = matmulBackInput(dFieldStates, block.w2, t, h, ffn); // dFfnOut @ W2
    const dH1p = reluBackward(s.h1, dH1r);
    const dFfnIn = matmulBackInput(dH1p, block.w1, t, ffn, h); // dH1p @ W1
    const dAttnOut = new Float32Array(t * h);
    for (let i = 0; i < t * h; i++) dAttnOut[i] = dFieldStates[i]! + dFfnIn[i]!;
    const dScores = attentionBackwardScores(dAttnOut, s.v, s.p, t, t, heads, headDim);
    const { dQ, dK, dV } = attentionBackwardQkv(dScores, s.q, s.k, s.p, dAttnOut, t, t, heads, headDim);
    dFieldStates.set(dAttnOut);
    addInPlace(dFieldStates, matmulBackInput(dQ, block.wq, t, h, h));
    addInPlace(dFieldStates, matmulBackInput(dK, block.wk, t, h, h));
    addInPlace(dFieldStates, matmulBackInput(dV, block.wv, t, h, h));
  }

  return {
    dFieldStates,
    dQueryValues,
    dBankKeys,
    dBankValues,
    dPool,
    dSelectorWq,
    dSelectorWk,
    dDecisionWh: dh.dWh,
    dQueryOutput: dh.dQueryOutput,
    dIntentGather: dh.dIntentGather,
    dArgGather: dh.dArgGather,
  };
}
