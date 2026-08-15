// Plain TypeScript f32 CPU references for the M1 training ops.
//
// These are test references, not a fallback runtime (WEBGPU_BACKWARD_PLAN.md
// §14.1). Each function mirrors one GPU shader's contract exactly, so the GPU
// output can be compared bit-for-bit within the declared f32 tolerance.

export interface CeResult {
  /** lossRows [M] = logZ - logits[m, target[m]] (no /M). */
  lossRows: Float32Array;
  /** dLogits [M,V] = (prob - one_hot) / M. */
  dLogits: Float32Array;
}

/**
 * Numerically stable softmax cross-entropy + dLogits, mean-reduced over M.
 * Mirrors cross_entropy_forward_backward.wgsl.
 */
export function crossEntropyForwardBackward(logits: Float32Array, targets: readonly number[], v: number): CeResult {
  const m = logits.length / v;
  if (!Number.isInteger(m)) throw new Error("logits length must be M*V");
  const lossRows = new Float32Array(m);
  const dLogits = new Float32Array(m * v);
  for (let row = 0; row < m; row++) {
    const base = row * v;
    let rowMax = -Infinity;
    for (let i = 0; i < v; i++) rowMax = Math.max(rowMax, logits[base + i]!);
    let sumExp = 0;
    for (let i = 0; i < v; i++) sumExp += Math.exp(logits[base + i]! - rowMax);
    const logZ = rowMax + Math.log(sumExp);
    const target = targets[row]!;
    lossRows[row] = logZ - logits[base + target]!;
    for (let i = 0; i < v; i++) {
      const prob = Math.exp(logits[base + i]! - logZ);
      const oneHot = i === target ? 1 : 0;
      dLogits[base + i] = (prob - oneHot) / m;
    }
  }
  return { lossRows, dLogits };
}

/** Mean of lossRows [M] -> scalar. Mirrors loss_reduce.wgsl. */
export function lossReduce(lossRows: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < lossRows.length; i++) sum += lossRows[i]!;
  return sum / lossRows.length;
}

/**
 * dX = dY @ W, W row-major [N,K]. Mirrors matmul_backward_input.wgsl.
 */
export function matmulBackwardInput(dY: Float32Array, w: Float32Array, m: number, n: number, k: number): Float32Array {
  if (dY.length !== m * n) throw new Error("dY length must be M*N");
  if (w.length !== n * k) throw new Error("W length must be N*K");
  const dX = new Float32Array(m * k);
  for (let row = 0; row < m; row++) {
    for (let col = 0; col < k; col++) {
      let sum = 0;
      for (let mid = 0; mid < n; mid++) {
        sum += dY[row * n + mid]! * w[mid * k + col]!;
      }
      dX[row * k + col] = sum;
    }
  }
  return dX;
}

/**
 * dW = dY^T @ X, X row-major [M,K]. Mirrors matmul_backward_weight.wgsl.
 */
export function matmulBackwardWeight(dY: Float32Array, x: Float32Array, m: number, n: number, k: number): Float32Array {
  if (dY.length !== m * n) throw new Error("dY length must be M*N");
  if (x.length !== m * k) throw new Error("X length must be M*K");
  const dW = new Float32Array(n * k);
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < k; col++) {
      let sum = 0;
      for (let mid = 0; mid < m; mid++) {
        sum += dY[mid * n + row]! * x[mid * k + col]!;
      }
      dW[row * k + col] = sum;
    }
  }
  return dW;
}

/**
 * dEmbedding[v,h] = sum over m where tokens[m]==v of dHidden[m,h].
 * Mirrors embedding_backward.wgsl.
 */
export function embeddingBackward(
  dHidden: Float32Array,
  tokens: readonly number[],
  m: number,
  v: number,
  h: number,
): Float32Array {
  if (dHidden.length !== m * h) throw new Error("dHidden length must be M*H");
  const dEmbedding = new Float32Array(v * h);
  for (let row = 0; row < v; row++) {
    for (let col = 0; col < h; col++) {
      let sum = 0;
      for (let mid = 0; mid < m; mid++) {
        if (tokens[mid] === row) sum += dHidden[mid * h + col]!;
      }
      dEmbedding[row * h + col] = sum;
    }
  }
  return dEmbedding;
}

/** Plain SGD: params -= lr * grads. Mirrors sgd_step.wgsl. */
export function sgdStep(params: Float32Array, grads: Float32Array, learningRate: number): Float32Array {
  if (params.length !== grads.length) throw new Error("param/grad length mismatch");
  const out = new Float32Array(params.length);
  for (let i = 0; i < params.length; i++) out[i] = params[i]! - learningRate * grads[i]!;
  return out;
}

/** Forward logits = X @ W^T, W row-major [N,K]. Mirrors matmul_f32.wgsl. */
export function matmulForward(x: Float32Array, w: Float32Array, m: number, k: number, n: number): Float32Array {
  if (x.length !== m * k) throw new Error("X length must be M*K");
  if (w.length !== n * k) throw new Error("W length must be N*K");
  const y = new Float32Array(m * n);
  for (let row = 0; row < m; row++) {
    for (let col = 0; col < n; col++) {
      let sum = 0;
      for (let mid = 0; mid < k; mid++) {
        sum += x[row * k + mid]! * w[col * k + mid]!;
      }
      y[row * n + col] = sum;
    }
  }
  return y;
}

// ---------------------------------------------------------------------------
// Attention (§17 item 6) — Krystal encoder semantics: bidirectional,
// host-masked, multi-head, no KV cache. All tensors are f32, row-major.
//
// Layouts (mirror attention_*.wgsl):
//   Q, K, V, out, dOut, dQ, dK, dV  [M, H]; head h owns columns
//     [h*headDim, (h+1)*headDim)
//   P, dScores  [headCount, M, M]; index (h*M + i)*M + j
//   mask        [M, M]; 0.0 = allowed, -1e30 = blocked
// ---------------------------------------------------------------------------

export interface AttentionForwardResult {
  /** P [headCount, M, M] softmax probabilities (persisted for backward). */
  P: Float32Array;
  /** out [M, H] context vectors. */
  out: Float32Array;
}

/**
 * Masked multi-head attention forward. Mirrors attention_forward.wgsl.
 * scale = 1/sqrt(headDim); the mask is added to raw scores before softmax.
 */
export function attentionForward(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  mask: Float32Array,
  headCount: number,
  headDim: number,
): AttentionForwardResult {
  const m = q.length / (headCount * headDim);
  const h = headCount * headDim;
  const scale = 1 / Math.sqrt(headDim);
  const P = new Float32Array(headCount * m * m);
  const out = new Float32Array(m * h);
  for (let head = 0; head < headCount; head++) {
    for (let i = 0; i < m; i++) {
      // Raw scores.
      const scores = new Float32Array(m);
      for (let j = 0; j < m; j++) {
        let s = 0;
        for (let d = 0; d < headDim; d++) {
          s += q[i * h + head * headDim + d]! * k[j * h + head * headDim + d]!;
        }
        scores[j] = s * scale + mask[i * m + j]!;
      }
      // Row softmax (masked entries collapse to exp(-1e30) ~ 0).
      let rowMax = -Infinity;
      for (let j = 0; j < m; j++) rowMax = Math.max(rowMax, scores[j]!);
      let sumExp = 0;
      for (let j = 0; j < m; j++) sumExp += Math.exp(scores[j]! - rowMax);
      const inv = 1 / Math.max(sumExp, 1e-20);
      for (let j = 0; j < m; j++) {
        const p = Math.exp(scores[j]! - rowMax) * inv;
        P[(head * m + i) * m + j] = p;
      }
      // Context vector.
      for (let d = 0; d < headDim; d++) {
        let value = 0;
        for (let j = 0; j < m; j++) {
          value += P[(head * m + i) * m + j]! * v[j * h + head * headDim + d]!;
        }
        out[i * h + head * headDim + d] = value;
      }
    }
  }
  return { P, out };
}

/**
 * Softmax-score gradient. Mirrors attention_backward_scores.wgsl:
 *   dScores[h][i][j] = P[h][i][j] * (dP[h][i][j] - rowSum[h][i])
 * with dP[h][i][j] = dot(dOut[i][h], V[j][h]) and
 * rowSum[h][i] = sum_j P[h][i][j] * dP[h][i][j].
 */
export function attentionBackwardScores(
  dOut: Float32Array,
  v: Float32Array,
  p: Float32Array,
  headCount: number,
  headDim: number,
): Float32Array {
  const m = dOut.length / (headCount * headDim);
  const h = headCount * headDim;
  const dScores = new Float32Array(headCount * m * m);
  for (let head = 0; head < headCount; head++) {
    for (let i = 0; i < m; i++) {
      const dP = new Float32Array(m);
      for (let j = 0; j < m; j++) {
        let dp = 0;
        for (let d = 0; d < headDim; d++) {
          dp += dOut[i * h + head * headDim + d]! * v[j * h + head * headDim + d]!;
        }
        dP[j] = dp;
      }
      let rowSum = 0;
      for (let j = 0; j < m; j++) rowSum += p[(head * m + i) * m + j]! * dP[j]!;
      for (let j = 0; j < m; j++) {
        dScores[(head * m + i) * m + j] = p[(head * m + i) * m + j]! * (dP[j]! - rowSum);
      }
    }
  }
  return dScores;
}

/**
 * Q/K/V gradients. Mirrors attention_backward_qkv.wgsl:
 *   dQ[i,h,d] = scale * sum_j dScores[h,i,j] * K[j,h,d]
 *   dK[j,h,d] = scale * sum_i dScores[h,i,j] * Q[i,h,d]
 *   dV[j,h,d] =        sum_i P[h,i,j]       * dOut[i,h,d]
 */
export function attentionBackwardQkv(
  dScores: Float32Array,
  q: Float32Array,
  k: Float32Array,
  p: Float32Array,
  dOut: Float32Array,
  headCount: number,
  headDim: number,
): { dQ: Float32Array; dK: Float32Array; dV: Float32Array } {
  const m = q.length / (headCount * headDim);
  const h = headCount * headDim;
  const scale = 1 / Math.sqrt(headDim);
  const dQ = new Float32Array(m * h);
  const dK = new Float32Array(m * h);
  const dV = new Float32Array(m * h);
  for (let i = 0; i < m; i++) {
    for (let col = 0; col < h; col++) {
      const head = Math.floor(col / headDim);
      let acc = 0;
      for (let j = 0; j < m; j++) {
        acc += dScores[(head * m + i) * m + j]! * k[j * h + col]!;
      }
      dQ[i * h + col] = acc * scale;
    }
  }
  for (let j = 0; j < m; j++) {
    for (let col = 0; col < h; col++) {
      const head = Math.floor(col / headDim);
      let acc = 0;
      for (let i = 0; i < m; i++) {
        acc += dScores[(head * m + i) * m + j]! * q[i * h + col]!;
      }
      dK[j * h + col] = acc * scale;
    }
  }
  for (let j = 0; j < m; j++) {
    for (let col = 0; col < h; col++) {
      const head = Math.floor(col / headDim);
      let acc = 0;
      for (let i = 0; i < m; i++) {
        acc += p[(head * m + i) * m + j]! * dOut[i * h + col]!;
      }
      dV[j * h + col] = acc;
    }
  }
  return { dQ, dK, dV };
}

/** Full toy-graph forward: hidden -> logits -> loss rows + dLogits. */
export function forwardGraph(
  embedding: Float32Array,
  classifier: Float32Array,
  tokens: readonly number[],
  targets: readonly number[],
  v: number,
  h: number,
): { hidden: Float32Array; logits: Float32Array; lossRows: Float32Array; dLogits: Float32Array; loss: number } {
  const m = tokens.length;
  const hidden = new Float32Array(m * h);
  for (let row = 0; row < m; row++) {
    const token = tokens[row]!;
    for (let col = 0; col < h; col++) {
      hidden[row * h + col] = embedding[token * h + col]!;
    }
  }
  const logits = matmulForward(hidden, classifier, m, h, v);
  const ce = crossEntropyForwardBackward(logits, targets, v);
  return { hidden, logits, lossRows: ce.lossRows, dLogits: ce.dLogits, loss: lossReduce(ce.lossRows) };
}
