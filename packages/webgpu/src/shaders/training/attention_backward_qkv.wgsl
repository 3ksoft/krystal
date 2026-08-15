// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: attention_backward_qkv
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid
//
// Q/K/V gradients of masked multi-head attention (§17 item 6).
//
//   dQ[i,h,d] = scale * sum_j dScores[h,i,j] * K[j,h,d]
//   dK[j,h,d] = scale * sum_i dScores[h,i,j] * Q[i,h,d]
//   dV[j,h,d] =        sum_i P[h,i,j]       * dOut[i,h,d]
//
// where scale = 1/sqrt(headDim). Every output element has one owner (a linear
// gid over the three [M,H] outputs), so no atomics are needed.
//
// Layouts (all f32 in the shared arena):
//   dScores [headCount, M, M], dScores[(h*M + i)*M + j]
//   P       [headCount, M, M], same layout
//   Q, K, V, dOut, dQ, dK, dV  [M, H], head h at columns [h*headDim, ...)
//
// OpParams:
//   tokenCount = M, inputDim = H, outputDim = headDim, u0 = headCount
//   inputOffset  = dScores [headCount,M,M]
//   auxOffset    = Q  [M,H]
//   aux2Offset   = K  [M,H]
//   aux3Offset   = P  [headCount,M,M]
//   aux4Offset   = dOut [M,H]
//   outputOffset = dQ  [M,H]
//   aux5Offset   = dK  [M,H]
//   aux6Offset   = dV  [M,H]

  let M = op.tokenCount;
  let H = op.inputDim;
  let headDim = op.outputDim;
  let headCount = op.u0;
  let total = M * H;
  let linear = gid.x;
  if (linear >= 3u * total) { return; }

  let scale = 1.0 / sqrt(f32(headDim));
  let h = (linear % H) / headDim;
  let d = (linear % H) % headDim;
  let col = linear % H;        // h*headDim + d

  if (linear < total) {
    // dQ[i, col] = scale * sum_j dScores[h,i,j] * K[j, col]
    let i = linear / H;
    var sum = 0.0;
    for (var j = 0u; j < M; j++) {
      let s = arena[op.inputOffset + (h * M + i) * M + j];
      let k = arena[op.aux2Offset + j * H + col];
      sum += s * k;
    }
    arena[op.outputOffset + i * H + col] = sum * scale;
    return;
  }
  if (linear < 2u * total) {
    // dK[j, col] = scale * sum_i dScores[h,i,j] * Q[i, col]
    let j = (linear - total) / H;
    var sum = 0.0;
    for (var i2 = 0u; i2 < M; i2++) {
      let s = arena[op.inputOffset + (h * M + i2) * M + j];
      let q = arena[op.auxOffset + i2 * H + col];
      sum += s * q;
    }
    arena[op.aux5Offset + j * H + col] = sum * scale;
    return;
  }
  // dV[j, col] = sum_i P[h,i,j] * dOut[i, col]
  let j = (linear - 2u * total) / H;
  var sum = 0.0;
  for (var i2 = 0u; i2 < M; i2++) {
    let p = arena[op.aux3Offset + (h * M + i2) * M + j];
    let dy = arena[op.aux4Offset + i2 * H + col];
    sum += p * dy;
  }
  arena[op.aux6Offset + j * H + col] = sum;
