// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: krystal_attention_backward_qkv
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid
//
// Q/K/V gradients of masked multi-head Krystal attention (M3, §17 order).
// Cross-capable: qRows and kRows are independent (record encoder local
// attention qRows == kRows; mixer cross-attention qRows != kRows).
//
//   dQ[i,h,d] = scale * sum_j dScores[h,i,j] * K[j,h,d]
//   dK[j,h,d] = scale * sum_i dScores[h,i,j] * Q[i,h,d]
//   dV[j,h,d] =        sum_i P[h,i,j]       * dOut[i,h,d]
//
// where scale = 1/sqrt(headDim). dQ has qRows rows while dK/dV have kRows
// rows, so the three outputs have different sizes; the linear gid is split
// into three blocks: [0, qRows*H) = dQ, [qRows*H, qRows*H + kRows*H) = dK,
// the remainder = dV. Every output element has one owner, so no atomics.
//
// Layouts (all f32 in the shared arena):
//   dScores [headCount, qRows, kRows], dScores[(h*qRows + i)*kRows + j]
//   P       [headCount, qRows, kRows], same layout
//   Q, dOut [qRows, H], head h at columns [h*headDim, ...)
//   K, V    [kRows, H], same head layout
//   dQ      [qRows, H], dK/dV [kRows, H]
//
// OpParams:
//   tokenCount = qRows, inputDim = H, outputDim = headDim
//   u0 = kRows, u1 = headCount
//   inputOffset  = dScores [headCount, qRows, kRows]
//   auxOffset    = Q  [qRows, H]
//   aux2Offset   = K  [kRows, H]
//   aux3Offset   = P  [headCount, qRows, kRows]
//   aux4Offset   = dOut [qRows, H]
//   outputOffset = dQ  [qRows, H]
//   aux5Offset   = dK  [kRows, H]
//   aux6Offset   = dV  [kRows, H]
// Optional record-local mode for self-attention:
//   u2 = activeTokens arena offset, u3 = recordCompactOffset arena offset,
//   u4 = recordCompactCount arena offset, u5 != 0 enables it.

  let qRows = op.tokenCount;
  let kRows = op.u0;
  let H = op.inputDim;
  let headDim = op.outputDim;
  let headCount = op.u1;
  let qTotal = qRows * H;
  let kTotal = kRows * H;
  let linear = gid.x;
  if (linear >= qTotal + 2u * kTotal) { return; }

  let scale = 1.0 / sqrt(f32(headDim));
  let col = linear % H;        // h*headDim + d
  let h = (linear % H) / headDim;
  let d = (linear % H) % headDim;

  if (linear < qTotal) {
    // dQ[i, col] = scale * sum_j dScores[h,i,j] * K[j, col]
    let i = linear / H;
    var keyStart = 0u;
    var keyEnd = kRows;
    if (op.u5 != 0u) {
      let frameTok = bitcast<u32>(arena[op.u2 + i]);
      let slot = frameTok >> 3u;
      keyStart = bitcast<u32>(arena[op.u3 + slot]);
      keyEnd = keyStart + bitcast<u32>(arena[op.u4 + slot]);
    }
    var sum = 0.0;
    for (var j = keyStart; j < keyEnd; j++) {
      let s = arena[op.inputOffset + (h * qRows + i) * kRows + j];
      let k = arena[op.aux2Offset + j * H + col];
      sum += s * k;
    }
    arena[op.outputOffset + i * H + col] = sum * scale;
    return;
  }
  if (linear < qTotal + kTotal) {
    // dK[j, col] = scale * sum_i dScores[h,i,j] * Q[i, col]
    let j = (linear - qTotal) / H;
    var queryStart = 0u;
    var queryEnd = qRows;
    if (op.u5 != 0u) {
      let frameTok = bitcast<u32>(arena[op.u2 + j]);
      let slot = frameTok >> 3u;
      queryStart = bitcast<u32>(arena[op.u3 + slot]);
      queryEnd = queryStart + bitcast<u32>(arena[op.u4 + slot]);
    }
    var sum = 0.0;
    for (var i2 = queryStart; i2 < queryEnd; i2++) {
      let s = arena[op.inputOffset + (h * qRows + i2) * kRows + j];
      let q = arena[op.auxOffset + i2 * H + col];
      sum += s * q;
    }
    arena[op.aux5Offset + j * H + col] = sum * scale;
    return;
  }
  // dV[j, col] = sum_i P[h,i,j] * dOut[i, col]
  let j = (linear - qTotal - kTotal) / H;
  var queryStart = 0u;
  var queryEnd = qRows;
  if (op.u5 != 0u) {
    let frameTok = bitcast<u32>(arena[op.u2 + j]);
    let slot = frameTok >> 3u;
    queryStart = bitcast<u32>(arena[op.u3 + slot]);
    queryEnd = queryStart + bitcast<u32>(arena[op.u4 + slot]);
  }
  var sum = 0.0;
  for (var i2 = queryStart; i2 < queryEnd; i2++) {
    let p = arena[op.aux3Offset + (h * qRows + i2) * kRows + j];
    let dy = arena[op.aux4Offset + i2 * H + col];
    sum += p * dy;
  }
  arena[op.aux6Offset + j * H + col] = sum;
