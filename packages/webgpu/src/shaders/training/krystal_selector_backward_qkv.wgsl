// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: krystal_selector_backward_qkv
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid
//
// Soft-gather gradients of the selector projections and bank values (M3,
// §17 item 8), mirroring the attention_backward_qkv ownership pattern:
//
//   dQProj[i,d] = scale * sum_j dScore[i,j] * kProj[j,d]
//   dKProj[j,d] = scale * sum_i dScore[i,j] * qProj[i,d]
//   dValue[j,d] =        sum_i p[i,j]       * dGather[i,d]
//
// where scale = 1/sqrt(H). dQProj has Q rows while dKProj/dValue have R rows,
// so the linear gid is split into three blocks:
//   [0, Q*H) = dQProj, [Q*H, Q*H + R*H) = dKProj, the remainder = dValue.
// Every output element has one owner, so no atomics. The pointer-loss and
// softmax gradients live in dScore (computed by
// krystal_selector_backward_scores); this pass only routes them.
//
// OpParams:
//   tokenCount = Q, inputDim = H, u0 = R
//   inputOffset  = dScore  [Q, R]
//   auxOffset    = qProj   [Q, H]
//   aux2Offset   = kProj   [R, H]
//   aux3Offset   = p       [Q, R]
//   aux4Offset   = dGather [Q, H]
//   outputOffset = dQProj  [Q, H]
//   aux5Offset   = dKProj  [R, H]
//   aux6Offset   = dValue  [R, H]

  let Q = op.tokenCount;
  let R = op.u0;
  let H = op.inputDim;
  let qTotal = Q * H;
  let rTotal = R * H;
  let linear = gid.x;
  if (linear >= qTotal + 2u * rTotal) { return; }

  let scale = 1.0 / sqrt(f32(H));
  let col = linear % H;

  if (linear < qTotal) {
    // dQProj[i, col] = scale * sum_j dScore[i,j] * kProj[j, col]
    let i = linear / H;
    var sum = 0.0;
    for (var j = 0u; j < R; j++) {
      sum += arena[op.inputOffset + i * R + j] * arena[op.aux2Offset + j * H + col];
    }
    arena[op.outputOffset + i * H + col] = sum * scale;
    return;
  }
  if (linear < qTotal + rTotal) {
    // dKProj[j, col] = scale * sum_i dScore[i,j] * qProj[i, col]
    let j = (linear - qTotal) / H;
    var sum = 0.0;
    for (var i2 = 0u; i2 < Q; i2++) {
      sum += arena[op.inputOffset + i2 * R + j] * arena[op.auxOffset + i2 * H + col];
    }
    arena[op.aux5Offset + j * H + col] = sum * scale;
    return;
  }
  // dValue[j, col] = sum_i p[i,j] * dGather[i, col]
  let j = (linear - qTotal - rTotal) / H;
  var sum = 0.0;
  for (var i2 = 0u; i2 < Q; i2++) {
    sum += arena[op.aux3Offset + i2 * R + j] * arena[op.aux4Offset + i2 * H + col];
  }
  arena[op.aux6Offset + j * H + col] = sum;
