// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: matmul_backward_input
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid
//
// dX = dY @ W  (input gradient of Y = X @ W^T, W row-major [N,K]).
//
//   dX[m,k] = sum_n dY[m,n] * W[n,k]
//
// OpParams:
//   tokenCount = M, inputDim = N, outputDim = K
//   inputOffset  = dY [M,N]
//   outputOffset = dX [M,K]
//   weight32     = W  [N,K] f32 (row-major, same layout as matmul_f32)
//
// Each output element has one owner; no atomics are needed.

  let total = op.tokenCount * op.outputDim;
  let linear = gid.x;
  if (linear >= total) { return; }

  let m = linear / op.outputDim;
  let k = linear % op.outputDim;
  var sum = 0.0;
  for (var n = 0u; n < op.inputDim; n++) {
    sum += arena[op.inputOffset + m * op.inputDim + n] * weight32[n * op.outputDim + k];
  }
  arena[op.outputOffset + linear] = sum;
