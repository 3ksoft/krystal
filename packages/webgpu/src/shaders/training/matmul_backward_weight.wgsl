// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: matmul_backward_weight
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid
//
// dW = dY^T @ X  (weight gradient of Y = X @ W^T, W row-major [N,K]).
//
//   dW[n,k] = sum_m dY[m,n] * X[m,k]
//
// OpParams:
//   tokenCount = M, inputDim = N, outputDim = K
//   inputOffset  = dY [M,N]
//   auxOffset    = X  [M,K]
//   outputOffset = dW [N,K]
//
// Each output element has one owner and reduces over M; no atomics are needed.
// This shader never touches parameters — gradient computation and optimizer
// mutation are separate dispatches.

  let total = op.inputDim * op.outputDim;
  let linear = gid.x;
  if (linear >= total) { return; }

  let n = linear / op.outputDim;
  let k = linear % op.outputDim;
  var sum = 0.0;
  for (var m = 0u; m < op.tokenCount; m++) {
    sum += arena[op.inputOffset + m * op.inputDim + n] * arena[op.auxOffset + m * op.outputDim + k];
  }
  arena[op.outputOffset + linear] = sum;
