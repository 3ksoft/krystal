// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: embedding_backward
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid
//
// Scatter-add embedding gradient:
//
//   dEmbedding[v,h] = sum_(m where tokens[u0 + m] == v) dHidden[m,h]
//
// OpParams:
//   tokenCount = M, inputDim = V, outputDim = H
//   inputOffset  = dHidden [M,H]
//   outputOffset = dEmbedding [V,H]
//   u0 = token-id base offset inside the tokens buffer
//
// Deterministic MVP: one invocation owns (v,h), scans all m, writes exactly one
// dEmbedding element (including 0 for unused vocabulary rows). Repeated token
// ids accumulate correctly without floating-point atomics.

  let total = op.inputDim * op.outputDim;
  let linear = gid.x;
  if (linear >= total) { return; }

  let v = linear / op.outputDim;
  let h = linear % op.outputDim;
  var sum = 0.0;
  for (var m = 0u; m < op.tokenCount; m++) {
    if (tokens[op.u0 + m] == v) {
      sum += arena[op.inputOffset + m * op.outputDim + h];
    }
  }
  arena[op.outputOffset + linear] = sum;
