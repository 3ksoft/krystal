// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: krystal_pool_dpool
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid
//
// Reduce the per-record pool-query gradient partials written by
// krystal_pool_backward into the final dPool [2, H]:
//
//   dPool[which*H + d] = sum_r dPoolPartial[r*2*H + which*H + d]
//
// The pool queries are shared across every record, so accumulation over
// records is a plain reduction — one invocation owns (which, d) and scans the
// records, deterministic with no atomics.
//
// OpParams:
//   tokenCount = recordCount, inputDim = H
//   inputOffset  = dPoolPartial [recordCount, 2, H]
//   outputOffset = dPool [2, H]

  let total = 2 * op.inputDim;
  let linear = gid.x;
  if (linear >= total) { return; }

  var sum = 0.0;
  for (var r = 0u; r < op.tokenCount; r++) {
    sum += arena[op.inputOffset + r * 2 * op.inputDim + linear];
  }
  arena[op.outputOffset + linear] = sum;
