// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: relu_backward
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid
//
// Elementwise ReLU backward for the Krystal FFN (M3, §17 order):
//
//   dIn[i] = out[i] > 0.0 ? dOut[i] : 0.0
//
// The mask is read from the saved relu OUTPUT rather than the pre-activation:
// the runner applies relu in place, so out = max(0, x) is all that survives.
// out > 0 iff x > 0 except at exactly x == 0, where the conventional
// subgradient is 0 anyway (out == 0 there), so this is exact.
//
// OpParams:
//   tokenCount = element count
//   inputOffset = relu output (post-activation)
//   auxOffset   = dOut (upstream gradient)
//   outputOffset = dIn (gradient wrt the pre-activation)

  let linear = gid.x;
  if (linear >= op.tokenCount) { return; }
  let mask = select(0.0, 1.0, arena[op.inputOffset + linear] > 0.0);
  arena[op.outputOffset + linear] = arena[op.auxOffset + linear] * mask;
