// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: relu
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid
//
// Elementwise ReLU for the Krystal FFN (M2b): out[i] = max(0, in[i]).
//
// OpParams:
//   tokenCount = element count
//   inputOffset = in
//   outputOffset = out

  let linear = gid.x;
  if (linear >= op.tokenCount) { return; }
  let value = arena[op.inputOffset + linear];
  // WGSL has no ternary; select(false, true, cond).
  arena[op.outputOffset + linear] = select(0.0, value, value > 0.0);
