// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: silu_mul
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid

  let total = op.tokenCount * op.inputDim;
  let i = gid.x;
  if (i >= total) { return; }
  let gate = arena[op.inputOffset + i];
  let up = arena[op.auxOffset + i];
  arena[op.outputOffset + i] = (gate / (1.0 + exp(-gate))) * up;
