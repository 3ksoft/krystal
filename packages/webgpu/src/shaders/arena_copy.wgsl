// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: arena_copy
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid

  let total = op.tokenCount * op.inputDim;
  let i = gid.x;
  if (i >= total) { return; }
  arena[op.outputOffset + i] = arena[op.inputOffset + i];
