// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: zero_f32
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid
//
// Zero `tokenCount` f32 elements at `outputOffset`.
//
// This exists for gradient buffers that truly accumulate (embedding scatter
// variants, later fused kernels). The current training graph fully overwrites
// every gradient, so trainStep does not dispatch it; it is part of the
// operator contract and tested standalone.

  let i = gid.x;
  if (i >= op.tokenCount) { return; }
  arena[op.outputOffset + i] = 0.0;
