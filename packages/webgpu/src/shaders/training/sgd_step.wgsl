// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: sgd_step
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid
//
// Plain SGD without momentum:
//
//   weight32[i] = weight32[i] - learningRate * arena[inputOffset + i]
//
// OpParams:
//   tokenCount = element count
//   inputOffset = gradient offset (arena)
//   f0 = learning rate
//   weight32 = the trainable parameter page (read-write storage)
//
// Each invocation owns one parameter element. Only explicitly registered
// trainable pages are ever bound here; frozen buffers are never dispatched
// through this shader.

  let i = gid.x;
  if (i >= op.tokenCount) { return; }
  weight32[i] = weight32[i] - op.f0 * arena[op.inputOffset + i];
