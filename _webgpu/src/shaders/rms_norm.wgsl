// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: rms_norm
// workgroupSize: [64, 1, 1]
// builtins: workgroup_id -> wid, local_invocation_id -> lid

  let tokenIndex = wid.x;
  if (tokenIndex >= op.tokenCount) { return; }

  var sum = 0.0;
  var d = lid.x;
  loop {
    if (d >= op.inputDim) { break; }
    let x = arena[arena_index(op.inputOffset, tokenIndex, d, op.inputDim)];
    sum += x * x;
    d += WG;
  }
  reduceF32[lid.x] = sum;
  workgroupBarrier();

  var width = WG >> 1u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) { reduceF32[lid.x] += reduceF32[lid.x + width]; }
    workgroupBarrier();
    width >>= 1u;
  }

  let invRms = inverseSqrt(reduceF32[0] / f32(op.inputDim) + op.f0);
  d = lid.x;
  loop {
    if (d >= op.inputDim) { break; }
    let x = arena[arena_index(op.inputOffset, tokenIndex, d, op.inputDim)];
    arena[arena_index(op.outputOffset, tokenIndex, d, op.inputDim)] = x * invRms * weight32[d];
    d += WG;
  }
