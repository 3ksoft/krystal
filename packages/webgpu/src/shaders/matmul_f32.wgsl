// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: matmul_f32
// workgroupSize: [64, 1, 1]
// builtins: workgroup_id -> wid, local_invocation_id -> lid

  let localRow = wid.x;
  let tokenIndex = wid.y;
  if (localRow >= op.rowCount || tokenIndex >= op.tokenCount) { return; }

  var sum = 0.0;
  var k = lid.x;
  loop {
    if (k >= op.inputDim) { break; }
    let x = arena[arena_index(op.inputOffset, tokenIndex, k, op.inputDim)];
    sum += x * weight32[localRow * op.inputDim + k];
    k += WG;
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

  if (lid.x == 0u) {
    let logicalRow = op.rowStart + localRow;
    arena[op.outputOffset + tokenIndex * op.outputDim + logicalRow] = reduceF32[0];
  }
