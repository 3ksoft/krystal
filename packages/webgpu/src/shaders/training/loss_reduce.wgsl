// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: loss_reduce
// workgroupSize: [64, 1, 1]
// builtins: local_invocation_id -> lid
//
// Deterministic single-workgroup mean reduction of lossRows [M] to one scalar:
//   arena[outputOffset] = sum_m lossRows[inputOffset + m] / M
//
// OpParams:
//   tokenCount = M
//   inputOffset  = lossRows [M]
//   outputOffset = scalar loss [1]
//
// The scalar is telemetry/debug only; the gradient already carries /M from the
// cross-entropy shader and never depends on this reduction.

  let M = op.tokenCount;
  var sum = 0.0;
  var i = lid.x;
  loop {
    if (i >= M) { break; }
    sum += arena[op.inputOffset + i];
    i += WG;
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
    let mean = reduceF32[0] / f32(M);
    arena[op.outputOffset] = mean;
    lossTelemetry[0] = mean;
  }
