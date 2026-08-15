// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: attention_forward
// workgroupSize: [64, 1, 1]
// builtins: workgroup_id -> wid, local_invocation_id -> lid
//
// Masked multi-head attention forward for the Krystal encoder
// (WEBGPU_BACKWARD_PLAN.md §17 item 6). Bidirectional over the M rows, with a
// host-compiled mask: 0.0 = allowed, -1e30 = blocked. No KV cache, no GQA.
//
// Layouts (all f32 in the shared arena):
//   Q, K, V    [M, H] row-major; head h owns columns [h*headDim, (h+1)*headDim)
//   mask       [M, M] f32, mask[i*M+j]
//   out        [M, H] row-major, same head layout as Q/K/V
//   P (probs)  [headCount, M, M], P[(h*M + i)*M + j]
//
// OpParams:
//   tokenCount = M, inputDim = H, outputDim = headDim, u0 = headCount
//   inputOffset  = Q   [M,H]
//   auxOffset    = K   [M,H]
//   aux2Offset   = V   [M,H]
//   aux3Offset   = mask [M,M]
//   outputOffset = out [M,H]
//   aux4Offset   = P   [headCount,M,M]
//
// One workgroup per (head, query row i): lanes cover the M keys with a strided
// loop, then the row's softmax is computed in workgroup memory and the context
// vector is reduced per output dimension.
//
// scale = 1 / sqrt(headDim), computed in WGSL.

  let head = wid.x;
  let i = wid.y;
  let M = op.tokenCount;
  let H = op.inputDim;
  let headDim = op.outputDim;
  let headCount = op.u0;
  if (head >= headCount || i >= M) { return; }

  let scale = 1.0 / sqrt(f32(headDim));
  let qBase = op.inputOffset + i * H + head * headDim;

  // Pass 1: raw scores into workgroup memory (keys looped by lanes).
  var j = lid.x;
  loop {
    if (j >= M) { break; }
    var s = 0.0;
    for (var d = 0u; d < headDim; d++) {
      let q = arena[qBase + d];
      let k = arena[op.auxOffset + j * H + head * headDim + d];
      s += q * k;
    }
    attentionScores[j] = s * scale + arena[op.aux3Offset + i * M + j];
    j += WG;
  }
  workgroupBarrier();

  // Pass 2: row max + sum-exp (single-lane reduction over the M keys).
  if (lid.x == 0u) {
    var rowMax = -3.402823466e+38;
    for (var k = 0u; k < M; k++) { rowMax = max(rowMax, attentionScores[k]); }
    var sumExp = 0.0;
    for (var k = 0u; k < M; k++) {
      let e = exp(attentionScores[k] - rowMax);
      attentionScores[k] = e;
      sumExp += e;
    }
    let inv = 1.0 / max(sumExp, 1e-20);
    for (var k = 0u; k < M; k++) { attentionScores[k] *= inv; }
  }
  workgroupBarrier();

  // Pass 3: persist probs (backward needs them) and reduce the context vector.
  j = lid.x;
  loop {
    if (j >= M) { break; }
    arena[op.aux4Offset + (head * M + i) * M + j] = attentionScores[j];
    j += WG;
  }
  workgroupBarrier();

  var dim = lid.x;
  loop {
    if (dim >= headDim) { break; }
    var value = 0.0;
    for (var k = 0u; k < M; k++) {
      value += attentionScores[k] * arena[op.aux2Offset + k * H + head * headDim + dim];
    }
    arena[op.outputOffset + i * H + head * headDim + dim] = value;
    dim += WG;
  }
