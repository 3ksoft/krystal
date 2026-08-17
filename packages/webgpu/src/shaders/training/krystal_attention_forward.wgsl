// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: krystal_attention_forward
// workgroupSize: [64, 1, 1]
// builtins: workgroup_id -> wid, local_invocation_id -> lid
//
// Masked multi-head attention for the Krystal record encoder and mixer (M2b).
// Bidirectional with a host-compiled mask: 0.0 = allowed, -1e30 = blocked.
// qRows and kRows are independent so the same body serves both the record
// encoder's block-diagonal local attention (qRows == kRows) and the mixer's
// query->record-bank cross-attention (qRows != kRows). No KV cache, no GQA.
// Forward-only: no P persistence (M3 adds backward).
//
// Layouts (all f32 in the shared arena):
//   Q    [qRows, H]  head h owns columns [h*headDim, (h+1)*headDim)
//   K, V [kRows, H]  same head layout
//   mask [qRows, kRows] f32, mask[i*kRows + j]
//   out  [qRows, H]  same head layout as Q
//   P    [headCount, qRows, kRows], P[(h*qRows + i)*kRows + j] (persisted
//        for backward, mirroring attention_forward's aux4Offset contract)
//
// OpParams:
//   tokenCount = qRows, inputDim = H, outputDim = headDim
//   u0 = kRows, u1 = headCount
//   inputOffset = Q [qRows, H]
//   auxOffset   = K [kRows, H]
//   aux2Offset  = V [kRows, H]
//   aux3Offset  = mask [qRows, kRows]
//   outputOffset = out [qRows, H]
//   aux4Offset  = P [headCount, qRows, kRows]
// Optional record-local mode (u3 != 0), used by the Krystal encoder:
//   aux5Offset = activeTokens [qRows]
//   aux6Offset = recordCompactOffset [record slots]
//   u2 = arena offset of recordCompactCount [record slots]
//
// scale = 1 / sqrt(headDim), computed in WGSL.

  let head = wid.x;
  let i = wid.y;
  let qRows = op.tokenCount;
  let kRows = op.u0;
  let H = op.inputDim;
  let headDim = op.outputDim;
  let headCount = op.u1;
  if (head >= headCount || i >= qRows) { return; }

  let scale = 1.0 / sqrt(f32(headDim));
  let qBase = op.inputOffset + i * H + head * headDim;
  var keyStart = 0u;
  var keyEnd = kRows;
  if (op.u3 != 0u) {
    let frameTok = bitcast<u32>(arena[op.aux5Offset + i]);
    let slot = frameTok >> 3u;
    keyStart = bitcast<u32>(arena[op.aux6Offset + slot]);
    keyEnd = keyStart + bitcast<u32>(arena[op.u2 + slot]);
  }

  // Pass 1: raw scores into workgroup memory (keys looped by lanes).
  var j = keyStart + lid.x;
  loop {
    if (j >= keyEnd) { break; }
    var s = 0.0;
    for (var d = 0u; d < headDim; d++) {
      s += arena[qBase + d] * arena[op.auxOffset + j * H + head * headDim + d];
    }
    attentionScores[j] = s * scale + arena[op.aux3Offset + i * kRows + j];
    j += WG;
  }
  workgroupBarrier();

  // Pass 2: row max + sum-exp (single-lane reduction over the kRows keys).
  // An entirely blocked row (every score ~= -1e30, i.e. the host mask left no
  // open position) produces a zero distribution instead of the degenerate
  // uniform-over-everything softmax: the attention output and the persisted P
  // are both zero, keeping the forward bank-independent (FOLLOW_UP2
  // metamorphic equivalence). Real unblocked scores are bounded well above
  // -1e29, so rowMax cleanly separates the two cases.
  if (lid.x == 0u) {
    var rowMax = -3.402823466e+38;
    for (var k = keyStart; k < keyEnd; k++) { rowMax = max(rowMax, attentionScores[k]); }
    let allBlocked = rowMax < -1e29;
    var sumExp = 0.0;
    for (var k = keyStart; k < keyEnd; k++) {
      let e = exp(attentionScores[k] - rowMax);
      attentionScores[k] = e;
      sumExp += e;
    }
    var inv = 1.0 / max(sumExp, 1e-20);
    if (allBlocked) { inv = 0.0; }
    for (var k = keyStart; k < keyEnd; k++) { attentionScores[k] *= inv; }
  }
  workgroupBarrier();

  // Pass 3: persist probs (backward needs them), then reduce the context
  // vector per output dimension.
  j = keyStart + lid.x;
  loop {
    if (j >= keyEnd) { break; }
    arena[op.aux4Offset + (head * qRows + i) * kRows + j] = attentionScores[j];
    j += WG;
  }
  workgroupBarrier();
  var dim = lid.x;
  loop {
    if (dim >= headDim) { break; }
    var value = 0.0;
    for (var k = keyStart; k < keyEnd; k++) {
      value += attentionScores[k] * arena[op.aux2Offset + k * H + head * headDim + dim];
    }
    arena[op.outputOffset + i * H + head * headDim + dim] = value;
    dim += WG;
  }
