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
//
// OpParams:
//   tokenCount = qRows, inputDim = H, outputDim = headDim
//   u0 = kRows, u1 = headCount
//   inputOffset = Q [qRows, H]
//   auxOffset   = K [kRows, H]
//   aux2Offset  = V [kRows, H]
//   aux3Offset  = mask [qRows, kRows]
//   outputOffset = out [qRows, H]
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

  // Pass 1: raw scores into workgroup memory (keys looped by lanes).
  var j = lid.x;
  loop {
    if (j >= kRows) { break; }
    var s = 0.0;
    for (var d = 0u; d < headDim; d++) {
      s += arena[qBase + d] * arena[op.auxOffset + j * H + head * headDim + d];
    }
    attentionScores[j] = s * scale + arena[op.aux3Offset + i * kRows + j];
    j += WG;
  }
  workgroupBarrier();

  // Pass 2: row max + sum-exp (single-lane reduction over the kRows keys).
  if (lid.x == 0u) {
    var rowMax = -3.402823466e+38;
    for (var k = 0u; k < kRows; k++) { rowMax = max(rowMax, attentionScores[k]); }
    var sumExp = 0.0;
    for (var k = 0u; k < kRows; k++) {
      let e = exp(attentionScores[k] - rowMax);
      attentionScores[k] = e;
      sumExp += e;
    }
    let inv = 1.0 / max(sumExp, 1e-20);
    for (var k = 0u; k < kRows; k++) { attentionScores[k] *= inv; }
  }
  workgroupBarrier();

  // Pass 3: reduce the context vector per output dimension.
  var dim = lid.x;
  loop {
    if (dim >= headDim) { break; }
    var value = 0.0;
    for (var k = 0u; k < kRows; k++) {
      value += attentionScores[k] * arena[op.aux2Offset + k * H + head * headDim + dim];
    }
    arena[op.outputOffset + i * H + head * headDim + dim] = value;
    dim += WG;
  }
