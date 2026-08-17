// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: krystal_selector
// workgroupSize: [64, 1, 1]
// builtins: workgroup_id -> wid, local_invocation_id -> lid
//
// One typed selector slot over the record bank (architecture v2 §7, concerns
// answer 26):
//
//   score[i,j] = dot(qProj[i], kProj[j]) / sqrt(H) + mask[i,j]
//   p[i]       = softmax(score[i])          (0.0 allowed, -1e30 blocked)
//   gather[i]  = sum_j p[i,j] * value[j]
//   index[i]   = argmax_j p[i,j]            (first max on ties)
//
// qProj/kProj are the selector's Wq/Wk projections of the mixed query output
// and the pooled bank keys (computed by matmul_f32 before this dispatch).
// One workgroup per query row; lanes cover the R bank records, then the row
// softmax and the H-dim gather are reduced in workgroup memory. The selected
// index is stored as a u32 payload bitcast into the f32 arena.
//
// OpParams:
//   tokenCount = Q (query rows), inputDim = H, u0 = R (bank records)
//   inputOffset = qProj [Q, H]
//   auxOffset   = kProj [R, H]
//   aux2Offset  = value [R, H]  (pooled bank values)
//   aux3Offset  = mask [Q, R]
//   outputOffset = gather [Q, H]
//   aux4Offset  = p [Q, R]
//   aux5Offset  = selectedIndex [Q] (u32 payload)
//
// scale = 1 / sqrt(H) (answer 26).

  let i = wid.y;
  let Q = op.tokenCount;
  let R = op.u0;
  let H = op.inputDim;
  if (i >= Q) { return; }

  let scale = 1.0 / sqrt(f32(H));
  let qBase = op.inputOffset + i * H;

  // Pass 1: raw scores into workgroup memory (records looped by lanes).
  var j = lid.x;
  loop {
    if (j >= R) { break; }
    var s = 0.0;
    for (var d = 0u; d < H; d++) {
      s += arena[qBase + d] * arena[op.auxOffset + j * H + d];
    }
    attentionScores[j] = s * scale + arena[op.aux3Offset + i * R + j];
    j += WG;
  }
  workgroupBarrier();

  // Pass 2: row softmax + first-max argmax (single lane). An entirely
  // blocked row (every score ~= -1e30, no open mask position) produces a zero
  // distribution and an INVALID pointer instead of the degenerate
  // uniform-over-everything softmax: the gather is zero and the emit side
  // never fabricates a reference (S2-S10 contract, FOLLOW_UP2 metamorphic
  // equivalence). Real unblocked scores are bounded well above -1e29.
  if (lid.x == 0u) {
    var rowMax = -3.402823466e+38;
    for (var k = 0u; k < R; k++) { rowMax = max(rowMax, attentionScores[k]); }
    let allBlocked = rowMax < -1e29;
    var sumExp = 0.0;
    for (var k = 0u; k < R; k++) {
      let e = exp(attentionScores[k] - rowMax);
      attentionScores[k] = e;
      sumExp += e;
    }
    var inv = 1.0 / max(sumExp, 1e-20);
    if (allBlocked) { inv = 0.0; }
    var best = 0u;
    var bestScore = attentionScores[0] * inv;
    for (var k = 0u; k < R; k++) {
      let normalized = attentionScores[k] * inv;
      attentionScores[k] = normalized;
      if (normalized > bestScore) { bestScore = normalized; best = k; }
    }
    if (allBlocked) { best = 0xffffffffu; }
    arena[op.aux5Offset + i] = bitcast<f32>(best);
  }
  workgroupBarrier();

  // Pass 3: persist the distribution and gather the value vectors.
  j = lid.x;
  loop {
    if (j >= R) { break; }
    arena[op.aux4Offset + i * R + j] = attentionScores[j];
    j += WG;
  }
  workgroupBarrier();

  var dim = lid.x;
  loop {
    if (dim >= H) { break; }
    var value = 0.0;
    for (var k = 0u; k < R; k++) {
      value += attentionScores[k] * arena[op.aux2Offset + k * H + dim];
    }
    arena[op.outputOffset + i * H + dim] = value;
    dim += WG;
  }
