// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: krystal_selector_backward_scores
// workgroupSize: [64, 1, 1]
// builtins: workgroup_id -> wid, local_invocation_id -> lid
//
// Soft-gather score gradient for one typed selector slot (M3, §17 item 8):
//
//   dP[i,j]     = dot(dGather[i], value[j])      (through the gather)
//   rowSum[i]   = sum_j p[i,j] * dP[i,j]
//   dScore[i,j] = p[i,j] * (dP[i,j] - rowSum[i])
//                 + pointerLossGrad[i,j]
//   pointerLossGrad[i,j] = p[i,j] - onehot(j == gold[i])
//                          when gold[i] is valid (not 0xffffffff)
//
// p is the persisted softmax distribution from the forward; masked positions
// carry p == 0, so their dScore gradient is exactly the pointer-loss term (0
// when that slot is not the gold). dScore [Q, R] is row-owned, so it is
// written directly here; dQProj/dKProj/dValue are computed by the companion
// krystal_selector_backward_qkv pass.
//
// OpParams:
//   tokenCount = Q, inputDim = H, u0 = R, u1 = zeroInvalidRows (0/1)
//   inputOffset  = dGather [Q, H]
//   auxOffset    = value   [R, H]
//   aux2Offset   = p       [Q, R]
//   aux3Offset   = gold    [Q] (u32 payloads; 0xffffffff = no pointer loss)
//   outputOffset = dScore  [Q, R]
//
// One workgroup per query row (mirrors krystal_selector). When
// zeroInvalidRows is set, rows with an INVALID target contribute exactly
// zero gradient (FOLLOW_UP2): an arity-0 / unlabelled argument row must not
// push the shared selector parameters through its degenerate all-masked
// softmax-backprop term.

  let i = wid.y;
  let Q = op.tokenCount;
  let R = op.u0;
  let H = op.inputDim;
  if (i >= Q) { return; }

  // Pass 1: dP[j] = dot(dGather[i], value[j]) into workgroup memory + rowSum.
  var j = lid.x;
  var localSum = 0.0;
  loop {
    if (j >= R) { break; }
    var dp = 0.0;
    for (var d = 0u; d < H; d++) {
      dp += arena[op.inputOffset + i * H + d] * arena[op.auxOffset + j * H + d];
    }
    attentionScores[j] = dp;
    let p = arena[op.aux2Offset + i * R + j];
    localSum += p * dp;
    j += WG;
  }
  reduceF32[lid.x] = localSum;
  workgroupBarrier();
  var width = WG >> 1u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) { reduceF32[lid.x] += reduceF32[lid.x + width]; }
    workgroupBarrier();
    width >>= 1u;
  }
  let rowSum = reduceF32[0];
  workgroupBarrier();

  // Pass 2: dScore = p * (dP - rowSum) + pointer-loss gradient.
  let gold = bitcast<u32>(arena[op.aux3Offset + i]);
  if (op.u1 != 0u && gold == 0xffffffffu) {
    // No pointer target: the branch contributes exactly zero gradient.
    j = lid.x;
    loop {
      if (j >= R) { break; }
      arena[op.outputOffset + i * R + j] = 0.0;
      j += WG;
    }
    return;
  }
  j = lid.x;
  loop {
    if (j >= R) { break; }
    let p = arena[op.aux2Offset + i * R + j];
    var pointerGrad = 0.0;
    if (gold != 0xffffffffu && j == gold) { pointerGrad = p - 1.0; }
    else if (gold != 0xffffffffu) { pointerGrad = p; }
    arena[op.outputOffset + i * R + j] = p * (attentionScores[j] - rowSum) + pointerGrad;
    j += WG;
  }
