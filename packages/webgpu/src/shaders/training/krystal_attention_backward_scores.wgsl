// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: krystal_attention_backward_scores
// workgroupSize: [64, 1, 1]
// builtins: workgroup_id -> wid, local_invocation_id -> lid
//
// Softmax-score gradient of masked multi-head Krystal attention (M3, §17
// order). Cross-capable: qRows and kRows are independent, so the same body
// serves the record encoder's local attention (qRows == kRows) and the
// mixer's query->bank cross-attention (qRows != kRows).
//
// Given the saved attention probabilities P and the upstream gradient dOut:
//   dP[h][i][j]    = dot(dOut[i][h], V[j][h])
//   rowSum[h][i]   = sum_j P[h][i][j] * dP[h][i][j]
//   dScores[h][i][j] = P[h][i][j] * (dP[h][i][j] - rowSum[h][i])
//
// Masked positions carry P == 0.0, so their dScores gradient is exactly 0 and
// the mask never needs to be re-read here.
//
// Layouts (all f32 in the shared arena):
//   dOut   [qRows, H], head h at columns [h*headDim, (h+1)*headDim)
//   V      [kRows, H], same head layout
//   P      [headCount, qRows, kRows], P[(h*qRows + i)*kRows + j]
//   dScores [headCount, qRows, kRows], same layout as P
//
// OpParams:
//   tokenCount = qRows, inputDim = H, outputDim = headDim
//   u0 = kRows, u1 = headCount
//   inputOffset  = dOut   [qRows, H]
//   auxOffset    = V      [kRows, H]
//   aux2Offset   = P      [headCount, qRows, kRows]
//   outputOffset = dScores [headCount, qRows, kRows]
//
// One workgroup per (head, query row i); lanes cover the kRows keys.

  let head = wid.x;
  let i = wid.y;
  let qRows = op.tokenCount;
  let kRows = op.u0;
  let H = op.inputDim;
  let headDim = op.outputDim;
  let headCount = op.u1;
  if (head >= headCount || i >= qRows) { return; }

  let dOutBase = op.inputOffset + i * H + head * headDim;

  // Pass 1: dP[j] = dot(dOut[i,h], V[j,h]) into workgroup memory + rowSum.
  var j = lid.x;
  var localSum = 0.0;
  loop {
    if (j >= kRows) { break; }
    var dp = 0.0;
    for (var d = 0u; d < headDim; d++) {
      let dy = arena[dOutBase + d];
      let v = arena[op.auxOffset + j * H + head * headDim + d];
      dp += dy * v;
    }
    attentionScores[j] = dp;
    let p = arena[op.aux2Offset + (head * qRows + i) * kRows + j];
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

  // Pass 2: dScores = P * (dP - rowSum).
  j = lid.x;
  loop {
    if (j >= kRows) { break; }
    let p = arena[op.aux2Offset + (head * qRows + i) * kRows + j];
    arena[op.outputOffset + (head * qRows + i) * kRows + j] = p * (attentionScores[j] - rowSum);
    j += WG;
  }
