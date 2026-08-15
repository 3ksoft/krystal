// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: attention_backward_scores
// workgroupSize: [64, 1, 1]
// builtins: workgroup_id -> wid, local_invocation_id -> lid
//
// Softmax-score gradient of masked multi-head attention (§17 item 6).
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
//   dOut  [M, H], head h at columns [h*headDim, (h+1)*headDim)
//   V     [M, H], same head layout
//   P     [headCount, M, M], P[(h*M + i)*M + j]
//   dScores [headCount, M, M], same layout as P
//
// OpParams:
//   tokenCount = M, inputDim = H, outputDim = headDim, u0 = headCount
//   inputOffset  = dOut  [M,H]
//   auxOffset    = V     [M,H]
//   aux2Offset   = P     [headCount,M,M]
//   outputOffset = dScores [headCount,M,M]
//
// One workgroup per (head, query row i); lanes cover the M keys.

  let head = wid.x;
  let i = wid.y;
  let M = op.tokenCount;
  let H = op.inputDim;
  let headDim = op.outputDim;
  let headCount = op.u0;
  if (head >= headCount || i >= M) { return; }

  let dOutBase = op.inputOffset + i * H + head * headDim;

  // Pass 1: dP[j] = dot(dOut[i,h], V[j,h]) into workgroup memory + rowSum.
  var j = lid.x;
  var localSum = 0.0;
  loop {
    if (j >= M) { break; }
    var dp = 0.0;
    for (var d = 0u; d < headDim; d++) {
      let dy = arena[dOutBase + d];
      let v = arena[op.auxOffset + j * H + head * headDim + d];
      dp += dy * v;
    }
    attentionScores[j] = dp;
    let p = arena[op.aux2Offset + (head * M + i) * M + j];
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
    if (j >= M) { break; }
    let p = arena[op.aux2Offset + (head * M + i) * M + j];
    arena[op.outputOffset + (head * M + i) * M + j] = p * (attentionScores[j] - rowSum);
    j += WG;
  }
