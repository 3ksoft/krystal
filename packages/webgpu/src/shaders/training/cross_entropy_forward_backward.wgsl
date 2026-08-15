// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: cross_entropy_forward_backward
// workgroupSize: [64, 1, 1]
// builtins: workgroup_id -> wid, local_invocation_id -> lid
//
// Fused numerically-stable softmax cross-entropy forward + dLogits backward.
// One workgroup per row m; WG lanes cover the vocabulary with a strided loop.
//
// Per row m (mean reduction, /M applied exactly once to dLogits):
//   rowMax = max_v logits[m,v]
//   sumExp = sum_v exp(logits[m,v] - rowMax)
//   logZ   = rowMax + log(sumExp)
//   lossRows[m] = logZ - logits[m, targets[u1 + m]]
//   dLogits[m,v] = (exp(logits[m,v] - logZ) - one_hot(targets[u1 + m], v)) / M
//
// OpParams:
//   tokenCount = M, outputDim = V
//   inputOffset  = logits  [M,V]
//   outputOffset = dLogits [M,V]
//   auxOffset    = lossRows [M]
//   u1 = target-id base offset inside the targets buffer
//
// A target outside [0, V) writes NaN to its loss row so any test that passes
// one fails loudly; the host validates targets before dispatch.

  let m = wid.x;
  if (m >= op.tokenCount) { return; }
  let V = op.outputDim;
  let base = op.inputOffset + m * V;
  let targetId = targets[op.u1 + m];

  // Pass 1: row max.
  var localMax = -1.0e30;
  var v = lid.x;
  loop {
    if (v >= V) { break; }
    localMax = max(localMax, arena[base + v]);
    v += WG;
  }
  reduceF32[lid.x] = localMax;
  workgroupBarrier();
  var width = WG >> 1u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) { reduceF32[lid.x] = max(reduceF32[lid.x], reduceF32[lid.x + width]); }
    workgroupBarrier();
    width >>= 1u;
  }
  let rowMax = reduceF32[0];
  workgroupBarrier();

  // Pass 2: sum of exp(logit - rowMax).
  var localSum = 0.0;
  v = lid.x;
  loop {
    if (v >= V) { break; }
    localSum += exp(arena[base + v] - rowMax);
    v += WG;
  }
  reduceF32[lid.x] = localSum;
  workgroupBarrier();
  width = WG >> 1u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) { reduceF32[lid.x] += reduceF32[lid.x + width]; }
    workgroupBarrier();
    width >>= 1u;
  }
  let logZ = rowMax + log(reduceF32[0]);
  workgroupBarrier();

  // Pass 3: dLogits for the owned slice + the row's loss.
  if (lid.x == 0u) {
    if (targetId < V) {
      arena[op.auxOffset + m] = logZ - arena[base + targetId];
    } else {
      // NaN via runtime division (a literal NaN is rejected at parse time).
      let zero = 0.0;
      arena[op.auxOffset + m] = zero / zero;
    }
  }
  v = lid.x;
  loop {
    if (v >= V) { break; }
    let prob = exp(arena[base + v] - logZ);
    let oneHot = select(0.0, 1.0, v == targetId);
    arena[op.outputOffset + m * V + v] = (prob - oneHot) / f32(op.tokenCount);
    v += WG;
  }
