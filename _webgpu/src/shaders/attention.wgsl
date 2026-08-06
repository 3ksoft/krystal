// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: attention
// workgroupSize: [64, 1, 1]
// builtins: workgroup_id -> wid, local_invocation_id -> lid

  let head = wid.x;
  let tokenIndex = wid.y;
  if (head >= QUERY_HEADS || tokenIndex >= op.tokenCount) { return; }

  let qBase = op.inputOffset + tokenIndex * (QUERY_HEADS * HEAD_DIM) + head * HEAD_DIM;
  let kvHead = head / (QUERY_HEADS / KV_HEADS);
  let position = token_position(tokenIndex);
  // Clamp instead of early-returning: storage-buffer loads are non-uniform to
  // WGSL's uniformity analysis, so a return here would put the workgroupBarrier()
  // below in non-uniform control flow (rejected by Dawn). position < MAX_CONTEXT
  // is guaranteed by the runtime (contextCapacity <= MAX_CONTEXT), so clamping
  // is lossless.
  let contextCount = min(position + 1u, MAX_CONTEXT);

  let layerStride = runtime.contextCapacity * KV_DIM;
  let cacheK = (op.attentionSlot * 2u) * layerStride;
  let cacheV = cacheK + layerStride;

  var p = lid.x;
  loop {
    if (p >= contextCount) { break; }
    var dot = 0.0;
    for (var d = 0u; d < HEAD_DIM; d++) {
      let q = arena[qBase + d];
      let k = kvCache[cacheK + p * KV_DIM + kvHead * HEAD_DIM + d];
      dot += q * k;
    }
    attentionScores[p] = dot * 0.125; // 1 / sqrt(64)
    p += WG;
  }
  workgroupBarrier();

  if (lid.x == 0u) {
    var maxScore = -3.402823466e+38;
    for (var i = 0u; i < contextCount; i++) { maxScore = max(maxScore, attentionScores[i]); }
    var sumExp = 0.0;
    for (var i = 0u; i < contextCount; i++) {
      let e = exp(attentionScores[i] - maxScore);
      attentionScores[i] = e;
      sumExp += e;
    }
    let inv = 1.0 / max(sumExp, 1e-20);
    for (var i = 0u; i < contextCount; i++) { attentionScores[i] *= inv; }
  }
  workgroupBarrier();

  let d = lid.x;
  var value = 0.0;
  for (var i = 0u; i < contextCount; i++) {
    value += attentionScores[i] * kvCache[cacheV + i * KV_DIM + kvHead * HEAD_DIM + d];
  }
  let outIndex = tokenIndex * (QUERY_HEADS * HEAD_DIM) + head * HEAD_DIM + d;
  arena[op.outputOffset + outIndex] = value;
