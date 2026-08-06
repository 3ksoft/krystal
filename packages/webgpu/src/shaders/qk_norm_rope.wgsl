// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: qk_norm_rope
// workgroupSize: [64, 1, 1]
// builtins: workgroup_id -> wid, local_invocation_id -> lid

  let head = wid.x;
  let tokenIndex = wid.y;
  if (tokenIndex >= op.tokenCount) { return; }
  let isK = op.u0 != 0u;
  let headCount = select(QUERY_HEADS, KV_HEADS, isK);
  if (head >= headCount) { return; }

  let base = select(op.inputOffset, op.auxOffset, isK);
  var x = rope_component(base, tokenIndex, head, lid.x, headCount);
  reduceF32[lid.x] = x * x;
  workgroupBarrier();

  var width = WG >> 1u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) { reduceF32[lid.x] += reduceF32[lid.x + width]; }
    workgroupBarrier();
    width >>= 1u;
  }

  let norm = inverseSqrt(reduceF32[0] / 64.0 + op.f0);
  x *= norm * weight32[lid.x];

  // Liquid's RoPE uses rotate_half over the two 32-wide halves.
  let half = HEAD_DIM / 2u;
  let pairDim = select(lid.x + half, lid.x - half, lid.x >= half);
  var pair = rope_component(base, tokenIndex, head, pairDim, headCount);
  pair *= norm * weight32[pairDim];
  pair = select(-pair, pair, lid.x >= half);
  // Every lane must capture its unrotated partner before any lane overwrites Q/K.
  workgroupBarrier();

  let freqDim = lid.x % half;
  let invFreq = pow(op.f1, -2.0 * f32(freqDim) / f32(HEAD_DIM));
  let angle = f32(token_position(tokenIndex)) * invFreq;
  let rotated = x * cos(angle) + pair * sin(angle);

  let dim = headCount * HEAD_DIM;
  arena[base + tokenIndex * dim + head * HEAD_DIM + lid.x] = rotated;
