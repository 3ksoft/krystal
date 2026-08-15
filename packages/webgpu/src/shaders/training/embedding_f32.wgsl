// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: embedding_f32
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid
//
// f32 embedding lookup for training (the legacy embedding.wgsl reads f16/WQ4
// weight pages, which is the inference path; training keeps f32 parameters).
//
//   hidden[m,h] = embedding[tokens[u0 + m], h]
//
// OpParams:
//   tokenCount = M, inputDim = V, outputDim = H
//   outputOffset = hidden [M,H]
//   u0 = token-id base offset inside the tokens buffer
//   weight32 = embedding table [V,H] f32 (row-major)
//
// Out-of-range token ids write 0 so the row is deterministic, not stale.

  let total = op.tokenCount * op.outputDim;
  let linear = gid.x;
  if (linear >= total) { return; }

  let m = linear / op.outputDim;
  let h = linear % op.outputDim;
  let tokenId = tokens[op.u0 + m];
  if (tokenId < op.inputDim) {
    arena[op.outputOffset + linear] = weight32[tokenId * op.outputDim + h];
  } else {
    arena[op.outputOffset + linear] = 0.0;
  }
