// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: embedding
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid

  let total = op.tokenCount * op.outputDim;
  let linear = gid.x;
  if (linear >= total) { return; }

  let tokenIndex = linear / op.outputDim;
  let dim = linear % op.outputDim;
  let sourceToken = op.u0 + tokenIndex;
  let tokenId = select(tokens[sourceToken], runtime.currentToken, op.mode != 0u);

  if (tokenId < op.rowStart || tokenId >= op.rowStart + op.rowCount) { return; }
  let localRow = tokenId - op.rowStart;
  let value = load_f16(localRow * op.outputDim + dim);
  arena[arena_index(op.outputOffset, tokenIndex, dim, op.outputDim)] = value;
