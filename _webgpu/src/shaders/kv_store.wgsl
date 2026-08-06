// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: kv_store
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid

  let total = op.tokenCount * KV_DIM;
  let linear = gid.x;
  if (linear >= total) { return; }
  let t = linear / KV_DIM;
  let d = linear % KV_DIM;
  let position = token_position(t);
  if (position >= runtime.contextCapacity) { return; }

  let layerStride = runtime.contextCapacity * KV_DIM;
  let baseK = (op.attentionSlot * 2u) * layerStride;
  let baseV = baseK + layerStride;
  kvCache[baseK + position * KV_DIM + d] = arena[op.inputOffset + t * KV_DIM + d];
  kvCache[baseV + position * KV_DIM + d] = arena[op.auxOffset + t * KV_DIM + d];
