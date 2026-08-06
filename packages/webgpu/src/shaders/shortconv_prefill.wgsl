// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: shortconv_prefill
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid

  let total = op.tokenCount * op.inputDim;
  let linear = gid.x;
  if (linear >= total) { return; }
  let t = linear / op.inputDim;
  let d = linear % op.inputDim;

  // in_proj output layout is [B | C | x], each hidden-sized.
  let base = op.inputOffset + t * (op.inputDim * 3u);
  let B = arena[base + d];
  let C = arena[base + op.inputDim + d];
  let x = arena[base + op.inputDim * 2u + d];
  let bx0 = B * x;

  var y = bx0 * weight32[d * 3u + 2u];
  if (t >= 1u) {
    let p = op.inputOffset + (t - 1u) * (op.inputDim * 3u);
    y += arena[p + d] * arena[p + op.inputDim * 2u + d] * weight32[d * 3u + 1u];
  }
  if (t >= 2u) {
    let p = op.inputOffset + (t - 2u) * (op.inputDim * 3u);
    y += arena[p + d] * arena[p + op.inputDim * 2u + d] * weight32[d * 3u];
  }

  arena[op.outputOffset + t * op.inputDim + d] = C * y;

  // Persist only the tail needed by the first decode step.
  let keepStart = select(0u, op.tokenCount - 3u, op.tokenCount > 3u);
  if (t >= keepStart) {
    let slot = select(3u - op.tokenCount + t, t - keepStart, op.tokenCount >= 3u);
    let cacheBase = op.layerIndex * op.inputDim * 3u + d * 3u;
    convCache[cacheBase + slot] = bx0;
  }
