// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: shortconv_decode
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid

  let d = gid.x;
  if (d >= op.inputDim) { return; }
  let base = op.inputOffset;
  let B = arena[base + d];
  let C = arena[base + op.inputDim + d];
  let x = arena[base + op.inputDim * 2u + d];
  let bx = B * x;

  let cacheBase = op.layerIndex * op.inputDim * 3u + d * 3u;
  let a = convCache[cacheBase + 1u];
  let b = convCache[cacheBase + 2u];
  convCache[cacheBase] = a;
  convCache[cacheBase + 1u] = b;
  convCache[cacheBase + 2u] = bx;

  let y = a * weight32[d * 3u] +
          b * weight32[d * 3u + 1u] +
          bx * weight32[d * 3u + 2u];
  arena[op.outputOffset + d] = C * y;
