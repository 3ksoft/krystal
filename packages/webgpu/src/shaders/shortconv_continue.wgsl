// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: shortconv_continue
// workgroupSize: [1, 1, 1]
// builtins: workgroup_id -> wid

  let d = wid.x;
  if (d >= op.inputDim || op.tokenCount == 0u) { return; }

  let cacheBase = op.layerIndex * op.inputDim * 3u + d * 3u;
  var c0 = convCache[cacheBase];
  var c1 = convCache[cacheBase + 1u];
  var c2 = convCache[cacheBase + 2u];

  for (var t = 0u; t < op.tokenCount; t++) {
    let base = op.inputOffset + t * (op.inputDim * 3u);
    let B = arena[base + d];
    let C = arena[base + op.inputDim + d];
    let x = arena[base + op.inputDim * 2u + d];
    let bx = B * x;

    let y = c1 * weight32[d * 3u] +
            c2 * weight32[d * 3u + 1u] +
            bx * weight32[d * 3u + 2u];
    arena[op.outputOffset + t * op.inputDim + d] = C * y;

    c0 = c1;
    c1 = c2;
    c2 = bx;
  }

  convCache[cacheBase] = c0;
  convCache[cacheBase + 1u] = c1;
  convCache[cacheBase + 2u] = c2;
