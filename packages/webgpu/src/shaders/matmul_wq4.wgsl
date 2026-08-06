// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: matmul_wq4
// workgroupSize: [64, 1, 1]
// builtins: workgroup_id -> wid, local_invocation_id -> lid

  let localRow = wid.x;
  let tokenIndex = wid.y;
  if (localRow >= op.rowCount || tokenIndex >= op.tokenCount) { return; }

  let inputBase = op.inputOffset + tokenIndex * op.inputDim;
  let blocksPerRow = op.inputDim / 32u;
  // Weight pages are row-local. rowStart is only the logical output-row offset.
  let rowBlockStart = localRow * blocksPerRow;

  var sum: f32 = 0.0;
  var b = lid.x;
  loop {
    if (b >= blocksPerRow) { break; }

    let blockIdx = rowBlockStart + b;
    let baseU32 = blockIdx * 5u; // 4 packed words + i32 pow2 exponent = 20 B
    let expVal = bitcast<i32>(weightRaw[baseU32 + 4u]);
    let scale = exp2(f32(expVal));
    let kStart = b * 32u;

    for (var w = 0u; w < 4u; w++) {
      let packed = weightRaw[baseU32 + w];
      let kBase = kStart + w * 8u;

      sum += (f32((packed >>  0u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 0u];
      sum += (f32((packed >>  4u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 1u];
      sum += (f32((packed >>  8u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 2u];
      sum += (f32((packed >> 12u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 3u];
      sum += (f32((packed >> 16u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 4u];
      sum += (f32((packed >> 20u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 5u];
      sum += (f32((packed >> 24u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 6u];
      sum += (f32((packed >> 28u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 7u];
    }

    b += WG;
  }

  reduceF32[lid.x] = sum;
  workgroupBarrier();

  var width = WG >> 1u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) { reduceF32[lid.x] += reduceF32[lid.x + width]; }
    workgroupBarrier();
    width >>= 1u;
  }

  if (lid.x == 0u) {
    let logicalRow = op.rowStart + localRow;
    arena[op.outputOffset + tokenIndex * op.outputDim + logicalRow] = reduceF32[0];
  }
