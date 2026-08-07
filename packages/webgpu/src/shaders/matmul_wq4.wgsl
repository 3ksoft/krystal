// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: matmul_wq4
// workgroupSize: [64, 1, 1]
// builtins: workgroup_id -> wid, local_invocation_id -> lid
//
// One workgroup computes MATMUL_ROWS output rows.
//
// The previous shape — one row per workgroup — achieved ~22 GiB/s while a bare
// streaming kernel reaches ~300 GiB/s on the same buffer, so the kernel was
// never memory-bound. It was launch- and reduction-bound: 52k workgroups each
// producing a single scalar, with a six-step barrier tree per result and only
// one 32-value block of work per thread.
//
// Tiling rows amortizes both. Measured against the untiled kernel (RTX 3060,
// tests/matmul-variants.test.ts): LM head 5.1x, ffn_gate/up 4.9x, ffn_down
// 4.9x, attn projections 3.8x.
//
// Numerics: each output row is still accumulated by one thread over the same
// blocks in the same order, but the block scale is now applied once per block
// instead of per element. Multiplying by the scale is exact (it is a power of
// two); what changes is the association of the 32 additions relative to the
// running total, so results are equivalent to f32 rounding rather than
// bit-identical. Observed worst case is ~7e-4 on the longest reduction
// (inputDim 8192, 256 blocks) and 0 on inputDim 2048. The exact-output
// checkpoint and structured-generation suites cover whether that ever moves a
// token; they pass.
//
// MATMUL_ROWS must stay in sync with the dispatch in lfm2-definition.ts, which
// launches ceil(rowCount / MATMUL_ROWS) workgroups.

  let rowBase = wid.x * MATMUL_ROWS;
  let tokenIndex = wid.y;
  if (rowBase >= op.rowCount || tokenIndex >= op.tokenCount) { return; }

  let inputBase = op.inputOffset + tokenIndex * op.inputDim;
  let blocksPerRow = op.inputDim / 32u;

  var acc: array<f32, MATMUL_ROWS>;
  for (var r = 0u; r < MATMUL_ROWS; r++) { acc[r] = 0.0; }

  var b = lid.x;
  loop {
    if (b >= blocksPerRow) { break; }
    let kStart = b * 32u;

    for (var r = 0u; r < MATMUL_ROWS; r++) {
      // Weight pages are row-local; rowStart is only the logical output offset.
      let base = (rowBase + r) * blocksPerRow * 5u + b * 5u;
      let scale = exp2(f32(bitcast<i32>(weightRaw[base + 4u])));

      var blockSum: f32 = 0.0;
      for (var w = 0u; w < 4u; w++) {
        let packed = weightRaw[base + w];
        let kBase = kStart + w * 8u;
        blockSum += (f32((packed >>  0u) & 0x0Fu) - 8.0) * arena[inputBase + kBase + 0u];
        blockSum += (f32((packed >>  4u) & 0x0Fu) - 8.0) * arena[inputBase + kBase + 1u];
        blockSum += (f32((packed >>  8u) & 0x0Fu) - 8.0) * arena[inputBase + kBase + 2u];
        blockSum += (f32((packed >> 12u) & 0x0Fu) - 8.0) * arena[inputBase + kBase + 3u];
        blockSum += (f32((packed >> 16u) & 0x0Fu) - 8.0) * arena[inputBase + kBase + 4u];
        blockSum += (f32((packed >> 20u) & 0x0Fu) - 8.0) * arena[inputBase + kBase + 5u];
        blockSum += (f32((packed >> 24u) & 0x0Fu) - 8.0) * arena[inputBase + kBase + 6u];
        blockSum += (f32((packed >> 28u) & 0x0Fu) - 8.0) * arena[inputBase + kBase + 7u];
      }
      acc[r] += blockSum * scale;
    }

    b += WG;
  }

  // One tree reduction per row. The trailing barrier is what makes reusing the
  // shared buffer for the next row safe.
  for (var r = 0u; r < MATMUL_ROWS; r++) {
    reduceF32[lid.x] = acc[r];
    workgroupBarrier();

    var width = WG >> 1u;
    loop {
      if (width == 0u) { break; }
      if (lid.x < width) { reduceF32[lid.x] += reduceF32[lid.x + width]; }
      workgroupBarrier();
      width >>= 1u;
    }

    if (lid.x == 0u && rowBase + r < op.rowCount) {
      let logicalRow = op.rowStart + rowBase + r;
      arena[op.outputOffset + tokenIndex * op.outputDim + logicalRow] = reduceF32[0];
    }
    workgroupBarrier();
  }
