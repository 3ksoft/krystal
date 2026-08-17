// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: krystal_field_embed_sgd
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid
//
// Fused sparse field-embedding backward + SGD for the composed trainer.
// The host supplies the distinct concatenated embedding rows referenced by
// this frame. Each invocation owns one (row, hidden-column), scans active
// tokens in the same order as krystal_field_embed_backward, and updates that
// weight directly. This is mathematically identical to the dense
// gradient-then-SGD pair but does no work for unused rows.
//
// OpParams:
//   tokenCount = U, number of distinct active embedding rows
//   inputDim = H, outputDim = T_active
//   inputOffset  = dFieldStates [T_active, H]
//   outputOffset = activeRows [U] (u32 payloads in the f32 arena)
//   auxOffset..aux6Offset = frame metadata, matching the dense backward pass
//   u0..u5 = cumulative table row counts
//   f0 = learning rate
//   weight32 = concatenated embedding page (read-write)

  let rowCount = op.tokenCount;
  let H = op.inputDim;
  let tCount = op.outputDim;
  let linear = gid.x;
  if (linear >= rowCount * H) { return; }

  let sparseRow = linear / H;
  let h = linear % H;
  let row = bitcast<u32>(arena[op.outputOffset + sparseRow]);

  var tableId = 0u;
  var startRow = 0u;
  if (row < op.u0) { tableId = 0u; startRow = 0u; }
  else if (row < op.u1) { tableId = 1u; startRow = op.u0; }
  else if (row < op.u2) { tableId = 2u; startRow = op.u1; }
  else if (row < op.u3) { tableId = 3u; startRow = op.u2; }
  else if (row < op.u4) { tableId = 4u; startRow = op.u3; }
  else { tableId = 5u; startRow = op.u4; }
  let localRow = row - startRow;

  var sum = 0.0;
  for (var t = 0u; t < tCount; t++) {
    let frameTok = bitcast<u32>(arena[op.aux6Offset + t]);
    let slot = frameTok >> 3u;
    var index = 0u;
    if (tableId == 0u) {
      index = bitcast<u32>(arena[op.auxOffset + frameTok]);
    } else if (tableId == 1u) {
      index = bitcast<u32>(arena[op.aux2Offset + frameTok]);
    } else if (tableId == 2u) {
      index = bitcast<u32>(arena[op.aux3Offset + slot]);
    } else if (tableId == 3u) {
      index = bitcast<u32>(arena[op.aux4Offset + slot]);
    } else if (tableId == 4u) {
      index = bitcast<u32>(arena[op.aux5Offset + slot]);
    } else {
      index = frameTok & 7u;
    }
    if (index == localRow) {
      sum += arena[op.inputOffset + t * H + h];
    }
  }

  let weightIndex = row * H + h;
  weight32[weightIndex] = weight32[weightIndex] - op.f0 * sum;
