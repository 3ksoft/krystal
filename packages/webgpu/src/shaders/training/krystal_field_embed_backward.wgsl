// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: krystal_field_embed_backward
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid
//
// Scatter-add gradient of the Krystal field embedding (M3, §17 order):
//
//   dEmbedding[table, row, h] = sum over active tokens t where the table's
//     index for t == row of dFieldStates[t, h]
//
// The six tables (token/field/schema/band/stream/pos) are concatenated in one
// f32 page; a linear gid owns one (table, row, h) element and scans the active
// tokens, so every dEmbedding element has exactly one owner (deterministic,
// no atomics, zero for unused rows). The table is identified from the row by
// the cumulative row counts passed in u0..u5.
//
// Token index per table (mirrors krystal_field_embed forward):
//   table 0 token  -> tokenIds[frameTok]
//   table 1 field  -> fieldRoles[frameTok]
//   table 2 schema -> schemaIds[slot]
//   table 3 band   -> bandIds[slot]
//   table 4 stream -> streamIds[slot]
//   table 5 pos    -> frameTok & 7 (learned record-local position)
// with slot = frameTok >> 3 (record width is the frozen ABI value 8).
//
// OpParams:
//   tokenCount = totalTableRows (sum of the six table row spaces)
//   inputDim = H, outputDim = T_active
//   u0..u5 = cumulative row counts: u0 = rows(table0), u1 = rows(table0+1),
//     ... u5 = total rows
//   inputOffset  = dFieldStates [T_active, H]
//   outputOffset = dEmbedding region base (whole concatenated page)
//   auxOffset    = tokenIds   [maxTokens]
//   aux2Offset   = fieldRoles [maxTokens]
//   aux3Offset   = schemaIds  [maxRecords]
//   aux4Offset   = bandIds    [maxRecords]
//   aux5Offset   = streamIds  [maxRecords]
//   aux6Offset   = activeTokens [T_active]

  let totalRows = op.tokenCount;
  let H = op.inputDim;
  let tCount = op.outputDim;
  let linear = gid.x;
  if (linear >= totalRows * H) { return; }

  let row = linear / H;
  let h = linear % H;

  // Identify the table from cumulative row counts; startRow is the table's
  // first row in the concatenation, and word base = startRow * H, so the
  // write offset is simply outputOffset + linear.
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
  arena[op.outputOffset + linear] = sum;
