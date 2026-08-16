// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: krystal_field_embed
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid
//
// Krystal field embedding (M2b). One invocation per active field token writes
//
//   fieldStates[t, h] = E_token[tok] + E_field[role] + E_schema[schema]
//                       + E_band[band] + E_stream[stream] + E_pos[pos]
//
// (KRYSTAL_BRAIN_ARCHITECTURE_V2.md §4.3; concerns answers 14/24: five
// additive embeddings plus the learned record-local position table 0..7).
// The six tables are concatenated into one f32 weight page; this shader reads
// the table bases from u0..u5. The SoA frame payloads are u32 and arrive in
// the shared arena (bitcast here); tokenIds/fieldRoles are indexed by frame
// token id, schemaIds/bandIds/streamIds by record slot, and activeTokens is
// the compact list of active frame token ids. Record width is the frozen ABI
// value 8, so slot = frameToken >> 3 and local position = frameToken & 7.
//
// OpParams:
//   tokenCount = T_active, inputDim = H
//   inputOffset = tokenIds   [maxTokens]
//   auxOffset   = fieldRoles [maxTokens]
//   aux2Offset  = schemaIds  [maxRecords]
//   aux3Offset  = bandIds    [maxRecords]
//   aux4Offset  = activeTokens [T_active]
//   aux5Offset  = streamIds  [maxRecords]
//   outputOffset = fieldStates [T_active, H]
//   u0..u5 = E_token/E_field/E_schema/E_band/E_stream/E_pos table bases (f32 words)
//   weight32 = concatenated embedding tables [rows, H]

  let total = op.tokenCount * op.inputDim;
  let linear = gid.x;
  if (linear >= total) { return; }

  let t = linear / op.inputDim;
  let h = linear % op.inputDim;

  let frameTok = bitcast<u32>(arena[op.aux4Offset + t]);
  let slot = frameTok >> 3u;
  let local = frameTok & 7u;

  let tok = bitcast<u32>(arena[op.inputOffset + frameTok]);
  let role = bitcast<u32>(arena[op.auxOffset + frameTok]);
  let schema = bitcast<u32>(arena[op.aux2Offset + slot]);
  let band = bitcast<u32>(arena[op.aux3Offset + slot]);
  let stream = bitcast<u32>(arena[op.aux5Offset + slot]);

  var value = 0.0;
  value += weight32[op.u0 + tok * op.inputDim + h];
  value += weight32[op.u1 + role * op.inputDim + h];
  value += weight32[op.u2 + schema * op.inputDim + h];
  value += weight32[op.u3 + band * op.inputDim + h];
  value += weight32[op.u4 + stream * op.inputDim + h];
  value += weight32[op.u5 + local * op.inputDim + h];
  arena[op.outputOffset + linear] = value;
