// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: argmax_candidates
// workgroupSize: [256, 1, 1]
// builtins: local_invocation_id -> lid

  // Fixed-size guide tables are padded with op.u0 (= EMPTY_TOKEN / 0xffff).
  let emptyToken = op.u0;
  var bestValue = -3.402823466e+38;
  var bestToken = emptyToken;
  var i = lid.x;
  loop {
    if (i >= op.inputDim) { break; }
    let token = candidateTokens[i];
    if (token != emptyToken) {
      let v = arena[op.inputOffset + token];
      if (bestToken == emptyToken || v > bestValue || (v == bestValue && token < bestToken)) {
        bestValue = v;
        bestToken = token;
      }
    }
    i += ARGMAX_WG;
  }
  reduceF32[lid.x] = bestValue;
  reduceU32[lid.x] = bestToken;
  workgroupBarrier();

  var width = ARGMAX_WG >> 1u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) {
      let rv = reduceF32[lid.x + width];
      let rt = reduceU32[lid.x + width];
      let lv = reduceF32[lid.x];
      let lt = reduceU32[lid.x];
      if (lt == emptyToken || (rt != emptyToken && (rv > lv || (rv == lv && rt < lt)))) {
        reduceF32[lid.x] = rv;
        reduceU32[lid.x] = rt;
      }
    }
    workgroupBarrier();
    width >>= 1u;
  }

  if (lid.x == 0u && runtime.status == 1u) {
    let token = reduceU32[0];
    if (token == emptyToken) {
      runtime.status = 4u;
      runtime.errorCode = 0x47554944u; // "GUID"
      runtime.telemetryRevision += 1u;
    } else {
      if (op.mode != 0u) { runtime.position += 1u; }
      let step = runtime.generatedCount;
      let outIndex = runtime.contextCapacity + step;
      tokens[outIndex] = token;
      runtime.currentToken = token;
      runtime.lastToken = token;
      runtime.generatedCount += 1u;
      runtime.telemetryRevision += 1u;

      if (token == runtime.eosToken) {
        runtime.status = 2u;
      } else if (runtime.generatedCount >= runtime.maxNewTokens) {
        runtime.status = 3u;
      }
      emit_decode_telemetry(step, token);
    }
  }
