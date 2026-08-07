// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: argmax
// workgroupSize: [256, 1, 1]
// builtins: local_invocation_id -> lid

  // op.u0 is the globally forbidden EMPTY_TOKEN sentinel.
  let emptyToken = op.u0;

  // No early return before the barriers below: runtime.status is a storage-buffer
  // load (non-uniform to WGSL), which would put workgroupBarrier() in non-uniform
  // control flow. The status gate is applied on the final lane-0 write instead.
  var bestValue = -3.402823466e+38;
  var bestToken = emptyToken;
  var i = lid.x;
  loop {
    if (i >= op.inputDim) { break; }
    if (i != emptyToken) {
      let v = arena[op.inputOffset + i];
      if (bestToken == emptyToken || v > bestValue || (v == bestValue && i < bestToken)) {
        bestValue = v;
        bestToken = i;
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

  if (lid.x == 0u) {
    if (runtime.status == 1u) {
      let token = reduceU32[0];
      if (token == emptyToken) {
        // No sampleable token is a guide/runtime error, never token 0.
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
  }
