// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: constraint_argmax
// workgroupSize: [256, 1, 1]
// builtins: local_invocation_id -> lid

  // Keep every lane alive until all workgroup barriers have completed.
  // op.u0 is the globally forbidden EMPTY_TOKEN sentinel, matching ordinary argmax.
  let emptyToken = op.u0;
  var bestValue = -3.402823466e+38;
  var bestToken = CONSTRAINT_ARGMAX_INVALID_TOKEN;
  var token = lid.x;
  loop {
    if (token >= op.inputDim) { break; }
    let maskWord = constraintMask[token >> 5u];
    let allowed = (maskWord & (1u << (token & 31u))) != 0u;
    if (token != emptyToken && allowed) {
      let value = arena[op.inputOffset + token];
      if (
        value > bestValue ||
        (value == bestValue && token < bestToken)
      ) {
        bestValue = value;
        bestToken = token;
      }
    }
    token += ARGMAX_WG;
  }

  reduceF32[lid.x] = bestValue;
  reduceU32[lid.x] = bestToken;
  workgroupBarrier();

  var width = ARGMAX_WG >> 1u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) {
      let rightValue = reduceF32[lid.x + width];
      let rightToken = reduceU32[lid.x + width];
      let leftValue = reduceF32[lid.x];
      let leftToken = reduceU32[lid.x];
      if (
        rightValue > leftValue ||
        (rightValue == leftValue && rightToken < leftToken)
      ) {
        reduceF32[lid.x] = rightValue;
        reduceU32[lid.x] = rightToken;
      }
    }
    workgroupBarrier();
    width >>= 1u;
  }

  if (lid.x == 0u && runtime.status == 1u) {
    let selected = reduceU32[0];
    if (selected == CONSTRAINT_ARGMAX_INVALID_TOKEN) {
      runtime.status = 4u;
      runtime.errorCode = CONSTRAINT_ERROR_DEAD_END;
      runtime.telemetryRevision += 1u;
    } else if (!commit_constraint_token(selected)) {
      runtime.status = 4u;
      runtime.errorCode = CONSTRAINT_ERROR_COMMIT;
      runtime.telemetryRevision += 1u;
    } else {
      if (op.mode != 0u) { runtime.position += 1u; }
      let step = runtime.generatedCount;
      let outIndex = runtime.contextCapacity + step;
      tokens[outIndex] = selected;
      runtime.currentToken = selected;
      runtime.lastToken = selected;
      runtime.generatedCount += 1u;
      runtime.telemetryRevision += 1u;

      if (selected == runtime.eosToken) {
        runtime.status = 2u;
      } else if (runtime.generatedCount >= runtime.maxNewTokens) {
        runtime.status = 3u;
      }
      emit_decode_telemetry(step, selected);
    }
  }
