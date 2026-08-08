// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: argmax
// workgroupSize: [256, 1, 1]
// builtins: local_invocation_id -> lid
//
// Greedy decoding and seeded top-k sampling are one kernel: greedy is the
// k = 1 case of the tournament below and emits bit-identical tokens to the
// pre-sampling version. See includes/sampling.wgsl for why Gumbel-max needs no
// softmax, no prefix scan and no second dispatch.
//
// OpParams has no dedicated sampling fields — pass.ts freezes its 64-byte
// record and checks the ABI at startup — so the configuration rides the
// pass-local slots, exactly as qk_norm_rope carries eps and the rope base:
//
//   u0        = EMPTY_TOKEN sentinel, globally forbidden as an output
//   u1        = RNG seed
//   f0        = temperature; <= 0 selects greedy
//   f1        = top-k count, integer-valued; <= 1 selects greedy
//   auxOffset = arena base of the per-lane candidate lists
//
// A fourth sampling parameter (top-p, repetition penalty) does not fit and
// would need OpParams widened.

  let emptyToken = op.u0;
  let temperature = op.f0;
  let requestedTopK = u32(max(op.f1, 1.0));
  let sampling = temperature > 0.0 && requestedTopK > 1u;
  // Every lane derives this from the same uniform record, so it is uniform and
  // the barriers inside the tournament stay in uniform control flow.
  let topK = select(1u, min(requestedTopK, SAMPLE_TOP_K_MAX), sampling);
  let scratch = op.auxOffset;

  // Phase 1 — each lane scans its stride of the vocabulary into a private
  // sorted top-k. worstLogit/worstToken mirror the last slot, so a candidate
  // that cannot enter the list costs one compare and no memory traffic beyond
  // the logit read. That keeps the scan the same cost as the old argmax.
  var slot = 0u;
  loop {
    if (slot >= topK) { break; }
    arena[sample_slot_index(scratch, slot, lid.x)] = -3.402823466e+38;
    arena[sample_token_index(scratch, slot, lid.x)] = f32(emptyToken);
    slot += 1u;
  }
  var worstLogit = -3.402823466e+38;
  var worstToken = emptyToken;

  var i = lid.x;
  loop {
    if (i >= op.inputDim) { break; }
    if (i != emptyToken) {
      let v = arena[op.inputOffset + i];
      if (sample_outranks(v, i, worstLogit, worstToken, emptyToken)) {
        // Insertion sort from the tail: shift every entry this candidate beats
        // one slot down, then drop it into the hole that opens up.
        var hole = topK - 1u;
        loop {
          if (hole == 0u) { break; }
          let aboveLogit = arena[sample_slot_index(scratch, hole - 1u, lid.x)];
          let aboveToken = u32(arena[sample_token_index(scratch, hole - 1u, lid.x)]);
          if (!sample_outranks(v, i, aboveLogit, aboveToken, emptyToken)) { break; }
          arena[sample_slot_index(scratch, hole, lid.x)] = aboveLogit;
          arena[sample_token_index(scratch, hole, lid.x)] = f32(aboveToken);
          hole -= 1u;
        }
        arena[sample_slot_index(scratch, hole, lid.x)] = v;
        arena[sample_token_index(scratch, hole, lid.x)] = f32(i);
        worstLogit = arena[sample_slot_index(scratch, topK - 1u, lid.x)];
        worstToken = u32(arena[sample_token_index(scratch, topK - 1u, lid.x)]);
      }
    }
    i += ARGMAX_WG;
  }

  // Phase 2 — k-way merge of the 256 sorted lists. Each round reduces the
  // lists' current heads and only the winning lane advances its cursor, so the
  // merge reads one slot per lane per round instead of rescanning the 65k
  // logits k times. The tournament is exact: a lane can contribute at most k
  // entries to the global top-k, and its list holds exactly those.
  //
  // No early return before the barriers below: runtime.status is a
  // storage-buffer load (non-uniform to WGSL), which would put
  // workgroupBarrier() in non-uniform control flow. The status gate is applied
  // on the final lane-0 write instead.
  var cursor = 0u;
  var round = 0u;
  loop {
    if (round >= topK) { break; }

    var headLogit = -3.402823466e+38;
    var headToken = emptyToken;
    if (cursor < topK) {
      headLogit = arena[sample_slot_index(scratch, cursor, lid.x)];
      headToken = u32(arena[sample_token_index(scratch, cursor, lid.x)]);
    }
    reduceF32[lid.x] = headLogit;
    reduceU32[lid.x] = headToken;
    workgroupBarrier();

    var width = ARGMAX_WG >> 1u;
    loop {
      if (width == 0u) { break; }
      if (lid.x < width) {
        let rv = reduceF32[lid.x + width];
        let rt = reduceU32[lid.x + width];
        if (sample_outranks(rv, rt, reduceF32[lid.x], reduceU32[lid.x], emptyToken)) {
          reduceF32[lid.x] = rv;
          reduceU32[lid.x] = rt;
        }
      }
      workgroupBarrier();
      width >>= 1u;
    }

    if (lid.x == 0u) {
      sampleTopLogit[round] = reduceF32[0];
      sampleTopToken[round] = reduceU32[0];
    }
    workgroupBarrier();

    // Token t is scanned by lane t % ARGMAX_WG, so the lane that just lost an
    // entry identifies itself without another reduction.
    let winner = sampleTopToken[round];
    if (winner != emptyToken && (winner % ARGMAX_WG) == lid.x) { cursor += 1u; }
    round += 1u;
  }

  if (lid.x == 0u) {
    if (runtime.status == 1u) {
      var token = sampleTopToken[0];

      if (sampling && token != emptyToken) {
        // Gumbel-max over the k candidates. Logits are re-centred on the peak
        // first: only differences decide the argmax, and the shift keeps the
        // divided logits away from large magnitudes.
        let step = runtime.generatedCount;
        let inverseTemperature = 1.0 / temperature;
        let peak = sampleTopLogit[0];
        var bestKey = 0.0;
        var bestToken = emptyToken;
        var candidateIndex = 0u;
        loop {
          if (candidateIndex >= topK) { break; }
          let candidate = sampleTopToken[candidateIndex];
          // The list is dense from the front; a hole means the vocabulary ran
          // out of eligible tokens before k were found.
          if (candidate == emptyToken) { break; }
          let key = (sampleTopLogit[candidateIndex] - peak) * inverseTemperature
            + sample_gumbel(op.u1, step, candidate);
          if (bestToken == emptyToken || key > bestKey) {
            bestKey = key;
            bestToken = candidate;
          }
          candidateIndex += 1u;
        }
        token = bestToken;
      }

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
