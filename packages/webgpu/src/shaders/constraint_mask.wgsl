// Body-only compute shader source. Sandblaster owns the entry point and bindings.

  let maskWord = gid.x;
  let vocab = tokenizer_vocab_size();
  let maskWords = (vocab + 31u) >> 5u;
  if (maskWord >= maskWords) { return; }

  let baseToken = maskWord << 5u;
  var bits = 0u;
  var lane = 0u;
  loop {
    if (lane >= 32u) { break; }
    let token = baseToken + lane;
    if (token < vocab && token_survives(token)) { bits |= 1u << lane; }
    lane += 1u;
  }
  constraintMask[maskWord] = bits;
