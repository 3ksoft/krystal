// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: krystal_pool
// workgroupSize: [64, 1, 1]
// builtins: workgroup_id -> wid, local_invocation_id -> lid
//
// Record key/value pooling (M2b, concerns answer 25): two learned pooling
// queries derive a record's key and value from its encoded field states:
//
//   key[r,h]   = sum_j softmax(qk . state_j / sqrt(H)) * state_j[h]
//   value[r,h] = sum_j softmax(qv . state_j / sqrt(H)) * state_j[h]
//
// where j runs over the record's active (non-padding) field tokens. One
// workgroup per record; record width is the frozen ABI value 8, so the score
// arrays fit in workgroup memory. The pool queries live in weight32 [2, H]
// (qk row 0, qv row 1).
//
// OpParams:
//   tokenCount = recordCount, inputDim = H
//   inputOffset = fieldStates [T_active, H]
//   auxOffset   = recordIndices [recordCount] (u32 payloads)
//   aux2Offset  = recordCompactOffset [maxRecords] (u32)
//   aux3Offset  = recordCompactCount [maxRecords] (u32)
//   outputOffset = keys [recordCount, H]
//   aux4Offset  = values [recordCount, H]
//   weight32 = pool queries [2, H]
//   pool-scores include: workgroup arrays poolKeyScores/poolValueScores (8 each)

  let r = wid.x;
  let recordCount = op.tokenCount;
  let H = op.inputDim;
  if (r >= recordCount) { return; }

  // Empty records (count == 0) fall through every pass and produce zeros in
  // pass 3; the host never pools empty records anyway. No early return here:
  // an arena-dependent branch before workgroupBarrier would be non-uniform.
  let slot = bitcast<u32>(arena[op.auxOffset + r]);
  let start = bitcast<u32>(arena[op.aux2Offset + slot]);
  let count = bitcast<u32>(arena[op.aux3Offset + slot]);

  let scale = 1.0 / sqrt(f32(H));

  // Pass 1: score every active token (one lane per token).
  var j = lid.x;
  loop {
    if (j >= count) { break; }
    var ks = 0.0;
    var vs = 0.0;
    let base = op.inputOffset + (start + j) * H;
    for (var d = 0u; d < H; d++) {
      let s = arena[base + d];
      ks += weight32[d] * s;
      vs += weight32[H + d] * s;
    }
    poolKeyScores[j] = ks * scale;
    poolValueScores[j] = vs * scale;
    j += WG;
  }
  workgroupBarrier();

  // Pass 2: independent softmax for key and value score rows (single lane).
  if (lid.x == 0u) {
    var kMax = -3.402823466e+38;
    var vMax = -3.402823466e+38;
    for (var k = 0u; k < count; k++) {
      kMax = max(kMax, poolKeyScores[k]);
      vMax = max(vMax, poolValueScores[k]);
    }
    var kSum = 0.0;
    var vSum = 0.0;
    for (var k = 0u; k < count; k++) {
      let ke = exp(poolKeyScores[k] - kMax);
      let ve = exp(poolValueScores[k] - vMax);
      poolKeyScores[k] = ke;
      poolValueScores[k] = ve;
      kSum += ke;
      vSum += ve;
    }
    let kInv = 1.0 / max(kSum, 1e-20);
    let vInv = 1.0 / max(vSum, 1e-20);
    for (var k = 0u; k < count; k++) {
      poolKeyScores[k] *= kInv;
      poolValueScores[k] *= vInv;
    }
  }
  workgroupBarrier();

  // Pass 3: weighted sums over the H dims (lanes over dims).
  var dim = lid.x;
  loop {
    if (dim >= H) { break; }
    var keyVal = 0.0;
    var valVal = 0.0;
    for (var k = 0u; k < count; k++) {
      let s = arena[op.inputOffset + (start + k) * H + dim];
      keyVal += poolKeyScores[k] * s;
      valVal += poolValueScores[k] * s;
    }
    arena[op.outputOffset + r * H + dim] = keyVal;
    arena[op.aux4Offset + r * H + dim] = valVal;
    dim += WG;
  }
