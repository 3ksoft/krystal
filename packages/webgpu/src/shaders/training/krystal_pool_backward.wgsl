// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: krystal_pool_backward
// workgroupSize: [64, 1, 1]
// builtins: workgroup_id -> wid, local_invocation_id -> lid
//
// Record key/value pooling backward (M3, §17 item 7 — record mixer training
// path). Mirrors krystal_pool forward:
//
//   key[r,h]   = sum_j pk[j] * s[j,h],   pk = softmax(qk . s_j / sqrt(H))
//   value[r,h] = sum_j pv[j] * s[j,h],   pv = softmax(qv . s_j / sqrt(H))
//
// Given upstream dKeys/dValues, compute:
//   dPk[j]          = dot(dKeys[r], s_j)      (through the weighted sum)
//   dScoreK[j]      = pk[j] * (dPk[j] - rowSumK), rowSumK = sum pk*dPk
//   dQk[d]          = scale * sum_j dScoreK[j] * s[j,d]
//   dS[j,d]        += pk[j]*dKeys[r,d] + scale*dScoreK[j]*qk[d]
//   (same for the value side with qv/pv/dValues)
//
// The fieldStates gradient contributions are disjoint per record (each record
// owns its compact token range), so this shader writes dFieldStates directly.
// The pool-query gradient dPool accumulates across records, so each workgroup
// writes its per-record partial into dPoolPartial [R, 2, H]; the
// krystal_pool_dpool pass reduces those into the final dPool [2, H].
// Recomputed softmaxes are deterministic (same math as the forward).
//
// OpParams:
//   tokenCount = recordCount, inputDim = H
//   inputOffset  = fieldStates [T_active, H]
//   auxOffset    = recordIndices [recordCount] (u32 payloads)
//   aux2Offset   = recordCompactOffset [maxRecords] (u32)
//   aux3Offset   = recordCompactCount [maxRecords] (u32)
//   aux4Offset   = dKeys   [recordCount, H]
//   aux5Offset   = dValues [recordCount, H]
//   outputOffset = dFieldStates [T_active, H]
//   aux6Offset   = dPoolPartial [recordCount, 2, H]  (key row 0, value row 1)
//   weight32     = pool queries [2, H] (qk row 0, qv row 1)
//   pool-backward-scores include: workgroup arrays (8 slots each)

  let r = wid.x;
  let recordCount = op.tokenCount;
  let H = op.inputDim;
  if (r >= recordCount) { return; }

  let slot = bitcast<u32>(arena[op.auxOffset + r]);
  let start = bitcast<u32>(arena[op.aux2Offset + slot]);
  let count = bitcast<u32>(arena[op.aux3Offset + slot]);

  let scale = 1.0 / sqrt(f32(H));

  // Pass 1: scores + dPk/dPv per active token (one lane per token).
  var j = lid.x;
  loop {
    if (j >= count) { break; }
    var ks = 0.0;
    var vs = 0.0;
    var dPk = 0.0;
    var dPv = 0.0;
    let base = op.inputOffset + (start + j) * H;
    let dkBase = op.aux4Offset + r * H;
    let dvBase = op.aux5Offset + r * H;
    for (var d = 0u; d < H; d++) {
      let s = arena[base + d];
      ks += weight32[d] * s;
      vs += weight32[H + d] * s;
      dPk += arena[dkBase + d] * s;
      dPv += arena[dvBase + d] * s;
    }
    poolKeyScores[j] = ks * scale;
    poolValueScores[j] = vs * scale;
    poolKeyGrads[j] = dPk;
    poolValueGrads[j] = dPv;
    j += WG;
  }
  workgroupBarrier();

  // Pass 2: independent softmax for key/value rows (single lane), then the
  // softmax backward: dScore = p * (dP - rowSum).
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
    var kRowSum = 0.0;
    var vRowSum = 0.0;
    for (var k = 0u; k < count; k++) {
      poolKeyScores[k] *= kInv;
      poolValueScores[k] *= vInv;
      kRowSum += poolKeyScores[k] * poolKeyGrads[k];
      vRowSum += poolValueScores[k] * poolValueGrads[k];
    }
    for (var k = 0u; k < count; k++) {
      poolKeyGrads[k] = poolKeyScores[k] * (poolKeyGrads[k] - kRowSum);
      poolValueGrads[k] = poolValueScores[k] * (poolValueGrads[k] - vRowSum);
    }
  }
  workgroupBarrier();

  // Pass 3: fieldStates gradient (disjoint per record) + per-record dPool
  // partials (lanes over dims).
  var dim = lid.x;
  loop {
    if (dim >= H) { break; }
    var dQk = 0.0;
    var dQv = 0.0;
    for (var k = 0u; k < count; k++) {
      let base = op.inputOffset + (start + k) * H + dim;
      let s = arena[base];
      dQk += poolKeyGrads[k] * s;
      dQv += poolValueGrads[k] * s;
      let dS = poolKeyScores[k] * arena[op.aux4Offset + r * H + dim] +
        poolValueScores[k] * arena[op.aux5Offset + r * H + dim] +
        scale * (poolKeyGrads[k] * weight32[dim] + poolValueGrads[k] * weight32[H + dim]);
      arena[op.outputOffset + (start + k) * H + dim] += dS;
    }
    arena[op.aux6Offset + r * 2 * H + dim] = dQk * scale;
    arena[op.aux6Offset + r * 2 * H + H + dim] = dQv * scale;
    dim += WG;
  }
