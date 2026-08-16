// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: krystal_decision_head_backward
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid
//
// Typed decision head backward (M3, §17 item 9): the final linear head that
// maps the gathered context to route-kind logits (architecture v2 §12.9,
// TypedPlan.routeKind; concerns answer 27):
//
//   logits[q,c] = sum_{d in [0,3H)} ctx[q,d] * Wh[c,d]
//
// ctx[q] = concat(queryOutput[q], intentGather[q], argGather[q]) — read from
// three regions directly, so no materialized concat is needed. Given upstream
// dLogits (e.g. from cross_entropy_forward_backward):
//
//   dQueryOutput[q,d]  = sum_c dLogits[q,c] * Wh[c, d]
//   dIntentGather[q,d] = sum_c dLogits[q,c] * Wh[c, H + d]
//   dArgGather[q,d]    = sum_c dLogits[q,c] * Wh[c, 2H + d]
//   dWh[c,d']          = sum_q dLogits[q,c] * ctx[q, d']   (d' in [0, 3H))
//
// The linear gid is split into four blocks like the qkv backward:
//   [0, QH)                    -> dQueryOutput
//   [QH, 2QH)                  -> dIntentGather
//   [2QH, 3QH)                 -> dArgGather
//   [3QH, 3QH + C*3H)          -> dWh
// Every output element has exactly one owner; no atomics.
//
// OpParams:
//   tokenCount = Q, inputDim = H, outputDim = C
//   inputOffset  = dLogits [Q, C]
//   auxOffset    = queryOutput  [Q, H]
//   aux2Offset   = intentGather [Q, H]
//   aux3Offset   = argGather    [Q, H]
//   outputOffset = dQueryOutput  [Q, H]
//   aux4Offset   = dIntentGather [Q, H]
//   aux5Offset   = dArgGather    [Q, H]
//   aux6Offset   = dWh [C, 3H]
//   weight32     = Wh [C, 3H]  (row-major [outDim, inDim], like matmul_f32)

  let Q = op.tokenCount;
  let H = op.inputDim;
  let C = op.outputDim;
  let qh = Q * H;
  let whTotal = C * 3u * H;
  let linear = gid.x;
  if (linear >= 3u * qh + whTotal) { return; }

  let col = linear % H;

  if (linear < qh) {
    // dQueryOutput[q, col] = sum_c dLogits[q,c] * Wh[c, col]
    let q = linear / H;
    var sum = 0.0;
    for (var c = 0u; c < C; c++) {
      sum += arena[op.inputOffset + q * C + c] * weight32[c * 3u * H + col];
    }
    arena[op.outputOffset + q * H + col] = sum;
    return;
  }
  if (linear < 2u * qh) {
    // dIntentGather[q, col] = sum_c dLogits[q,c] * Wh[c, H + col]
    let q = (linear - qh) / H;
    var sum = 0.0;
    for (var c = 0u; c < C; c++) {
      sum += arena[op.inputOffset + q * C + c] * weight32[c * 3u * H + H + col];
    }
    arena[op.aux4Offset + q * H + col] = sum;
    return;
  }
  if (linear < 3u * qh) {
    // dArgGather[q, col] = sum_c dLogits[q,c] * Wh[c, 2H + col]
    let q = (linear - 2u * qh) / H;
    var sum = 0.0;
    for (var c = 0u; c < C; c++) {
      sum += arena[op.inputOffset + q * C + c] * weight32[c * 3u * H + 2u * H + col];
    }
    arena[op.aux5Offset + q * H + col] = sum;
    return;
  }

  // dWh[c, col'] = sum_q dLogits[q,c] * ctx[q, col'], col' in [0, 3H). The ctx
  // value is read from the region matching col' < H / < 2H.
  let whLinear = linear - 3u * qh;
  let c = whLinear / (3u * H);
  let dp = whLinear % (3u * H);
  var sum = 0.0;
  for (var q = 0u; q < Q; q++) {
    let base = q * H + (dp % H);
    var ctxVal = 0.0;
    if (dp < H) { ctxVal = arena[op.auxOffset + base]; }
    else if (dp < 2u * H) { ctxVal = arena[op.aux2Offset + base]; }
    else { ctxVal = arena[op.aux3Offset + base]; }
    sum += arena[op.inputOffset + q * C + c] * ctxVal;
  }
  arena[op.aux6Offset + whLinear] = sum;
