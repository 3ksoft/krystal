// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: krystal_decision_head
// workgroupSize: [64, 1, 1]
// builtins: workgroup_id -> wid, local_invocation_id -> lid
//
// Typed decision head forward (architecture v2 §12.9, TypedPlan.routeKind;
// concerns answer 27): the final linear head that maps the gathered context
// to route-kind logits.
//
//   logits[q,c] = sum_{d in [0,3H)} ctx[q,d] * Wh[c,d]
//
// ctx[q] = concat(queryOutput[q], intentGather[q], argGather[q]) — read from
// three regions directly, so no materialized concat is needed. Wh [C, 3H] is
// row-major [outDim, inDim], same layout as matmul_f32's weight. No bias
// (this graph has no bias terms). One workgroup per query row; lanes cover
// the C route kinds. Every (q, c) has exactly one owner; no atomics.
//
// OpParams:
//   tokenCount = Q, inputDim = H, outputDim = C
//   inputOffset  = queryOutput  [Q, H]
//   auxOffset    = intentGather [Q, H]
//   aux2Offset   = argGather    [Q, H]
//   outputOffset = logits [Q, C]
//   weight32     = Wh [C, 3H]

  let q = wid.y;
  let Q = op.tokenCount;
  let H = op.inputDim;
  let C = op.outputDim;
  if (q >= Q) { return; }

  var c = lid.x;
  loop {
    if (c >= C) { break; }
    var sum = 0.0;
    for (var d = 0u; d < H; d++) {
      sum += arena[op.inputOffset + q * H + d] * weight32[c * 3u * H + d];
    }
    for (var d = 0u; d < H; d++) {
      sum += arena[op.auxOffset + q * H + d] * weight32[c * 3u * H + H + d];
    }
    for (var d = 0u; d < H; d++) {
      sum += arena[op.aux2Offset + q * H + d] * weight32[c * 3u * H + 2u * H + d];
    }
    arena[op.outputOffset + q * C + c] = sum;
    c += WG;
  }
