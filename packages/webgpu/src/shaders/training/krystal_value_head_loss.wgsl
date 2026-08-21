// Body-only compute shader source.
// Sandblaster owns @compute, @workgroup_size and the entry-point signature.
// entryPoint: krystal_value_head_loss
// workgroupSize: [256, 1, 1]
// builtins: global_invocation_id -> gid
//
// Squared-error loss of the value head against the observed change in valence:
//
//   loss     = mean_q 0.5 * (pred[q] - target)^2
//   dPred[q] = (pred[q] - target) / Q
//
// The target is one number for the whole frame — valence describes the actor,
// not a query row — so every row is trained toward the same observation, and
// the /Q that makes the loss a mean rides in the gradient here rather than in
// the reduction (same convention as cross_entropy_forward_backward).
//
// A frame with NO target contributes nothing rather than being pushed toward
// zero: the first frame of a life has nothing to difference against, and
// treating "unknown" as "no change" would teach the creature that beginnings
// are uneventful. That is what u0 = 0 selects, and it writes explicit zeros
// rather than leaving whatever the last frame left behind.
//
// OpParams:
//   tokenCount = Q
//   f0 = observed target, u0 = 1 when the frame carries one (0 = none)
//   inputOffset  = predictions  [Q] (value head logits, C = 1)
//   outputOffset = dPredictions [Q]
//   auxOffset    = per-row loss [Q] (0.5 * err^2; loss_reduce means it)

  let i = gid.x;
  let Q = op.tokenCount;
  if (i >= Q) { return; }

  if (op.u0 == 0u) {
    arena[op.outputOffset + i] = 0.0;
    arena[op.auxOffset + i] = 0.0;
    return;
  }

  let err = arena[op.inputOffset + i] - op.f0;
  arena[op.outputOffset + i] = err / f32(Q);
  arena[op.auxOffset + i] = 0.5 * err * err;
