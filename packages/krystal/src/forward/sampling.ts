/**
 * Where the creature's variability comes from.
 *
 * The selection heads take an argmax, which is the right answer to "what does
 * this policy believe" and the wrong answer to "what does this creature do".
 * A deterministic policy has no behaviour to reinforce: it emits one action per
 * frame, gets one outcome, and has nothing to compare it against. No reward
 * signal can fix that, because a reward chooses among variations and cannot
 * create them — which is why an untrained creature repeated the same LOOK for
 * as long as anyone watched it.
 *
 * A baby does not flail because flailing is rewarding. It flails because its
 * motor output is noisy, and the outcomes attach themselves to the movements
 * afterwards. Sampling is that noise, and it has one property worth the whole
 * change: an untrained selector's distribution is nearly uniform, so sampling
 * from it tries everything, and as learning sharpens the distribution the
 * exploration decays on its own. There is no temperature to schedule and no
 * epsilon to decay — the policy's own certainty is the schedule.
 *
 * Where the numbers come from is the HOST's business: `choose` takes a
 * `sample` closure and this module only turns one of its uniforms into a
 * choice. A generator lived here once, counter-based so a run stayed
 * reproducible — and reproducibility is exactly why it belongs to whoever owns
 * the run, not to the brain being replayed.
 */

/**
 * Inverse-CDF sample from one row of a [Q, R] distribution.
 *
 * Falls back to the last positive-probability entry, which matters only for
 * the rounding case where the row's probabilities sum to slightly under the
 * drawn uniform. Returning the argmax there would quietly bias the sample
 * toward the mode exactly when the distribution is flattest.
 *
 * A weight that is not a number is not a weight: skipped like a zero, so a
 * row of NaN chooses nothing (-1) rather than its last entry. Measured in a
 * world: a brain with NaN in it "chose" the last record of every frame, which
 * happened to be the one act nothing may do, a hundred and twenty times over,
 * and looked like a decision.
 */
export function sampleRow(
  p: Float32Array,
  offset: number,
  count: number,
  uniform: number,
): number {
  let cumulative = 0;
  let last = -1;
  for (let j = 0; j < count; j++) {
    const weight = p[offset + j]!;
    if (!(weight > 0)) continue;
    last = j;
    cumulative += weight;
    if (uniform < cumulative) return j;
  }
  return last;
}
