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
 * The randomness is counter-based rather than a stateful generator, so a run
 * stays reproducible: the same (seed, tick, slot, row) always draws the same
 * number, no matter what else the process did in between.
 */

/** Counter-based 32-bit mixer (the one the simulation uses). */
export function mix32(value: number): number {
  let x = value >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}

/** Stable 32-bit hash of a string, for turning an agent id into a seed. */
export function hashString(text: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash = Math.imul(hash ^ text.charCodeAt(i), 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

/** Which selector a draw belongs to, so the two never share a number. */
export const SAMPLE_STREAM = {
  intent: 0x1,
  /**
   * Base for the per-role selection streams; a role's stream is this plus its
   * index in `RELATION_ROLES`.
   *
   * Separate streams per role on purpose: sharing one would correlate the agent
   * and the patient draws, so a relation would tend to pick both from the same
   * region of the distribution and the pair would explore far less than either
   * head alone.
   */
  role: 0x10,
} as const;

/**
 * A uniform in [0, 1) for one (stream, row) of one decision.
 *
 * Pure in its arguments on purpose. Threading a mutable generator through the
 * forward pass would make a decision depend on how many draws happened before
 * it, and a replayed frame would then diverge from the frame it replays.
 */
export function drawUniform(seed: number, stream: number, row: number): number {
  const mixed = mix32((seed >>> 0) ^ mix32(Math.imul(stream, 0x9e3779b1) ^ row));
  // 2^-32, so the result is in [0, 1) and never exactly 1.
  return mixed * 2.3283064365386963e-10;
}

/**
 * Inverse-CDF sample from one row of a [Q, R] distribution.
 *
 * Falls back to the last positive-probability entry, which matters only for
 * the rounding case where the row's probabilities sum to slightly under the
 * drawn uniform. Returning the argmax there would quietly bias the sample
 * toward the mode exactly when the distribution is flattest.
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
    if (weight <= 0) continue;
    last = j;
    cumulative += weight;
    if (uniform < cumulative) return j;
  }
  return last;
}
