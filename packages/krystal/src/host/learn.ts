/**
 * Learning from what followed.
 *
 * The same reduction the forward already went through. `policy-training.ts`
 * knows about experience buffers, compiled catalogs, intent schema ids and the
 * fixed frame packer — all of it the simulation bridge's vocabulary, none of it
 * the math's. What the update actually needs is far smaller: the records the
 * creature was shown, which one it picked for each question it was asked, the
 * grammar that limited the picking, and what became of things afterwards.
 *
 * REINFORCE with the value head as baseline. The trick that makes it reuse the
 * supervised machinery unchanged: the gradient of cross-entropy toward the
 * choice that was ACTUALLY MADE is exactly the direction that makes that choice
 * more likely, so scaling it by the advantage turns "push toward the right
 * answer" into "push toward what was done, in proportion to how much better
 * than expected it turned out". A negative advantage flips the sign.
 *
 * The baseline is what stops this from being useless. Reinforcing by the raw
 * outcome credits every choice made in a good moment equally, including the
 * ones that had nothing to do with it; subtracting what was expected leaves
 * only the part the choice is answerable for.
 *
 * Actor and critic move together on purpose. A baseline that lags behind the
 * policy is worse than no baseline: it reports stale expectations as surprise,
 * and the policy chases its own shadow.
 */
import { brainBackwardOracle } from "../forward/backward.ts";
import type { ActiveFrame } from "../forward/masks.ts";
import type { BrainForwardConfig, BrainForwardWeights } from "../forward/model.ts";
import { packHostFrame, type HostRecord } from "./frame.ts";

/**
 * One remembered turn: what was shown, what was chosen, and what followed.
 *
 * `chosen` is in the same order and the same numbering as `think` answered in —
 * record slots, as the host sent them — because the two have to describe the
 * same event or the update trains against a frame that never happened.
 */
export interface HostExperience {
  readonly records: readonly HostRecord[];
  /** The record slot each question chose, by question index. `undefined` for a
   *  question the creature did not answer; the frame still teaches the critic. */
  readonly chosen?: readonly (number | undefined)[];
  /**
   * The grammar that was in force when the choice was made — the SAME predicate
   * `think` was given. A different one here would compute the gradient of a
   * distribution the creature never sampled from.
   */
  readonly allows?: (query: number, record: number) => boolean;
  /**
   * What became of things: the change in how the creature was doing. Omitted
   * when there is nothing to difference against — the first frame of a life —
   * and then the frame is skipped rather than counted as a zero, which is a
   * claim that nothing happened.
   */
  readonly reward?: number;
}

export interface LearnOptions {
  readonly learningRate?: number;
  /**
   * Scale on the policy step relative to the value step.
   *
   * The critic can afford a steady rate; the actor cannot, because every update
   * changes the distribution the next batch is drawn from.
   */
  readonly policyScale?: number;
  /**
   * Cap on a standardised advantage, in batch deviations.
   *
   * One unusually good outcome should shift the policy, not rewrite it. Without
   * this the batch's own outlier sets the step size, which is how an actor and
   * its baseline chase each other into divergence.
   */
  readonly advantageClip?: number;
}

export interface LearnReport {
  readonly framesSeen: number;
  readonly meanAdvantage: number;
  readonly meanValueLoss: number;
  /** Choices that beat expectation, and were pushed toward. */
  readonly reinforced: number;
  /** Choices that fell short, and were pushed away from. */
  readonly discouraged: number;
}

const NEG_INF = -1e30;
/** No target for this row. */
const NO_TARGET = 0xffff_ffff;

/**
 * The host's grammar as the selector reads it: one row per question, NEG_INF on
 * every record that question may not choose.
 *
 * A question nothing answers is left OPEN rather than uniformly forbidden — a
 * row of NEG_INF comes back as a uniform over the impossible, which looks like
 * an answer and is not one.
 */
export function selectionMask(
  active: ActiveFrame,
  allows: ((query: number, record: number) => boolean) | undefined,
): Float32Array {
  const queries = active.queryRecords.length;
  const bank = active.bankRecords.length;
  const mask = new Float32Array(queries * bank);
  if (!allows) return mask;
  for (let q = 0; q < queries; q++) {
    let admitted = 0;
    for (let r = 0; r < bank; r++) {
      if (allows(q, active.bankRecords[r]!)) admitted++;
      else mask[q * bank + r] = NEG_INF;
    }
    if (!admitted) for (let r = 0; r < bank; r++) mask[q * bank + r] = 0;
  }
  return mask;
}

/**
 * One pass over remembered turns, updating the actor and the critic.
 *
 * Gradients are collected before any of them is applied. A per-frame step would
 * make each update depend on the ones before it inside the same batch and —
 * worse — leaves the advantage on whatever scale the world's own sense of
 * well-being happens to use. Standardising across the batch is what makes one
 * learning rate work for a world where eating is worth 0.35 and for one where
 * it is worth 3.5.
 */
export function learnFromExperience(
  experiences: readonly HostExperience[],
  weights: BrainForwardWeights,
  config: BrainForwardConfig,
  options: LearnOptions = {},
): LearnReport {
  const lr = options.learningRate ?? 0.05;
  const policyScale = options.policyScale ?? 1;
  const clip = options.advantageClip ?? 3;

  interface Pending {
    readonly advantage: number;
    readonly dSelectorWq: Float32Array;
    readonly dSelectorWk: Float32Array;
  }
  const pending: Pending[] = [];
  const dValue = new Float32Array(weights.valueHeadWv.length);
  let seen = 0;
  let advantageTotal = 0;
  let lossTotal = 0;

  for (const experience of experiences) {
    if (experience.reward === undefined || !experience.records.length) continue;
    const frame = packHostFrame(experience.records);
    const { active } = frame;
    const queries = active.queryRecords.length;
    const bank = active.bankRecords.length;
    if (!queries || !bank) continue;

    // Slots back to bank rows: the host numbers records, the selector numbers
    // the ones it may choose from, and the target is in the second numbering.
    const targets = new Array<number>(queries).fill(NO_TARGET);
    let answered = false;
    for (let q = 0; q < queries; q++) {
      const slot = experience.chosen?.[q];
      if (slot === undefined || slot < 0) continue;
      const row = active.bankRecords.indexOf(slot);
      if (row < 0) continue;
      targets[q] = row;
      answered = true;
    }

    const mask = selectionMask(active, experience.allows);
    const result = brainBackwardOracle({
      frame: frame.gpu,
      active,
      weights,
      config,
      recordMask: frame.recordMask,
      // Unconstrained, exactly as `think` runs it: what a question may attend to
      // while it thinks is not what it may CHOOSE.
      mixerMask: new Float32Array(queries * bank),
      intentMask: mask,
      // There is one selector question here, not an intent and an argument. The
      // second slot is given nothing to aim at, and contributes nothing.
      argMask: new Float32Array(queries * bank),
      ...(answered ? { intentTargets: targets } : {}),
      valenceTarget: experience.reward,
    });

    let baseline = 0;
    for (const prediction of result.valuePrediction) baseline += prediction;
    baseline = result.valuePrediction.length ? baseline / result.valuePrediction.length : 0;
    const advantage = experience.reward - baseline;

    // The critic's gradient is accumulated, not applied here.
    //
    // Applying it per frame would move the baseline in the middle of the batch,
    // so a later frame's advantage would be measured against expectations an
    // earlier frame had just changed — the batch would depend on the order it
    // happened to be in, and two identical outcomes would come out looking
    // different from each other.
    for (let i = 0; i < dValue.length; i++) dValue[i]! += result.dValueWv[i]!;

    if (answered && Number.isFinite(advantage))
      pending.push({ advantage, dSelectorWq: result.dSelectorWq, dSelectorWk: result.dSelectorWk });

    advantageTotal += advantage;
    lossTotal += result.valueLoss;
    seen++;
  }

  // Critic first, and on its own rate: it is a regression onto a number that
  // exists whether or not the creature chose anything, so it is safe to move
  // steadily. One step per batch, so the rate does not secretly scale with how
  // much was remembered.
  if (seen)
    for (let i = 0; i < weights.valueHeadWv.length; i++)
      weights.valueHeadWv[i] = weights.valueHeadWv[i]! - (lr * dValue[i]!) / seen;

  let reinforced = 0;
  let discouraged = 0;
  if (pending.length) {
    const mean = pending.reduce((sum, item) => sum + item.advantage, 0) / pending.length;
    const variance = pending.reduce((sum, item) => sum + (item.advantage - mean) ** 2, 0) / pending.length;
    const deviation = Math.sqrt(variance);
    // A batch where everything went equally well says nothing about which choice
    // was responsible, so it must not push at all — a guard rather than a small
    // epsilon, which would amplify noise into confidence.
    if (deviation > 1e-6) {
      for (const item of pending) {
        const standardised = Math.max(-clip, Math.min(clip, (item.advantage - mean) / deviation));
        const step = (lr * policyScale * standardised) / pending.length;
        for (let i = 0; i < weights.selector.wq.length; i++)
          weights.selector.wq[i] = weights.selector.wq[i]! - step * item.dSelectorWq[i]!;
        for (let i = 0; i < weights.selector.wk.length; i++)
          weights.selector.wk[i] = weights.selector.wk[i]! - step * item.dSelectorWk[i]!;
        if (standardised > 0) reinforced++;
        else if (standardised < 0) discouraged++;
      }
    }
  }

  return {
    framesSeen: seen,
    meanAdvantage: seen ? advantageTotal / seen : 0,
    meanValueLoss: seen ? lossTotal / seen : 0,
    reinforced,
    discouraged,
  };
}
