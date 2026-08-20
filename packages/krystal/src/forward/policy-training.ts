import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import type { ExperienceEntry } from "../bridge/experience.ts";
import { packBrainFrame } from "../frame/packer.ts";
import {
  compileActiveFrame,
  compileIntentMask,
  compileMixerMask,
  compilePerRowArgumentMask,
  compileRecordMask,
} from "./masks.ts";
import { brainBackwardOracle } from "./backward.ts";
import { BRAIN_FORWARD_CONFIG, type BrainForwardConfig, type BrainForwardWeights } from "./model.ts";
import type { CompiledCatalog } from "../bridge/agent.ts";

/**
 * Learning from consequences.
 *
 * The other half of the loop from `teaching.ts`. A lesson needs someone who
 * knows the answer; this needs only that something happened afterwards, which
 * is the only signal a creature has once the curriculum stops.
 *
 * REINFORCE with the value head as baseline. The trick that makes it reuse the
 * supervised machinery unchanged: the gradient of cross-entropy toward the
 * action that was ACTUALLY TAKEN is exactly the direction that makes that
 * action more likely, so scaling it by the advantage turns "push toward the
 * right answer" into "push toward what was done, in proportion to how much
 * better than expected it turned out". A negative advantage flips the sign and
 * pushes away.
 *
 * The baseline is what stops this from being useless. Reinforcing by the raw
 * outcome credits every action taken in a good moment equally, including the
 * ones that had nothing to do with it; subtracting what was expected leaves
 * only the part the choice is answerable for.
 *
 * Actor and critic are updated together on purpose. A baseline that lags behind
 * the policy is worse than no baseline: it reports stale expectations as
 * surprise, and the policy chases its own shadow.
 */

export interface PolicyTrainingInput {
  readonly entries: readonly ExperienceEntry[];
  readonly weights: BrainForwardWeights;
  readonly config?: BrainForwardConfig;
  readonly catalog: CompiledCatalog;
  readonly intentSchemaId: number;
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

export interface PolicyTrainingReport {
  readonly framesSeen: number;
  readonly meanAdvantage: number;
  readonly meanValueLoss: number;
  /** Frames whose outcome beat expectation, and were pushed toward. */
  readonly reinforced: number;
  /** Frames that fell short, and were pushed away from. */
  readonly discouraged: number;
}

export function trainPolicy(input: PolicyTrainingInput): PolicyTrainingReport {
  const config = input.config ?? BRAIN_FORWARD_CONFIG;
  const lr = input.learningRate ?? 0.05;
  const policyScale = input.policyScale ?? 1;
  const clip = input.advantageClip ?? 3;
  const weights = input.weights;

  // Gradients are collected before any of them is applied.
  //
  // A per-frame update would make each step depend on the ones before it inside
  // the same batch, and — worse — leaves the advantage on whatever scale the
  // world's valence happens to use. Standardising across the batch is what
  // makes one learning rate work for a world where eating is worth 0.35 and for
  // one where it is worth 3.5, and it is what stopped this from diverging.
  interface Pending {
    readonly advantage: number;
    readonly dSelectorWq: Float32Array;
    readonly dSelectorWk: Float32Array;
  }
  const pending: Pending[] = [];

  let seen = 0;
  let advantageTotal = 0;
  let lossTotal = 0;

  for (const entry of input.entries) {
    // No consequence, nothing to learn from in either direction.
    if (entry.target === undefined) continue;

    const packed = packBrainFrame(entry.frame).frame;
    const active = compileActiveFrame(packed);
    const q = active.queryRecords.length;
    const r = active.bankRecords.length;
    if (q === 0 || r === 0) continue;

    // A frame the creature did not act in still teaches the critic what such a
    // situation is worth. Only the actor step needs a choice to push on.
    const choice =
      entry.choice && entry.choice.intentBank < r ? entry.choice : undefined;

    const argMask = compilePerRowArgumentMask(
      packed,
      active,
      input.catalog,
      new Array(q).fill(choice?.intentId ?? 0),
      "patient",
    );

    const patientBank =
      choice?.patientBank !== undefined && choice.patientBank < r ? choice.patientBank : undefined;

    const result = brainBackwardOracle({
      frame: packed,
      active,
      weights,
      config,
      recordMask: compileRecordMask(active.activeTokens).mask,
      mixerMask: compileMixerMask(packed, active),
      intentMask: compileIntentMask(packed, active, input.intentSchemaId),
      argMask,
      routeKinds: new Array<number>(q).fill(0),
      // The action taken stands in for the gold label. See the note above.
      ...(choice === undefined
        ? {}
        : { intentTargets: new Array<number>(q).fill(choice.intentBank) }),
      ...(patientBank === undefined
        ? {}
        : { argumentTargets: new Array<number>(q).fill(patientBank) }),
      valenceTarget: entry.target,
    });

    let baseline = 0;
    for (const prediction of result.valuePrediction) baseline += prediction;
    baseline = result.valuePrediction.length === 0 ? 0 : baseline / result.valuePrediction.length;
    const advantage = entry.target - baseline;

    // Critic first, and on its own rate: it is a regression onto a number that
    // exists whether or not the creature acted, so it is safe to move steadily.
    for (let i = 0; i < weights.valueHeadWv.length; i++) {
      weights.valueHeadWv[i] = weights.valueHeadWv[i]! - lr * result.dValueWv[i]!;
    }

    if (choice !== undefined && Number.isFinite(advantage)) {
      pending.push({
        advantage,
        dSelectorWq: result.dSelectorWq,
        dSelectorWk: result.dSelectorWk,
      });
    }

    advantageTotal += advantage;
    lossTotal += result.valueLoss;
    seen++;
  }

  let reinforced = 0;
  let discouraged = 0;

  if (pending.length > 0) {
    const mean = pending.reduce((sum, item) => sum + item.advantage, 0) / pending.length;
    const variance =
      pending.reduce((sum, item) => sum + (item.advantage - mean) ** 2, 0) / pending.length;
    // A batch where everything went equally well says nothing about which
    // choice was responsible, so it must not push at all — hence the guard
    // rather than a small epsilon that would amplify noise into confidence.
    const deviation = Math.sqrt(variance);

    if (deviation > 1e-6) {
      for (const item of pending) {
        const standardised = Math.max(
          -clip,
          Math.min(clip, (item.advantage - mean) / deviation),
        );
        const step = (lr * policyScale * standardised) / pending.length;
        for (let i = 0; i < weights.selector.wq.length; i++) {
          weights.selector.wq[i] = weights.selector.wq[i]! - step * item.dSelectorWq[i]!;
        }
        for (let i = 0; i < weights.selector.wk.length; i++) {
          weights.selector.wk[i] = weights.selector.wk[i]! - step * item.dSelectorWk[i]!;
        }
        if (standardised > 0) reinforced++;
        else if (standardised < 0) discouraged++;
      }
    }
  }

  return {
    framesSeen: seen,
    meanAdvantage: seen === 0 ? 0 : advantageTotal / seen,
    meanValueLoss: seen === 0 ? 0 : lossTotal / seen,
    reinforced,
    discouraged,
  };
}
