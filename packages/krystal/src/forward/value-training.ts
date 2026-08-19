/**
 * Learning from buffered experience.
 *
 * Lives here rather than in `bridge/` on purpose: it needs weights, the forward
 * and the backward, and pulling those across the simulation boundary would undo
 * the separation that boundary exists for. Bridge owns the buffer, which is
 * plain data; the model owns what to do with it.
 *
 * This is the half of the loop the gold curriculum cannot supply. Once labels
 * stop, the only target left is the one the world hands back for free: what
 * actually happened to valence after the creature acted.
 */
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import type { ExperienceEntry } from "../bridge/experience.ts";
import { packBrainFrame } from "../frame/packer.ts";
import {
  compileActiveFrame,
  compileIntentMask,
  compileMixerMask,
  compileRecordMask,
} from "./masks.ts";
import { brainBackwardOracle } from "./backward.ts";
import { BRAIN_FORWARD_CONFIG, type BrainForwardConfig, type BrainForwardWeights } from "./model.ts";

export interface ValueTrainingInput {
  readonly entries: readonly ExperienceEntry[];
  readonly weights: BrainForwardWeights;
  readonly config?: BrainForwardConfig;
  /** Schema id of the ActionIntent catalog records, for the intent mask. */
  readonly intentSchemaId: number;
  /** Per-role candidate mask for the argument selector. */
  readonly argMask: (frame: v1_0_0.BrainFrameGpu, active: ReturnType<typeof compileActiveFrame>) => Float32Array;
  readonly learningRate?: number;
}

export interface ValueTrainingReport {
  readonly framesSeen: number;
  readonly meanValueLoss: number;
  /** Loss on the first and last frame, so a caller can see direction cheaply. */
  readonly firstLoss: number;
  readonly lastLoss: number;
}

/**
 * One pass over settled experience, updating the value head only.
 *
 * Deliberately narrow. The composed backward returns gradients for the whole
 * graph, but applying them here would mean this function silently retrains the
 * perception stack from a signal whose usefulness has not been demonstrated
 * yet. Establishing that the value head can predict at all comes first; letting
 * that signal reshape representations is a separate decision with its own way
 * of going wrong.
 *
 * The route-kind targets are absent, so the cross-entropy term contributes its
 * usual gradient over an unlabelled row and is discarded along with everything
 * else; only `dValueWv` is applied.
 */
export function trainValueHead(input: ValueTrainingInput): ValueTrainingReport {
  const config = input.config ?? BRAIN_FORWARD_CONFIG;
  const lr = input.learningRate ?? 0.05;
  const weights = input.weights;

  let total = 0;
  let firstLoss = 0;
  let lastLoss = 0;
  let seen = 0;

  for (const entry of input.entries) {
    if (entry.target === undefined) continue;
    const packed = packBrainFrame(entry.frame).frame;
    const active = compileActiveFrame(packed);
    if (active.queryRecords.length === 0 || active.bankRecords.length === 0) continue;

    const result = brainBackwardOracle({
      frame: packed,
      active,
      weights,
      config,
      recordMask: compileRecordMask(active.activeTokens).mask,
      mixerMask: compileMixerMask(packed, active),
      intentMask: compileIntentMask(packed, active, input.intentSchemaId),
      argMask: input.argMask(packed, active),
      // No behavioural labels: live play has none. The route-kind row is
      // unlabelled and its gradient is not applied.
      routeKinds: new Array<number>(active.queryRecords.length).fill(0),
      valenceTarget: entry.target,
    });

    for (let i = 0; i < weights.valueHeadWv.length; i++) {
      weights.valueHeadWv[i] = weights.valueHeadWv[i]! - lr * result.dValueWv[i]!;
    }

    if (seen === 0) firstLoss = result.valueLoss;
    lastLoss = result.valueLoss;
    total += result.valueLoss;
    seen++;
  }

  return {
    framesSeen: seen,
    meanValueLoss: seen === 0 ? 0 : total / seen,
    firstLoss,
    lastLoss,
  };
}
