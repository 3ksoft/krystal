/**
 * S2-S10 deterministic transition simulator (docs/S2_S10_CURRICULUM_TASK.md).
 *
 * This is a test oracle, not a second Pira world implementation: given an
 * episode and the model's prediction for the current frame, it decides
 * whether the episode advances to the next frame. Stages S3/S5/S6/S8/S9 are
 * temporal, so they advance only when the previous selected intent/ref is
 * correct; a wrong prediction blocks the transition (returns the same frame,
 * terminal flag false).
 *
 * The frame content of the next frame is fully determined by the episode
 * (generatePolicyEpisode), so the oracle only gates on correctness:
 *
 *   advance(episode, frameIndex, prediction)
 *     -> { nextFrame?, advanced, terminal, correct }
 */
import type { PolicyAction, PolicyEpisode, PolicyRawFrame } from "./policy.ts";

export interface PolicyPrediction {
  readonly action: PolicyAction;
  /** Exact ref; undefined/absent for arity-0 actions. */
  readonly refToken?: number;
}

export interface TransitionResult {
  /** True when the prediction matched the frame's gold (intent + ref). */
  readonly correct: boolean;
  /** True when a correct prediction advanced to a later frame. */
  readonly advanced: boolean;
  /** True when the episode has no further frame after a correct prediction. */
  readonly terminal: boolean;
  /** The next frame when advanced, else the current frame. */
  readonly frame: PolicyRawFrame;
}

/** Stages whose episodes are multi-frame (temporal transitions). */
export const TEMPORAL_STAGES: readonly string[] = ["S3", "S5", "S6", "S8", "S9"];

/**
 * Does the prediction match the gold decision of `frame`? Intent must match
 * exactly; when the gold carries a reference argument, the predicted ref must
 * equal it; an arity-0 gold ignores the predicted ref entirely.
 */
export function predictionMatchesGold(prediction: PolicyPrediction, gold: PolicyRawFrame["gold"]): boolean {
  if (prediction.action !== gold.action) return false;
  if (gold.refToken === undefined) return true;
  return prediction.refToken === gold.refToken;
}

/** Is the frame's gold action arity-0 (no reference argument)? */
export function goldHasReference(gold: PolicyRawFrame["gold"]): boolean {
  return gold.refToken !== undefined;
}

/**
 * Advance an episode given the model's prediction for `frameIndex`. A correct
 * prediction moves to the next frame (terminal when none remains); a wrong
 * prediction keeps the episode at the same frame. Frame 0 of any episode is
 * the entry point; the oracle never invents content — the next frame comes
 * from the episode itself.
 */
export function advanceEpisode(
  episode: PolicyEpisode,
  frameIndex: number,
  prediction: PolicyPrediction,
): TransitionResult {
  const frame = episode.frames[frameIndex]!;
  const correct = predictionMatchesGold(prediction, frame.gold);
  if (!correct) {
    return { correct, advanced: false, terminal: false, frame };
  }
  const next = episode.frames[frameIndex + 1];
  if (!next) {
    return { correct, advanced: false, terminal: true, frame };
  }
  return { correct, advanced: true, terminal: false, frame: next };
}

/**
 * The temporal stage rule, expressed directly for the transition-oracle test:
 * what must be true of the previous prediction for the episode to advance?
 * (S3/S5/S6/S8/S9 only; the single-frame stages have no transition.)
 */
export function stageTransitionRule(stage: PolicyEpisode["stage"]): string {
  switch (stage) {
    case "S3":
      return "the previous frame's action (CRY with no Apple) must be correct before Mother delivers the Apple";
    case "S5":
      return "the previous MOVE_TOWARDS(ref) must point at the exact ref before reachability flips to EAT(same ref)";
    case "S6":
      return "the previous LOOK(ref) must point at the exact ref before the reveal decides EAT(ref) or reject";
    case "S8":
      return "the previous EAT(ref) must target the exact ref before the consequence frame (improved or worsened comfort)";
    case "S9":
      return "the previous MOVE_TOWARDS/EAT(ref) must carry the exact ref before vision drops it and memory retains it";
    default:
      return "single-frame stage: no temporal transition";
  }
}
