/**
 * Where an update gets its gradients.
 *
 * `learn` and `teach` are written once, against this: hand over a frame and
 * what it is being pushed toward, get back the arrays the update consumes. The
 * CPU oracle below is one source; a device backend that can differentiate is
 * another; and the update cannot tell them apart, which is the point — the
 * numbers it applies are the same numbers, and only who computed them changed.
 */
import { brainBackwardOracle } from "../forward/backward.ts";
import type { BrainForwardConfig, BrainForwardWeights } from "../forward/model.ts";
import type { BackwardRequest, BackwardResult, WeightChanges } from "./backend.ts";
import type { HostFrame } from "./frame.ts";

export interface GradientSource {
  /** Differentiate one frame against the weights as they stand right now. */
  backward(frame: HostFrame, request: BackwardRequest): BackwardResult | Promise<BackwardResult>;
  /**
   * Told after the host's arrays changed, before the next frame may be
   * differentiated. A source that reads those arrays directly needs nothing;
   * one holding a copy takes up exactly what moved.
   */
  wrote?(changes: WeightChanges): void;
}

/**
 * The CPU oracle, on the host's own arrays. Reads them at call time, so an
 * update applied between two frames is already in force for the second.
 */
export function cpuGradients(weights: BrainForwardWeights, config: BrainForwardConfig): GradientSource {
  return {
    backward(frame, request) {
      const { active } = frame;
      const queries = active.queryRecords.length;
      const bank = active.bankRecords.length;
      const result = brainBackwardOracle({
        frame: frame.gpu,
        active,
        weights,
        config,
        // Unconstrained, exactly as `think` runs it: what a question may attend
        // to while it thinks is not what it may CHOOSE.
        mixerMask: new Float32Array(queries * bank),
        intentMask: request.selection,
        // There is one selector question here, not an intent and an argument.
        // The second slot is given nothing to aim at, and does not run at all.
        argMask: new Float32Array(queries * bank),
        ...(request.targets ? { intentTargets: request.targets } : {}),
        // The third block of the critic's context is what this question was
        // allowed to choose from, not a soft gather under a slot with no mask —
        // which averaged the whole bank, forbidden records included.
        context: "available",
        ...(request.valenceTarget === undefined ? {} : { valenceTarget: request.valenceTarget }),
      });
      return {
        policy: result.policy,
        valuePrediction: result.valuePrediction,
        valueLoss: result.valueLoss,
        dSelectorWq: result.dSelectorWq,
        dSelectorWk: result.dSelectorWk,
        dPool: result.dPool,
        dFieldStates: result.dFieldStates,
        dValueWv: result.dValueWv,
      };
    },
  };
}
