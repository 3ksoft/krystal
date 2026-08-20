import { INVALID_U32, RELATION_ROLE_INDEX } from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import { packBrainFrame } from "../frame/packer.ts";
import {
  compileActiveFrame,
  compileIntentMask,
  compileMixerMask,
  compilePerRowArgumentMask,
  compileRecordMask,
} from "./masks.ts";
import { brainBackwardOracle } from "./backward.ts";
import { brainForwardOracle, selectorOracle } from "./oracle.ts";
import { BRAIN_FORWARD_CONFIG, type BrainForwardConfig, type BrainForwardWeights } from "./model.ts";
import type { CompiledCatalog } from "../bridge/agent.ts";

/**
 * Learning from a teacher, as opposed to learning from consequences.
 *
 * A lesson names what should have been chosen in a scene, and the selectors are
 * pushed toward it. This is the golden path: it needs someone to know the
 * answer, which is exactly what makes it unsuitable as the only way to learn —
 * but it is also the only signal available before a creature has generated any
 * experience worth differencing.
 *
 * Only the selector projections are updated. The composed backward returns
 * gradients for the whole graph, and applying them here would let a handful of
 * authored lessons reshape perception itself — a much larger claim than "this
 * scene called for that action", and one with its own ways of going wrong.
 */

export interface TeachExpectation {
  /** Catalog position of the relation the teacher wants chosen. */
  readonly intentId: number;
  /**
   * Runtime reference token of the participant the patient role should bind.
   *
   * Omitted for a relation with no patient, and then only the relation itself
   * is taught.
   */
  readonly patientRefToken?: number | undefined;
}

export interface TeachInput {
  readonly frame: v1_0_0.BrainFrame;
  readonly weights: BrainForwardWeights;
  readonly config?: BrainForwardConfig;
  readonly catalog: CompiledCatalog;
  readonly intentSchemaId: number;
  readonly expect: TeachExpectation;
  readonly learningRate?: number;
}

export interface TeachReport {
  /** What the policy would have chosen BEFORE this update. */
  readonly chosenIntentId: number | undefined;
  readonly intentHit: boolean;
  readonly patientHit: boolean;
  /** Probability the policy gave the taught relation, before the update. */
  readonly intentProbability: number;
  readonly patientProbability: number;
  /** Whether the lesson could be applied at all, and why not when it could not. */
  readonly applied: boolean;
  readonly skipped?: string;
}

export function teach(input: TeachInput): TeachReport {
  const config = input.config ?? BRAIN_FORWARD_CONFIG;
  const lr = input.learningRate ?? 0.05;
  const { hiddenSize: h } = config;
  const weights = input.weights;

  const packed = packBrainFrame(input.frame).frame;
  const active = compileActiveFrame(packed);
  const q = active.queryRecords.length;
  const r = active.bankRecords.length;

  const miss = (reason: string): TeachReport => ({
    chosenIntentId: undefined,
    intentHit: false,
    patientHit: false,
    intentProbability: 0,
    patientProbability: 0,
    applied: false,
    skipped: reason,
  });

  if (q === 0 || r === 0) return miss("frame has no query row or no bank");

  // Where the taught relation sits in the bank. A lesson naming a relation the
  // frame does not carry cannot be applied: there is no candidate to point at,
  // and pushing toward an absent one would only flatten the row.
  const catalogBanks: number[] = [];
  active.bankRecords.forEach((slot, bank) => {
    if (packed.schemaIds[slot] === input.intentSchemaId) catalogBanks.push(bank);
  });
  const intentBank = catalogBanks[input.expect.intentId];
  if (intentBank === undefined) return miss(`relation ${input.expect.intentId} is not in the catalog band`);

  // Where the taught participant sits. Matched by runtime reference, never by
  // slot: slots are shuffled per frame, and a lesson has to survive that.
  let patientBank: number | undefined;
  if (input.expect.patientRefToken !== undefined) {
    for (let bank = 0; bank < r; bank++) {
      const slot = active.bankRecords[bank]!;
      const handle = packed.runtimeRefs[slot * 8];
      if (handle === undefined || handle === INVALID_U32) continue;
      if ((handle & 0xffff) === input.expect.patientRefToken) {
        patientBank = bank;
        break;
      }
    }
    if (patientBank === undefined) return miss("the taught participant is not in this frame");
  }

  const intentMask = compileIntentMask(packed, active, input.intentSchemaId);
  const argMask = compilePerRowArgumentMask(
    packed,
    active,
    input.catalog,
    new Array(q).fill(input.expect.intentId),
    "patient",
  );

  // What the policy believes before being told. Reported rather than used: a
  // curriculum that cannot see whether it is teaching anything is a curriculum
  // nobody can debug.
  const forward = brainForwardOracle(
    packed,
    active,
    weights,
    config,
    compileRecordMask(active.activeTokens).mask,
    compileMixerMask(packed, active),
  );
  const intentSelection = selectorOracle(
    forward.queryOutput, forward.bankKeys, forward.bankValues, intentMask, weights.selector, h,
  );
  const argSelection = selectorOracle(
    forward.queryOutput, forward.bankKeys, forward.bankValues, argMask, weights.selector, h,
  );
  const chosenBank = intentSelection.index[0]!;
  const chosenIntentId = catalogBanks.indexOf(chosenBank);
  const chosenPatient = argSelection.index[0]!;

  const result = brainBackwardOracle({
    frame: packed,
    active,
    weights,
    config,
    recordMask: compileRecordMask(active.activeTokens).mask,
    mixerMask: compileMixerMask(packed, active),
    intentMask,
    argMask,
    routeKinds: new Array<number>(q).fill(0),
    intentTargets: new Array<number>(q).fill(intentBank),
    ...(patientBank === undefined
      ? {}
      : { argumentTargets: new Array<number>(q).fill(patientBank) }),
  });

  for (let i = 0; i < weights.selector.wq.length; i++) {
    weights.selector.wq[i] = weights.selector.wq[i]! - lr * result.dSelectorWq[i]!;
  }
  for (let i = 0; i < weights.selector.wk.length; i++) {
    weights.selector.wk[i] = weights.selector.wk[i]! - lr * result.dSelectorWk[i]!;
  }

  return {
    chosenIntentId: chosenIntentId < 0 ? undefined : chosenIntentId,
    intentHit: chosenBank === intentBank,
    patientHit: patientBank === undefined ? true : chosenPatient === patientBank,
    intentProbability: intentSelection.p[intentBank] ?? 0,
    patientProbability: patientBank === undefined ? 0 : (argSelection.p[patientBank] ?? 0),
    applied: true,
  };
}
