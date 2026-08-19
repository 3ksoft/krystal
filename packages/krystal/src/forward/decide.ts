/**
 * One turn of the creature: perception in, proposals out.
 *
 * Two passes rather than one, because the argument mask depends on which intent
 * won. What may fill "the thing eaten" is not knowable until EAT has been
 * chosen, so the intent selector runs first and its result conditions the
 * candidate set the argument selector then scores.
 *
 * Returns the schema-level `IntentSet`. Mapping references back to the
 * simulation's own instance ids belongs to the boundary, not here — this module
 * knows about weights and masks, and deliberately nothing about contracts.
 */
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import { packBrainFrame } from "../frame/packer.ts";
import {
  compileActiveFrame,
  compileIntentMask,
  compileMixerMask,
  compilePerRowArgumentMask,
  compileRecordMask,
  type ActiveFrame,
} from "./masks.ts";
import { brainForwardOracle, selectorOracle } from "./oracle.ts";
import { SAMPLE_STREAM, drawUniform } from "./sampling.ts";
import { emitIntentSet } from "./intentset.ts";
import { BRAIN_FORWARD_CONFIG, type BrainForwardConfig, type BrainForwardWeights } from "./model.ts";
import type { CompiledActionCatalog } from "../fixtures/action-intents.ts";

export interface DecideInput {
  /** Lowered frame, before packing. */
  readonly frame: v1_0_0.BrainFrame;
  readonly weights: BrainForwardWeights;
  readonly config?: BrainForwardConfig;
  readonly catalog: CompiledActionCatalog;
  /** Schema id stamped on catalog records. */
  readonly intentSchemaId: number;
  readonly tick: number;
  /**
   * Draw the action from the policy instead of taking its mode.
   *
   * Omitted, both selectors take the argmax: that is what a supervised
   * evaluation and a gradient check want, since they ask what the policy
   * believes and an answer that moves between runs is not an answer.
   *
   * Supplied, the creature acts. See `sampling.ts` — the short version is that
   * a deterministic policy produces one action per frame and therefore has
   * nothing to compare against, so no reward rule can teach it anything. The
   * seed keeps a run reproducible; derive it from the agent id so two
   * creatures in the same world do not flail in lockstep.
   */
  readonly explore?: { readonly seed: number };
}

export interface DecideResult {
  readonly intentSet: v1_0_0.IntentSet;
  readonly packed: v1_0_0.BrainFrameGpu;
  readonly active: ActiveFrame;
  /** Catalog bank index chosen per query row, for telemetry. */
  readonly chosenIntents: readonly number[];
  /**
   * What each head actually chose, and how likely it was to.
   *
   * The record of the draw, kept because a policy-gradient update needs the
   * action that was taken rather than the one that would have won: reinforcing
   * the argmax of a sampled policy credits a choice the creature did not make.
   */
  readonly chosen: readonly ChosenAction[];
  /**
   * Why no proposal came out, when none did.
   *
   * "It proposes nothing" has several causes that look identical from outside:
   * an empty catalog, a frame with no actor record, a mask that admits no
   * candidate, or a policy that simply has not committed. Only the last is
   * about learning; the rest are malformed input, and saying which is which is
   * the difference between a question and a mystery.
   */
  readonly diagnostics: DecideDiagnostics;
}

export interface ChosenAction {
  /** Query row this choice belongs to. */
  readonly row: number;
  /** Bank index of the chosen catalog record, INVALID_U32 if none. */
  readonly intentBank: number;
  /** Bank index of the chosen object record, INVALID_U32 if none. */
  readonly objectBank: number;
  /** Probability the policy assigned to each, at the moment of choosing. */
  readonly intentProbability: number;
  readonly objectProbability: number;
  /**
   * How many bank records the object role admitted for this row.
   *
   * Zero is the whole explanation of a dropped proposal, and without it the
   * drop is mute: "no admissible object" leaves open whether the world held
   * nothing suitable, the role was declared too narrowly, or the candidates
   * were there and something downstream refused them. The count separates the
   * first two from the third, and the chosen intent says which role to look at.
   */
  readonly objectCandidates: number;
}

export interface DecideDiagnostics {
  /** Catalog records the intent selector could score. Zero means no options. */
  readonly catalogCandidates: number;
  /** Query rows. Zero means nothing was asked. */
  readonly queryRows: number;
  /** Proposals lost because the actor could not be resolved in the frame. */
  readonly droppedNoSubject: number;
  /** Proposals lost because no admissible object was found. */
  readonly droppedNoObject: number;
}

export function decide(input: DecideInput): DecideResult {
  const config = input.config ?? BRAIN_FORWARD_CONFIG;
  const { hiddenSize: h } = config;
  const packed = packBrainFrame(input.frame).frame;
  const active = compileActiveFrame(packed);

  const empty: v1_0_0.IntentSet = {
    tick: input.tick,
    count: 0,
    revision: 0,
    flags: 0,
    proposals: [],
  };
  // No query row means nothing was asked, and no bank means nothing to choose
  // between. Either way there is no proposal to make, and inventing one would
  // be worse than saying so.
  if (active.queryRecords.length === 0 || active.bankRecords.length === 0) {
    return {
      intentSet: empty,
      packed,
      active,
      chosenIntents: [],
      chosen: [],
      diagnostics: {
        catalogCandidates: 0,
        queryRows: active.queryRecords.length,
        droppedNoSubject: 0,
        droppedNoObject: 0,
      },
    };
  }

  const forward = brainForwardOracle(
    packed,
    active,
    input.weights,
    config,
    compileRecordMask(active.activeTokens).mask,
    compileMixerMask(packed, active),
  );

  // Pass one: which action.
  const intentMask = compileIntentMask(packed, active, input.intentSchemaId);
  const explore = input.explore;
  const intent = selectorOracle(
    forward.queryOutput, forward.bankKeys, forward.bankValues, intentMask, input.weights.selector, h,
    explore && ((row) => drawUniform(explore.seed, SAMPLE_STREAM.intent, row)),
  );

  // The winning bank index names a catalog record; its position within the
  // catalog band is the intent id.
  const catalogBanks: number[] = [];
  active.bankRecords.forEach((slot, bank) => {
    if (packed.schemaIds[slot] === input.intentSchemaId) catalogBanks.push(bank);
  });
  const chosenIntents = Array.from(intent.index, (bank) => {
    const position = catalogBanks.indexOf(bank);
    return position < 0 ? 0 : position;
  });

  // Pass two: what to act on, conditioned per row on the action just chosen.
  const argMask = compilePerRowArgumentMask(packed, active, input.catalog, chosenIntents, "object");
  const argument = selectorOracle(
    forward.queryOutput, forward.bankKeys, forward.bankValues, argMask, input.weights.selector, h,
    explore && ((row) => drawUniform(explore.seed, SAMPLE_STREAM.object, row)),
  );

  const emission = emitIntentSet({
    frame: packed,
    active,
    catalog: input.catalog,
    intentSchemaId: input.intentSchemaId,
    intent,
    argument,
    tick: input.tick,
  });

  const r = active.bankRecords.length;
  const chosen: ChosenAction[] = [];
  for (let row = 0; row < active.queryRecords.length; row++) {
    const intentBank = intent.index[row]!;
    const objectBank = argument.index[row]!;
    let objectCandidates = 0;
    for (let j = 0; j < r; j++) if (argMask[row * r + j]! > -1e29) objectCandidates++;
    chosen.push({
      row,
      intentBank,
      objectBank,
      intentProbability: intentBank < r ? intent.p[row * r + intentBank]! : 0,
      objectProbability: objectBank < r ? argument.p[row * r + objectBank]! : 0,
      objectCandidates,
    });
  }

  return {
    intentSet: emission.intentSet,
    packed,
    active,
    chosenIntents,
    chosen,
    diagnostics: {
      catalogCandidates: catalogBanks.length,
      queryRows: active.queryRecords.length,
      droppedNoSubject: emission.droppedNoSubject,
      droppedNoObject: emission.droppedNoObject,
    },
  };
}
