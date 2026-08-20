/**
 * One turn of the creature: perception in, proposals out.
 *
 * Two stages rather than one, because a role's candidate set depends on which
 * relation won. What may fill "the thing eaten" is not knowable until EAT has
 * been chosen, so the relation selector runs first and its result conditions
 * every role selector that follows.
 *
 * The second stage is a loop, not a single pass: a reified relation binds up to
 * six roles and each is scored against the bank in its own right. A role no
 * chosen relation declares is skipped entirely rather than scored and
 * discarded.
 *
 * Returns the schema-level `IntentSet`. Mapping references back to the
 * simulation's own instance ids belongs to the boundary, not here — this module
 * knows about weights and masks, and deliberately nothing about contracts.
 */
import {
  RELATION_ROLES,
  RELATION_ROLE_FLAGS,
  RELATION_ROLE_INDEX,
  type RelationRoleName,
} from "../../../schema/src/krystal-engine-schema.ts";
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
import { emitIntentSet, type RoleSelections } from "./intentset.ts";
import { BRAIN_FORWARD_CONFIG, type BrainForwardConfig, type BrainForwardWeights } from "./model.ts";
import type { CompiledCatalog } from "../bridge/agent.ts";

export interface DecideInput {
  /** Lowered frame, before packing. */
  readonly frame: v1_0_0.BrainFrame;
  readonly weights: BrainForwardWeights;
  readonly config?: BrainForwardConfig;
  readonly catalog: CompiledCatalog;
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

/** What one role's head chose for one query row. */
export interface ChosenRole {
  readonly bank: number;
  readonly probability: number;
  /**
   * How many bank records this role admitted for this row.
   *
   * Zero is the whole explanation of a dropped proposal, and without it the
   * drop is mute: "no admissible filler" leaves open whether the world held
   * nothing suitable, the role was declared too narrowly, or the candidates
   * were there and something downstream refused them.
   */
  readonly candidates: number;
}

export interface ChosenAction {
  /** Query row this choice belongs to. */
  readonly row: number;
  /** Bank index of the chosen catalog record, INVALID_U32 if none. */
  readonly intentBank: number;
  /** Probability the policy assigned to the relation, at the moment of choosing. */
  readonly intentProbability: number;
  /** What each declared role's head chose. Absent roles were not scored. */
  readonly roles: Partial<Record<RelationRoleName, ChosenRole>>;
}

export interface DecideDiagnostics {
  /** Catalog records the relation selector could score. Zero means no options. */
  readonly catalogCandidates: number;
  /** Query rows. Zero means nothing was asked. */
  readonly queryRows: number;
  /** Proposals lost because no agent could be resolved in the frame. */
  readonly droppedNoAgent: number;
  /** Proposals lost because no admissible patient was found. */
  readonly droppedNoPatient: number;
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
        droppedNoAgent: 0,
        droppedNoPatient: 0,
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

  // Stage one: which relation.
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

  // Stage two: who and what, one selector per role, each conditioned per row on
  // the relation that row chose. Only roles some chosen relation actually
  // declares are scored — running all six over a world of binary relations
  // would spend most of the pass on slots nothing can fill.
  const declaredRoles = new Set<RelationRoleName>();
  for (const intentId of chosenIntents) {
    const descriptor = input.catalog.descriptors.find(
      (candidate) => candidate.intentId === intentId,
    );
    if (!descriptor) continue;
    for (const role of RELATION_ROLES) {
      const roleDesc = descriptor.roles[RELATION_ROLE_INDEX[role]];
      if (roleDesc && (roleDesc.flags & RELATION_ROLE_FLAGS.present) !== 0) declaredRoles.add(role);
    }
  }

  const roleSelections: RoleSelections = {};
  const roleMasks = new Map<RelationRoleName, Float32Array>();
  for (const role of RELATION_ROLES) {
    if (!declaredRoles.has(role)) continue;
    const mask = compilePerRowArgumentMask(packed, active, input.catalog, chosenIntents, role);
    roleMasks.set(role, mask);
    roleSelections[role] = selectorOracle(
      forward.queryOutput, forward.bankKeys, forward.bankValues, mask, input.weights.selector, h,
      explore &&
        ((row) => drawUniform(explore.seed, SAMPLE_STREAM.role + RELATION_ROLE_INDEX[role], row)),
    );
  }

  const emission = emitIntentSet({
    frame: packed,
    active,
    catalog: input.catalog,
    intentSchemaId: input.intentSchemaId,
    intent,
    roleSelections,
    tick: input.tick,
  });

  const r = active.bankRecords.length;
  const chosen: ChosenAction[] = [];
  for (let row = 0; row < active.queryRecords.length; row++) {
    const intentBank = intent.index[row]!;
    const roles: Partial<Record<RelationRoleName, ChosenRole>> = {};
    for (const [role, selection] of Object.entries(roleSelections) as [
      RelationRoleName,
      (typeof roleSelections)[RelationRoleName],
    ][]) {
      if (!selection) continue;
      const mask = roleMasks.get(role)!;
      const bank = selection.index[row]!;
      let candidates = 0;
      for (let j = 0; j < r; j++) if (mask[row * r + j]! > -1e29) candidates++;
      roles[role] = {
        bank,
        probability: bank < r ? selection.p[row * r + bank]! : 0,
        candidates,
      };
    }
    chosen.push({
      row,
      intentBank,
      intentProbability: intentBank < r ? intent.p[row * r + intentBank]! : 0,
      roles,
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
      droppedNoAgent: emission.droppedNoAgent,
      droppedNoPatient: emission.droppedNoPatient,
    },
  };
}
