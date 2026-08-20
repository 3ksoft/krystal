/**
 * Host-compiled active lists and masks for the Krystal forward (M2b).
 *
 * concerns answers 15/16: candidate/record masks are compiled on the host
 * from ABI metadata and uploaded with the frame; dispatch runs over active
 * records/tokens only (fixed bands can produce non-contiguous active slots,
 * so the frame's activeRecordIndices list drives everything).
 *
 * The packed SoA frame (M2a packer) carries ascending activeRecordIndices.
 * This module compiles:
 *   - activeTokens: compact list of active (non-padding) frame token ids;
 *   - recordCompactOffset/Count: per-slot compact token ranges;
 *   - streamIds: per-slot stream id (query band -> 1, else 0);
 *   - the block-diagonal record mask for the local encoder attention.
 */
import {
  BRAIN_LIMITS,
  INVALID_U32,
  RECORD_FLAGS,
  RELATION_ROLE_FLAGS,
  RELATION_ROLE_INDEX,
  type RelationRoleName,
} from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import { bandIndex } from "../binary-layout-plan.ts";
import { PAD_TOKEN_ID } from "../frame/packer.ts";
import { STREAM_QUERY } from "./model.ts";
import type { CompiledCatalog } from "../bridge/agent.ts";

export const QUERY_BAND_INDEX = bandIndex("query");
const RECORD_WIDTH = BRAIN_LIMITS.recordWidth;

export interface ActiveFrame {
  /** Compact list of active frame token ids (ascending, record-major). */
  readonly activeTokens: Uint32Array;
  /** activeTokens[recordCompactOffset[slot] .. +recordCompactCount[slot]) is record slot. */
  readonly recordCompactOffset: Uint32Array;
  readonly recordCompactCount: Uint32Array;
  /** Per-slot stream id (STREAM_RECORD / STREAM_QUERY). */
  readonly streamIds: Uint32Array;
  /** Active record slots (ascending) — the frame's activeRecordIndices. */
  readonly activeRecords: Uint32Array;
  /** Bank record slots (non-query active records, ascending). */
  readonly bankRecords: Uint32Array;
  /** Query record slots (ascending). */
  readonly queryRecords: Uint32Array;
}

/**
 * Compile the active token/record lists and stream ids from the packed frame.
 * Padding positions (PAD token) are excluded; the record slot is derived from
 * the frame token id (slot = frameToken / recordWidth).
 */
export function compileActiveFrame(frame: v1_0_0.BrainFrameGpu): ActiveFrame {
  const recordSlots = frame.schemaIds.length;
  const activeTokens: number[] = [];
  const recordCompactOffset = new Uint32Array(recordSlots).fill(0xffff_ffff);
  const recordCompactCount = new Uint32Array(recordSlots);
  const streamIds = new Uint32Array(recordSlots);

  const activeRecords: number[] = [];
  for (const slot of frame.activeRecordIndices) {
    if (slot === 0xffff_ffff || slot >= recordSlots) break; // unoccupied tail
    activeRecords.push(slot);
    recordCompactOffset[slot] = activeTokens.length;
    let count = 0;
    for (let local = 0; local < RECORD_WIDTH; local++) {
      const frameTok = slot * RECORD_WIDTH + local;
      if (frame.tokenIds[frameTok] !== PAD_TOKEN_ID) {
        activeTokens.push(frameTok);
        count++;
      }
    }
    recordCompactCount[slot] = count;
    streamIds[slot] = frame.bandIds[slot] === QUERY_BAND_INDEX ? STREAM_QUERY : 0;
  }

  const bankRecords = activeRecords.filter((slot) => streamIds[slot] !== STREAM_QUERY);
  const queryRecords = activeRecords.filter((slot) => streamIds[slot] === STREAM_QUERY);

  return {
    activeTokens: Uint32Array.from(activeTokens),
    recordCompactOffset,
    recordCompactCount,
    streamIds,
    activeRecords: Uint32Array.from(activeRecords),
    bankRecords: Uint32Array.from(bankRecords),
    queryRecords: Uint32Array.from(queryRecords),
  };
}

/**
 * Compile the block-diagonal record mask for the local encoder attention over
 * T_active compacted tokens: 0.0 when two tokens belong to the same record,
 * -1e30 otherwise (no token ever attends across a record boundary). The
 * padding positions are already absent from the compact list.
 */
export const INVALID_WORD_ID = 0xffff_ffff;

/**
 * Optional same-word structural bias (docs/word_attention_bias.md).
 *
 * `wordIds` is indexed by FRAME token index, exactly like the entries of
 * `activeTokens`, and carries an arbitrary local label per token;
 * INVALID_WORD_ID means "no word" (KEY/VALUE/pooling/query control slots),
 * which never receives the bias. Labels are compared only inside one record,
 * so the same local id may be reused freely across records.
 *
 * The bias is additive on the attention logit, which is why it rides in the
 * existing record mask: the shader already computes
 * `score = q·k*scale + mask[i][j]`, and the backward never re-reads the mask.
 * Nothing in the device path changes, and `alpha = 0` is bit-identical to the
 * unbiased mask by construction.
 */
export interface WordBias {
  readonly wordIds: Readonly<Record<number, number>> | Uint32Array;
  readonly alpha: number;
}

export function compileRecordMask(
  activeTokens: readonly number[] | Uint32Array,
  wordBias?: WordBias,
): { mask: Float32Array; tokenRecords: Uint32Array } {
  const t = activeTokens.length;
  const tokenRecords = new Uint32Array(t);
  for (let i = 0; i < t; i++) tokenRecords[i] = activeTokens[i]! >> 3; // slot = frameToken / recordWidth

  const words = new Uint32Array(t).fill(INVALID_WORD_ID);
  if (wordBias && wordBias.alpha !== 0) {
    for (let i = 0; i < t; i++) {
      const frameToken = activeTokens[i]!;
      const id = (wordBias.wordIds as Record<number, number>)[frameToken];
      words[i] = id === undefined ? INVALID_WORD_ID : id;
    }
  }
  const alpha = wordBias?.alpha ?? 0;

  const mask = new Float32Array(t * t);
  for (let i = 0; i < t; i++) {
    for (let j = 0; j < t; j++) {
      if (tokenRecords[i] !== tokenRecords[j]) {
        mask[i * t + j] = -1e30;
        continue;
      }
      const sameWord =
        alpha !== 0 && words[i] !== INVALID_WORD_ID && words[i] === words[j];
      mask[i * t + j] = sameWord ? alpha : 0;
    }
  }
  return { mask, tokenRecords };
}

/**
 * Compile the typed mixer cross-attention mask [Q, R] (S2-S10 contract,
 * FOLLOW_UP.md §5): the query mixer may attend only to dynamic bank records
 * that carry an exact runtime-ref binding — real world entities relevant to
 * current state, needs and perception. Static ActionIntent catalog records
 * and unrelated distractor/noise records (no runtime ref) do not participate
 * in query-state mixing, so their bulk cannot dilute the decision-relevant
 * signal in the query state.
 *
 * This filters only the query-state mixing: intent selection still addresses
 * the ActionIntent catalog (`compileIntentMask`) and argument selection its
 * normal candidate bank (`argMaskFor`); neither is feasibility-masked as a
 * workaround.
 */
export function compileMixerMask(
  frame: v1_0_0.BrainFrameGpu,
  active: ActiveFrame,
): Float32Array {
  const q = active.queryRecords.length;
  const r = active.bankRecords.length;
  const maxRefs = BRAIN_LIMITS.maxReferencesPerRecord;
  const mask = new Float32Array(q * r); // zeros by default; fill blocked cells
  for (let i = 0; i < q; i++) {
    for (let j = 0; j < r; j++) {
      const slot = active.bankRecords[j]!;
      if (frame.runtimeRefs[slot * maxRefs] === INVALID_U32) mask[i * r + j] = -1e30;
    }
  }
  return mask;
}

/**
 * Compile the intent-selector mask [Q, R] (architecture v2 §7, answer 15):
 * 0.0 for bank records that are ActionIntent catalog records (matching
 * `intentSchemaId`), -1e30 otherwise. The selector must never point to
 * padding, truncated or invalid types.
 */
export function compileIntentMask(
  frame: v1_0_0.BrainFrameGpu,
  active: ActiveFrame,
  intentSchemaId: number,
): Float32Array {
  const q = active.queryRecords.length;
  const r = active.bankRecords.length;
  const mask = new Float32Array(q * r); // zeros by default; fill blocked cells
  for (let i = 0; i < q; i++) {
    for (let j = 0; j < r; j++) {
      const slot = active.bankRecords[j]!;
      if (frame.schemaIds[slot] !== intentSchemaId) mask[i * r + j] = -1e30;
    }
  }
  return mask;
}

/**
 * A role's constraints, resolved once so nobody re-derives them.
 *
 * `undefined` means unconstrained — and it is a distinct value rather than an
 * empty set on purpose. An empty constraint reads identically to a total
 * prohibition when the check is a bare `set.has(...)`, and that ambiguity has
 * now cost four separate bugs in this file and the emitter: every role a world
 * left open resolved to no candidate at all, silently. Making "no constraint"
 * unrepresentable as an empty set is what stops the fifth.
 */
export interface RoleFilter {
  readonly bands: ReadonlySet<number> | undefined;
  /**
   * Whether a candidate must carry a live runtime reference.
   *
   * True for a role that names a world entity (`context_ref`), because such a
   * role is filled by an exact handle read from the record's sidecar and a
   * record without one cannot be acted upon. Leaving it out of the mask let
   * the selector score candidates that could never become a proposal: the
   * choice was made, the emitter then refused it, and the creature spent the
   * tick doing nothing for a reason no one could see from the policy. A
   * structural role (`record_ref` — Self, a body part) is addressed by slot
   * and legitimately has no sidecar, so it is exempt.
   */
  readonly requiresReference: boolean;
}

export function roleFilter(
  candidateBandIds: readonly number[],
  valueKind?: v1_0_0.BrainValueKind,
): RoleFilter {
  return {
    bands: candidateBandIds.length === 0 ? undefined : new Set(candidateBandIds),
    requiresReference: valueKind !== undefined && valueKind !== "record_ref",
  };
}

/**
 * Does this role admit the record in `slot`?
 *
 * The single definition of acceptance, shared by the mask compiler and the
 * emitter. They used to decide it separately and had drifted: the mask opened
 * every record for an unconstrained role while the emitter rejected every one,
 * so the selector was trained on candidates whose selection could never be
 * turned into a proposal.
 *
 * Acceptance is matched against the record's TOKENS, not its schema id. A
 * record carries its identity in `tokens[0]` and its categories after it, so
 * one rule covers both: `accepts: ["resource:Apple"]` names an individual,
 * `accepts: ["category:Edible"]` names a class, and an unseen berry that
 * carries the category is admitted without anyone touching the catalog.
 */
export function roleAdmitsRecord(
  frame: v1_0_0.BrainFrameGpu,
  slot: number,
  filter: RoleFilter,
): boolean {
  if (filter.bands && !filter.bands.has(frame.bandIds[slot]!)) return false;
  // A participant is a thing, never an event. Nothing in the engine binds a
  // relation as a participant — WANT is an operator rather than a relation
  // taking one — so a relation record in a role could only produce "eat the
  // eating". Grammar, not physics: the stone stays admissible.
  if ((frame.recordFlags[slot]! & RECORD_FLAGS.relation) !== 0) return false;
  if (filter.requiresReference && frame.runtimeRefs[slot * BRAIN_LIMITS.maxReferencesPerRecord] === INVALID_U32) {
    return false;
  }
  return true;
}

/**
 * Compile the argument-selector mask [Q, R] for a typed reference slot: 0.0
 * for bank records the role admits, -1e30 otherwise. `acceptedTokens` /
 * `candidateBandIds` come from the relation role descriptor (host-compiled
 * from ABI metadata).
 */
export function compileArgumentMask(
  frame: v1_0_0.BrainFrameGpu,
  active: ActiveFrame,
  candidateBandIds: readonly number[],
  valueKind?: v1_0_0.BrainValueKind,
): Float32Array {
  const q = active.queryRecords.length;
  const r = active.bankRecords.length;
  const filter = roleFilter(candidateBandIds, valueKind);
  const mask = new Float32Array(q * r);
  for (let i = 0; i < q; i++) {
    for (let j = 0; j < r; j++) {
      const slot = active.bankRecords[j]!;
      if (!roleAdmitsRecord(frame, slot, filter)) mask[i * r + j] = -1e30;
    }
  }
  return mask;
}

// ---------------------------------------------------------------------------
// Selected-intent conditional argument masks (S2-S10 contract)
// ---------------------------------------------------------------------------
//
// The three-part supervision contract (S2_S10_CURRICULUM_TASK.md):
//   intentMask        structural legality only (ActionIntent catalog records);
//   argMask           conditioned on (selectedIntent, argumentIndex) and the
//                     catalog argument descriptor — it must exclude Mother,
//                     distractors and incompatible records;
//   argumentTarget    gold bank record / exact runtime-ref sidecar, INVALID_U32
//                     only for arity-0 or explicitly unlabelled rows.
//
// The temporary implementation keeps one pointer head (one argument selector
// dispatch) while every training action has at most one reference argument;
// its mask and loss target are nevertheless selected-intent conditional. The
// public shape is `argMaskFor(intentId, argumentIndex)` (the equivalent of
// argMask[q][intent][argument][record], flattened to the concrete lowering)
// plus `argumentTarget[q][argument]` per query row. See training/policy.ts for
// the curriculum wiring that feeds these host-compiled masks into the runner.

function allBlocked(q: number, r: number): Float32Array {
  return new Float32Array(q * r).fill(-1e30);
}

/** Band ids (bit positions) set in a compiled candidateBandMask. */
export function bandIdsFromMask(candidateBandMask: number): number[] {
  const ids: number[] = [];
  for (let bit = 0; bit < 32; bit++) {
    if ((candidateBandMask >>> bit) & 1) ids.push(bit);
  }
  return ids;
}

/**
 * Compile the [Q, R] mask for one role: 0.0 for bank records the role admits,
 * -1e30 otherwise.
 *
 * What is left after acceptance sets were removed is structural: a candidate
 * must sit in an admissible band and, for a role that names a world entity,
 * must carry a live reference — a record without one cannot be acted upon, and
 * scoring it would let the selector choose something the emitter must then
 * refuse. Whether the candidate makes any SENSE in the role is not asked here.
 * The creature is free to think about eating a stone, and will find out.
 */
export function argMaskFor(
  frame: v1_0_0.BrainFrameGpu,
  active: ActiveFrame,
  catalog: CompiledCatalog,
  intentId: number,
  role: RelationRoleName = "patient",
): Float32Array {
  const q = active.queryRecords.length;
  const r = active.bankRecords.length;
  const descriptor = catalog.descriptors.find((candidate) => candidate.intentId === intentId);
  if (!descriptor) return allBlocked(q, r);
  const roleDesc = descriptor.roles[RELATION_ROLE_INDEX[role]];
  // A relation that does not declare this role has nothing to select for it.
  if (!roleDesc || (roleDesc.flags & RELATION_ROLE_FLAGS.present) === 0) return allBlocked(q, r);
  return compileArgumentMask(
    frame,
    active,
    bandIdsFromMask(roleDesc.candidateBandMask),
    roleDesc.valueKind,
  );
}

/** The compiled filter for one role. */
export function roleFilterFor(
  catalog: CompiledCatalog,
  intentId: number,
  role: RelationRoleName,
  roleDesc: v1_0_0.RelationRoleDescriptor,
): RoleFilter {
  return roleFilter(bandIdsFromMask(roleDesc.candidateBandMask), roleDesc.valueKind);
}

/**
 * Compile a per-query-row argument mask: row i uses `intents[i]`'s argument
 * descriptor (all rows share one selector dispatch, but the mask is
 * conditioned per query on its own selected intent).
 */
export function compilePerRowArgumentMask(
  frame: v1_0_0.BrainFrameGpu,
  active: ActiveFrame,
  catalog: CompiledCatalog,
  intents: readonly number[],
  role: RelationRoleName = "patient",
): Float32Array {
  const q = active.queryRecords.length;
  const r = active.bankRecords.length;
  if (intents.length !== q) throw new ForwardMasksError(`per-row intents must be [Q] = ${q}`);
  const mask = new Float32Array(q * r);
  for (let i = 0; i < q; i++) {
    const row = argMaskFor(frame, active, catalog, intents[i]!, role);
    for (let j = 0; j < r; j++) mask[i * r + j] = row[j]!;
  }
  return mask;
}

export class ForwardMasksError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ForwardMasksError";
  }
}
