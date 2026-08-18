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
import { BRAIN_LIMITS, INVALID_U32 } from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import { bandIndex } from "../binary-layout-plan.ts";
import { PAD_TOKEN_ID } from "../frame/packer.ts";
import { STREAM_QUERY } from "./model.ts";
import {
  argumentRequiredCapability,
  argumentNameById,
  intentNameById,
  schemaIdsWithCapability,
} from "../fixtures/capabilities.ts";
import type { CompiledActionCatalog } from "../fixtures/action-intents.ts";

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
 * Compile the argument-selector mask [Q, R] for a typed reference slot: 0.0
 * for bank records whose schemaId is accepted and whose band is a candidate
 * band, -1e30 otherwise. `acceptedSchemaIds`/`candidateBandIds` come from the
 * ActionIntent argument descriptor (host-compiled from ABI metadata).
 */
export function compileArgumentMask(
  frame: v1_0_0.BrainFrameGpu,
  active: ActiveFrame,
  acceptedSchemaIds: readonly number[],
  candidateBandIds: readonly number[],
): Float32Array {
  const q = active.queryRecords.length;
  const r = active.bankRecords.length;
  const accepted = new Set(acceptedSchemaIds);
  const bands = new Set(candidateBandIds);
  const mask = new Float32Array(q * r);
  for (let i = 0; i < q; i++) {
    for (let j = 0; j < r; j++) {
      const slot = active.bankRecords[j]!;
      if (!accepted.has(frame.schemaIds[slot]!) || !bands.has(frame.bandIds[slot]!)) {
        mask[i * r + j] = -1e30;
      }
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
// plus `argumentTarget[q][argument]` per query row. See bridge/policy.ts for
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
 * Compile the [Q, R] argument mask for one (selectedIntent, argumentIndex)
 * pair from its compiled catalog argument descriptor: 0.0 for bank records
 * whose schema is accepted by the argument's capability/identity set and
 * whose band is a candidate band, -1e30 otherwise. Arity-0 intents and
 * out-of-range argument indexes produce an all-blocked row (the selector can
 * never fabricate a pointer for them).
 */
export function argMaskFor(
  frame: v1_0_0.BrainFrameGpu,
  active: ActiveFrame,
  catalog: CompiledActionCatalog,
  intentId: number,
  argumentIndex: number,
): Float32Array {
  const q = active.queryRecords.length;
  const r = active.bankRecords.length;
  const descriptor = catalog.descriptors.find((candidate) => candidate.intentId === intentId);
  if (!descriptor || argumentIndex >= descriptor.argumentCount) return allBlocked(q, r);
  const argDesc = catalog.arguments[descriptor.argumentOffset + argumentIndex]!;
  const accepted = argAcceptedSchemaIds(catalog, intentId, argumentIndex, argDesc);
  return compileArgumentMask(frame, active, accepted, bandIdsFromMask(argDesc.candidateBandMask));
}

/**
 * Accepted schema ids for one argument: capability-derived when the argument
 * declares a required capability (S7 `TARGET_OF(EAT)`), else the single
 * acceptedSchema identity from the compiled descriptor.
 */
export function argAcceptedSchemaIds(
  catalog: CompiledActionCatalog,
  intentId: number,
  argumentIndex: number,
  argDesc?: v1_0_0.ActionArgumentDescriptor,
): number[] {
  const descriptor = catalog.descriptors.find((candidate) => candidate.intentId === intentId);
  if (!descriptor || argumentIndex >= descriptor.argumentCount) return [];
  const desc = argDesc ?? catalog.arguments[descriptor.argumentOffset + argumentIndex]!;
  const intentName = intentNameById(intentId);
  const argName = argumentNameById(intentId, argumentIndex);
  const capability = argumentRequiredCapability(intentName, argName);
  if (capability) return schemaIdsWithCapability(capability);
  return desc.acceptedSchemaId === 0 && desc.valueKind === "context_ref" ? [] : [desc.acceptedSchemaId];
}

/**
 * Compile a per-query-row argument mask: row i uses `intents[i]`'s argument
 * descriptor (all rows share one selector dispatch, but the mask is
 * conditioned per query on its own selected intent).
 */
export function compilePerRowArgumentMask(
  frame: v1_0_0.BrainFrameGpu,
  active: ActiveFrame,
  catalog: CompiledActionCatalog,
  intents: readonly number[],
  argumentIndex = 0,
): Float32Array {
  const q = active.queryRecords.length;
  const r = active.bankRecords.length;
  if (intents.length !== q) throw new ForwardMasksError(`per-row intents must be [Q] = ${q}`);
  const mask = new Float32Array(q * r);
  for (let i = 0; i < q; i++) {
    const row = argMaskFor(frame, active, catalog, intents[i]!, argumentIndex);
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
