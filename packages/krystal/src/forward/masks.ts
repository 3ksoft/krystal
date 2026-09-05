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
  BRAIN_FRAME_BANDS,
  BRAIN_LIMITS,
  KRYSTAL_SENTINEL_TOKENS,
} from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
// The pad sentinel is an ABI fact, taken from the schema that defines it — the
// old frame packer merely re-exported it, and importing it from there is what
// tied every mask to the fixed-geometry world.
const PAD_TOKEN_ID = KRYSTAL_SENTINEL_TOKENS.pad;
import { STREAM_QUERY } from "./model.ts";

/**
 * A band kind's stable id: its index in the frame's band list.
 *
 * Lived in the binary-layout module until that module turned out to be a plan
 * nobody executed. This is the one thing anything asked of it, and it is asked
 * exactly once — the query band is what tells a question from a fact.
 */
export function bandIndex(kind: (typeof BRAIN_FRAME_BANDS)[number]["kind"]): number {
  const index = BRAIN_FRAME_BANDS.findIndex((band) => band.kind === kind);
  if (index < 0) throw new Error(`Unknown BrainBandKind: ${kind}`);
  return index;
}

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
 * Each compact token's own record, as a range into the compact list.
 *
 * The encoder's attention is block-diagonal by construction, and the blocks are
 * eight tokens wide at most. Handing the ranges to the attention lets it visit
 * only the keys that can matter, instead of computing a full T x T score matrix
 * and then adding -1e30 to almost all of it — which is what the device shader
 * has always done, and what made the CPU oracle quadratic in a frame's size for
 * no reason a reader could see.
 */
export interface RecordRanges {
  /** First compact index of this token's record. */
  readonly start: Uint32Array;
  /** How many tokens that record has. */
  readonly count: Uint32Array;
}

export function recordRanges(active: ActiveFrame): RecordRanges {
  const t = active.activeTokens.length;
  const start = new Uint32Array(t);
  const count = new Uint32Array(t);
  for (const slot of active.activeRecords) {
    const from = active.recordCompactOffset[slot]!;
    const size = active.recordCompactCount[slot]!;
    for (let i = from; i < from + size; i++) {
      start[i] = from;
      count[i] = size;
    }
  }
  return { start, count };
}

/**
 * Compile the block-diagonal record mask for the local encoder attention over
 * T_active compacted tokens: 0.0 when two tokens belong to the same record,
 * -1e30 otherwise (no token ever attends across a record boundary). The
 * padding positions are already absent from the compact list.
 *
 * Only the device path builds this now: given the ranges above, the CPU never
 * looks across a record boundary, so every cell it would have read is zero.
 * The mask survives because the same-word bias rides in it (see WordBias).
 */
export const INVALID_WORD_ID = 0xffff_ffff;

/**
 * Optional same-word structural bias (docs/archive/word_attention_bias.md).
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

// ---------------------------------------------------------------------------
// Selected-intent conditional argument masks (S2-S10 contract)
// ---------------------------------------------------------------------------
//
// The three-part supervision contract (docs/archive/S2_S10_CURRICULUM_TASK.md, historical):
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

export function allBlocked(q: number, r: number): Float32Array {
  return new Float32Array(q * r).fill(-1e30);
}


