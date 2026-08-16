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
import { BRAIN_LIMITS } from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import { bandIndex } from "../binary-layout-plan.ts";
import { PAD_TOKEN_ID } from "../frame/packer.ts";
import { STREAM_QUERY } from "./model.ts";

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
export function compileRecordMask(
  activeTokens: readonly number[] | Uint32Array,
): { mask: Float32Array; tokenRecords: Uint32Array } {
  const t = activeTokens.length;
  const tokenRecords = new Uint32Array(t);
  for (let i = 0; i < t; i++) tokenRecords[i] = activeTokens[i]! >> 3; // slot = frameToken / recordWidth

  const mask = new Float32Array(t * t);
  for (let i = 0; i < t; i++) {
    for (let j = 0; j < t; j++) {
      mask[i * t + j] = tokenRecords[i] === tokenRecords[j] ? 0 : -1e30;
    }
  }
  return { mask, tokenRecords };
}

/**
 * Compile the mixer cross-attention mask [Q, R]: the query may attend to every
 * bank record in the first forward. Candidate masks per selector slot arrive
 * with catalog selection (next M2b step).
 */
export function compileMixerMask(qRows: number, bankRows: number): Float32Array {
  return new Float32Array(qRows * bankRows); // zeros = all allowed
}

export class ForwardMasksError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ForwardMasksError";
  }
}
