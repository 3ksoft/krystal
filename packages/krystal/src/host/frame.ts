/**
 * A frame, from a list of token records.
 *
 * The forward pass reads six things per token: the token's own row, its field
 * role's row, its record's schema and band, whether the record is a query or
 * part of the bank, and the token's position inside its record. Everything else
 * the frozen `BrainFrame` carries — headers, band states, reference tables,
 * revisions — is host bookkeeping the model never looks at.
 *
 * So the host hands over exactly that: records of tokens. Krystal serializes
 * them and says nothing about what a band means, how many records a world may
 * have, or which of them is perception and which is memory. A simulation that
 * wants ten memory records and two hundred perceptual ones simply sends them;
 * the geometry is the list it sent, not a constant compiled in here.
 *
 * The one structural rule left is the record width: eight token positions, one
 * learned position embedding each. A record with more is a record that would
 * change the model's shape, so it is refused rather than truncated.
 */
import {
  BRAIN_LIMITS,
  INVALID_U32,
  KRYSTAL_SENTINEL_TOKENS,
} from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import { QUERY_BAND_INDEX, compileActiveFrame, compileRecordMask, type ActiveFrame } from "../forward/masks.ts";

export const RECORD_WIDTH = BRAIN_LIMITS.recordWidth;
export const PAD_TOKEN = KRYSTAL_SENTINEL_TOKENS.pad;
/** The band a query row must sit in — the one band index the model's own
 *  plumbing reads, because it is what tells a question from a fact. */
export const QUERY_BAND = QUERY_BAND_INDEX;

export class HostFrameError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "HostFrameError";
  }
}

/** A token and, optionally, the field it fills. Both are ids in the host's own
 *  manifest; the session maps them to embedding rows. */
export interface HostToken {
  readonly token: number;
  readonly role?: number;
}

export interface HostRecord {
  /** Which kind of record this is. Free for the host to assign; the model only
   *  learns to tell them apart. */
  readonly schemaId?: number;
  /** Which band it belongs to. Meaningless to Krystal beyond its own embedding
   *  row — and beyond `QUERY_BAND`, which marks a question. */
  readonly band?: number;
  /** A question this frame asks. Shorthand for `band: QUERY_BAND`. */
  readonly query?: boolean;
  readonly tokens: readonly (number | HostToken)[];
}

export interface HostFrame {
  readonly gpu: v1_0_0.BrainFrameGpu;
  readonly active: ActiveFrame;
  /** Block-diagonal encoder mask: no token attends across a record boundary. */
  readonly recordMask: Float32Array;
  /** Record slots, in the order the host sent them. */
  readonly slots: number;
}

const tokenOf = (entry: number | HostToken): HostToken => (typeof entry === "number" ? { token: entry } : entry);

/**
 * Serialize a list of records into the arrays the forward reads.
 *
 * Sized to what it was given: a frame of three records is three records, not
 * three records padded into a fixed 432-slot geometry. Padding was never the
 * model's requirement — it was the old fixed-layout ABI's.
 */
export function packHostFrame(records: readonly HostRecord[]): HostFrame {
  const slots = records.length;
  if (!slots) throw new HostFrameError("a frame with no records has nothing to think about");

  const tokenIds = new Array<number>(slots * RECORD_WIDTH).fill(PAD_TOKEN);
  const fieldRoles = new Array<number>(slots * RECORD_WIDTH).fill(0);
  const attentionMask = new Array<number>(slots * RECORD_WIDTH).fill(0);
  const schemaIds = new Array<number>(slots).fill(0);
  const bandIds = new Array<number>(slots).fill(0);
  const recordFlags = new Array<number>(slots).fill(0);
  // The reference table is the host's business now; the frame carries an empty
  // one so anything reading the shape finds what it expects.
  const runtimeRefs = new Array<number>(slots * BRAIN_LIMITS.maxReferencesPerRecord).fill(INVALID_U32);
  const activeRecordIndices = new Array<number>(slots).fill(INVALID_U32);

  for (let slot = 0; slot < slots; slot++) {
    const record = records[slot]!;
    const tokens = record.tokens ?? [];
    if (tokens.length > RECORD_WIDTH) {
      throw new HostFrameError(
        `record ${slot} carries ${tokens.length} tokens; ${RECORD_WIDTH} is the width the model has positions for — split it into a base record and a continuation`,
      );
    }
    schemaIds[slot] = record.schemaId ?? 0;
    bandIds[slot] = record.query ? QUERY_BAND : (record.band ?? 0);
    activeRecordIndices[slot] = slot;
    for (let local = 0; local < tokens.length; local++) {
      const { token, role } = tokenOf(tokens[local]!);
      if (!Number.isInteger(token) || token < 0) throw new HostFrameError(`record ${slot} token ${local} is not a token id`);
      if (token === PAD_TOKEN) continue; // a hole in a record is simply absent
      const at = slot * RECORD_WIDTH + local;
      tokenIds[at] = token;
      fieldRoles[at] = role ?? 0;
      attentionMask[at] = 1;
    }
  }

  const gpu = {
    header: {} as any,
    tokenIds,
    fieldRoles,
    attentionMask,
    schemaIds,
    bandIds,
    runtimeRefs,
    recordFlags,
    activeRecordIndices,
  } as unknown as v1_0_0.BrainFrameGpu;

  const active = compileActiveFrame(gpu);
  const { mask: recordMask } = compileRecordMask(active.activeTokens);
  return { gpu, active, recordMask, slots };
}
