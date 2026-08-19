/**
 * Frame packer: canonical AoS `BrainFrame` -> SoA `BrainFrameGpu`.
 *
 * The SoA lowering follows the frozen BinaryLayoutPlan (M2a). Record-local
 * token positions and record slots are logical ABI positions; only the
 * physical packing changes here, never semantics. The packer is mechanical:
 * band membership, 12-bit token range and reference-consistency violations
 * are errors; placement/overflow policy decisions belong to the frame
 * builder, not to this module.
 */
import {
  BRAIN_FRAME_BANDS,
  BRAIN_LIMITS,
  INVALID_U32,
  KRYSTAL_ABI,
  KRYSTAL_SENTINEL_TOKENS,
  RECORD_FLAGS,
  TOKEN_FLAGS,
} from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import {
  BINARY_LAYOUT_PLAN,
  PlanMismatchError,
  bandIndex,
  validatePlan,
} from "../binary-layout-plan.ts";

export { PlanMismatchError };

/** PAD token used for unused token positions. */
export const PAD_TOKEN_ID = KRYSTAL_SENTINEL_TOKENS.pad;

/** Reference half of the 16-bit token space: 0x8000..0xFFFF. */
export const DYNAMIC_REF_TOKEN_START = KRYSTAL_ABI.refSpaceStart;
export const DYNAMIC_REF_TOKEN_END = KRYSTAL_ABI.refSpaceEnd;

/**
 * A packed handle is still one u32, now split 16/16 instead of 12/20.
 *
 * NOTE the narrowing: generation drops from 20 bits (~1M epochs) to 16 (~65k).
 * That is mitigated rather than solved by the wider reference space — with 32k
 * reference slots instead of 256, a given slot is reused ~128x less often, so
 * the wrap horizon is far away in practice. If long-lived autobiographical
 * memory ever needs to outlive 65k rebindings of one slot, the fix is to widen
 * `runtimeRefs` to two words per reference (token in word 0, full u32
 * generation in word 1) rather than to steal bits back from the token.
 */
export const RUNTIME_REF_GENERATION_BITS = 16;
export const RUNTIME_REF_TOKEN_BITS = KRYSTAL_ABI.tokenBits;

export class FramePackerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FramePackerError";
  }
}

/**
 * Pack a runtime handle into one u32: tokenId in the low 16 bits, generation
 * in the high 16. This is the compiler-selected identity projection exposed to
 * the model (generation protects against reference-slot ABA reuse).
 */
export function packRuntimeHandle(handle: v1_0_0.RuntimeRefHandle): number {
  if (handle.tokenId < DYNAMIC_REF_TOKEN_START || handle.tokenId > DYNAMIC_REF_TOKEN_END) {
    throw new FramePackerError(
      `Runtime handle tokenId 0x${handle.tokenId.toString(16)} outside the reference half 0x${DYNAMIC_REF_TOKEN_START.toString(16)}..0x${DYNAMIC_REF_TOKEN_END.toString(16)}`,
    );
  }
  if (handle.generation >= 1 << RUNTIME_REF_GENERATION_BITS) {
    throw new FramePackerError(
      `Runtime handle generation ${handle.generation} exceeds ${RUNTIME_REF_GENERATION_BITS} bits`,
    );
  }
  return ((handle.tokenId | (handle.generation << RUNTIME_REF_TOKEN_BITS)) >>> 0) >>> 0;
}

export function unpackRuntimeHandle(packed: number): v1_0_0.RuntimeRefHandle {
  if (packed === INVALID_U32) return { tokenId: 0, generation: 0, kind: "none", status: "invalid" };
  return {
    tokenId: packed & 0xffff,
    generation: packed >>> RUNTIME_REF_TOKEN_BITS,
    kind: "none",
    status: "live",
  };
}

export interface PackedBrainFrame {
  /** The packed SoA frame. */
  frame: v1_0_0.BrainFrameGpu;
  /** Occupied record slots in ascending order. */
  activeRecordIndices: number[];
}

/**
 * Pack an AoS BrainFrame into the SoA BrainFrameGpu buffers described by the
 * frozen BinaryLayoutPlan. Throws FramePackerError on any structural
 * violation; throws PlanMismatchError when the supplied plan disagrees with
 * the frozen plan.
 */
export function packBrainFrame(
  input: v1_0_0.BrainFrame,
  plan: v1_0_0.BinaryLayoutPlan = BINARY_LAYOUT_PLAN,
): PackedBrainFrame {
  validatePlan(plan);

  const { frameRecordSlots, recordWidth, maxReferencesPerRecord, frameTokens, frameBands } =
    BRAIN_LIMITS;
  if (input.records.length !== frameRecordSlots) {
    throw new FramePackerError(
      `BrainFrame has ${input.records.length} record slots, expected ${frameRecordSlots}`,
    );
  }
  if (input.bands.length !== frameBands) {
    throw new FramePackerError(
      `BrainFrame has ${input.bands.length} band states, expected ${frameBands}`,
    );
  }

  const tokenIds = new Array<number>(frameTokens).fill(PAD_TOKEN_ID);
  const fieldRoles = new Array<number>(frameTokens).fill(0);
  // Default 0: an unoccupied slot is masked out entirely, so the vast empty
  // remainder of every sensory band never reaches attention.
  const attentionMask = new Array<number>(frameTokens).fill(0);
  const schemaIds = new Array<number>(frameRecordSlots).fill(INVALID_U32);
  const bandIds = new Array<number>(frameRecordSlots).fill(INVALID_U32);
  const runtimeRefs = new Array<number>(frameRecordSlots * maxReferencesPerRecord).fill(
    INVALID_U32,
  );
  const recordFlags = new Array<number>(frameRecordSlots).fill(0);
  const activeRecordIndices = new Array<number>(frameRecordSlots).fill(INVALID_U32);

  const active: number[] = [];
  let activeTokenCount = 0;

  for (let slot = 0; slot < frameRecordSlots; slot++) {
    const record = input.records[slot]!;
    const header = record.header;
    if ((header.flags & RECORD_FLAGS.occupied) === 0) continue;

    // Band membership: the record's band must own this slot range.
    const band = BRAIN_FRAME_BANDS.find((candidate) => candidate.kind === header.band);
    if (!band) {
      throw new FramePackerError(`Record ${slot}: unknown band ${header.band}`);
    }
    if (slot < band.recordOffset || slot >= band.recordOffset + band.recordCapacity) {
      throw new FramePackerError(
        `Record ${slot}: band ${header.band} owns slots ${band.recordOffset}..${band.recordOffset + band.recordCapacity - 1}`,
      );
    }

    active.push(slot);
    schemaIds[slot] = header.schemaId;
    bandIds[slot] = bandIndex(header.band);
    recordFlags[slot] = header.flags;

    for (let localToken = 0; localToken < recordWidth; localToken++) {
      const token = record.tokens[localToken] ?? PAD_TOKEN_ID;
      if (token > KRYSTAL_ABI.refSpaceEnd) {
        throw new FramePackerError(
          `Record ${slot} token ${localToken}: 0x${token.toString(16)} exceeds the 16-bit token space`,
        );
      }
      tokenIds[slot * recordWidth + localToken] = token;
      const meta = record.tokenMeta[localToken];
      fieldRoles[slot * recordWidth + localToken] = meta?.roleToken ?? 0;
      if ((meta?.flags ?? 0) & TOKEN_FLAGS.padding) {
        // Padding positions stay PAD and are masked by the model mask.
        if (token !== PAD_TOKEN_ID) {
          throw new FramePackerError(
            `Record ${slot} token ${localToken}: flagged padding but holds 0x${token.toString(16)}`,
          );
        }
      }
      if (token !== PAD_TOKEN_ID) {
        activeTokenCount++;
        // Only structural absence is masked. `void` (sensed emptiness) and
        // `unavailable` (the sense is not reporting) are ordinary percepts and
        // must stay visible — a creature that cannot tell silence from a
        // blocked ear has lost real information.
        attentionMask[slot * recordWidth + localToken] = 1;
      }
    }

    for (let localRef = 0; localRef < maxReferencesPerRecord; localRef++) {
      const binding = record.references[localRef];
      if (!binding || binding.handle.tokenId === 0) continue;
      if (binding.localTokenIndex === INVALID_U32) {
        throw new FramePackerError(`Record ${slot} reference ${localRef}: missing localTokenIndex`);
      }
      const tokenIndex = slot * recordWidth + binding.localTokenIndex;
      if (tokenIds[tokenIndex] !== binding.handle.tokenId) {
        throw new FramePackerError(
          `Record ${slot} reference ${localRef}: token at local index ${binding.localTokenIndex} is 0x${tokenIds[tokenIndex]!.toString(16)}, expected handle 0x${binding.handle.tokenId.toString(16)}`,
        );
      }
      runtimeRefs[slot * maxReferencesPerRecord + localRef] = packRuntimeHandle(binding.handle);
    }
  }

  active.sort((a, b) => a - b);
  for (let i = 0; i < active.length; i++) activeRecordIndices[i] = active[i]!;

  const frame: v1_0_0.BrainFrameGpu = {
    header: { ...plan.header },
    tokenIds,
    fieldRoles,
    attentionMask,
    schemaIds,
    bandIds,
    runtimeRefs,
    recordFlags,
    activeRecordIndices,
  };

  return { frame, activeRecordIndices: active };
}
