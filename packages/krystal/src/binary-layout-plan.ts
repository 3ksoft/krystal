/**
 * Versioned SoA BinaryLayoutPlan.
 *
 * The buffer set is NOT defined here. It is generated from `BrainFrameGpu` by
 * the schema build (`generated/krystal.buffers.ts`): the struct's array fields
 * are the buffers, in declaration order, and the analyzer already knows each
 * one's exact length and byte size. This module only versions, hashes and
 * validates that generated table.
 *
 * That split is the point. A buffer set maintained by hand next to the struct
 * it describes cannot be checked by anything — the type system does not relate
 * the two, and a test that enumerates buffers by name cannot notice one that is
 * missing. Deriving it means a buffer added, removed or resized in the schema
 * propagates on rebuild and cannot silently disagree.
 *
 * Indexing (all u32; byteSize = 4 * elementCount):
 *   tokenIds / fieldRoles / attentionMask   [slot * recordWidth + localToken]
 *   runtimeRefs                             [slot * maxReferencesPerRecord + localRef]
 *   schemaIds / bandIds / recordFlags       [slot]
 *   activeRecordIndices                     [0..activeRecordCount)
 */
import {
  BINARY_LAYOUT_PLAN_VERSION,
  BRAIN_FRAME_BANDS,
  BRAIN_LIMITS,
  KRYSTAL_ABI,
} from "../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../schema/generated/krystal.types.ts";
import {
  BINARY_LAYOUT_BUFFER_COUNT,
  BINARY_LAYOUT_BUFFER_IDS,
  BRAIN_FRAME_GPU_BUFFERS,
} from "../../schema/generated/krystal.buffers.ts";
import { hashU32s } from "./hash.ts";

export { BINARY_LAYOUT_BUFFER_COUNT, BINARY_LAYOUT_BUFFER_IDS };

type BinaryLayoutBufferDesc = v1_0_0.BinaryLayoutBufferDesc;
type BinaryLayoutPlan = v1_0_0.BinaryLayoutPlan;

/** BrainBandKind -> index in BRAIN_FRAME_BANDS (stable band id). */
export function bandIndex(kind: (typeof BRAIN_FRAME_BANDS)[number]["kind"]): number {
  const index = BRAIN_FRAME_BANDS.findIndex((band) => band.kind === kind);
  if (index < 0) throw new Error(`Unknown BrainBandKind: ${kind}`);
  return index;
}

/** Bitmask over band ids; bit `i` is set for band with index `i`. */
export function bandMask(kinds: readonly (typeof BRAIN_FRAME_BANDS)[number]["kind"][]): number {
  let mask = 0;
  for (const kind of kinds) mask |= 1 << bandIndex(kind);
  return mask >>> 0;
}

/** Element counts of every SoA buffer, in buffer-id order (schema-derived). */
export function brainFrameGpuElementCounts(): number[] {
  return BRAIN_FRAME_GPU_BUFFERS.map((buffer) => buffer.elementCount);
}

/**
 * Canonical SoA buffer descriptors, in buffer-id order.
 *
 * `byteSize` is taken from the generated table rather than recomputed as
 * `4 * elementCount`, so that an element type that stops being u32 surfaces as
 * a plan mismatch instead of being silently mis-sized here.
 */
export function brainFrameGpuBuffers(): BinaryLayoutBufferDesc[] {
  return BRAIN_FRAME_GPU_BUFFERS.map((buffer) => ({
    bufferId: buffer.bufferId,
    elementCount: buffer.elementCount,
    byteSize: buffer.byteSize,
    flags: 0,
  }));
}

/** Hash over the descriptor list and geometry. */
export function computePlanHash(plan: BinaryLayoutPlan): { lo: number; hi: number } {
  const words: number[] = [plan.header.bufferCount];
  for (const buffer of plan.buffers) {
    words.push(buffer.bufferId, buffer.elementCount, buffer.byteSize, buffer.flags);
  }
  words.push(
    plan.header.recordSlots,
    plan.header.recordWidth,
    plan.header.tokenCapacity,
    plan.header.maxReferencesPerRecord,
  );
  return hashU32s(words);
}

/** Build the canonical frozen plan for BINARY_LAYOUT_PLAN_VERSION. */
export function buildBinaryLayoutPlan(): BinaryLayoutPlan {
  const buffers = brainFrameGpuBuffers();
  const plan: BinaryLayoutPlan = {
    header: {
      planVersion: BINARY_LAYOUT_PLAN_VERSION,
      layoutVersion: KRYSTAL_ABI.frameLayoutVersion,
      bufferCount: buffers.length,
      recordSlots: BRAIN_LIMITS.frameRecordSlots,
      recordWidth: BRAIN_LIMITS.recordWidth,
      tokenCapacity: BRAIN_LIMITS.frameTokens,
      maxReferencesPerRecord: BRAIN_LIMITS.maxReferencesPerRecord,
      planHashLo: 0,
      planHashHi: 0,
      flags: 0,
      reserved0: 0,
    },
    buffers,
  };
  const hash = computePlanHash(plan);
  plan.header.planHashLo = hash.lo;
  plan.header.planHashHi = hash.hi;
  return plan;
}

/** Canonical frozen plan instance (singleton; buffers are immutable data). */
export const BINARY_LAYOUT_PLAN: BinaryLayoutPlan = buildBinaryLayoutPlan();

export class PlanMismatchError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PlanMismatchError";
  }
}

/**
 * Validate a plan against the frozen ABI. Throws PlanMismatchError when the
 * plan version, geometry or buffer shape disagrees with the canonical plan.
 */
export function validatePlan(plan: BinaryLayoutPlan): void {
  if (plan.header.planVersion !== BINARY_LAYOUT_PLAN_VERSION) {
    throw new PlanMismatchError(
      `BinaryLayoutPlan version ${plan.header.planVersion} does not match frozen version ${BINARY_LAYOUT_PLAN_VERSION}`,
    );
  }
  const canonical = buildBinaryLayoutPlan();
  if (
    plan.header.layoutVersion !== canonical.header.layoutVersion ||
    plan.header.recordSlots !== canonical.header.recordSlots ||
    plan.header.recordWidth !== canonical.header.recordWidth ||
    plan.header.tokenCapacity !== canonical.header.tokenCapacity ||
    plan.header.maxReferencesPerRecord !== canonical.header.maxReferencesPerRecord
  ) {
    throw new PlanMismatchError("BinaryLayoutPlan geometry does not match the frozen plan");
  }
  const expected = brainFrameGpuBuffers();
  if (plan.buffers.length !== expected.length) {
    throw new PlanMismatchError(
      `BinaryLayoutPlan buffer count ${plan.buffers.length} does not match ${expected.length}`,
    );
  }
  for (let i = 0; i < expected.length; i++) {
    const buffer = plan.buffers[i]!;
    const want = expected[i]!;
    if (
      buffer.bufferId !== want.bufferId ||
      buffer.elementCount !== want.elementCount ||
      buffer.byteSize !== want.byteSize
    ) {
      throw new PlanMismatchError(
        `BinaryLayoutPlan buffer ${i} (id ${buffer.bufferId}) does not match the frozen plan`,
      );
    }
  }
  const hash = computePlanHash(plan);
  if (hash.lo !== plan.header.planHashLo || hash.hi !== plan.header.planHashHi) {
    throw new PlanMismatchError("BinaryLayoutPlan hash does not match its descriptor list");
  }
}
