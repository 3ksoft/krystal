/**
 * Versioned SoA BinaryLayoutPlan — M2a freeze.
 *
 * This module is the single implementation of the frozen plan: the canonical
 * `buildBinaryLayoutPlan()` output must not be hand-edited beside the schema
 * (concerns answer 18). The schema owns the *types* (`BinaryLayoutPlan`,
 * `BrainFrameGpu`); this module owns the concrete buffer IDs, geometry and
 * hash so the schema file stays declarative and the plan stays testable.
 *
 * Buffer layout (all u32 in v1, byteSize = 4 * elementCount):
 *
 *   bufferId  name                  elementCount          index
 *   0         tokenIds              128 * 8 = 1024        [slot * 8 + localToken]
 *   1         fieldRoles            1024                  [slot * 8 + localToken]
 *   2         schemaIds             128                   [slot]
 *   3         bandIds               128                   [slot]
 *   4         runtimeRefs           128 * 8 = 1024        [slot * 8 + localRef]
 *   5         recordFlags           128                   [slot]
 *   6         activeRecordIndices   128                   [0..activeRecordCount)
 */
import {
  BINARY_LAYOUT_PLAN_VERSION,
  BRAIN_FRAME_BANDS,
  BRAIN_LIMITS,
  KRYSTAL_ABI,
} from "../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../schema/generated/krystal.types.ts";
import { hashU32s } from "./hash.ts";

type BinaryLayoutBufferDesc = v1_0_0.BinaryLayoutBufferDesc;
type BinaryLayoutPlan = v1_0_0.BinaryLayoutPlan;

export const BINARY_LAYOUT_BUFFER_IDS = {
  tokenIds: 0,
  fieldRoles: 1,
  schemaIds: 2,
  bandIds: 3,
  runtimeRefs: 4,
  recordFlags: 5,
  activeRecordIndices: 6,
} as const;

export const BINARY_LAYOUT_BUFFER_COUNT = Object.keys(BINARY_LAYOUT_BUFFER_IDS).length;

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

/** Element counts of every SoA buffer, in buffer-id order. */
export function brainFrameGpuElementCounts(): number[] {
  const { recordWidth, frameRecordSlots, maxReferencesPerRecord } = BRAIN_LIMITS;
  return [
    frameRecordSlots * recordWidth, // tokenIds
    frameRecordSlots * recordWidth, // fieldRoles
    frameRecordSlots, // schemaIds
    frameRecordSlots, // bandIds
    frameRecordSlots * maxReferencesPerRecord, // runtimeRefs
    frameRecordSlots, // recordFlags
    frameRecordSlots, // activeRecordIndices
  ];
}

/** Canonical SoA buffer descriptors, in buffer-id order. */
export function brainFrameGpuBuffers(): BinaryLayoutBufferDesc[] {
  return brainFrameGpuElementCounts().map((elementCount, bufferId) => ({
    bufferId,
    elementCount,
    byteSize: elementCount * 4,
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
