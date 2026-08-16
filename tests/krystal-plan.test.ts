// Frozen SoA BinaryLayoutPlan tests (M2a, concerns answer 18): the plan is
// versioned, its geometry matches the schema limits, and any change to buffer
// shape or geometry changes the hash so an older runtime fails the check
// instead of misreading buffers.
import { expect, test } from "bun:test";
import {
  BRAIN_LIMITS,
  BINARY_LAYOUT_PLAN_VERSION,
  KRYSTAL_ABI,
} from "../packages/schema/src/krystal-engine-schema.ts";
import {
  BINARY_LAYOUT_BUFFER_IDS,
  BINARY_LAYOUT_PLAN,
  BINARY_LAYOUT_BUFFER_COUNT,
  bandIndex,
  bandMask,
  brainFrameGpuBuffers,
  brainFrameGpuElementCounts,
  buildBinaryLayoutPlan,
  computePlanHash,
  validatePlan,
  PlanMismatchError,
} from "../packages/krystal/src/binary-layout-plan.ts";

test("plan version and geometry match the frozen ABI", () => {
  const plan = buildBinaryLayoutPlan();
  expect(plan.header.planVersion).toBe(BINARY_LAYOUT_PLAN_VERSION);
  expect(plan.header.layoutVersion).toBe(KRYSTAL_ABI.frameLayoutVersion);
  expect(plan.header.recordSlots).toBe(BRAIN_LIMITS.frameRecordSlots);
  expect(plan.header.recordWidth).toBe(BRAIN_LIMITS.recordWidth);
  expect(plan.header.tokenCapacity).toBe(BRAIN_LIMITS.frameTokens);
  expect(plan.header.maxReferencesPerRecord).toBe(BRAIN_LIMITS.maxReferencesPerRecord);
});

test("buffer descriptors cover the BrainFrameGpu SoA shape exactly", () => {
  const counts = brainFrameGpuElementCounts();
  expect(counts[BINARY_LAYOUT_BUFFER_IDS.tokenIds]).toBe(BRAIN_LIMITS.frameTokens);
  expect(counts[BINARY_LAYOUT_BUFFER_IDS.fieldRoles]).toBe(BRAIN_LIMITS.frameTokens);
  expect(counts[BINARY_LAYOUT_BUFFER_IDS.schemaIds]).toBe(BRAIN_LIMITS.frameRecordSlots);
  expect(counts[BINARY_LAYOUT_BUFFER_IDS.bandIds]).toBe(BRAIN_LIMITS.frameRecordSlots);
  expect(counts[BINARY_LAYOUT_BUFFER_IDS.runtimeRefs]).toBe(
    BRAIN_LIMITS.frameRecordSlots * BRAIN_LIMITS.maxReferencesPerRecord,
  );
  expect(counts[BINARY_LAYOUT_BUFFER_IDS.recordFlags]).toBe(BRAIN_LIMITS.frameRecordSlots);
  expect(counts[BINARY_LAYOUT_BUFFER_IDS.activeRecordIndices]).toBe(BRAIN_LIMITS.frameRecordSlots);

  const buffers = brainFrameGpuBuffers();
  expect(buffers).toHaveLength(BINARY_LAYOUT_BUFFER_COUNT);
  for (let i = 0; i < buffers.length; i++) {
    expect(buffers[i]!.bufferId).toBe(i);
    expect(buffers[i]!.byteSize).toBe(buffers[i]!.elementCount * 4);
  }
});

test("buffer count and element counts cover the whole token capacity", () => {
  const counts = brainFrameGpuElementCounts();
  const tokenSized = counts[BINARY_LAYOUT_BUFFER_IDS.tokenIds];
  expect(tokenSized).toBe(BRAIN_LIMITS.frameRecordSlots * BRAIN_LIMITS.recordWidth);
});

test("the plan hash is deterministic and covers buffer shape", () => {
  const planA = buildBinaryLayoutPlan();
  const planB = buildBinaryLayoutPlan();
  expect(planA.header.planHashLo).toBe(planB.header.planHashLo);
  expect(planA.header.planHashHi).toBe(planB.header.planHashHi);

  // Mutating a descriptor shape must change the hash.
  const tampered = structuredClone(planA);
  tampered.buffers[1]!.elementCount += 1;
  tampered.buffers[1]!.byteSize += 4;
  const hash = computePlanHash(tampered);
  expect(hash.lo).not.toBe(planA.header.planHashLo);
  expect(hash.hi).not.toBe(planA.header.planHashHi);

  // Mutating geometry must change the hash.
  const tamperedGeometry = structuredClone(planA);
  tamperedGeometry.header.recordWidth += 1;
  expect(computePlanHash(tamperedGeometry).lo).not.toBe(planA.header.planHashLo);
});

test("the singleton plan passes validation and rejects tampering", () => {
  expect(() => validatePlan(BINARY_LAYOUT_PLAN)).not.toThrow();

  const wrongVersion = structuredClone(BINARY_LAYOUT_PLAN);
  wrongVersion.header.planVersion = BINARY_LAYOUT_PLAN_VERSION + 1;
  expect(() => validatePlan(wrongVersion)).toThrow(PlanMismatchError);

  const wrongCount = structuredClone(BINARY_LAYOUT_PLAN);
  wrongCount.buffers.pop();
  expect(() => validatePlan(wrongCount)).toThrow(PlanMismatchError);

  const wrongShape = structuredClone(BINARY_LAYOUT_PLAN);
  wrongShape.buffers[0]!.elementCount -= 1;
  wrongShape.buffers[0]!.byteSize -= 4;
  expect(() => validatePlan(wrongShape)).toThrow(PlanMismatchError);
});

test("band index and band mask helpers are stable", () => {
  expect(bandIndex("system")).toBe(0);
  expect(bandIndex("homeostasis")).toBe(1);
  expect(bandIndex("body")).toBe(2);
  expect(bandIndex("vision")).toBe(3);
  expect(bandIndex("memory")).toBe(8);
  expect(bandIndex("query")).toBe(10);

  expect(bandMask(["vision", "memory"])).toBe((1 << 3) | (1 << 8));
  expect(bandMask([])).toBe(0);
});
