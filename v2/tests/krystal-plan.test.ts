// Frozen SoA BinaryLayoutPlan tests (M2a, concerns answer 18): the plan is
// versioned, its geometry matches the schema limits, and any change to buffer
// shape or geometry changes the hash so an older runtime fails the check
// instead of misreading buffers.
import { expect, test } from "bun:test";
import {
  BRAIN_LIMITS,
  BINARY_LAYOUT_PLAN_VERSION,
  KRYSTAL_ABI,
} from "../../packages/schema/src/krystal-engine-schema.ts";
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
} from "../../packages/krystal/src/binary-layout-plan.ts";
import { BRAIN_FRAME_GPU_BUFFERS } from "../../packages/schema/generated/krystal.buffers.ts";
import * as codec from "../../packages/schema/generated/krystal.codec.ts";

test("plan version and geometry match the frozen ABI", () => {
  const plan = buildBinaryLayoutPlan();
  expect(plan.header.planVersion).toBe(BINARY_LAYOUT_PLAN_VERSION);
  expect(plan.header.layoutVersion).toBe(KRYSTAL_ABI.frameLayoutVersion);
  expect(plan.header.recordSlots).toBe(BRAIN_LIMITS.frameRecordSlots);
  expect(plan.header.recordWidth).toBe(BRAIN_LIMITS.recordWidth);
  expect(plan.header.tokenCapacity).toBe(BRAIN_LIMITS.frameTokens);
  expect(plan.header.maxReferencesPerRecord).toBe(BRAIN_LIMITS.maxReferencesPerRecord);
});

test("the buffer table covers every BrainFrameGpu array field, by derivation", () => {
  // The old version of this test named the buffers it expected, which meant a
  // buffer MISSING from the table was invisible to it — the failure mode that
  // actually happened when `attentionMask` was added to the struct. Cross-check
  // the two independently generated artifacts instead: the codec emits one
  // `BRAIN_FRAME_GPU_<FIELD>_LEN` per array field, the buffer table emits one
  // descriptor per array field, and neither is written by hand.
  const codecLengths: [string, number][] = Object.entries(codec)
    .filter(([name]) => name.startsWith("BRAIN_FRAME_GPU_") && name.endsWith("_LEN"))
    .map(([name, value]) => [name.slice("BRAIN_FRAME_GPU_".length, -"_LEN".length), value as number]);

  expect(codecLengths).toHaveLength(BINARY_LAYOUT_BUFFER_COUNT);

  for (const buffer of BRAIN_FRAME_GPU_BUFFERS) {
    const key = buffer.name.toUpperCase();
    const match = codecLengths.find(([name]) => name === key);
    if (!match) throw new Error(`no codec length for buffer ${buffer.name}`);
    expect(match[1]).toBe(buffer.elementCount);
    const declaredId: number =
      BINARY_LAYOUT_BUFFER_IDS[buffer.name as keyof typeof BINARY_LAYOUT_BUFFER_IDS];
    expect(declaredId).toBe(buffer.bufferId);
  }

  // Geometry the ABI fixes, spot-checked through the derived table.
  const counts = brainFrameGpuElementCounts();
  expect(counts[BINARY_LAYOUT_BUFFER_IDS.tokenIds]).toBe(BRAIN_LIMITS.frameTokens);
  expect(counts[BINARY_LAYOUT_BUFFER_IDS.attentionMask]).toBe(BRAIN_LIMITS.frameTokens);
  expect(counts[BINARY_LAYOUT_BUFFER_IDS.runtimeRefs]).toBe(
    BRAIN_LIMITS.frameRecordSlots * BRAIN_LIMITS.maxReferencesPerRecord,
  );
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
