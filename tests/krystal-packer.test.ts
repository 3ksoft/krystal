// Frame packer tests (M2a): the mechanical AoS BrainFrame -> SoA BrainFrameGpu
// lowering. Band membership, 12-bit token range, padding consistency and
// reference-identity consistency are enforced; placement/overflow policy is
// out of scope here (it belongs to the frame builder).
import { expect, test } from "bun:test";
import {
  BRAIN_FIXED_RECORDS,
  BRAIN_LIMITS,
  INVALID_U32,
  RECORD_FLAGS,
  TOKEN_FLAGS,
} from "../packages/schema/src/krystal-engine-schema.ts";
import {
  buildFixtureFrame,
  buildFixtureRecords,
  FIXTURE_APPLE_REF,
  FIXTURE_MEMORY_REF,
} from "../packages/krystal/src/fixtures/frame.ts";
import { fixtureTokenId } from "../packages/krystal/src/fixtures/vocabulary.ts";
import {
  DYNAMIC_REF_TOKEN_END,
  DYNAMIC_REF_TOKEN_START,
  FramePackerError,
  PAD_TOKEN_ID,
  packBrainFrame,
  packRuntimeHandle,
  unpackRuntimeHandle,
} from "../packages/krystal/src/frame/packer.ts";
import {
  BINARY_LAYOUT_PLAN,
  PlanMismatchError,
  bandIndex,
  buildBinaryLayoutPlan,
} from "../packages/krystal/src/binary-layout-plan.ts";

const { recordWidth, frameRecordSlots, maxReferencesPerRecord } = BRAIN_LIMITS;

test("packer: fixture frame packs to the expected SoA buffers", () => {
  const input = buildFixtureFrame();
  const { frame, activeRecordIndices } = packBrainFrame(input, BINARY_LAYOUT_PLAN);

  // Active records in ascending slot order.
  expect(activeRecordIndices).toEqual([4, 12, 24, 90]);
  // BrainFrameGpu header is the plan header.
  expect(frame.header.planVersion).toBe(BINARY_LAYOUT_PLAN.header.planVersion);
  expect(frame.header.bufferCount).toBe(BINARY_LAYOUT_PLAN.header.bufferCount);

  // tokenIds/fieldRoles at each occupied slot.
  const slots = new Map(buildFixtureRecords().map((spec) => [spec.slot, spec]));
  for (let slot = 0; slot < frameRecordSlots; slot++) {
    const spec = slots.get(slot);
    const base = slot * recordWidth;
    if (!spec) {
      expect(frame.tokenIds.slice(base, base + recordWidth)).toEqual(
        new Array(recordWidth).fill(PAD_TOKEN_ID),
      );
      expect(frame.schemaIds[slot]).toBe(INVALID_U32);
      expect(frame.bandIds[slot]).toBe(INVALID_U32);
      expect(frame.recordFlags[slot]).toBe(0);
      continue;
    }
    expect(frame.tokenIds.slice(base, base + recordWidth)).toEqual([...spec.tokens]);
    expect(frame.fieldRoles.slice(base, base + recordWidth)).toEqual([...spec.roleTokens]);
    expect(frame.schemaIds[slot]).toBe(spec.schemaId);
    expect(frame.bandIds[slot]).toBe(bandIndex(spec.band));
    expect(frame.recordFlags[slot]! & RECORD_FLAGS.occupied).toBe(RECORD_FLAGS.occupied);
  }
});

test("packer: reference bindings pack tokenId + generation and stay consistent", () => {
  const input = buildFixtureFrame();
  const { frame } = packBrainFrame(input);

  // Apple record at slot 24, reference at local index 1.
  const appleBase = 24 * maxReferencesPerRecord;
  const applePacked = frame.runtimeRefs[appleBase]!;
  expect(applePacked).toBe((FIXTURE_APPLE_REF | (3 << 12)) >>> 0);
  const appleHandle = unpackRuntimeHandle(applePacked);
  expect(appleHandle.tokenId).toBe(FIXTURE_APPLE_REF);
  expect(appleHandle.generation).toBe(3);

  // Memory record at slot 90, reference at local index 1.
  const memoryBase = 90 * maxReferencesPerRecord;
  const memoryPacked = frame.runtimeRefs[memoryBase]!;
  expect(unpackRuntimeHandle(memoryPacked).tokenId).toBe(FIXTURE_MEMORY_REF);

  // Every other ref slot stays INVALID_U32.
  expect(frame.runtimeRefs.filter((value) => value === INVALID_U32).length).toBe(
    frameRecordSlots * maxReferencesPerRecord - 2,
  );
});

test("packer: packRuntimeHandle validates the dynamic range and generation bits", () => {
  expect(packRuntimeHandle({ tokenId: DYNAMIC_REF_TOKEN_START, generation: 0, kind: "entity", status: "live" })).toBe(DYNAMIC_REF_TOKEN_START);
  expect(packRuntimeHandle({ tokenId: DYNAMIC_REF_TOKEN_END, generation: 0xfffff, kind: "entity", status: "live" })).toBe((DYNAMIC_REF_TOKEN_END | (0xfffff << 12)) >>> 0);
  expect(() =>
    packRuntimeHandle({ tokenId: 0x123, generation: 0, kind: "entity", status: "live" }),
  ).toThrow(FramePackerError);
  expect(() =>
    packRuntimeHandle({ tokenId: DYNAMIC_REF_TOKEN_START, generation: 1 << 20, kind: "entity", status: "live" }),
  ).toThrow(FramePackerError);
});

test("packer: an empty frame packs with zero active records", () => {
  const input = buildFixtureFrame();
  for (const spec of buildFixtureRecords()) {
    input.records[spec.slot]!.header.flags = 0;
  }
  const { frame, activeRecordIndices } = packBrainFrame(input);
  expect(activeRecordIndices).toEqual([]);
  expect(frame.activeRecordIndices.every((value) => value === INVALID_U32)).toBe(true);
  expect(frame.tokenIds.every((value) => value === PAD_TOKEN_ID)).toBe(true);
  expect(frame.schemaIds.every((value) => value === INVALID_U32)).toBe(true);
});

test("packer: rejects records outside their band's slot range", () => {
  const input = buildFixtureFrame();
  // Move the Self record (body band, slots 12..23) into the vision band.
  input.records[24]!.header = {
    ...input.records[24]!.header,
    band: "body",
  };
  expect(() => packBrainFrame(input)).toThrow(FramePackerError);
});

test("packer: rejects tokens outside the 12-bit token space", () => {
  const input = buildFixtureFrame();
  input.records[24]!.tokens[5] = 0x1000;
  expect(() => packBrainFrame(input)).toThrow(FramePackerError);
});

test("packer: rejects padding-flagged positions that hold non-PAD tokens", () => {
  const input = buildFixtureFrame();
  const meta = input.records[24]!.tokenMeta[5]!;
  meta.flags = TOKEN_FLAGS.padding;
  input.records[24]!.tokens[5] = fixtureTokenId("RED");
  expect(() => packBrainFrame(input)).toThrow(FramePackerError);
});

test("packer: rejects reference bindings whose token position disagrees", () => {
  const input = buildFixtureFrame();
  const binding = input.records[24]!.references[0]!;
  binding.handle.tokenId = FIXTURE_MEMORY_REF; // token at index 1 is FIXTURE_APPLE_REF
  expect(() => packBrainFrame(input)).toThrow(FramePackerError);
});

test("packer: rejects a stale plan version", () => {
  const input = buildFixtureFrame();
  const stale = buildBinaryLayoutPlan();
  stale.header.planVersion += 1;
  expect(() => packBrainFrame(input, stale)).toThrow(PlanMismatchError);
});

test("packer: rejects wrong record-slot count", () => {
  const input = buildFixtureFrame();
  input.records.pop();
  expect(() => packBrainFrame(input)).toThrow(FramePackerError);
});

test("packer: fixture frame active token counts agree with band states", () => {
  const input = buildFixtureFrame();
  const expected = buildFixtureRecords().reduce(
    (sum, spec) => sum + spec.tokens.filter((token) => token !== PAD_TOKEN_ID).length,
    0,
  );
  expect(input.header.activeTokenCount).toBe(expected);
  expect(input.header.activeRecordCount).toBe(buildFixtureRecords().length);
  // Fixed records use the documented bindings.
  expect(input.records[BRAIN_FIXED_RECORDS.self]!.header.flags & RECORD_FLAGS.occupied).toBe(
    RECORD_FLAGS.occupied,
  );
});
