// M2b IntentSet emission tests (concerns answer 27): the host resolver that
// turns the GPU selection heads into a typed IntentSet with exact argument
// handles resolved from the record sidecars. Pure CPU — no GPU needed.
import { expect, test } from "bun:test";
import {
  BRAIN_LIMITS,
  INVALID_U32,
} from "../packages/schema/src/krystal-engine-schema.ts";
import { packBrainFrame, unpackRuntimeHandle } from "../packages/krystal/src/frame/packer.ts";
import { buildFixtureActionCatalog, fixtureIntent } from "../packages/krystal/src/fixtures/action-intents.ts";
import { ACTION_INTENT_SCHEMA_ID, buildFixtureFrame, FIXTURE_APPLE_REF, FIXTURE_MEMORY_REF } from "../packages/krystal/src/fixtures/frame.ts";
import { fixtureTokenId } from "../packages/krystal/src/fixtures/vocabulary.ts";
import { compileActiveFrame } from "../packages/krystal/src/forward/masks.ts";
import { emitIntentSet, emptyProposal, type IntentSetEmissionInput } from "../packages/krystal/src/forward/intentset.ts";
import { selectorOracle, softmaxRow } from "../packages/krystal/src/forward/oracle.ts";
import type { v1_0_0 } from "../packages/schema/generated/krystal.types.ts";

const H = 8;

function fixtureBase(): {
  frame: v1_0_0.BrainFrameGpu;
  active: ReturnType<typeof compileActiveFrame>;
  catalog: ReturnType<typeof buildFixtureActionCatalog>;
} {
  const frame = packBrainFrame(buildFixtureFrame()).frame;
  const active = compileActiveFrame(frame);
  const catalog = buildFixtureActionCatalog();
  return { frame, active, catalog };
}

/**
 * Build a SelectionSlotResult whose argmax lands on the given bank index and
 * whose distribution is peaked there (mask-free: all zeros except a spike).
 */
function peakedSlot(bankIdx: number, r: number, q = 1): ReturnType<typeof selectorOracle> {
  const p = new Float32Array(q * r);
  p[bankIdx] = 1;
  const gather = new Float32Array(q * H);
  const index = Uint32Array.from({ length: q }, () => bankIdx);
  return { p, gather, index };
}

function emit(
  input: Pick<IntentSetEmissionInput, "intent" | "argument"> & Partial<Omit<IntentSetEmissionInput, "intent" | "argument">>,
): v1_0_0.IntentSet {
  const base = fixtureBase();
  return emitIntentSet({
    frame: base.frame,
    active: base.active,
    catalog: base.catalog,
    intentSchemaId: ACTION_INTENT_SCHEMA_ID,
    tick: 10,
    ...input,
  }).intentSet;
}

test("IntentSet: EAT selected with Apple argument resolves exact handles", () => {
  const base = fixtureBase();
  // Bank order (non-query active slots ascending): homeostasis(4), self(12),
  // apple(24), memory(90), LOOK(116), EAT(117), WAIT(118). Query is slot 122.
  const EAT_BANK = base.active.bankRecords.indexOf(117);
  const APPLE_BANK = base.active.bankRecords.indexOf(24);
  const r = base.active.bankRecords.length;

  const intent = peakedSlot(EAT_BANK, r);
  const argument = peakedSlot(APPLE_BANK, r);
  const set = emit({ frame: base.frame, active: base.active, catalog: base.catalog, intent, argument });

  expect(set.tick).toBe(10);
  expect(set.count).toBe(1);
  expect(set.proposals.length).toBe(BRAIN_LIMITS.maxIntentProposals);

  const proposal = set.proposals[0]!;
  expect(proposal.lifecycle).toBe("start");
  // EAT is the second intent in the fixture catalog.
  expect(proposal.intentId).toBe(fixtureIntent("EAT").intentId);
  expect(proposal.confidence).toBe(1);

  // One typed argument: EAT(target: Apple context_ref).
  const arg = proposal.arguments[0]!;
  expect(arg.kind).toBe("context_ref");
  expect(arg.token).toBe(FIXTURE_APPLE_REF);
  expect(arg.handle.tokenId).toBe(FIXTURE_APPLE_REF);
  expect(arg.handle.generation).toBe(3);
  expect(arg.selector.status).toBe("selected");
  expect(arg.selector.selectedRecord).toBe(24);
  expect(arg.selector.probability).toBe(1);

  // Unused argument slots stay empty/invalid.
  for (let k = 1; k < BRAIN_LIMITS.maxActionArguments; k++) {
    expect(proposal.arguments[k]!.kind).toBe("none");
    expect(proposal.arguments[k]!.selector.status).toBe("empty");
  }
});

test("IntentSet: LOOK selected with Apple yields masked (Apple is not VisionObject)", () => {
  const base = fixtureBase();
  // LOOK accepts VisionObject (schema 1); the Apple record is schema 2.
  const LOOK_BANK = base.active.bankRecords.indexOf(116);
  const APPLE_BANK = base.active.bankRecords.indexOf(24);
  const r = base.active.bankRecords.length;

  const set = emit({
    frame: base.frame,
    active: base.active,
    catalog: base.catalog,
    intent: peakedSlot(LOOK_BANK, r),
    argument: peakedSlot(APPLE_BANK, r),
  });

  expect(set.count).toBe(1);
  const proposal = set.proposals[0]!;
  expect(proposal.intentId).toBe(fixtureIntent("LOOK").intentId);
  const arg = proposal.arguments[0]!;
  expect(arg.selector.status).toBe("masked");
  expect(arg.handle.tokenId).toBe(0);
});

test("IntentSet: WAIT emits a proposal with no typed arguments", () => {
  const base = fixtureBase();
  const WAIT_BANK = base.active.bankRecords.indexOf(118);
  const r = base.active.bankRecords.length;

  const set = emit({
    frame: base.frame,
    active: base.active,
    catalog: base.catalog,
    intent: peakedSlot(WAIT_BANK, r),
    argument: peakedSlot(0, r), // unused: WAIT has no argument slots
  });

  expect(set.count).toBe(1);
  const proposal = set.proposals[0]!;
  expect(proposal.intentId).toBe(fixtureIntent("WAIT").intentId);
  expect(proposal.arguments[0]!.kind).toBe("none");
  expect(proposal.arguments[0]!.selector.status).toBe("empty");
});

test("IntentSet: argument selector landing on a wrong schema yields masked, not fabricated handle", () => {
  const base = fixtureBase();
  // LOOK accepts VisionObject (schema id 1). Memory is schema id 4.
  const LOOK_BANK = base.active.bankRecords.indexOf(116);
  const MEMORY_BANK = base.active.bankRecords.indexOf(90);
  const r = base.active.bankRecords.length;

  const set = emit({
    frame: base.frame,
    active: base.active,
    catalog: base.catalog,
    intent: peakedSlot(LOOK_BANK, r),
    argument: peakedSlot(MEMORY_BANK, r),
  });

  expect(set.count).toBe(1);
  const arg = set.proposals[0]!.arguments[0]!;
  expect(arg.selector.status).toBe("masked");
  expect(arg.handle.tokenId).toBe(0);
  expect(arg.selector.selectedRecord).toBe(INVALID_U32);
});

test("IntentSet: masked/absent intent selection yields count 0", () => {
  const base = fixtureBase();
  const r = base.active.bankRecords.length;
  // All-masked row: softmax over -1e30 is uniform, argmax falls to index 0
  // (homeostasis, schema 3 != ACTION_INTENT_SCHEMA_ID) -> not a proposal.
  const mask = new Float32Array(r).fill(-1e30);
  const scores = new Float32Array(r).fill(-1e30);
  softmaxRow(scores, 0, r);
  const p = new Float32Array(r);
  p.set(scores);
  const index = new Uint32Array([0]);
  const intent = { p, gather: new Float32Array(H), index };

  const set = emit({
    frame: base.frame,
    active: base.active,
    catalog: base.catalog,
    intent,
    argument: peakedSlot(0, r),
  });

  expect(set.count).toBe(0);
  for (const proposal of set.proposals) {
    expect(proposal.lifecycle).toBe("empty");
    expect(proposal.intentId).toBe(0);
  }
});

test("IntentSet: count 0 preserves transport capacity and empty slots", () => {
  const base = fixtureBase();
  const r = base.active.bankRecords.length;
  const set = emit({
    frame: base.frame,
    active: base.active,
    catalog: base.catalog,
    intent: peakedSlot(0, r), // homeostasis: not a catalog record
    argument: peakedSlot(0, r),
  });
  expect(set.count).toBe(0);
  expect(set.proposals.length).toBe(BRAIN_LIMITS.maxIntentProposals);
  expect(set.revision).toBe(0);
});

test("IntentSet: emitted handles round-trip through the packer's sidecar", () => {
  const base = fixtureBase();
  const EAT_BANK = base.active.bankRecords.indexOf(117);
  const APPLE_BANK = base.active.bankRecords.indexOf(24);
  const r = base.active.bankRecords.length;

  const set = emit({
    frame: base.frame,
    active: base.active,
    catalog: base.catalog,
    intent: peakedSlot(EAT_BANK, r),
    argument: peakedSlot(APPLE_BANK, r),
  });
  const arg = set.proposals[0]!.arguments[0]!;
  // The packed sidecar for Apple (slot 24) is the same handle the packer stored.
  const packed = base.frame.runtimeRefs[24 * BRAIN_LIMITS.maxReferencesPerRecord]!;
  const unpacked = unpackRuntimeHandle(packed);
  expect(arg.handle.tokenId).toBe(unpacked.tokenId);
  expect(arg.handle.generation).toBe(unpacked.generation);
  expect(arg.handle.tokenId).toBe(FIXTURE_APPLE_REF);
});

test("IntentSet: selector distributions with entropy/candidate counts", () => {
  const base = fixtureBase();
  const LOOK_BANK = base.active.bankRecords.indexOf(116);
  const APPLE_BANK = base.active.bankRecords.indexOf(24);
  const r = base.active.bankRecords.length;

  // A soft distribution over the three catalog records (others masked).
  const scores = new Float32Array(r).fill(-1e30);
  scores[LOOK_BANK] = 2;
  scores[base.active.bankRecords.indexOf(117)] = 1;
  scores[base.active.bankRecords.indexOf(118)] = 0.5;
  const p = new Float32Array(r).fill(-1e30);
  softmaxRow(p, 0, r);
  p[LOOK_BANK] = Math.exp(2 - 2);
  p[base.active.bankRecords.indexOf(117)] = Math.exp(1 - 2);
  p[base.active.bankRecords.indexOf(118)] = Math.exp(0.5 - 2);
  const norm = p[LOOK_BANK]! + p[base.active.bankRecords.indexOf(117)]! + p[base.active.bankRecords.indexOf(118)]!;
  for (let j = 0; j < r; j++) p[j] = p[j]! / norm;
  const index = new Uint32Array([LOOK_BANK]);
  const intent = { p, gather: new Float32Array(H), index };

  const argP = new Float32Array(r);
  argP[APPLE_BANK] = 1;
  const argIndex = new Uint32Array([APPLE_BANK]);
  const argument = { p: argP, gather: new Float32Array(H), index: argIndex };

  const set = emit({ frame: base.frame, active: base.active, catalog: base.catalog, intent, argument });
  const proposal = set.proposals[0]!;
  expect(proposal.intentId).toBe(fixtureIntent("LOOK").intentId);
  expect(proposal.confidence).toBeGreaterThan(0.5);
  expect(proposal.arguments[0]!.selector.candidateCount).toBe(1);
  expect(proposal.arguments[0]!.selector.probability).toBe(1);
  // Entropy is strictly positive on a non-degenerate distribution.
  expect(proposal.arguments[0]!.selector.entropy).toBe(0);
});

test("IntentSet: emptyProposal factory matches the schema envelope", () => {
  const proposal = emptyProposal(3);
  expect(proposal.proposalSlot).toBe(3);
  expect(proposal.lifecycle).toBe("empty");
  expect(proposal.arguments.length).toBe(BRAIN_LIMITS.maxActionArguments);
  for (const arg of proposal.arguments) {
    expect(arg.kind).toBe("none");
    expect(arg.selector.status).toBe("empty");
  }
});
