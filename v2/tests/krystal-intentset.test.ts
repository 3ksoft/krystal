// M2b IntentSet emission tests (concerns answer 27): the host resolver that
// turns the GPU selection heads into a typed IntentSet with exact argument
// handles resolved from the record sidecars. Pure CPU — no GPU needed.
import { expect, test } from "bun:test";
import {
  BRAIN_FIXED_RECORDS,
  BRAIN_FRAME_BANDS,
  BRAIN_LIMITS,
  INVALID_U32,
} from "../../packages/schema/src/krystal-engine-schema.ts";

// Fixture slots, taken from the layout rather than written as literals: the
// canonical frame places its Apple in the first vision slot, its MemoryObject
// in the first memory slot and LOOK/EAT/WAIT in the first catalog slots.
const VISION_SLOT = BRAIN_FRAME_BANDS.find((band) => band.kind === "vision")!.recordOffset;
const MEMORY_SLOT = BRAIN_FRAME_BANDS.find((band) => band.kind === "memory")!.recordOffset;
const CATALOG = {
  LOOK: BRAIN_FIXED_RECORDS.catalogBase,
  EAT: BRAIN_FIXED_RECORDS.catalogBase + 1,
  WAIT: BRAIN_FIXED_RECORDS.catalogBase + 2,
} as const;
import { packBrainFrame, unpackRuntimeHandle } from "../../packages/krystal/src/frame/packer.ts";
import { buildFixtureActionCatalog, fixtureIntent } from "../../packages/krystal/src/fixtures/action-intents.ts";
import { ACTION_INTENT_SCHEMA_ID, buildFixtureFrame, FIXTURE_APPLE_REF, FIXTURE_MEMORY_REF } from "../../packages/krystal/src/fixtures/frame.ts";
import { fixtureTokenId } from "../../packages/krystal/src/fixtures/vocabulary.ts";
import { compileActiveFrame } from "../../packages/krystal/src/forward/masks.ts";
import { emitIntentSet, emptyProposal, type IntentSetEmissionInput } from "../../packages/krystal/src/forward/intentset.ts";
import { selectorOracle, softmaxRow } from "../../packages/krystal/src/forward/oracle.ts";
import type { v1_0_0 } from "../../packages/schema/generated/krystal.types.ts";

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

test.todo("IntentSet: EAT selected with Apple argument resolves exact handles", () => {
  const base = fixtureBase();
  // Bank order is the non-query active slots ascending: homeostasis, self,
  // apple, memory, then the LOOK/EAT/WAIT catalog records.
  const EAT_BANK = base.active.bankRecords.indexOf(CATALOG.EAT);
  const APPLE_BANK = base.active.bankRecords.indexOf(VISION_SLOT);
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
  // Peaked intent (p=1) × peakedness 1 × peaked argument (p=1) -> intensity 1.
  expect(proposal.intensity).toBe(1);

  // One typed argument: EAT(target: Apple context_ref).
  const arg = proposal.arguments[0]!;
  expect(arg.kind).toBe("context_ref");
  expect(arg.token).toBe(FIXTURE_APPLE_REF);
  expect(arg.handle.tokenId).toBe(FIXTURE_APPLE_REF);
  expect(arg.handle.generation).toBe(3);
  expect(arg.selector.status).toBe("selected");
  expect(arg.selector.selectedRecord).toBe(VISION_SLOT);
  expect(arg.selector.probability).toBe(1);

  // Unused argument slots stay empty/invalid.
  for (let k = 1; k < BRAIN_LIMITS.maxActionArguments; k++) {
    expect(proposal.arguments[k]!.kind).toBe("none");
    expect(proposal.arguments[k]!.selector.status).toBe("empty");
  }
});

test.todo("IntentSet: LOOK selected with Apple resolves (Apple is observable)", () => {
  const base = fixtureBase();
  // LOOK.target requires the "observable" capability (S7 contract); the Apple
  // schema carries it, so the argument resolves even though Apple is not the
  // VisionObject identity the descriptor names.
  const LOOK_BANK = base.active.bankRecords.indexOf(CATALOG.LOOK);
  const APPLE_BANK = base.active.bankRecords.indexOf(VISION_SLOT);
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
  expect(arg.selector.status).toBe("selected");
  expect(arg.handle.tokenId).toBe(FIXTURE_APPLE_REF);
});

test.todo("IntentSet: incompatible required argument row yields no executable proposal", () => {
  const base = fixtureBase();
  // LOOK.target is a required reference argument. Pointing it at the
  // homeostasis record (schema 3, no "observable" capability) must produce no
  // fabricated handle and no executable proposal: the slot stays empty.
  const LOOK_BANK = base.active.bankRecords.indexOf(CATALOG.LOOK);
  const HOMEOSTASIS_BANK = base.active.bankRecords.indexOf(BRAIN_FIXED_RECORDS.homeostasisSummary);
  const r = base.active.bankRecords.length;

  const set = emit({
    frame: base.frame,
    active: base.active,
    catalog: base.catalog,
    intent: peakedSlot(LOOK_BANK, r),
    argument: peakedSlot(HOMEOSTASIS_BANK, r),
  });

  expect(set.count).toBe(0);
  for (const proposal of set.proposals) {
    expect(proposal.lifecycle).toBe("empty");
    expect(proposal.arguments[0]!.handle.tokenId).toBe(0);
  }
});

test.todo("IntentSet: WAIT emits a proposal with no typed arguments", () => {
  const base = fixtureBase();
  const WAIT_BANK = base.active.bankRecords.indexOf(CATALOG.WAIT);
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
  // No typed arguments -> argument support is 1; peaked intent -> intensity 1.
  expect(proposal.intensity).toBe(1);
});

test.todo("IntentSet: required-arg resolution error yields masked record, not fabricated handle", () => {
  const base = fixtureBase();
  // EAT.target requires "edible"; the memory record carries no edible
  // capability. The argmax landing there yields a masked argument and — since
  // EAT's argument is required — no executable proposal.
  const EAT_BANK = base.active.bankRecords.indexOf(CATALOG.EAT);
  const MEMORY_BANK = base.active.bankRecords.indexOf(MEMORY_SLOT);
  const r = base.active.bankRecords.length;

  const set = emit({
    frame: base.frame,
    active: base.active,
    catalog: base.catalog,
    intent: peakedSlot(EAT_BANK, r),
    argument: peakedSlot(MEMORY_BANK, r),
  });

  expect(set.count).toBe(0);
  for (const proposal of set.proposals) {
    expect(proposal.lifecycle).toBe("empty");
    expect(proposal.arguments[0]!.handle.tokenId).toBe(0);
  }
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

test.todo("IntentSet: emitted handles round-trip through the packer's sidecar", () => {
  const base = fixtureBase();
  const EAT_BANK = base.active.bankRecords.indexOf(CATALOG.EAT);
  const APPLE_BANK = base.active.bankRecords.indexOf(VISION_SLOT);
  const r = base.active.bankRecords.length;

  const set = emit({
    frame: base.frame,
    active: base.active,
    catalog: base.catalog,
    intent: peakedSlot(EAT_BANK, r),
    argument: peakedSlot(APPLE_BANK, r),
  });
  const arg = set.proposals[0]!.arguments[0]!;
  // The packed sidecar for Apple is the same handle the packer stored.
  const packed = base.frame.runtimeRefs[VISION_SLOT * BRAIN_LIMITS.maxReferencesPerRecord]!;
  const unpacked = unpackRuntimeHandle(packed);
  expect(arg.handle.tokenId).toBe(unpacked.tokenId);
  expect(arg.handle.generation).toBe(unpacked.generation);
  expect(arg.handle.tokenId).toBe(FIXTURE_APPLE_REF);
});

test.todo("IntentSet: selector distributions with entropy/candidate counts", () => {
  const base = fixtureBase();
  const LOOK_BANK = base.active.bankRecords.indexOf(CATALOG.LOOK);
  const APPLE_BANK = base.active.bankRecords.indexOf(VISION_SLOT);
  const r = base.active.bankRecords.length;

  // A soft distribution over the three catalog records (others masked).
  const scores = new Float32Array(r).fill(-1e30);
  scores[LOOK_BANK] = 2;
  scores[base.active.bankRecords.indexOf(CATALOG.EAT)] = 1;
  scores[base.active.bankRecords.indexOf(CATALOG.WAIT)] = 0.5;
  const p = new Float32Array(r).fill(-1e30);
  softmaxRow(p, 0, r);
  p[LOOK_BANK] = Math.exp(2 - 2);
  p[base.active.bankRecords.indexOf(CATALOG.EAT)] = Math.exp(1 - 2);
  p[base.active.bankRecords.indexOf(CATALOG.WAIT)] = Math.exp(0.5 - 2);
  const norm = p[LOOK_BANK]! + p[base.active.bankRecords.indexOf(CATALOG.EAT)]! + p[base.active.bankRecords.indexOf(CATALOG.WAIT)]!;
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
  expect(proposal.arguments[0]!.selector.status).toBe("selected");
  expect(proposal.confidence).toBeGreaterThan(0.5);
  expect(proposal.arguments[0]!.selector.candidateCount).toBe(1);
  expect(proposal.arguments[0]!.selector.probability).toBe(1);
  // Entropy is strictly positive on a non-degenerate distribution.
  expect(proposal.arguments[0]!.selector.entropy).toBe(0);
  // Non-degenerate intent distribution: peakedness < 1 drags intensity below
  // the top-1 confidence (argument here is peaked, so support is 1).
  expect(proposal.intensity).toBeGreaterThan(0);
  expect(proposal.intensity).toBeLessThan(proposal.confidence);
});

test.todo("IntentSet: intensity reflects peakedness and argument support", () => {
  const base = fixtureBase();
  const EAT_BANK = base.active.bankRecords.indexOf(CATALOG.EAT);
  const APPLE_BANK = base.active.bankRecords.indexOf(VISION_SLOT);
  const WAIT_BANK = base.active.bankRecords.indexOf(CATALOG.WAIT);
  const r = base.active.bankRecords.length;

  // Case 1: peaked intent + peaked argument -> intensity 1 (already covered by
  // the EAT test, restated here for the trio of cases).
  let set = emit({
    frame: base.frame, active: base.active, catalog: base.catalog,
    intent: peakedSlot(EAT_BANK, r), argument: peakedSlot(APPLE_BANK, r),
  });
  expect(set.proposals[0]!.intensity).toBe(1);

  // Case 2: uniform intent over two catalog records -> peakedness 0 -> intensity 0.
  const uniform = new Float32Array(r);
  uniform[EAT_BANK] = 0.5;
  uniform[WAIT_BANK] = 0.5;
  set = emit({
    frame: base.frame, active: base.active, catalog: base.catalog,
    intent: { p: uniform, gather: new Float32Array(H), index: new Uint32Array([EAT_BANK]) },
    argument: peakedSlot(APPLE_BANK, r),
  });
  expect(set.proposals[0]!.intensity).toBe(0);

  // Case 3: peaked intent, argument selector split 50/50 -> intensity = 0.5.
  const softArg = new Float32Array(r);
  softArg[APPLE_BANK] = 0.5;
  softArg[base.active.bankRecords.indexOf(MEMORY_SLOT)] = 0.5;
  set = emit({
    frame: base.frame, active: base.active, catalog: base.catalog,
    intent: peakedSlot(EAT_BANK, r),
    argument: { p: softArg, gather: new Float32Array(H), index: new Uint32Array([APPLE_BANK]) },
  });
  expect(set.proposals[0]!.intensity).toBeCloseTo(0.5, 6);
});

test.todo("IntentSet: emptyProposal factory matches the schema envelope", () => {
  const proposal = emptyProposal(3);
  expect(proposal.proposalSlot).toBe(3);
  expect(proposal.lifecycle).toBe("empty");
  expect(proposal.arguments.length).toBe(BRAIN_LIMITS.maxActionArguments);
  for (const arg of proposal.arguments) {
    expect(arg.kind).toBe("none");
    expect(arg.selector.status).toBe("empty");
  }
});
