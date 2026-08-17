// S2-S10 policy lowering contract tests (docs/S2_S10_CURRICULUM_TASK.md):
// the three-part supervision contract and the curriculum/transition oracles.
// Pure CPU — no GPU needed.
//
//   intentMask        structural legality only (ActionIntent catalog records);
//   argMaskFor(...)   intent-conditional (selectedIntent, argumentIndex);
//   argumentTarget    gold bank record / exact runtime-ref sidecar, INVALID_U32
//                     only for arity-0 or explicitly unlabelled rows.
import { expect, test } from "bun:test";
import {
  BRAIN_LIMITS,
  INVALID_U32,
  RECORD_FLAGS,
} from "../packages/schema/src/krystal-engine-schema.ts";
import { packBrainFrame, unpackRuntimeHandle } from "../packages/krystal/src/frame/packer.ts";
import { buildFixtureActionCatalog } from "../packages/krystal/src/fixtures/action-intents.ts";
import { ACTION_INTENT_SCHEMA_ID } from "../packages/krystal/src/fixtures/frame.ts";
import { emitIntentSet } from "../packages/krystal/src/forward/intentset.ts";
import {
  argMaskFor,
  compileActiveFrame,
  compileIntentMask,
  compilePerRowArgumentMask,
} from "../packages/krystal/src/forward/masks.ts";
import { selectorOracle } from "../packages/krystal/src/forward/oracle.ts";
import {
  generatePolicyEpisode,
  lowerPolicyFrame,
  resourceSchemaId,
  type PolicyEpisode,
} from "../packages/krystal/src/bridge/policy.ts";
import {
  advanceEpisode,
  predictionMatchesGold,
  stageTransitionRule,
} from "../packages/krystal/src/bridge/transition.ts";
import {
  ADVERSARIAL_KINDS,
  buildCurriculum,
  generateAdversarialEpisode,
} from "../packages/krystal/src/bridge/curriculum.ts";
import type { v1_0_0 } from "../packages/schema/generated/krystal.types.ts";

const H = 8;

/** Blocked cells carry -1e30 (f32 rounds it); any value < -1e29 is blocked. */
function isBlocked(value: number): boolean {
  return value < -1e29;
}

function packedEpisode(episode: PolicyEpisode, frameIndex = 0): {
  frame: v1_0_0.BrainFrameGpu;
  active: ReturnType<typeof compileActiveFrame>;
} {
  const frame = packBrainFrame(lowerPolicyFrame(episode.frames[frameIndex]!, episode)).frame;
  const active = compileActiveFrame(frame);
  return { frame, active };
}

function peakedSlot(bankIdx: number, r: number, q = 1): ReturnType<typeof selectorOracle> {
  const p = new Float32Array(q * r);
  p[bankIdx] = 1;
  const gather = new Float32Array(q * H);
  const index = Uint32Array.from({ length: q }, () => bankIdx);
  return { p, gather, index };
}

/** Bank index of the record whose runtime-ref sidecar holds `refToken`. */
function bankIndexOfRef(
  frame: v1_0_0.BrainFrameGpu,
  active: ReturnType<typeof compileActiveFrame>,
  refToken: number,
): number {
  for (let j = 0; j < active.bankRecords.length; j++) {
    const slot = active.bankRecords[j]!;
    const packed = frame.runtimeRefs[slot * BRAIN_LIMITS.maxReferencesPerRecord]!;
    if (packed !== INVALID_U32 && unpackRuntimeHandle(packed).tokenId === refToken) return j;
  }
  throw new Error(`ref 0x${refToken.toString(16)} not found in the frame bank`);
}

test("lowering: intentMask is structural; argMask is intent-conditional; argumentTarget is exact", () => {
  const episode = generatePolicyEpisode("S4", 5); // Apple + Mother + Stone + Feces
  const { frame, active } = packedEpisode(episode);
  const catalog = buildFixtureActionCatalog();

  const intentMask = compileIntentMask(frame, active, ACTION_INTENT_SCHEMA_ID);
  const r = active.bankRecords.length;

  // intentMask: every catalog record open, nothing else (structural only).
  for (let j = 0; j < r; j++) {
    const slot = active.bankRecords[j]!;
    const expectOpen = frame.schemaIds[slot] === ACTION_INTENT_SCHEMA_ID;
    for (let i = 0; i < active.queryRecords.length; i++) {
      const v = intentMask[i * r + j]!;
      if (expectOpen) expect(v).toBe(0);
      else expect(isBlocked(v)).toBe(true);
    }
  }

  // argMaskFor(EAT, 0): only edible schemas (Apple=2, Berry=5, Bread=6) open;
  // Mother/Stone/Feces are blocked even though they are visible records.
  const eatMask = argMaskFor(frame, active, catalog, catalog.descriptors.find((d) => d.actionToken === 0x601)!.intentId, 0);
  for (let j = 0; j < r; j++) {
    const schema = frame.schemaIds[active.bankRecords[j]!]!;
    const open = eatMask[j] === 0;
    if (schema === 2 || schema === 5 || schema === 6) expect(open).toBe(true);
    else expect(open).toBe(false);
  }

  // argumentTarget: exact bank index of the gold ref, resolved through the
  // packed sidecar after the seeded slot shuffle.
  const appleRef = episode.frames[0]!.resources.find((res) => res.kind === "apple")!.refToken;
  const goldBank = bankIndexOfRef(frame, active, appleRef);
  const goldSlot = active.bankRecords[goldBank]!;
  expect(unpackRuntimeHandle(frame.runtimeRefs[goldSlot * BRAIN_LIMITS.maxReferencesPerRecord]!).tokenId).toBe(appleRef);
});

test("mask negatives: EAT cannot point to Mother/Stone/Feces; LOOK can point to an unknown", () => {
  const episode = generatePolicyEpisode("S7", 8); // edible + negative consumable
  const { frame, active } = packedEpisode(episode);
  const catalog = buildFixtureActionCatalog();

  const eat = catalog.descriptors.find((d) => d.actionToken === 0x601)!;
  const look = catalog.descriptors.find((d) => d.actionToken === 0x600)!;
  const eatMask = argMaskFor(frame, active, catalog, eat.intentId, 0);
  const lookMask = argMaskFor(frame, active, catalog, look.intentId, 0);

  for (let j = 0; j < active.bankRecords.length; j++) {
    const slot = active.bankRecords[j]!;
    const schema = frame.schemaIds[slot]!;
    const band = frame.bandIds[slot]!;
    if (schema === resourceSchemaId("mother") || schema === resourceSchemaId("stone") || schema === resourceSchemaId("feces")) {
      expect(isBlocked(eatMask[j]!)).toBe(true); // EAT never points at these
    }
    // LOOK opens observable records in the vision/memory candidate bands.
    if ((schema === resourceSchemaId("apple") || schema === resourceSchemaId("unknown")) && (band === 3 || band === 8)) {
      expect(lookMask[j]).toBe(0);
    }
  }

  // S6 unknown: LOOK's mask opens the UnknownObject record (vision band).
  const s6 = generatePolicyEpisode("S6", 3);
  const s6Packed = packedEpisode(s6);
  const s6LookMask = argMaskFor(s6Packed.frame, s6Packed.active, catalog, look.intentId, 0);
  for (let j = 0; j < s6Packed.active.bankRecords.length; j++) {
    const slot = s6Packed.active.bankRecords[j]!;
    const schema = s6Packed.frame.schemaIds[slot]!;
    if (schema === resourceSchemaId("unknown")) expect(s6LookMask[j]).toBe(0);
  }
});

test("sidecars: EAT(Apple#ref) resolves the exact runtime handle after shuffled packing", () => {
  const episode = generatePolicyEpisode("S4", 12);
  const { frame, active } = packedEpisode(episode);
  const catalog = buildFixtureActionCatalog();
  const appleRef = episode.frames[0]!.resources.find((res) => res.kind === "apple")!.refToken;

  const eat = catalog.descriptors.find((d) => d.actionToken === 0x601)!;
  const eatMask = argMaskFor(frame, active, catalog, eat.intentId, 0);
  const intentBank = active.bankRecords.findIndex((slot) => frame.schemaIds[slot] === ACTION_INTENT_SCHEMA_ID && frame.tokenIds[slot * BRAIN_LIMITS.recordWidth] === 0x601)!;
  const argBank = bankIndexOfRef(frame, active, appleRef);

  const r = active.bankRecords.length;
  const set = emitIntentSet({
    frame, active, catalog,
    intentSchemaId: ACTION_INTENT_SCHEMA_ID,
    intent: peakedSlot(intentBank, r),
    argument: peakedSlot(argBank, r),
    tick: 10,
  }).intentSet;

  expect(set.count).toBe(1);
  const proposal = set.proposals[0]!;
  expect(proposal.intentId).toBe(eat.intentId);
  const arg = proposal.arguments[0]!;
  expect(arg.selector.status).toBe("selected");
  expect(arg.handle.tokenId).toBe(appleRef);
  // The handle came from the packed sidecar, not from a synthesized token.
  const packed = frame.runtimeRefs[active.bankRecords[argBank]! * BRAIN_LIMITS.maxReferencesPerRecord]!;
  expect(arg.handle.tokenId).toBe(unpackRuntimeHandle(packed).tokenId);
});

test("no fabrication: an all-masked/incompatible argument row emits no executable proposal", () => {
  const episode = generatePolicyEpisode("S2", 0); // bad comfort, no apple -> CRY
  const { frame, active } = packedEpisode(episode);
  const catalog = buildFixtureActionCatalog();
  const eat = catalog.descriptors.find((d) => d.actionToken === 0x601)!;
  const r = active.bankRecords.length;

  // EAT's argument mask is all-blocked (no edible record in the frame), so a
  // peaked intent selection toward EAT must yield no executable proposal.
  const eatMask = argMaskFor(frame, active, catalog, eat.intentId, 0);
  expect(eatMask.every((v) => isBlocked(v))).toBe(true);
  const intentBank = active.bankRecords.findIndex((slot) => frame.schemaIds[slot] === ACTION_INTENT_SCHEMA_ID && frame.tokenIds[slot * BRAIN_LIMITS.recordWidth] === 0x601)!;

  const set = emitIntentSet({
    frame, active, catalog,
    intentSchemaId: ACTION_INTENT_SCHEMA_ID,
    intent: peakedSlot(intentBank, r),
    argument: peakedSlot(0, r),
    tick: 10,
  }).intentSet;

  expect(set.count).toBe(0);
  for (const proposal of set.proposals) {
    expect(proposal.lifecycle).toBe("empty");
    expect(proposal.arguments[0]!.handle.tokenId).toBe(0);
  }
});

test("per-row arg mask: compilePerRowArgumentMask conditions each query row on its own intent", () => {
  const episode = generatePolicyEpisode("S4", 3);
  const { frame, active } = packedEpisode(episode);
  const catalog = buildFixtureActionCatalog();
  const eat = catalog.descriptors.find((d) => d.actionToken === 0x601)!;
  const cry = catalog.descriptors.find((d) => d.actionToken === 0x605)!;
  const rowMask = compilePerRowArgumentMask(frame, active, catalog, [eat.intentId], 0);
  const single = argMaskFor(frame, active, catalog, eat.intentId, 0);
  expect(Array.from(rowMask)).toEqual(Array.from(single));
  // CRY (arity-0) produces an all-blocked arg row.
  const cryMask = argMaskFor(frame, active, catalog, cry.intentId, 0);
  expect(cryMask.every((v) => isBlocked(v))).toBe(true);
});

test("counterfactual: counterfactual-free policy frames are deterministic per seed", () => {
  const a = generatePolicyEpisode("S4", 7);
  const b = generatePolicyEpisode("S4", 7);
  expect(a.frames[0]!.resources.map((r) => r.refToken)).toEqual(b.frames[0]!.resources.map((r) => r.refToken));
  // Different seeds give different ref ids (held-out resource identity).
  const c = generatePolicyEpisode("S4", 8);
  expect(c.frames[0]!.resources.map((r) => r.refToken)).not.toEqual(a.frames[0]!.resources.map((r) => r.refToken));
});

test("S6 reveal: LOOK always exposes new evidence while preserving the exact ref", () => {
  const outcomes = new Set<string>();
  for (let seed = 0; seed < 64; seed++) {
    const episode = generatePolicyEpisode("S6", seed);
    const before = episode.frames[0]!.resources[0]!;
    const after = episode.frames[1]!.resources[0]!;
    expect(before.kind).toBe("unknown");
    expect(after.kind).not.toBe("unknown");
    expect(after.refToken).toBe(before.refToken);
    outcomes.add(episode.frames[1]!.gold.action);
  }
  expect(outcomes).toEqual(new Set(["EAT", "CRY"]));
});

test("S8 consequence: known poison is never paired with EAT and variants preserve the ref", () => {
  const outcomes = new Set<string>();
  const foods = new Set<string>();
  for (let seed = 0; seed < 96; seed++) {
    const episode = generatePolicyEpisode("S8", seed);
    const before = episode.frames[0]!;
    const after = episode.frames[1]!;
    expect(before.gold.action).toBe("EAT");
    expect(before.resources[0]!.properties).not.toContain("POISONED");
    expect(after.resources[0]!.refToken).toBe(before.resources[0]!.refToken);
    if (after.resources[0]!.properties.includes("POISONED")) {
      expect(after.gold.action).toBe("CRY");
    } else {
      expect(after.gold.action).toBe("LAUGH");
    }
    outcomes.add(after.gold.action);
    foods.add(before.resources[0]!.kind);
  }
  expect(outcomes).toEqual(new Set(["CRY", "LAUGH"]));
  expect(foods).toEqual(new Set(["apple", "berry", "bread"]));
});

test("S9 memory: Vision disappears while the exact ref and pending decision survive", () => {
  const catalog = buildFixtureActionCatalog();
  for (const seed of [2, 3]) { // FAR/MOVE_TOWARDS and NEAR/EAT
    const episode = generatePolicyEpisode("S9", seed);
    const vision = episode.frames[0]!;
    const memory = episode.frames[1]!;
    expect(vision.resources[0]!.band).toBe("vision");
    expect(memory.resources[0]!.band).toBe("memory");
    expect(memory.resources[0]!.refToken).toBe(vision.resources[0]!.refToken);
    expect(memory.gold).toEqual(vision.gold);

    const packed = packedEpisode(episode, 1);
    const expectedRef = memory.gold.refToken!;
    const refBank = bankIndexOfRef(packed.frame, packed.active, expectedRef);
    const refSlot = packed.active.bankRecords[refBank]!;
    expect(packed.frame.bandIds[refSlot]).toBe(8); // memory band
    expect(packed.frame.recordFlags[refSlot]! & RECORD_FLAGS.remembered).toBe(RECORD_FLAGS.remembered);
    expect(unpackRuntimeHandle(
      packed.frame.runtimeRefs[refSlot * BRAIN_LIMITS.maxReferencesPerRecord]!,
    ).tokenId).toBe(expectedRef);

    const actionToken = memory.gold.action === "EAT" ? 0x601 : 0x607;
    const intent = catalog.descriptors.find((descriptor) => descriptor.actionToken === actionToken)!;
    const mask = argMaskFor(packed.frame, packed.active, catalog, intent.intentId, 0);
    expect(mask[refBank]).toBe(0);
  }
});

test("transition oracle: S3/S5/S6/S8/S9 advance only when the previous intent/ref is correct", () => {
  for (const stage of ["S3", "S5", "S6", "S8", "S9"] as const) {
    const episode = generatePolicyEpisode(stage, 2);
    expect(episode.frames.length).toBeGreaterThan(1);
    const first = episode.frames[0]!;
    const second = episode.frames[1]!;

    // Correct prediction (intent + exact ref) advances.
    const ok = advanceEpisode(episode, 0, {
      action: first.gold.action,
      refToken: first.gold.refToken,
    });
    expect(ok.correct).toBe(true);
    expect(ok.advanced).toBe(true);
    expect(ok.frame).toBe(second);

    // Wrong intent blocks the transition.
    const wrongAction = advanceEpisode(episode, 0, {
      action: first.gold.action === "EAT" ? "CRY" : "EAT",
      refToken: first.gold.refToken,
    });
    expect(wrongAction.correct).toBe(false);
    expect(wrongAction.advanced).toBe(false);
    expect(wrongAction.frame).toBe(first);

    // Correct intent with a wrong ref blocks reference-bearing frames.
    if (first.gold.refToken !== undefined) {
      const wrongRef = advanceEpisode(episode, 0, {
        action: first.gold.action,
        refToken: (first.gold.refToken + 1) & 0xfff,
      });
      expect(wrongRef.correct).toBe(false);
      expect(wrongRef.advanced).toBe(false);
    }

    // Terminal: the last frame advances nowhere.
    const last = episode.frames[episode.frames.length - 1]!;
    const terminal = advanceEpisode(episode, episode.frames.length - 1, {
      action: last.gold.action,
      refToken: last.gold.refToken,
    });
    expect(terminal.correct).toBe(true);
    expect(terminal.terminal).toBe(true);
    expect(terminal.advanced).toBe(false);

    expect(stageTransitionRule(stage)).not.toContain("single-frame");
  }
});

test("transition oracle: single-frame stages never advance", () => {
  for (const stage of ["S1", "S2", "S4", "S7", "S10"] as const) {
    const episode = generatePolicyEpisode(stage, 1);
    const frame = episode.frames[0]!;
    const result = advanceEpisode(episode, 0, { action: frame.gold.action, refToken: frame.gold.refToken });
    expect(result.correct).toBe(true);
    expect(result.terminal).toBe(true);
    expect(stageTransitionRule(stage)).toContain("single-frame");
  }
});

test("curriculum: S2 variants are decoupled from the stage pick and all appear in train and eval", () => {
  // FOLLOW_UP.md §1: the stage pick uses `seed % stages.length` and S2's
  // internal variant used the same `seed % 3`, so the mixture always drew
  // S2 variant 0 (EAT) and never trained "bad + no Apple -> CRY" or
  // "good -> LAUGH". The variant now uses an independently salted hash.
  const split = buildCurriculum({
    stages: ["S2", "S3", "S4"],
    replayStages: ["S1"],
    trainSeeds: [0, 48],
    evalSeeds: [48, 80],
  });
  const count = (episodes: readonly PolicyEpisode[], stage: string, action: string) =>
    episodes.reduce((sum, ep) =>
      ep.stage === stage
        ? sum + ep.frames.filter((f) => f.gold.action === action).length
        : sum,
    0);

  // All three S2 variants occur in training and in held-out evaluation.
  for (const pool of [split.train, split.eval] as const) {
    expect(count(pool, "S2", "EAT")).toBeGreaterThan(0);
    expect(count(pool, "S2", "CRY")).toBeGreaterThan(0);
    expect(count(pool, "S2", "LAUGH")).toBeGreaterThan(0);
  }
  // Intended deterministic counts for these seed ranges.
  expect(count(split.train, "S2", "EAT")).toBe(3);
  expect(count(split.train, "S2", "CRY")).toBe(3);
  expect(count(split.train, "S2", "LAUGH")).toBe(3);
  expect(count(split.eval, "S2", "EAT")).toBe(4);
  expect(count(split.eval, "S2", "CRY")).toBe(4);
  expect(count(split.eval, "S2", "LAUGH")).toBe(3);

  // The 60/30/10 stage mixture is preserved.
  const train = split.train;
  const current = train.filter((e) => ["S2", "S3", "S4"].includes(e.stage)).length;
  const replay = train.filter((e) => e.stage === "S1").length;
  const adversarial = train.length - current - replay;
  expect(current / train.length).toBeGreaterThan(0.55);
  expect(current / train.length).toBeLessThan(0.75);
  expect(replay / train.length).toBeGreaterThan(0.15);
  expect(replay / train.length).toBeLessThan(0.35);
  expect(adversarial / train.length).toBeGreaterThan(0.02);
  expect(adversarial / train.length).toBeLessThan(0.15);

  // Train/eval resource ids and physical layouts remain disjoint (disjoint
  // seeds => disjoint refs and slot shuffles; the refs are derived from the
  // seed and the slot shuffle is seed-seeded).
  const trainRefs = new Set(
    split.train.flatMap((ep) => ep.frames.flatMap((f) => f.resources.map((r) => r.refToken))),
  );
  for (const ep of split.eval) {
    for (const f of ep.frames) {
      for (const r of f.resources) expect(trainRefs.has(r.refToken)).toBe(false);
    }
  }
});

test("curriculum: 60/30/10 mixture with logged seeds and disjoint train/eval", () => {
  const split = buildCurriculum({
    stages: ["S2", "S3", "S4"],
    replayStages: ["S1"],
    trainSeeds: [0, 32],
    evalSeeds: [32, 40],
  });
  expect(split.train.length).toBe(32);
  expect(split.eval.length).toBe(8);

  // Mixture coverage: adversarial kinds must appear across the train set.
  const adversarial = split.train.filter((ep) => "adversarialKind" in ep);
  expect(adversarial.length).toBeGreaterThan(0);

  // Eval uses only held-out seeds (>= 32), train only < 32.
  for (const ep of split.train) expect(ep.seed).toBeLessThan(32);
  for (const ep of split.eval) expect(ep.seed).toBeGreaterThanOrEqual(32);

  // Deterministic: same options -> same log + same episodes.
  const again = buildCurriculum({
    stages: ["S2", "S3", "S4"],
    replayStages: ["S1"],
    trainSeeds: [0, 32],
    evalSeeds: [32, 40],
  });
  expect(again.log).toEqual(split.log);
  expect(again.train.map((e) => `${e.stage}:${e.seed}`)).toEqual(split.train.map((e) => `${e.stage}:${e.seed}`));
  expect(split.log.join("\n")).toContain("mixture current=");
  expect(split.log.join("\n")).toContain("rootSeed");
});

test("curriculum: every adversarial kind is constructible", () => {
  const catalog = buildFixtureActionCatalog();
  for (const kind of ADVERSARIAL_KINDS) {
    const episode = generateAdversarialEpisode(kind, 4);
    expect(episode.frames.length).toBe(1);
    expect(episode.frames[0]!.gold.action).toBeDefined();
    // Deterministic per seed.
    const again = generateAdversarialEpisode(kind, 4);
    expect(again.frames[0]!.gold.action).toBe(episode.frames[0]!.gold.action);
    expect(again.frames[0]!.resources.map((r) => r.refToken)).toEqual(episode.frames[0]!.resources.map((r) => r.refToken));
  }
  expect(catalog.header.intentCount).toBeGreaterThanOrEqual(6);
});

test("lowering: held-out seeds produce unseen record layouts and ids", () => {
  const train = generatePolicyEpisode("S4", 3);
  const evalEp = generatePolicyEpisode("S4", 35);
  const trainRefs = train.frames[0]!.resources.map((r) => r.refToken);
  const evalRefs = evalEp.frames[0]!.resources.map((r) => r.refToken);
  expect(trainRefs.some((ref) => evalRefs.includes(ref))).toBe(false);
  // Slot order differs (shuffle is seed-dependent).
  const a = packedEpisode(train);
  const b = packedEpisode(evalEp);
  const appleSlotA = a.active.bankRecords.findIndex((slot) => a.frame.schemaIds[slot] === resourceSchemaId("apple"));
  const appleSlotB = b.active.bankRecords.findIndex((slot) => b.frame.schemaIds[slot] === resourceSchemaId("apple"));
  expect(appleSlotA).not.toBe(appleSlotB);
});
