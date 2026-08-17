// S2-S10 CPU/GPU parity (docs/S2_S10_CURRICULUM_TASK.md, test requirements):
// the new intent-conditional argument-mask path, pointer-loss with no-argument
// rows, and composed backward gradient parity — on a real policy frame with a
// visible Apple.
//
//   forward: GPU selector P/gather/argmax vs CPU oracle under
//            argMaskFor(selectedIntent, 0) (the capability-aware mask);
//   backward: GPU gradients vs brainBackwardOracle with argumentTargets[0]
//             carrying INVALID_U32 rows for arity-0 intents.
import { expect, test } from "bun:test";
import { INVALID_U32 } from "../packages/schema/src/krystal-engine-schema.ts";
import { getTrainingHarness, readArenaRegion } from "./training-harness.ts";
import {
  KRYSTAL_BACKWARD_ARENA,
  KRYSTAL_BACKWARD_ARENA_BASE,
} from "../packages/webgpu/src/krystal-layout.ts";
import { KrystalForward, type SelectionMasks } from "../packages/webgpu/src/krystal-forward.ts";
import { KrystalBackward } from "../packages/webgpu/src/krystal-backward.ts";
import { packBrainFrame } from "../packages/krystal/src/frame/packer.ts";
import { buildFixtureActionCatalog } from "../packages/krystal/src/fixtures/action-intents.ts";
import { ACTION_INTENT_SCHEMA_ID } from "../packages/krystal/src/fixtures/frame.ts";
import {
  argMaskFor,
  compileActiveFrame,
  compileIntentMask,
  compileMixerMask,
  compileRecordMask,
} from "../packages/krystal/src/forward/masks.ts";
import {
  BRAIN_FORWARD_CONFIG,
  createBrainForwardWeights,
} from "../packages/krystal/src/forward/model.ts";
import { brainForwardOracle, brainSelectionOracle, decisionHeadOracle } from "../packages/krystal/src/forward/oracle.ts";
import { brainBackwardOracle } from "../packages/krystal/src/forward/backward.ts";
import { generatePolicyEpisode, lowerPolicyFrame } from "../packages/krystal/src/bridge/policy.ts";

const ROUTE_EAT = 2; // policy route kinds: CRY=0 LAUGH=1 EAT=2

const POLICY_CONFIG = {
  ...BRAIN_FORWARD_CONFIG,
  routeKindCount: 6, // CRY/LAUGH/EAT/MOVE_TOWARDS/LOOK/WAIT
};

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let max = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) max = Math.max(max, Math.abs(a[i]! - b[i]!));
  return max;
}

function bwdRegion(name: keyof typeof KRYSTAL_BACKWARD_ARENA, elements: number): number {
  return KRYSTAL_BACKWARD_ARENA_BASE + KRYSTAL_BACKWARD_ARENA[name];
}

test("policy forward: intent-conditional arg mask matches the CPU oracle on a visible-Apple frame", async () => {
  const h = await getTrainingHarness();
  const episode = generatePolicyEpisode("S2", 6); // bad + Apple -> EAT (seed % 3 == 0)
  const frame = packBrainFrame(lowerPolicyFrame(episode.frames[0]!, episode)).frame;
  const active = compileActiveFrame(frame);
  const catalog = buildFixtureActionCatalog();
  const { mask: recordMask } = compileRecordMask(active.activeTokens);
  const mixerMask = compileMixerMask(frame, active);
  const intentMask = compileIntentMask(frame, active, ACTION_INTENT_SCHEMA_ID);
  // Selected intent = EAT (intentId 1 in catalog order); the arg mask is the
  // capability-aware edible mask, not a shared identity list.
  const eat = catalog.descriptors.find((d) => d.actionToken === 0x601)!;
  const argMask = argMaskFor(frame, active, catalog, eat.intentId, 0);
  const selection: SelectionMasks = { intentMask, argMask };

  const weights = createBrainForwardWeights(POLICY_CONFIG, 1337);
  const runner = new KrystalForward(weights, POLICY_CONFIG);
  runner.forward(frame, selection);
  await h.device.queue.onSubmittedWorkDone();

  const { hiddenSize: hDim } = POLICY_CONFIG;
  const r = active.bankRecords.length;
  const q = active.queryRecords.length;
  const gpu = await runner.readSelection(q, r, hDim);

  const cpu = brainForwardOracle(frame, active, weights, POLICY_CONFIG, recordMask, mixerMask);
  const cpuSel = brainSelectionOracle(
    cpu.queryOutput, cpu.bankKeys, cpu.bankValues,
    selection.intentMask, selection.argMask, weights.selector, hDim,
  );
  const cpuLogits = decisionHeadOracle(
    cpu.queryOutput, cpuSel.intent.gather, cpuSel.argument.gather,
    weights.decisionHeadWh, q, hDim, POLICY_CONFIG.routeKindCount,
  );
  const gpuLogits = await runner.readDecisionLogits(q, POLICY_CONFIG.routeKindCount);

  expect(maxAbsDiff(gpu.intent.p, cpuSel.intent.p)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.argument.p, cpuSel.argument.p)).toBeLessThanOrEqual(1e-4);
  expect(Array.from(gpu.intent.index)).toEqual(Array.from(cpuSel.intent.index));
  expect(Array.from(gpu.argument.index)).toEqual(Array.from(cpuSel.argument.index));
  expect(maxAbsDiff(gpuLogits, cpuLogits)).toBeLessThanOrEqual(1e-2);

  // The conditional arg mask only leaves edible records open, so the argument
  // selector can never land on the Mother/noise distractors.
  for (const idx of gpu.argument.index) {
    const schema = frame.schemaIds[active.bankRecords[idx]!]!;
    expect([2, 5, 6]).toContain(schema);
  }
  runner.destroy();
});

test("policy backward: argument pointer loss with no-argument rows matches the CPU oracle", async () => {
  const h = await getTrainingHarness();
  const episode = generatePolicyEpisode("S2", 6);
  const frame = packBrainFrame(lowerPolicyFrame(episode.frames[0]!, episode)).frame;
  const active = compileActiveFrame(frame);
  const catalog = buildFixtureActionCatalog();
  const { mask: recordMask } = compileRecordMask(active.activeTokens);
  const mixerMask = compileMixerMask(frame, active);
  const intentMask = compileIntentMask(frame, active, ACTION_INTENT_SCHEMA_ID);
  const eat = catalog.descriptors.find((d) => d.actionToken === 0x601)!;
  const selection: SelectionMasks = {
    intentMask,
    argMask: argMaskFor(frame, active, catalog, eat.intentId, 0),
  };
  const q = active.queryRecords.length;

  // Gold: EAT route; the argument target is the Apple record's bank index
  // (resolved through the packed sidecar). No-argument rows would be INVALID.
  const appleRef = episode.frames[0]!.resources.find((res) => res.kind === "apple")!.refToken;
  let argGold = INVALID_U32;
  for (let j = 0; j < active.bankRecords.length; j++) {
    const slot = active.bankRecords[j]!;
    const packed = frame.runtimeRefs[slot * 8]!;
    if ((packed & 0xfff) === appleRef) argGold = j;
  }
  expect(argGold).not.toBe(INVALID_U32);
  const routeKinds = new Uint32Array(q).fill(ROUTE_EAT);
  const intentGold = new Uint32Array(q).fill(
    active.bankRecords.findIndex((slot) => frame.schemaIds[slot] === ACTION_INTENT_SCHEMA_ID && frame.tokenIds[slot * 8] === 0x601),
  );

  const weights = createBrainForwardWeights(POLICY_CONFIG, 1337);
  const runner = new KrystalForward(weights, POLICY_CONFIG);
  const trainer = new KrystalBackward(runner);
  await trainer.trainStep({
    frame, selection, routeKinds,
    argumentTargets: [new Uint32Array(q).fill(argGold)],
    intentGold,
    learningRate: 0.1,
  });
  await h.device.queue.onSubmittedWorkDone();

  const { hiddenSize: hd, routeKindCount: C } = POLICY_CONFIG;
  const t = active.activeTokens.length;
  const r = active.bankRecords.length;
  const got = {
    dFieldStates: await readArenaRegion(h, bwdRegion("dFieldStates", t * hd), t * hd),
    dQueryValues: await readArenaRegion(h, bwdRegion("dDecisionQuery", q * hd), q * hd),
    dBankKeys: await readArenaRegion(h, bwdRegion("dBankKeys", r * hd), r * hd),
    dBankValues: await readArenaRegion(h, bwdRegion("dBankValues", r * hd), r * hd),
    dPool: await readArenaRegion(h, bwdRegion("dPool", 2 * hd), 2 * hd),
    dSelectorWq: await readArenaRegion(h, bwdRegion("dSelectorWq", hd * hd), hd * hd),
    dSelectorWk: await readArenaRegion(h, bwdRegion("dSelectorWk", hd * hd), hd * hd),
    dDecisionWh: await readArenaRegion(h, bwdRegion("dDecisionWh", C * 3 * hd), C * 3 * hd),
    dArgGather: await readArenaRegion(h, bwdRegion("dDecisionArg", q * hd), q * hd),
  };

  const want = brainBackwardOracle({
    frame, active, weights, config: POLICY_CONFIG,
    recordMask, mixerMask,
    intentMask: selection.intentMask, argMask: selection.argMask,
    routeKinds: Array.from(routeKinds),
    argumentTargets: Array.from(new Uint32Array(q).fill(argGold)),
    intentTargets: Array.from(intentGold),
  });

  expect(maxAbsDiff(got.dFieldStates, want.dFieldStates)).toBeLessThanOrEqual(1e-2);
  expect(maxAbsDiff(got.dQueryValues, want.dQueryValues)).toBeLessThanOrEqual(1e-2);
  expect(maxAbsDiff(got.dBankKeys, want.dBankKeys)).toBeLessThanOrEqual(1e-2);
  expect(maxAbsDiff(got.dBankValues, want.dBankValues)).toBeLessThanOrEqual(1e-2);
  expect(maxAbsDiff(got.dPool, want.dPool)).toBeLessThanOrEqual(1e-3);
  expect(maxAbsDiff(got.dSelectorWq, want.dSelectorWq)).toBeLessThanOrEqual(1e-2);
  expect(maxAbsDiff(got.dSelectorWk, want.dSelectorWk)).toBeLessThanOrEqual(1e-2);
  expect(maxAbsDiff(got.dDecisionWh, want.dDecisionWh)).toBeLessThanOrEqual(1e-2);
  expect(maxAbsDiff(got.dArgGather, want.dArgGather)).toBeLessThanOrEqual(1e-3);
  runner.destroy();
});

test("policy backward: arity-0 rows contribute no argument pointer loss (INVALID target)", async () => {
  const h = await getTrainingHarness();
  const episode = generatePolicyEpisode("S1", 4); // bad -> CRY (arity-0)
  const frame = packBrainFrame(lowerPolicyFrame(episode.frames[0]!, episode)).frame;
  const active = compileActiveFrame(frame);
  const catalog = buildFixtureActionCatalog();
  const { mask: recordMask } = compileRecordMask(active.activeTokens);
  const mixerMask = compileMixerMask(frame, active);
  const intentMask = compileIntentMask(frame, active, ACTION_INTENT_SCHEMA_ID);
  // CRY has no argument: its arg mask is all-blocked and the target INVALID.
  const cry = catalog.descriptors.find((d) => d.actionToken === 0x605)!;
  const selection: SelectionMasks = {
    intentMask,
    argMask: argMaskFor(frame, active, catalog, cry.intentId, 0),
  };
  const q = active.queryRecords.length;
  const routeKinds = new Uint32Array(q).fill(0); // CRY
  const intentGold = new Uint32Array(q).fill(
    active.bankRecords.findIndex((slot) => frame.schemaIds[slot] === ACTION_INTENT_SCHEMA_ID && frame.tokenIds[slot * 8] === 0x605),
  );

  const weights = createBrainForwardWeights(POLICY_CONFIG, 1337);
  const runner = new KrystalForward(weights, POLICY_CONFIG);
  const trainer = new KrystalBackward(runner);
  await trainer.trainStep({
    frame, selection, routeKinds,
    argumentTargets: [new Uint32Array(q).fill(INVALID_U32)],
    intentGold,
    learningRate: 0.1,
  });
  await h.device.queue.onSubmittedWorkDone();

  // With an INVALID target the argument selector contributes only the
  // softmax-side gradient (never a pointer term); parity with the oracle's
  // no-target default proves the row contributes no pointer loss.
  const { hiddenSize: hd } = POLICY_CONFIG;
  const qh = q * hd;
  const got = await readArenaRegion(h, bwdRegion("dDecisionArg", qh), qh);

  const want = brainBackwardOracle({
    frame, active, weights, config: POLICY_CONFIG,
    recordMask, mixerMask,
    intentMask: selection.intentMask, argMask: selection.argMask,
    routeKinds: Array.from(routeKinds),
    // No argumentTargets at all -> every row is INVALID (no pointer loss).
    // The intent slot still gets its catalog pointer loss (intentGold).
    intentTargets: Array.from(intentGold),
  });
  expect(maxAbsDiff(got, want.dArgGather)).toBeLessThanOrEqual(1e-3);
  runner.destroy();
});
