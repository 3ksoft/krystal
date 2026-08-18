// M3 close (WEBGPU_BACKWARD_PLAN.md §17 item 10): the composed Krystal
// backward runner. Two tests:
//
//   1. Composed GPU/CPU gradient parity: one trainStep's arena gradients vs
//      the composed CPU backward oracle (brainBackwardOracle), including the
//      argument-slot pointer-loss path.
//   2. Plan §19 overfit: repeated trainStep on the canonical fixture lowers
//      the route-kind cross-entropy loss.
import { expect, test } from "bun:test";
import { getTrainingHarness, readArenaRegion } from "./training-harness.ts";
import {
  KRYSTAL_BACKWARD_ARENA,
  KRYSTAL_BACKWARD_ARENA_BASE,
} from "../../packages/webgpu/src/krystal-layout.ts";
import { KrystalForward, type SelectionMasks } from "../../packages/webgpu/src/krystal-forward.ts";
import { KrystalBackward } from "../../packages/webgpu/src/krystal-backward.ts";
import { ACTION_INTENT_SCHEMA_ID, buildFixtureFrame } from "../../packages/krystal/src/fixtures/frame.ts";
import { packBrainFrame } from "../../packages/krystal/src/frame/packer.ts";
import {
  compileActiveFrame,
  compileArgumentMask,
  compileIntentMask,
  compileMixerMask,
  compileRecordMask,
} from "../../packages/krystal/src/forward/masks.ts";
import {
  BRAIN_FORWARD_CONFIG,
  createBrainForwardWeights,
} from "../../packages/krystal/src/forward/model.ts";
import { brainBackwardOracle } from "../../packages/krystal/src/forward/backward.ts";

function bwdRegion(name: keyof typeof KRYSTAL_BACKWARD_ARENA, elements: number): number {
  return KRYSTAL_BACKWARD_ARENA_BASE + KRYSTAL_BACKWARD_ARENA[name];
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let max = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) max = Math.max(max, Math.abs(a[i]! - b[i]!));
  return max;
}

test("composed trainStep: GPU gradients match the CPU composed backward oracle", async () => {
  const h = await getTrainingHarness();
  const config = BRAIN_FORWARD_CONFIG;
  const weights = createBrainForwardWeights(config, 1337);

  const frame = packBrainFrame(buildFixtureFrame()).frame;
  const active = compileActiveFrame(frame);
  const { mask: recordMask } = compileRecordMask(active.activeTokens);
  const mixerMask = compileMixerMask(frame, active);
  const selection: SelectionMasks = {
    intentMask: compileIntentMask(frame, active, ACTION_INTENT_SCHEMA_ID),
    argMask: compileArgumentMask(frame, active, [2, 1], [3, 8]),
  };
  const q = active.queryRecords.length;
  const gold = new Uint32Array(q).fill(1); // route kind 1
  // Argument pointer-loss target: record 0 (homeostasis) for every query row.
  const argGold = new Uint32Array(q).fill(0);

  const runner = new KrystalForward(weights, config);
  const trainer = new KrystalBackward(runner);
  await trainer.trainStep({
    frame, selection, routeKinds: gold, argumentTargets: [argGold], learningRate: 0.1,
  });
  await h.device.queue.onSubmittedWorkDone();

  const { hiddenSize: hd, routeKindCount: C } = config;
  const t = active.activeTokens.length;
  const r = active.bankRecords.length;

  const got = {
    // The composed runner aliases dQueryValues to the dDecisionQuery region
    // (seeded by the decision head, then accumulated by selector + mixer).
    dFieldStates: await readArenaRegion(h, bwdRegion("dFieldStates", t * hd), t * hd),
    dQueryValues: await readArenaRegion(h, bwdRegion("dDecisionQuery", q * hd), q * hd),
    dBankKeys: await readArenaRegion(h, bwdRegion("dBankKeys", r * hd), r * hd),
    dBankValues: await readArenaRegion(h, bwdRegion("dBankValues", r * hd), r * hd),
    dPool: await readArenaRegion(h, bwdRegion("dPool", 2 * hd), 2 * hd),
    dSelectorWq: await readArenaRegion(h, bwdRegion("dSelectorWq", hd * hd), hd * hd),
    dSelectorWk: await readArenaRegion(h, bwdRegion("dSelectorWk", hd * hd), hd * hd),
    dDecisionWh: await readArenaRegion(h, bwdRegion("dDecisionWh", C * 3 * hd), C * 3 * hd),
    dIntentGather: await readArenaRegion(h, bwdRegion("dDecisionIntent", q * hd), q * hd),
    dArgGather: await readArenaRegion(h, bwdRegion("dDecisionArg", q * hd), q * hd),
  };
  expect(got.dQueryValues.length).toBe(q * hd);

  const want = brainBackwardOracle({
    frame, active, weights, config,
    recordMask, mixerMask,
    intentMask: selection.intentMask, argMask: selection.argMask,
    routeKinds: Array.from(gold), argumentTargets: Array.from(argGold),
  });

  // Deep composed chain: activations use the same 1e-2 envelope as the
  // composed forward parity; gradient magnitudes are O(0.01..1).
  expect(maxAbsDiff(got.dFieldStates, want.dFieldStates)).toBeLessThanOrEqual(1e-2);
  expect(maxAbsDiff(got.dQueryValues, want.dQueryValues)).toBeLessThanOrEqual(1e-2);
  expect(maxAbsDiff(got.dBankKeys, want.dBankKeys)).toBeLessThanOrEqual(1e-2);
  expect(maxAbsDiff(got.dBankValues, want.dBankValues)).toBeLessThanOrEqual(1e-2);
  expect(maxAbsDiff(got.dPool, want.dPool)).toBeLessThanOrEqual(1e-3);
  expect(maxAbsDiff(got.dSelectorWq, want.dSelectorWq)).toBeLessThanOrEqual(1e-2);
  expect(maxAbsDiff(got.dSelectorWk, want.dSelectorWk)).toBeLessThanOrEqual(1e-2);
  expect(maxAbsDiff(got.dDecisionWh, want.dDecisionWh)).toBeLessThanOrEqual(1e-2);
  expect(maxAbsDiff(got.dIntentGather, want.dIntentGather)).toBeLessThanOrEqual(1e-3);
  expect(maxAbsDiff(got.dArgGather, want.dArgGather)).toBeLessThanOrEqual(1e-3);

  // Sanity: the pointer-loss arg target actually moved the arg gather grad
  // off the softmax-only value (nonzero everywhere the CE term is nonzero).
  expect(maxAbsDiff(got.dArgGather, new Float32Array(q * hd))).toBeGreaterThan(1e-6);
  runner.destroy();
});

test("trainStep: synthetic fixture overfits, loss decreases (plan §19)", async () => {
  const config = BRAIN_FORWARD_CONFIG;
  const weights = createBrainForwardWeights(config, 1337);

  const frame = packBrainFrame(buildFixtureFrame()).frame;
  const active = compileActiveFrame(frame);
  const selection: SelectionMasks = {
    intentMask: compileIntentMask(frame, active, ACTION_INTENT_SCHEMA_ID),
    argMask: compileArgumentMask(frame, active, [2, 1], [3, 8]),
  };
  const gold = new Uint32Array(active.queryRecords.length).fill(1); // route kind 1

  const runner = new KrystalForward(weights, config);
  const trainer = new KrystalBackward(runner);
  const losses: number[] = [];
  for (let i = 0; i < 40; i++) {
    const res = await trainer.trainStep({
      frame, selection, routeKinds: gold, learningRate: 0.05, telemetry: true,
    });
    expect(res.loss).toBeDefined();
    expect(Number.isFinite(res.loss)).toBe(true);
    losses.push(res.loss!);
  }

  // The single query sample must be learnable: meaningful overall descent and
  // a low final loss (CE on 4 route kinds starts near ln(4) ~ 1.39).
  expect(losses[39]).toBeLessThan(losses[0]! - 0.2);
  expect(losses[39]).toBeLessThan(0.6);
  expect(losses[39]).toBeLessThan(losses[20]!);
  expect(trainer.currentStep).toBe(40);
  runner.destroy();
});
