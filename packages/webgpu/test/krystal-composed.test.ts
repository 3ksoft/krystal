/**
 * Composed CPU/GPU parity on a frame the host actually builds.
 *
 * The old composed test hung on a fixture world — a compiled catalog, a fixed
 * 432-slot geometry, an intent schema id — and went with it. What replaces it
 * is the same comparison against the input the host has now: records of tokens,
 * one of them a question, and the grammar as a mask. Every mask here is
 * supplied by the caller, which is the property under test as much as the
 * numbers are: the runner must not compile one of its own.
 */
import { expect, test } from "bun:test";
import { getTrainingHarness } from "./harness.ts";
import { TEST_CONFIG, TEST_RECORDS, testFrame } from "./frame.ts";
import { KrystalForward } from "../src/krystal-forward.ts";
import { KrystalBackward } from "../src/krystal-backward.ts";
import { createBrainForwardWeights } from "../../krystal/src/forward/model.ts";
import {
  brainForwardOracle,
  decisionHeadOracle,
  selectorOracle,
} from "../../krystal/src/forward/oracle.ts";
import { brainBackwardOracle } from "../../krystal/src/forward/backward.ts";

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let max = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) max = Math.max(max, Math.abs(a[i]! - b[i]! ));
  return max;
}

const NEG_INF = -1e30;

/** The host's grammar: this question may choose the first two records only. */
function grammar(q: number, r: number): Float32Array {
  const mask = new Float32Array(q * r);
  for (let i = 0; i < q; i++) for (let j = 2; j < r; j++) mask[i * r + j] = NEG_INF;
  return mask;
}

test("composed forward: encoder, mixer and selection match the CPU oracle", async () => {
  const h = await getTrainingHarness();
  const config = TEST_CONFIG;
  const weights = createBrainForwardWeights(config, 7);
  const { frame, active, recordMask } = testFrame();
  const t = active.activeTokens.length;
  const r = active.bankRecords.length;
  const q = active.queryRecords.length;
  const H = config.hiddenSize;
  expect([t, r, q]).toEqual([8, 3, 1]);

  const selection = grammar(q, r);
  // Unconstrained, exactly as the host session runs it: what a question may
  // attend to while it thinks is not what it may choose.
  const mixer = new Float32Array(q * r);

  const cpu = brainForwardOracle(frame, active, weights, config, recordMask, mixer);
  const cpuSel = selectorOracle(cpu.queryOutput, cpu.bankKeys, cpu.bankValues, selection, weights.selector, H);

  const runner = new KrystalForward(weights, config);
  runner.forward(frame, { mixer, selection, context: "available" });
  await h.device.queue.onSubmittedWorkDone();

  expect(maxAbsDiff(await runner.readFieldStates(t, H), cpu.fieldStates)).toBeLessThanOrEqual(1e-3);
  expect(maxAbsDiff(await runner.readBankKeys(r, H), cpu.bankKeys)).toBeLessThanOrEqual(1e-3);
  expect(maxAbsDiff(await runner.readBankValues(r, H), cpu.bankValues)).toBeLessThanOrEqual(1e-3);
  expect(maxAbsDiff(await runner.readQueryOutput(q, H), cpu.queryOutput)).toBeLessThanOrEqual(1e-3);
  expect(maxAbsDiff(await runner.readIntentP(q, r), cpuSel.p)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(await runner.readIntentGather(q, H), cpuSel.gather)).toBeLessThanOrEqual(1e-3);
  expect(Array.from(await runner.readIntentIndices(q))).toEqual(Array.from(cpuSel.index));

  // What the mask forbids carries no probability and is never chosen.
  const p = await runner.readIntentP(q, r);
  expect(p[2]).toBeCloseTo(0, 6);
  for (const index of await runner.readIntentIndices(q)) expect(index).toBeLessThan(2);

  runner.destroy();
});

test("the available context is the mean bank value over what the grammar allows", async () => {
  const h = await getTrainingHarness();
  const config = TEST_CONFIG;
  const weights = createBrainForwardWeights(config, 11);
  const { frame, active, recordMask } = testFrame();
  const r = active.bankRecords.length;
  const q = active.queryRecords.length;
  const H = config.hiddenSize;
  const selection = grammar(q, r);
  const mixer = new Float32Array(q * r);

  const cpu = brainForwardOracle(frame, active, weights, config, recordMask, mixer);
  const runner = new KrystalForward(weights, config);
  runner.forward(frame, { mixer, selection, context: "available" });
  await h.device.queue.onSubmittedWorkDone();

  // Records 0 and 1 are open, record 2 is struck out.
  const want = new Float32Array(q * H);
  for (let d = 0; d < H; d++) want[d] = (cpu.bankValues[d]! + cpu.bankValues[H + d]!) / 2;
  expect(maxAbsDiff(await runner.readAvailableGather(q, H), want)).toBeLessThanOrEqual(1e-3);

  // And the critic reads it: prediction = valueHeadWv . concat(query, intent, offered).
  const cpuSel = selectorOracle(cpu.queryOutput, cpu.bankKeys, cpu.bankValues, selection, weights.selector, H);
  const wantPrediction = decisionHeadOracle(cpu.queryOutput, cpuSel.gather, want, weights.valueHeadWv, q, H, 1);
  expect(maxAbsDiff(await runner.readValuePrediction(q), wantPrediction)).toBeLessThanOrEqual(1e-3);
  runner.destroy();
});

test("composed backward: gradients match the CPU oracle on the host's shape", async () => {
  const h = await getTrainingHarness();
  const config = TEST_CONFIG;
  const weights = createBrainForwardWeights(config, 3);
  const { frame, active, recordMask } = testFrame();
  const t = active.activeTokens.length;
  const r = active.bankRecords.length;
  const q = active.queryRecords.length;
  const H = config.hiddenSize;

  const selection = grammar(q, r);
  const mixer = new Float32Array(q * r);
  const targets = [1]; // the question should have chosen bank record 1
  const valenceTarget = 0.25;

  const cpu = brainBackwardOracle({
    frame, active, weights, config,
    recordMask,
    mixerMask: mixer,
    intentMask: selection,
    argMask: new Float32Array(q * r),
    intentTargets: targets,
    context: "available",
    valenceTarget,
    // No route-kind labels: nothing supplies them, so the decision head stays
    // inert and only the value head trains this context.
  });

  const forward = new KrystalForward(weights, config);
  const backward = new KrystalBackward(forward);
  const result = await backward.trainStep({
    frame,
    masks: { mixer, selection, context: "available" },
    selectionTargets: targets,
    valenceTarget,
    learningRate: 0.01,
    optimizer: "none",
    telemetry: true,
  });
  await h.device.queue.onSubmittedWorkDone();
  const gpu = await backward.readGradients({ t, r, q });

  expect(result.loss).toBeUndefined();
  expect(result.valueLoss).toBeCloseTo(cpu.valueLoss, 4);
  expect(maxAbsDiff(gpu.dValueWv, cpu.dValueWv)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dSelectorWq, cpu.dSelectorWq)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dSelectorWk, cpu.dSelectorWk)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dBankKeys, cpu.dBankKeys)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dBankValues, cpu.dBankValues)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dQueryValues, cpu.dQueryValues)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dPool, cpu.dPool)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dFieldStates, cpu.dFieldStates)).toBeLessThanOrEqual(1e-4);
  // A gradient of exactly nothing would pass every comparison above.
  expect(maxAbsDiff(gpu.dSelectorWq, new Float32Array(H * H))).toBeGreaterThan(1e-6);
  forward.destroy();
});

test("optimizer none leaves every page exactly as it was", async () => {
  const h = await getTrainingHarness();
  const config = TEST_CONFIG;
  const weights = createBrainForwardWeights(config, 5);
  const { frame, active } = testFrame(TEST_RECORDS);
  const r = active.bankRecords.length;
  const q = active.queryRecords.length;
  const H = config.hiddenSize;
  const selection = grammar(q, r);

  const forward = new KrystalForward(weights, config);
  const backward = new KrystalBackward(forward);
  forward.forward(frame, { selection, context: "available" });
  await h.device.queue.onSubmittedWorkDone();
  const before = await forward.readQueryOutput(q, H);
  await backward.trainStep({
    frame, masks: { selection, context: "available" },
    selectionTargets: [1], valenceTarget: 0.5, learningRate: 0.5, optimizer: "none",
  });
  await h.device.queue.onSubmittedWorkDone();
  // Same weights, same frame, same answer: nothing was applied.
  forward.forward(frame, { selection, context: "available" });
  await h.device.queue.onSubmittedWorkDone();
  expect(maxAbsDiff(await forward.readQueryOutput(q, H), before)).toBeLessThanOrEqual(1e-6);
  forward.destroy();
});

test("the second slot and the route-kind head still work when a host asks for them", async () => {
  const h = await getTrainingHarness();
  const config = TEST_CONFIG;
  const weights = createBrainForwardWeights(config, 13);
  const { frame, active, recordMask } = testFrame();
  const t = active.activeTokens.length;
  const r = active.bankRecords.length;
  const q = active.queryRecords.length;

  const selection = grammar(q, r);
  const argument = new Float32Array(q * r);
  for (let i = 0; i < q; i++) argument[i * r + 1] = NEG_INF; // its own grammar
  const mixer = new Float32Array(q * r);
  const routeKinds = [1];
  const argumentTargets = [[0]];

  const cpu = brainBackwardOracle({
    frame, active, weights, config,
    recordMask, mixerMask: mixer,
    intentMask: selection, argMask: argument,
    intentTargets: [1], argumentTargets: argumentTargets[0],
    routeKinds,
    context: "argument",
  });

  const forward = new KrystalForward(weights, config);
  const backward = new KrystalBackward(forward);
  const result = await backward.trainStep({
    frame,
    masks: { mixer, selection, argument, context: "argument" },
    selectionTargets: [1], argumentTargets, routeKinds,
    learningRate: 0.01, optimizer: "none", telemetry: true,
  });
  await h.device.queue.onSubmittedWorkDone();
  const gpu = await backward.readGradients({ t, r, q });

  expect(result.loss).toBeGreaterThan(0); // route-kind cross-entropy, as before
  expect(result.valueLoss).toBeUndefined(); // no valence to difference against
  expect(maxAbsDiff(gpu.dSelectorWq, cpu.dSelectorWq)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dSelectorWk, cpu.dSelectorWk)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dPool, cpu.dPool)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dFieldStates, cpu.dFieldStates)).toBeLessThanOrEqual(1e-4);
  forward.destroy();
});

test("repeated steps move the policy toward what it is shown", async () => {
  const h = await getTrainingHarness();
  const config = TEST_CONFIG;
  const weights = createBrainForwardWeights(config, 17);
  const { frame, active } = testFrame();
  const r = active.bankRecords.length;
  const q = active.queryRecords.length;
  const selection = grammar(q, r);
  const target = 1;

  const forward = new KrystalForward(weights, config);
  const backward = new KrystalBackward(forward);
  const agreement = async (): Promise<number> => {
    forward.forward(frame, { selection, context: "available" });
    await h.device.queue.onSubmittedWorkDone();
    return (await forward.readIntentP(q, r))[target]!;
  };

  const before = await agreement();
  for (let step = 0; step < 40; step++) {
    await backward.trainStep({
      frame, masks: { selection, context: "available" },
      selectionTargets: [target], learningRate: 0.05,
    });
  }
  await h.device.queue.onSubmittedWorkDone();
  const after = await agreement();

  // It starts at chance between the two records the grammar allows.
  expect(before).toBeGreaterThan(0.3);
  expect(before).toBeLessThan(0.7);
  expect(after).toBeGreaterThan(before + 0.2);
  forward.destroy();
});
