/**
 * The GPU session against the CPU one, on the surface a simulation calls.
 *
 * Not an operator comparison: `consider(...).choose(...)`, the same records,
 * the same grammar closure, the same seed — and the same answer, or the swap is
 * not a swap. And then `learn` and `teach`, the same batch on both, and the
 * same brain afterwards: the device differentiates, the host applies, and
 * nothing about the update may depend on which of the two did the arithmetic.
 */
import { expect, test } from "bun:test";
import { getTrainingHarness } from "./harness.ts";
import { TEST_TOKEN_ROWS } from "./frame.ts";
import { gpuBackend } from "../src/backend.ts";
import { BrainSession, type HostExperience } from "../../krystal/src/host/index.ts";
import type { HostRecord } from "../../krystal/src/host/frame.ts";

const world = (): HostRecord[] => [
  { schemaId: 1, band: 3, tokens: [10, 11, 12] },
  { schemaId: 1, band: 3, tokens: [10, 13] },
  { schemaId: 2, band: 2, tokens: [20, 21] },
  { schemaId: 9, query: true, tokens: [30] },
];

/** A life where the first thing goes well and the others do not. */
const lived = (): HostExperience[] => [
  { records: world(), chosen: [0], reward: 0.4 },
  { records: world(), chosen: [1], reward: -0.4 },
  { records: world(), chosen: [0], reward: 0.3 },
  { records: world(), chosen: [2], reward: -0.3 },
];

/** The two backends, built exactly as a host builds them: one class, one
 *  option. */
const sessions = (device: GPUDevice) => ({
  cpu: new BrainSession({ tokenRows: TEST_TOKEN_ROWS, seed: 7 }),
  gpu: new BrainSession({ tokenRows: TEST_TOKEN_ROWS, seed: 7, backend: gpuBackend(device) }),
});

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let max = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) max = Math.max(max, Math.abs(a[i]! - b[i]!));
  return max;
}

test("the same records get the same choice on both", async () => {
  const { device } = await getTrainingHarness();
  const { cpu, gpu } = sessions(device);
  const want = await cpu.think(world());
  const got = await gpu.think(world());
  expect(got.selections).toHaveLength(1);
  expect(got.selections[0]!.record).toBe(want.selections[0]!.record);
  for (let i = 0; i < want.selections[0]!.distribution.length; i++) {
    expect(got.selections[0]!.distribution[i]!).toBeCloseTo(want.selections[0]!.distribution[i]!, 3);
  }
  gpu.destroy();
});

test("the host's grammar is obeyed the same way", async () => {
  const { device } = await getTrainingHarness();
  const { cpu, gpu } = sessions(device);
  const allows = (_query: number, record: number) => record === 2;
  const want = await cpu.think(world(), { allows });
  const got = await gpu.think(world(), { allows });
  expect(got.selections[0]!.record).toBe(2);
  expect(got.selections[0]!.record).toBe(want.selections[0]!.record);
  expect(got.selections[0]!.distribution[0]!).toBeCloseTo(0, 5);
  gpu.destroy();
});

test("one encode, many questions — and each question sees its own grammar", async () => {
  const { device } = await getTrainingHarness();
  const { cpu, gpu } = sessions(device);
  const deliberation = await gpu.consider(world());
  const cpuDeliberation = await cpu.consider(world());
  for (const allowed of [0, 1, 2]) {
    const allows = (_q: number, record: number) => record === allowed;
    expect(deliberation.choose({ allows }).selections[0]!.record).toBe(allowed);
    expect(deliberation.choose({ allows }).selections[0]!.probability)
      .toBeCloseTo(cpuDeliberation.choose({ allows }).selections[0]!.probability, 4);
  }
  gpu.destroy();
});

test("being taught moves the device's brain too, not just the host's", async () => {
  const { device } = await getTrainingHarness();
  const { gpu } = sessions(device);
  const records = world();
  const before = (await gpu.think(records)).selections[0]!;
  const target = before.record === 0 ? 1 : 0;
  await gpu.teach([{ records, gold: [target] }], { learningRate: 0.3 });
  const after = (await gpu.think(records)).selections[0]!;
  // The device pages were stale before uploadWeights existed: `after` came back
  // identical to `before`, and nothing anywhere said so.
  expect(after.record).toBe(target);
  expect(after.distribution[target]!).toBeGreaterThan(before.distribution[target]!);
  gpu.destroy();
});

test("being shown on the device climbs exactly as being shown here", async () => {
  const { device } = await getTrainingHarness();
  const { cpu, gpu } = sessions(device);
  const shown = [{ records: world(), gold: [1] }];
  let first: number | undefined;
  let last = 0;
  for (let round = 0; round < 12; round++) {
    const want = await cpu.teach(shown, { learningRate: 0.2 });
    const got = await gpu.teach(shown, { learningRate: 0.2 });
    // Agreement is measured BEFORE the pass changes anything, so on every round
    // it is the reading of the brain the previous round left: the device has
    // taken up what the host wrote, or these drift apart at once.
    expect(got.meanAgreement).toBeCloseTo(want.meanAgreement, 3);
    first ??= got.meanAgreement;
    last = got.meanAgreement;
  }
  expect(last).toBeGreaterThan(first! + 0.2);
  expect(maxAbsDiff(gpu.weights.selector.wq, cpu.weights.selector.wq)).toBeLessThanOrEqual(1e-3);
  expect(maxAbsDiff(gpu.weights.selector.wk, cpu.weights.selector.wk)).toBeLessThanOrEqual(1e-3);
  expect(maxAbsDiff(gpu.weights.embeddings, cpu.weights.embeddings)).toBeLessThanOrEqual(1e-3);
  gpu.destroy();
});

test("being shown what is NOT done, on the device, is the same update as here", async () => {
  const { device } = await getTrainingHarness();
  const { cpu, gpu } = sessions(device);
  const not = [{ records: world(), gold: [undefined], forbidden: [1] }];
  const before = (await gpu.think(world())).selections[0]!.distribution[1]!;
  let first: number | undefined;
  let last = 0;
  for (let round = 0; round < 12; round++) {
    const want = await cpu.teach(not, { learningRate: 0.2 });
    const got = await gpu.teach(not, { learningRate: 0.2 });
    expect(got.meanAgreement).toBeCloseTo(want.meanAgreement, 3);
    first ??= got.meanAgreement;
    last = got.meanAgreement;
  }
  // Agreement with a "no" is how far off it the policy already is: it climbs
  // too, and what it climbs away from is the record it was told not to choose.
  expect(last).toBeGreaterThan(first! + 0.05);
  expect((await gpu.think(world())).selections[0]!.distribution[1]!).toBeLessThan(before);
  expect(maxAbsDiff(gpu.weights.selector.wq, cpu.weights.selector.wq)).toBeLessThanOrEqual(1e-3);
  expect(maxAbsDiff(gpu.weights.selector.wk, cpu.weights.selector.wk)).toBeLessThanOrEqual(1e-3);
  expect(maxAbsDiff(gpu.weights.embeddings, cpu.weights.embeddings)).toBeLessThanOrEqual(1e-3);
  gpu.destroy();
});

test("living with what happened on the device is the same update as here", async () => {
  const { device } = await getTrainingHarness();
  const { cpu, gpu } = sessions(device);
  const options = { learningRate: 0.05, unfreeze: { tokens: true, tokenRate: 0.1 } };
  for (let round = 0; round < 8; round++) {
    const want = await cpu.learn(lived(), options);
    const got = await gpu.learn(lived(), options);
    expect(got.framesSeen).toBe(want.framesSeen);
    expect(got.reinforced).toBe(want.reinforced);
    expect(got.discouraged).toBe(want.discouraged);
    expect(got.updateApplied).toBe(true);
    expect(got.meanEntropy).toBeCloseTo(want.meanEntropy, 3);
    expect(got.meanConfidence).toBeCloseTo(want.meanConfidence, 3);
    expect(got.meanValueLoss).toBeCloseTo(want.meanValueLoss, 3);
    expect(got.meanAdvantage).toBeCloseTo(want.meanAdvantage, 3);
  }
  expect(maxAbsDiff(gpu.weights.selector.wq, cpu.weights.selector.wq)).toBeLessThanOrEqual(1e-3);
  expect(maxAbsDiff(gpu.weights.selector.wk, cpu.weights.selector.wk)).toBeLessThanOrEqual(1e-3);
  expect(maxAbsDiff(gpu.weights.valueHeadWv, cpu.weights.valueHeadWv)).toBeLessThanOrEqual(1e-3);
  expect(maxAbsDiff(gpu.weights.embeddings, cpu.weights.embeddings)).toBeLessThanOrEqual(1e-3);
  // And it moved. Two brains that agree because neither changed would pass
  // every line above.
  const fresh = new BrainSession({ tokenRows: TEST_TOKEN_ROWS, seed: 7 });
  expect(maxAbsDiff(gpu.weights.selector.wq, fresh.weights.selector.wq)).toBeGreaterThan(1e-4);
  // The device's copy answers as the host's arrays do: the pages it was told
  // about are the pages that moved.
  const want = await cpu.think(world());
  const got = await gpu.think(world());
  for (let i = 0; i < want.selections[0]!.distribution.length; i++) {
    expect(got.selections[0]!.distribution[i]!).toBeCloseTo(want.selections[0]!.distribution[i]!, 3);
  }
  gpu.destroy();
});

test("a rejected update leaves the device's brain where it was", async () => {
  const { device } = await getTrainingHarness();
  const { gpu } = sessions(device);
  const before = (await gpu.think(world())).selections[0]!.distribution;
  const report = await gpu.learn(lived(), { maxParameterAbs: 0.01, unfreeze: { tokens: true } });
  expect(report.updateApplied).toBe(false);
  expect(report.rejected).toBe("parameter-limit");
  const after = (await gpu.think(world())).selections[0]!.distribution;
  for (let i = 0; i < before.length; i++) expect(after[i]!).toBeCloseTo(before[i]!, 6);
  gpu.destroy();
});

test("a frame may ask many questions at once, and each gets its own answer", async () => {
  const { device } = await getTrainingHarness();
  const { cpu, gpu } = sessions(device);
  // Eleven questions is what the simulation actually sent when this first
  // failed; the device refused the frame and the CPU had never had a ceiling.
  const many: HostRecord[] = [
    { schemaId: 1, band: 3, tokens: [10, 11, 12] },
    { schemaId: 1, band: 3, tokens: [10, 13] },
    { schemaId: 2, band: 2, tokens: [20, 21] },
  ];
  for (let q = 0; q < 11; q++) many.push({ schemaId: 9, query: true, tokens: [30 + q] });

  const want = await cpu.think(many);
  const got = await gpu.think(many);
  expect(got.selections).toHaveLength(11);
  for (let q = 0; q < 11; q++) {
    expect(got.selections[q]!.record).toBe(want.selections[q]!.record);
    for (let r = 0; r < 3; r++) {
      expect(got.selections[q]!.distribution[r]!).toBeCloseTo(want.selections[q]!.distribution[r]!, 3);
    }
  }
  gpu.destroy();
});

test("a frame past the device's ceiling is refused by name, not by corruption", async () => {
  const { device } = await getTrainingHarness();
  const { gpu } = sessions(device);
  const tooMany: HostRecord[] = [{ schemaId: 1, band: 3, tokens: [10, 11] }];
  for (let q = 0; q < 129; q++) tooMany.push({ schemaId: 9, query: true, tokens: [30] });
  expect(gpu.think(tooMany)).rejects.toThrow(/query records 129 exceed capacity 128/);
  gpu.destroy();
});
