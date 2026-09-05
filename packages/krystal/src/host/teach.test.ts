import { describe, expect, test } from "bun:test";
import { BrainSession, type HostDemonstration, type HostRecord } from "./index.ts";
import { BRAIN_FORWARD_CONFIG } from "../forward/model.ts";

const rows = (count: number): Uint32Array => Uint32Array.from({ length: count }, (_, id) => id % 64);
const small = {
  ...BRAIN_FORWARD_CONFIG,
  hiddenSize: 32,
  headCount: 1,
  headDim: 32,
  ffnSize: 64,
  encoderBlocks: 1,
  mixerBlocks: 1,
};

/** Two things and one question about them. */
const frame = (): HostRecord[] => [
  { schemaId: 1, band: 3, tokens: [10, 11] }, // an apple
  { schemaId: 1, band: 3, tokens: [20, 21] }, // a stone
  { schemaId: 9, query: true, tokens: [40, 41] }, // eat · food
];

const session = () => new BrainSession({ tokenRows: rows(4096), seed: 5, config: small });

describe("being shown what can be said", () => {
  test("a handful of showings is enough, where finding out takes hundreds", async () => {
    const brain = session();
    const before = (await brain.think(frame())).selections[0]!.distribution[0]!;
    const shown: HostDemonstration[] = [{ records: frame(), gold: [0] }];
    let report = await brain.teach(shown);
    // What it already believed, before this pass changed anything: a creature
    // that has been shown nothing is guessing between two.
    expect(report.meanAgreement).toBeCloseTo(0.5, 1);
    for (let round = 0; round < 40; round++) report = await brain.teach(shown);
    const after = (await brain.think(frame())).selections[0]!.distribution[0]!;
    expect(after).toBeGreaterThan(before);
    expect(after).toBeGreaterThan(0.95);
  });

  test("agreement is what teaching is measured by, and it climbs", async () => {
    const brain = session();
    const shown: HostDemonstration[] = [{ records: frame(), gold: [0] }];
    const first = (await brain.teach(shown)).meanAgreement;
    for (let round = 0; round < 15; round++) await brain.teach(shown);
    expect((await brain.teach(shown)).meanAgreement).toBeGreaterThan(first);
  });

  test("it teaches the grammar, and says nothing about how things went", async () => {
    const brain = session();
    const value = Float32Array.from(brain.weights.valueHeadWv);
    for (let round = 0; round < 10; round++) await brain.teach([{ records: frame(), gold: [0] }]);
    // A demonstration is not an outcome. The critic has nothing to learn from
    // one, and a critic that moved here would be predicting that being taught
    // feels good.
    expect(brain.weights.valueHeadWv).toEqual(value);
  });

  test("a demonstration is shown under the grammar a choice would be made under", async () => {
    const brain = session();
    // The stone is the only thing this question admits, and the demonstration
    // agrees with it — so agreement starts at certainty, not at a half.
    const report = await brain.teach([{ records: frame(), gold: [1], allows: (_query, record) => record === 1 }]);
    expect(report.meanAgreement).toBeCloseTo(1, 5);
  });

  test("being shown what is NOT done here pushes it down, and agreement is how far off it already was", async () => {
    const brain = session();
    const before = (await brain.think(frame())).selections[0]!.distribution;
    const not: HostDemonstration[] = [{ records: frame(), gold: [undefined], forbidden: [1] }];
    const report = await brain.teach(not);
    expect(report.shown).toBe(1);
    expect(report.meanAgreement).toBeCloseTo(1 - before[1]!, 4);
    for (let round = 0; round < 40; round++) await brain.teach(not);
    const after = (await brain.think(frame())).selections[0]!.distribution;
    expect(after[1]!).toBeLessThan(before[1]!);
    expect(after[1]!).toBeLessThan(0.05);
    // A "no" names nothing to do instead; here the only other thing is the apple.
    expect(after[0]!).toBeGreaterThan(0.95);
  });

  test("what to do says more than what not to: a question with both keeps its gold", async () => {
    // Two brains from the same seed, one shown both, one shown only the gold:
    // the same pass, to the last weight.
    const a = session();
    const b = session();
    const both = await a.teach([{ records: frame(), gold: [0], forbidden: [0] }]);
    const toward = await b.teach([{ records: frame(), gold: [0] }]);
    expect(both.meanAgreement).toBe(toward.meanAgreement);
    expect(a.weights.selector.wq).toEqual(b.weights.selector.wq);
    expect(a.weights.embeddings).toEqual(b.weights.embeddings);
  });

  test("a showing whose arithmetic came back broken is thrown away, to the last weight", async () => {
    const { teachFromDemonstration, cpuGradients } = await import("./index.ts");
    const brain = session();
    const wq = Float32Array.from(brain.weights.selector.wq);
    const embeddings = Float32Array.from(brain.weights.embeddings);
    const cpu = cpuGradients(brain.weights, brain.config);
    // A source that answers like a lost device: numbers that are not numbers.
    const broken = {
      backward: async (frame: any, request: any) => {
        const result = await cpu.backward(frame, request);
        result.dSelectorWq[3] = Number.NaN;
        return result;
      },
    };
    const report = await teachFromDemonstration([{ records: frame(), gold: [0] }], brain.weights, brain.config, {}, broken);
    expect(report.rejected).toBe(1);
    expect(report.framesSeen).toBe(0);
    expect(brain.weights.selector.wq).toEqual(wq);
    expect(brain.weights.embeddings).toEqual(embeddings);
    // The reading taken before the pass is still a reading.
    expect(report.shown).toBe(1);
    expect(Number.isFinite(report.meanAgreement)).toBe(true);
  });

  test("a frame with nothing to show is not a lesson", async () => {
    const brain = session();
    const before = Float32Array.from(brain.weights.selector.wq);
    const report = await brain.teach([{ records: frame(), gold: [undefined] }]);
    expect(report.framesSeen).toBe(0);
    expect(report.shown).toBe(0);
    expect(brain.weights.selector.wq).toEqual(before);
  });
});

describe("what teaching costs when the tables cannot move", () => {
  test("frozen, showing the same thing two hundred times changes almost nothing", async () => {
    const brain = session();
    const shown: HostDemonstration[] = [{ records: frame(), gold: [0] }];
    for (let round = 0; round < 200; round++) await brain.teach(shown, { learningRate: 0.5, unfreezeTokens: false });
    // 0.498 → 0.510. A selector is one shared projection and cannot pull apart
    // two records whose representations are fixed random vectors, so teaching
    // with the tables frozen is a no-op that looks like slow teaching. This is
    // why `unfreezeTokens` defaults ON here and OFF in `learn`.
    expect((await brain.think(frame())).selections[0]!.distribution[0]!).toBeLessThan(0.6);
  }, 20_000);
});
