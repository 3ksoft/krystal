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
  test("a handful of showings is enough, where finding out takes hundreds", () => {
    const brain = session();
    const before = brain.think(frame()).selections[0]!.distribution[0]!;
    const shown: HostDemonstration[] = [{ records: frame(), gold: [0] }];
    let report = brain.teach(shown);
    // What it already believed, before this pass changed anything: a creature
    // that has been shown nothing is guessing between two.
    expect(report.meanAgreement).toBeCloseTo(0.5, 1);
    for (let round = 0; round < 40; round++) report = brain.teach(shown);
    const after = brain.think(frame()).selections[0]!.distribution[0]!;
    expect(after).toBeGreaterThan(before);
    expect(after).toBeGreaterThan(0.95);
  });

  test("agreement is what teaching is measured by, and it climbs", () => {
    const brain = session();
    const shown: HostDemonstration[] = [{ records: frame(), gold: [0] }];
    const first = brain.teach(shown).meanAgreement;
    for (let round = 0; round < 15; round++) brain.teach(shown);
    expect(brain.teach(shown).meanAgreement).toBeGreaterThan(first);
  });

  test("it teaches the grammar, and says nothing about how things went", () => {
    const brain = session();
    const value = Float32Array.from(brain.weights.valueHeadWv);
    for (let round = 0; round < 10; round++) brain.teach([{ records: frame(), gold: [0] }]);
    // A demonstration is not an outcome. The critic has nothing to learn from
    // one, and a critic that moved here would be predicting that being taught
    // feels good.
    expect(brain.weights.valueHeadWv).toEqual(value);
  });

  test("a demonstration is shown under the grammar a choice would be made under", () => {
    const brain = session();
    // The stone is the only thing this question admits, and the demonstration
    // agrees with it — so agreement starts at certainty, not at a half.
    const report = brain.teach([{ records: frame(), gold: [1], allows: (_query, record) => record === 1 }]);
    expect(report.meanAgreement).toBeCloseTo(1, 5);
  });

  test("a frame with nothing to show is not a lesson", () => {
    const brain = session();
    const before = Float32Array.from(brain.weights.selector.wq);
    const report = brain.teach([{ records: frame(), gold: [undefined] }]);
    expect(report.framesSeen).toBe(0);
    expect(report.shown).toBe(0);
    expect(brain.weights.selector.wq).toEqual(before);
  });
});

describe("what teaching costs when the tables cannot move", () => {
  test("frozen, showing the same thing two hundred times changes almost nothing", () => {
    const brain = session();
    const shown: HostDemonstration[] = [{ records: frame(), gold: [0] }];
    for (let round = 0; round < 200; round++) brain.teach(shown, { learningRate: 0.5, unfreezeTokens: false });
    // 0.498 → 0.510. A selector is one shared projection and cannot pull apart
    // two records whose representations are fixed random vectors, so teaching
    // with the tables frozen is a no-op that looks like slow teaching. This is
    // why `unfreezeTokens` defaults ON here and OFF in `learn`.
    expect(brain.think(frame()).selections[0]!.distribution[0]!).toBeLessThan(0.6);
  }, 20_000);
});
