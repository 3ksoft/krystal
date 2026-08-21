import { describe, expect, test } from "bun:test";
import { BrainSession, type HostExperience, type HostRecord } from "./index.ts";

/** A tiny manifest: token id N sits in row N. */
const rows = (count: number): Uint32Array => Uint32Array.from({ length: count }, (_, id) => id % 64);

/** Two things to pick between, and one question. */
const world = (): HostRecord[] => [
	{ schemaId: 1, band: 3, tokens: [10, 11] }, // the good one
	{ schemaId: 1, band: 3, tokens: [20, 21] }, // the bad one
	{ schemaId: 9, query: true, tokens: [30] },
];

const GOOD = 0;
const BAD = 1;

/** How likely the brain is to pick a given record, right now. */
const chanceOf = (session: BrainSession, record: number): number => {
	const { selections } = session.think(world());
	return selections[0]!.distribution[record]!;
};

/** A life where picking the good one goes well and the bad one does not. */
const lived = (): HostExperience[] => [
	{ records: world(), chosen: [GOOD], reward: 0.4 },
	{ records: world(), chosen: [BAD], reward: -0.4 },
	{ records: world(), chosen: [GOOD], reward: 0.3 },
	{ records: world(), chosen: [BAD], reward: -0.3 },
];

describe("living with what happened", () => {
	test("what went well is more likely afterwards", () => {
		const session = new BrainSession({ tokenRows: rows(4096), seed: 7 });
		const before = chanceOf(session, GOOD);
		for (let round = 0; round < 40; round++) session.learn(lived(), { learningRate: 0.05 });
		expect(chanceOf(session, GOOD)).toBeGreaterThan(before);
	});

	test("and what went badly is less likely", () => {
		const session = new BrainSession({ tokenRows: rows(4096), seed: 7 });
		const before = chanceOf(session, BAD);
		for (let round = 0; round < 40; round++) session.learn(lived(), { learningRate: 0.05 });
		expect(chanceOf(session, BAD)).toBeLessThan(before);
	});

	test("it says how many it pushed each way", () => {
		const session = new BrainSession({ tokenRows: rows(4096), seed: 7 });
		const report = session.learn(lived());
		expect(report.framesSeen).toBe(4);
		expect(report.reinforced).toBe(2);
		expect(report.discouraged).toBe(2);
	});
});

describe("what it refuses to learn from", () => {
	test("a turn with no consequence yet teaches nothing", () => {
		const session = new BrainSession({ tokenRows: rows(4096), seed: 7 });
		const report = session.learn([{ records: world(), chosen: [GOOD] }]);
		expect(report.framesSeen).toBe(0);
	});

	test("a batch where everything went equally well does not move the actor", () => {
		const session = new BrainSession({ tokenRows: rows(4096), seed: 7 });
		const before = chanceOf(session, GOOD);
		// Nothing here says WHICH choice was responsible, so nothing may be
		// credited to one — a batch of identical outcomes is not evidence.
		const report = session.learn([
			{ records: world(), chosen: [GOOD], reward: 0.2 },
			{ records: world(), chosen: [BAD], reward: 0.2 },
		]);
		expect(report.reinforced + report.discouraged).toBe(0);
		expect(chanceOf(session, GOOD)).toBe(before);
	});

	test("a turn nobody acted in still teaches the critic what such a moment is worth", () => {
		const session = new BrainSession({ tokenRows: rows(4096), seed: 7 });
		const before = Float32Array.from(session.weights.valueHeadWv);
		const report = session.learn([{ records: world(), reward: 0.5 }]);
		expect(report.framesSeen).toBe(1);
		expect(report.reinforced + report.discouraged).toBe(0);
		expect([...session.weights.valueHeadWv]).not.toEqual([...before]);
	});

	test("the actor is left alone by a frame with no choice in it", () => {
		const session = new BrainSession({ tokenRows: rows(4096), seed: 7 });
		const before = Float32Array.from(session.weights.selector.wq);
		session.learn([{ records: world(), reward: 0.5 }]);
		expect([...session.weights.selector.wq]).toEqual([...before]);
	});
});

describe("the grammar that was in force", () => {
	test("a choice is trained against the same mask it was made under", () => {
		const session = new BrainSession({ tokenRows: rows(4096), seed: 7 });
		// Only the good one was ever on offer, so the distribution the update
		// differentiates has one live option in it.
		const allows = (_query: number, record: number) => record === GOOD;
		const report = session.learn([
			{ records: world(), chosen: [GOOD], allows, reward: 0.4 },
			{ records: world(), chosen: [GOOD], allows, reward: -0.4 },
		]);
		expect(report.framesSeen).toBe(2);
	});

	test("learning predicts better than it did: the critic's loss falls", () => {
		const session = new BrainSession({ tokenRows: rows(4096), seed: 7 });
		const first = session.learn(lived(), { learningRate: 0.05 });
		let last = first;
		for (let round = 0; round < 40; round++) last = session.learn(lived(), { learningRate: 0.05 });
		expect(last.meanValueLoss).toBeLessThan(first.meanValueLoss);
	});
});
