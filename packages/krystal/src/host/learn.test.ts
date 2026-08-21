import { describe, expect, test } from "bun:test";
import { BrainSession, type HostExperience, type HostRecord } from "./index.ts";
import { BRAIN_FORWARD_CONFIG } from "../forward/model.ts";

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

	test("a runaway update is rejected as one transaction", () => {
		const session = new BrainSession({ tokenRows: rows(4096), seed: 7 });
		const before = {
			embeddings: Float32Array.from(session.weights.embeddings),
			wq: Float32Array.from(session.weights.selector.wq),
			wk: Float32Array.from(session.weights.selector.wk),
			value: Float32Array.from(session.weights.valueHeadWv),
		};
		const report = session.learn(lived(), {
			maxParameterAbs: 0.01,
			unfreeze: { tokens: true, tokenRate: 0.1 },
		});
		expect(report.updateApplied).toBe(false);
		expect(report.rejected).toBe("parameter-limit");
		expect(session.weights.embeddings).toEqual(before.embeddings);
		expect(session.weights.selector.wq).toEqual(before.wq);
		expect(session.weights.selector.wk).toEqual(before.wk);
		expect(session.weights.valueHeadWv).toEqual(before.value);
		expect(report.health.finite).toBe(true);
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

// ─── Two arguments of one type ───────────────────────────────────────────────
//
// A host asks about an argument with a RECORD, so two arguments of one kind are
// only two questions if their query rows differ. These tests pin down how far
// that gets with today's frozen model, because the answer is not "all the way"
// and the difference is the argument for unfreezing.

/** Two people to choose between, and one question per role. */
const roles = (named: boolean): HostRecord[] => [
	{ schemaId: 1, band: 3, tokens: [10, 11] }, // Ada
	{ schemaId: 1, band: 3, tokens: [20, 21] }, // Bo
	{ schemaId: 9, query: true, tokens: [40, 41] }, // give · giver
	{ schemaId: 9, query: true, tokens: [40, named ? 42 : 41] }, // give · receiver
];

/** How far each question leans toward Ada, from -1 to 1. */
const leanings = (session: BrainSession, named: boolean): number[] =>
	session.think(roles(named)).selections.map((selection) => selection.distribution[0]! - selection.distribution[1]!);

const trained = (
	named: boolean,
	batch: HostExperience[],
	options: { rounds?: number; unfreeze?: { tokens?: boolean; tokenRate?: number } } = {},
): { moved: number[] } => {
	const session = new BrainSession({ tokenRows: rows(4096), seed: 11 });
	const before = leanings(session, named);
	// A short capability probe, deliberately not a stability run. Replaying this
	// same batch after the policy has moved makes it stale/off-policy: eventually
	// it keeps crediting a bad action the current policy would never draw.
	for (let round = 0; round < (options.rounds ?? 150); round++)
		session.learn(batch, { learningRate: 0.05, ...(options.unfreeze ? { unfreeze: options.unfreeze } : {}) });
	const after = leanings(session, named);
	return { moved: after.map((lean, index) => lean - before[index]!) };
};

/** Ada gives and Bo receives; the other way round goes badly. */
const eachToItsOwn = (): HostExperience[] => [
	{ records: roles(true), chosen: [0, 1], reward: 0.4 },
	{ records: roles(true), chosen: [1, 0], reward: -0.4 },
];

describe("telling two arguments of one kind apart", () => {
	test("rows that say the same thing are one question asked twice", () => {
		const batch: HostExperience[] = [
			{ records: roles(false), chosen: [0, undefined], reward: 0.4 },
			{ records: roles(false), chosen: [1, undefined], reward: -0.4 },
		];
		const { moved } = trained(false, batch);
		// Not merely similar — identical. Nothing in the frame distinguishes them,
		// so no amount of training can, and a giver could never differ from a
		// receiver.
		expect(moved[0]).toBe(moved[1]!);
	});

	test("a question that names its argument is answerable on its own", () => {
		const batch: HostExperience[] = [
			{ records: roles(true), chosen: [0, undefined], reward: 0.4 },
			{ records: roles(true), chosen: [1, undefined], reward: -0.4 },
		];
		const { moved } = trained(true, batch);
		// The credited question learned decisively — the policy gradient is not
		// weak, which matters for reading the next test.
		expect(Math.abs(moved[0]!)).toBeGreaterThan(0.15);
		expect(moved[0]).not.toBe(moved[1]!);
	});

	test("but the two still move together, which is what freezing costs", () => {
		const batch: HostExperience[] = [
			{ records: roles(true), chosen: [0, undefined], reward: 0.4 },
			{ records: roles(true), chosen: [1, undefined], reward: -0.4 },
		];
		const { moved } = trained(true, batch);
		// Only the first question was ever credited; the second was asked and left.
		// It moved almost exactly as far, because the one thing still learning is a
		// single shared projection and a query row cannot yet shape its own
		// representation. Naming the argument reaches the model — it does not yet
		// separate it.
		//
		// This test documents a LIMIT, not a requirement. When the pool, encoder
		// and mixer come unfrozen it should start failing, and the right response
		// is to invert it.
		expect(Math.abs(moved[0]! - moved[1]!)).toBeLessThan(0.01);
	});

	test("opposing targets on one shared projection very nearly cancel", () => {
		const { moved } = trained(true, eachToItsOwn());
		// "Ada gives, Bo receives" pushes the two rows in opposite directions
		// through the same weights, and what survives is a residue three orders of
		// magnitude below what a single question learns.
		expect(Math.max(...moved.map(Math.abs))).toBeLessThan(0.01);
	});
});

describe("what unfreezing the tables buys", () => {
	test("two questions finally pull in opposite directions", () => {
		// 300, not the 150 the other assays use. The critic no longer pushes the
		// selector through a soft gather over the whole bank — a slice of gradient
		// that used to shape the representation and was made of records the
		// grammar had forbidden. The actor does it alone now, and takes about
		// twice as long: measured 150 → `+0.019 / +0.000`, 300 → `+0.910 / -0.880`.
		const { moved } = trained(true, eachToItsOwn(), { rounds: 300, unfreeze: { tokens: true, tokenRate: 0.1 } });
		// Ada was the giver and Bo the receiver, and each question moved toward its
		// own answer. Frozen, both drift the same way whatever the reward said:
		// what tells two questions apart is about a hundredth of a row's vector,
		// and nothing downstream can amplify a difference that small. The
		// difference itself has to be allowed to grow.
		expect(moved[0]).toBeGreaterThan(0);
		expect(moved[1]).toBeLessThan(0);
		// Twice the rounds on the full profile: this one is a few seconds.
	}, 30_000);

	test("frozen, the same life moves both the same way", () => {
		const { moved } = trained(true, eachToItsOwn());
		expect(Math.sign(moved[0]!)).toBe(Math.sign(moved[1]!));
	});

	test("fresh on-policy batches separate the roles without the replay runaway", () => {
		// Smaller geometry keeps this long behavioural assay cheap. It exercises
		// the same six embedding tables and update path as the full profile.
		const config = {
			...BRAIN_FORWARD_CONFIG,
			hiddenSize: 32,
			headCount: 1,
			headDim: 32,
			ffnSize: 64,
			encoderBlocks: 1,
			mixerBlocks: 1,
		};
		const session = new BrainSession({ tokenRows: rows(4096), seed: 11, config });
		let seed = 123456789;
		const random = (): number => {
			seed |= 0;
			seed = (seed + 0x6d2b79f5) | 0;
			let value = Math.imul(seed ^ (seed >>> 15), 1 | seed);
			value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
			return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
		};

		for (let round = 0; round < 700; round++) {
			const batch: HostExperience[] = [];
			for (let sample = 0; sample < 8; sample++) {
				const chosen = session.think(roles(true), { sample: () => random() }).selections.map((entry) => entry.record);
				batch.push({
					records: roles(true),
					chosen,
					reward: chosen[0] === 0 && chosen[1] === 1 ? 0.4 : -0.4,
				});
			}
			// 0.5 belongs to THIS geometry, not to a host. On the full profile it is
			// too hot: 107 of 500 fresh batches were rolled back and both questions
			// collapsed onto one record. The game runs 0.1, where the unfrozen rows
			// plateau at |0.20| over 2000 batches and never trip the ceiling.
			session.learn(batch, { learningRate: 0.05, unfreeze: { tokens: true, tokenRate: 0.5 } });
		}

		const [giver, receiver] = session.think(roles(true)).selections;
		// Converged, not pinned: measured `0.99997` and `0.98991`. The bar is what
		// separates "it learned the two roles" from "it did not", and a threshold
		// closer than that to the measurement is a test about arithmetic noise.
		expect(giver!.distribution[0]).toBeGreaterThan(0.95);
		expect(receiver!.distribution[1]).toBeGreaterThan(0.95);
		expect([...session.weights.embeddings].every(Number.isFinite)).toBe(true);
		expect([...session.weights.selector.wq].every(Number.isFinite)).toBe(true);
		expect([...session.weights.valueHeadWv].every(Number.isFinite)).toBe(true);
	});
});
