import { describe, expect, test } from "bun:test";
import { BrainSession, packHostFrame, HostFrameError, QUERY_BAND, RECORD_WIDTH } from "./index.ts";
import { recordRanges } from "../forward/masks.ts";
import { recordRanges } from "../forward/masks.ts";

/** A tiny manifest: token id N sits in row N. */
const rows = (count: number): Uint32Array => Uint32Array.from({ length: count }, (_, id) => id % 64);

const world = () => [
	{ schemaId: 1, band: 3, tokens: [10, 11, 12] }, // an apple
	{ schemaId: 1, band: 3, tokens: [10, 13] }, // a stone
	{ schemaId: 2, band: 2, tokens: [20, 21] }, // a comfort
	{ schemaId: 9, query: true, tokens: [30] }, // what next?
];

describe("packing a frame from records", () => {
	test("is sized to what it was given, not to a fixed geometry", () => {
		const frame = packHostFrame(world());
		expect(frame.slots).toBe(4);
		expect(frame.gpu.tokenIds.length).toBe(4 * RECORD_WIDTH);
		expect(frame.active.activeTokens.length).toBe(3 + 2 + 2 + 1);
	});

	test("a query record is a question and the rest are the bank", () => {
		const frame = packHostFrame(world());
		expect([...frame.active.queryRecords]).toEqual([3]);
		expect([...frame.active.bankRecords]).toEqual([0, 1, 2]);
		expect(frame.gpu.bandIds[3]).toBe(QUERY_BAND);
	});

	test("no token attends across a record boundary", () => {
		const frame = packHostFrame([{ tokens: [1, 2] }, { tokens: [3] }]);
		const { start, count } = recordRanges(frame.active);
		// tokens 0 and 1 share a record, token 2 is in another — and these ranges
		// are what the forward reads, in place of a T x T mask of -1e30.
		expect([start[0], count[0]]).toEqual([0, 2]);
		expect([start[1], count[1]]).toEqual([0, 2]);
		expect([start[2], count[2]]).toEqual([2, 1]);
	});

	test("a record wider than the model has positions for is refused, not truncated", () => {
		expect(() => packHostFrame([{ tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9] }])).toThrow(HostFrameError);
	});

	test("an empty frame is refused — there is nothing to think about", () => {
		expect(() => packHostFrame([])).toThrow(HostFrameError);
	});
});

describe("thinking", () => {
	test("every question gets one choice out of the bank", async () => {
		const session = new BrainSession({ tokenRows: rows(4096), seed: 7 });
		const { selections } = (await session.think(world()));
		expect(selections).toHaveLength(1);
		expect([0, 1, 2]).toContain(selections[0]!.record);
		expect(selections[0]!.distribution).toHaveLength(3);
	});

	test("the same weights and the same frame give the same answer", async () => {
		const session = new BrainSession({ tokenRows: rows(4096), seed: 7 });
		expect((await session.think(world())).selections[0]!.record).toBe((await session.think(world())).selections[0]!.record);
	});

	test("what the host forbids is never chosen, and never carries probability", async () => {
		const session = new BrainSession({ tokenRows: rows(4096), seed: 7 });
		const { selections } = (await session.think(world(), { allows: (_query, record) => record === 2 }));
		expect(selections[0]!.record).toBe(2);
		expect(selections[0]!.distribution[0]).toBeCloseTo(0, 6);
		expect(selections[0]!.distribution[1]).toBeCloseTo(0, 6);
	});

	test("a question nothing admits comes back open rather than as a false answer", async () => {
		const session = new BrainSession({ tokenRows: rows(4096), seed: 7 });
		const { selections } = (await session.think(world(), { allows: () => false }));
		const total = [...selections[0]!.distribution].reduce((sum, p) => sum + p, 0);
		expect(total).toBeCloseTo(1, 5);
	});

	test("a frame with no question asks nothing", async () => {
		const session = new BrainSession({ tokenRows: rows(4096), seed: 7 });
		expect((await session.think([{ tokens: [10] }])).selections).toEqual([]);
	});
});
