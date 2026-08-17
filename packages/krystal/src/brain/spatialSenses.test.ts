import { expect, test } from "bun:test";
import { allocateRecordBudget, buildSpatialSenseRecords } from "./spatialSenses";

const entity = (id: string, x: number, y: number, name = "Apple", state: Record<string, unknown> = {}) => ({
	id, name, state: { x, y, ...state },
});

test("compresses many equal entities into one sight record", () => {
	const records = buildSpatialSenseRecords(
		entity("agent", 0, 0, "Agent"),
		[
			entity("a1", 4, 0), entity("a2", 4, 0), entity("a3", 4, 0), entity("a4", 4, 0),
		],
		{ modalities: ["sight"] },
	);

	expect(records).toHaveLength(1);
	expect(records[0]).toMatchObject({ kind: "sight", payload: {
		distance: "near", angle: "front", count: 4,
		tokens: ["SIGHT", "DIST_NEAR", "ANG_FRONT", "Apple", "Amount.muchmany"],
	} });
});

test("vision omits behind while hearing keeps it", () => {
	const subject = entity("agent", 0, 0, "Agent");
	const cat = entity("cat", -4, 0, "Cat");

	expect(buildSpatialSenseRecords(subject, [cat], { modalities: ["sight"] })).toEqual([]);
	const hearing = buildSpatialSenseRecords(subject, [cat], { modalities: ["hearing"] });
	expect(hearing[0]).toMatchObject({ kind: "hearing", payload: { angle: "behind" }, target: "cat" });
});

test("aggregates only matching spatial and observable signatures", () => {
	const records = buildSpatialSenseRecords(
		entity("agent", 0, 0, "Agent"),
		[
			entity("red", 4, 0, "Apple", { color: "Red" }),
			entity("blue", 4, 0, "Apple", { color: "Blue" }),
		],
		{ modalities: ["sight"] },
	);

	expect(records).toHaveLength(2);
	expect(records.map((record) => record.payload.tokens)).toEqual([
		["SIGHT", "DIST_NEAR", "ANG_FRONT", "Apple", "Color.Blue", "Amount.littlefew"],
		["SIGHT", "DIST_NEAR", "ANG_FRONT", "Apple", "Color.Red", "Amount.littlefew"],
	]);
});

test("applies a deterministic record budget", () => {
	const records = buildSpatialSenseRecords(
		entity("agent", 0, 0, "Agent"),
		[
			entity("near", 2, 0, "Near"),
			entity("far", 20, 0, "Far"),
		],
		{ modalities: ["hearing"], maxRecords: 1 },
	);

	expect(records).toHaveLength(1);
	expect(records[0]?.payload.distance).toBe("near");
});

test("the budget is SHARED between modalities — a crowded sight cannot starve hearing", () => {
	// six identical things right in front, plus one behind: sight sees the six
	// (as six distinct records, one per name) and hearing also hears the one
	// behind. A first-come budget of 4 would be all sight and no hearing at all —
	// and ANG_BEHIND, which only hearing can produce, would vanish.
	const front = ["a", "b", "c", "d", "e", "f"].map((name, index) => entity(`f${index}`, 4, 0, name.toUpperCase()));
	const behind = entity("behind", -0.5, 0, "Cat"); // HERE, so hearing draws it first
	const records = buildSpatialSenseRecords(entity("agent", 0, 0, "Agent"), [...front, behind], { maxRecords: 4 });

	expect(records.filter((record) => record.kind === "sight")).toHaveLength(2);
	expect(records.filter((record) => record.kind === "hearing")).toHaveLength(2);
	expect(records.some((record) => record.payload.angle === "behind")).toBe(true);
	// selection only — the emitted order stays canonical (sight before hearing)
	expect(records.map((record) => record.kind)).toEqual(["sight", "sight", "hearing", "hearing"]);
});

test("an under-supplied modality gives its share back rather than wasting the budget", () => {
	const records = [
		{ kind: "sight" as const, id: 1 }, { kind: "sight" as const, id: 2 },
		{ kind: "sight" as const, id: 3 }, { kind: "hearing" as const, id: 4 },
	];
	expect(allocateRecordBudget(records, ["sight", "hearing"], 3).map((r) => r.id)).toEqual([1, 2, 4]);
	// nothing to allocate: under the cap, everything survives untouched
	expect(allocateRecordBudget(records, ["sight", "hearing"], 9)).toEqual(records);
	expect(allocateRecordBudget(records, ["sight", "hearing"], undefined)).toEqual(records);
	expect(allocateRecordBudget(records, ["sight", "hearing"], 0)).toEqual([]);
});
