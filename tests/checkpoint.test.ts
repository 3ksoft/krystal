// Combined local checkpoint test suite (bun:test).
//
// Runs on the REAL engine: loadModel() (root src/) opens the WQ4 model on the
// GPU (Dawn via the `webgpu` npm bindings), drives Lfm2Forward through
// createLfm2WebGpuTransport, and returns an engine-ts Engine. Checkpoints are
// exact physical KV/conv snapshots, so these tests verify real checkpoint
// semantics (no re-prefill of the checkpoint prefix). Without a model file /
// GPU the same suite falls back to the mock exe backend with a warning.
//
//   bun run packages/backend/src/run-local-tests.ts
//   bun test tests/checkpoint.test.ts
import { describe, expect, test } from "bun:test";
import { loadModel } from "../src";
import {
	GPU_CONSTRAINT_ABI,
	GPU_CONSTRAINT_NODE_KIND,
	GPU_CONSTRAINT_TOKEN_META,
	GPU_CONSTRAINT_STATE,
	gpuConstraintMaskReference,
	createGpuConstraintDecoderState,
	feedGpuConstraintBytes,
	gpuConstraintComplete,
	linkGpuConstraintProgram,
	linkGpuConstraintTokenizer,
} from "../packages/engine-ts/src/gpu-constraint.ts";
import {
	compileJsonSchemaProgram,
	type LayoutConstraintProgram,
} from "../packages/engine-ts/src/index.ts";
import {
	dispatchGpuConstraintMask,
	readGpuConstraintMask,
	readGpuConstraintState,
	uploadGpuConstraint,
	writeGpuConstraintState,
} from "../packages/webgpu/src/constraint.ts";
import { defineLfm2, LFM2_ARENA } from "../packages/webgpu/src/lfm2-definition.ts";
import { lfm2Artifact } from "../packages/webgpu/src/lfm2.artifact.generated.ts";

async function collect(iterable: AsyncIterable<number>): Promise<number[]> {
	const out: number[] = [];
	for await (const token of iterable) out.push(token);
	return out;
}

function tokens(...values: number[]): Uint32Array {
	return new Uint32Array(values);
}

const BOS = 1;

// Greedy jest istotny: wyniki mają być deterministyczne.
const GENERATE = {
	maxTokens: 4,
	sampler: "argmax",
} as const;


const constraintEncoder = new TextEncoder();

describe("gpu constraint linker", () => {
	test("collapses split prefixes into a deterministic trie", () => {
		const program: LayoutConstraintProgram = {
			entry: 3,
			accept: 0,
			summary: {
				rootType: "value",
				segments: 4,
				fields: 0,
				optionalIncluded: 0,
				optionalSkipped: 0,
				enums: 0,
				strings: 0,
				numbers: 0,
				booleans: 0,
				arrays: 0,
			},
			nodes: [
				{ kind: "accept", label: "done" },
				{
					kind: "literal",
					bytes: constraintEncoder.encode("abc"),
					text: "abc",
					label: "a",
					next: 0,
				},
				{
					kind: "literal",
					bytes: constraintEncoder.encode("abd"),
					text: "abd",
					label: "b",
					next: 0,
				},
				{ kind: "split", targets: [1, 2], label: "split" },
			],
		};

		const linked = linkGpuConstraintProgram(program);

		expect(linked.summary.switchNodes).toBe(1);
		expect(linked.summary.edges).toBe(2);
		expect(linked.summary.literalNodes).toBe(1);
		expect(linked.nodes[0]).toBe(GPU_CONSTRAINT_NODE_KIND.literal);

		const switchNode = linked.nodes.subarray(
			GPU_CONSTRAINT_ABI.nodeWords,
			GPU_CONSTRAINT_ABI.nodeWords * 2,
		);
		expect(switchNode[0]).toBe(GPU_CONSTRAINT_NODE_KIND.switch);
	});

	test("packs tokenizer byte offsets, lengths and special flag", () => {
		const linked = linkGpuConstraintTokenizer(
			[
				{ id: 0, bytes: constraintEncoder.encode("a"), special: false },
				{ id: 1, bytes: constraintEncoder.encode("bc"), special: false },
				{ id: 2, bytes: null, special: true },
			],
			2,
		);

		expect(linked.header[0]).toBe(3);
		expect(linked.header[1]).toBe(2);

		expect(linked.entries[0]).toBe(0);
		expect(linked.entries[1]).toBe(1);

		expect(linked.entries[2]).toBe(1);
		expect(linked.entries[3]).toBe(2);

		expect(linked.entries[4]).toBe(3);
		expect(linked.entries[5]! & GPU_CONSTRAINT_TOKEN_META.special).not.toBe(0);
		expect(linked.byteLength).toBe(3);
	});

	test("executes the upload blob with transactional 64-byte state", () => {
		const program: LayoutConstraintProgram = {
			entry: 5,
			accept: 0,
			summary: {
				rootType: "value",
				segments: 6,
				fields: 2,
				optionalIncluded: 0,
				optionalSkipped: 0,
				enums: 0,
				strings: 1,
				numbers: 1,
				booleans: 0,
				arrays: 0,
			},
			nodes: [
				{ kind: "accept", label: "done" },
				{ kind: "literal", bytes: constraintEncoder.encode("}"), text: "}", label: "close", next: 0 },
				{ kind: "string", minLength: 1, maxLength: 4, label: "string", next: 1 },
				{ kind: "literal", bytes: constraintEncoder.encode(',"s":'), text: ',"s":', label: "field-s", next: 2 },
				{ kind: "number", integer: false, min: 0, max: 10, maxChars: 32, label: "number", next: 3 },
				{ kind: "literal", bytes: constraintEncoder.encode('{"n":'), text: '{"n":', label: "field-n", next: 4 },
			],
		};

		const linked = linkGpuConstraintProgram(program);
		const state = createGpuConstraintDecoderState(linked);
		expect(state.byteLength).toBe(GPU_CONSTRAINT_STATE.byteLength);

		expect(feedGpuConstraintBytes(linked, state, constraintEncoder.encode('{"n":'))).toBe(true);
		const beforeInvalid = state.slice();
		expect(feedGpuConstraintBytes(linked, state, constraintEncoder.encode("11,"))).toBe(false);
		expect(Array.from(state)).toEqual(Array.from(beforeInvalid));

		expect(feedGpuConstraintBytes(linked, state, constraintEncoder.encode("3.5"))).toBe(true);
		expect(feedGpuConstraintBytes(linked, state, constraintEncoder.encode(',"s":"a\\n"}'))).toBe(true);
		expect(gpuConstraintComplete(linked, state)).toBe(true);
	});

	test("builds the exact packed mask from upload blobs", () => {
		const program: LayoutConstraintProgram = {
			entry: 5,
			accept: 0,
			summary: {
				rootType: "value", segments: 6, fields: 2,
				optionalIncluded: 0, optionalSkipped: 0, enums: 0,
				strings: 1, numbers: 1, booleans: 0, arrays: 0,
			},
			nodes: [
				{ kind: "accept", label: "done" },
				{ kind: "literal", bytes: constraintEncoder.encode("}"), text: "}", label: "close", next: 0 },
				{ kind: "string", minLength: 1, maxLength: 4, label: "string", next: 1 },
				{ kind: "literal", bytes: constraintEncoder.encode(',"s":'), text: ',"s":', label: "field-s", next: 2 },
				{ kind: "number", integer: false, min: 0, max: 10, maxChars: 32, label: "number", next: 3 },
				{ kind: "literal", bytes: constraintEncoder.encode('{"n":'), text: '{"n":', label: "field-n", next: 4 },
			],
		};
		const linked = linkGpuConstraintProgram(program);
		const tokenizer = linkGpuConstraintTokenizer([
			{ id: 0, bytes: constraintEncoder.encode('{"n":'), special: false },
			{ id: 1, bytes: constraintEncoder.encode("3.5"), special: false },
			{ id: 2, bytes: constraintEncoder.encode("11,"), special: false },
			{ id: 3, bytes: constraintEncoder.encode(',"s":"x"}'), special: false },
			{ id: 4, bytes: null, special: true },
		], 4);
		const state = createGpuConstraintDecoderState(linked);

		expect(Array.from(gpuConstraintMaskReference(linked, tokenizer, state))).toEqual([0b00001]);
		expect(feedGpuConstraintBytes(linked, state, constraintEncoder.encode('{"n":'))).toBe(true);
		expect(Array.from(gpuConstraintMaskReference(linked, tokenizer, state))).toEqual([0b00010]);
		expect(feedGpuConstraintBytes(linked, state, constraintEncoder.encode('3.5,"s":"x"}'))).toBe(true);
		expect(gpuConstraintComplete(linked, state)).toBe(true);
		expect(Array.from(gpuConstraintMaskReference(linked, tokenizer, state))).toEqual([0b10000]);
	});


	test("Dawn AOT mask matches the CPU oracle bit-for-bit", async () => {
		const { create, globals } = await import("webgpu");
		Object.assign(globalThis, globals);

		// Keep the Dawn instance alive until after the device is destroyed.
		const gpu = create([]);
		const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
		if (!adapter) throw new Error("Could not acquire a Dawn WebGPU adapter");
		const device = await adapter.requestDevice({ label: "constraint-mask-aot-test" });

		const definition = defineLfm2();
		definition.engine.deserialize(lfm2Artifact);

		try {
			const compiled = await definition.engine.compile({ device });
			expect(compiled.failed).toBe(0);

			const program: LayoutConstraintProgram = {
				entry: 5,
				accept: 0,
				summary: {
					rootType: "value", segments: 6, fields: 2,
					optionalIncluded: 0, optionalSkipped: 0, enums: 0,
					strings: 1, numbers: 1, booleans: 0, arrays: 0,
				},
				nodes: [
					{ kind: "accept", label: "done" },
					{ kind: "literal", bytes: constraintEncoder.encode("}"), text: "}", label: "close", next: 0 },
					{ kind: "string", minLength: 1, maxLength: 4, label: "string", next: 1 },
					{ kind: "literal", bytes: constraintEncoder.encode(',"s":'), text: ',"s":', label: "field-s", next: 2 },
					{ kind: "number", integer: false, min: 0, max: 10, maxChars: 32, label: "number", next: 3 },
					{ kind: "literal", bytes: constraintEncoder.encode('{"n":'), text: '{"n":', label: "field-n", next: 4 },
				],
			};
			const linked = linkGpuConstraintProgram(program);
			const tokenizer = linkGpuConstraintTokenizer([
				{ id: 0, bytes: constraintEncoder.encode('{"n":'), special: false },
				{ id: 1, bytes: constraintEncoder.encode("3.5"), special: false },
				{ id: 2, bytes: constraintEncoder.encode("11,"), special: false },
				{ id: 3, bytes: constraintEncoder.encode(',"s":"x"}'), special: false },
				{ id: 4, bytes: null, special: true },
			], 4);
			const state = createGpuConstraintDecoderState(linked);
			const upload = uploadGpuConstraint(definition, linked, tokenizer, state);

			const compare = async () => {
				const cpu = gpuConstraintMaskReference(linked, tokenizer, state);
				dispatchGpuConstraintMask(definition);
				await device.queue.onSubmittedWorkDone();
				const gpuMask = await readGpuConstraintMask(definition, upload.maskWords);
				expect(Array.from(gpuMask)).toEqual(Array.from(cpu));
			};

			await compare();

			expect(feedGpuConstraintBytes(linked, state, constraintEncoder.encode('{"n":'))).toBe(true);
			writeGpuConstraintState(definition, state);
			await compare();

			expect(feedGpuConstraintBytes(linked, state, constraintEncoder.encode('3.5,"s":"x"}'))).toBe(true);
			expect(gpuConstraintComplete(linked, state)).toBe(true);
			writeGpuConstraintState(definition, state);
			await compare();

			// Exercise the bounded-array control-flow added by the JSON-Schema
			// frontend. This specifically covers both the explicit jump barrier and
			// the replay switch edge used for `]` versus a dynamic string item.
			const arrayCpu = compileJsonSchemaProgram({
				type: "array",
				maxItems: 2,
				items: { type: "string", maxLength: 2 },
			});
			const arrayProgram = linkGpuConstraintProgram(arrayCpu);
			const arrayTokenizer = linkGpuConstraintTokenizer([
				{ id: 0, bytes: constraintEncoder.encode("["), special: false },
				{ id: 1, bytes: constraintEncoder.encode("]"), special: false },
				{ id: 2, bytes: constraintEncoder.encode('"a"'), special: false },
				{ id: 3, bytes: constraintEncoder.encode(',"b"'), special: false },
				{ id: 4, bytes: null, special: true },
			], 4);
			const arrayState = createGpuConstraintDecoderState(arrayProgram);
			const arrayUpload = uploadGpuConstraint(definition, arrayProgram, arrayTokenizer, arrayState);

			const compareArray = async () => {
				const cpu = gpuConstraintMaskReference(arrayProgram, arrayTokenizer, arrayState);
				dispatchGpuConstraintMask(definition);
				await device.queue.onSubmittedWorkDone();
				const gpuMask = await readGpuConstraintMask(definition, arrayUpload.maskWords);
				expect(Array.from(gpuMask)).toEqual(Array.from(cpu));
			};

			await compareArray();
			expect(feedGpuConstraintBytes(arrayProgram, arrayState, constraintEncoder.encode("["))).toBe(true);
			writeGpuConstraintState(definition, arrayState);
			await compareArray();
			expect(feedGpuConstraintBytes(arrayProgram, arrayState, constraintEncoder.encode('"a"'))).toBe(true);
			writeGpuConstraintState(definition, arrayState);
			await compareArray();
			expect(feedGpuConstraintBytes(arrayProgram, arrayState, constraintEncoder.encode("]"))).toBe(true);
			expect(gpuConstraintComplete(arrayProgram, arrayState)).toBe(true);
			writeGpuConstraintState(definition, arrayState);
			await compareArray();
		} finally {
			device.destroy();
			void gpu;
		}
	}, 30_000);
	test("Dawn constrained argmax respects the exact mask and commits VM state", async () => {
		const { create, globals } = await import("webgpu");
		Object.assign(globalThis, globals);

		const gpu = create([]);
		const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
		if (!adapter) throw new Error("Could not acquire a Dawn WebGPU adapter");
		const device = await adapter.requestDevice({ label: "constraint-argmax-aot-test" });

		const definition = defineLfm2();
		definition.engine.deserialize(lfm2Artifact);

		try {
			const compiled = await definition.engine.compile({ device });
			expect(compiled.failed).toBe(0);

			const cpuProgram = compileJsonSchemaProgram({ enum: ["A", "B", "C"] });
			const linked = linkGpuConstraintProgram(cpuProgram);
			const tokenizer = linkGpuConstraintTokenizer([
				{ id: 0, bytes: constraintEncoder.encode('"A"'), special: false },
				{ id: 1, bytes: constraintEncoder.encode('"B"'), special: false },
				{ id: 2, bytes: constraintEncoder.encode('"C"'), special: false },
				{ id: 3, bytes: constraintEncoder.encode('"D"'), special: false },
				{ id: 4, bytes: null, special: true },
			], 4);
			const state = createGpuConstraintDecoderState(linked);
			uploadGpuConstraint(definition, linked, tokenizer, state);

			definition.resources.runtime.write({
				contextCapacity: definition.capacities.context,
				maxNewTokens: 4,
				eosToken: 4,
				promptTokenCount: 1,
				position: 1,
				generatedCount: 0,
				currentToken: 0,
				status: "running",
				telemetryRevision: 0,
				lastToken: 0,
				errorCode: 0,
				pad0: 0,
			});
			definition.resources.op.write({
				inputOffset: LFM2_ARENA.logits,
				inputDim: 5,
				u0: 2, // synthetic EMPTY_TOKEN: schema allows token 2, sampler still must not emit it
				mode: "prefill",
			});

			// Token 3 is forbidden by the schema and token 2 is schema-allowed but reserved by op.u0.
			// Both outrank B, so selecting token 1 proves mask + global sentinel are applied together.
			device.queue.writeBuffer(
				definition.resources.arena.gpu,
				LFM2_ARENA.logits * 4,
				new Float32Array([2, 7, 10_000, 20_000, -1000]),
			);

			dispatchGpuConstraintMask(definition);
			definition.engine.submit((encoder) => {
				encoder.compute({ label: "constraint.argmax.test" }, (pass) => {
					pass.run(
						definition.programs.constraint_argmax,
						{ workgroups: [1, 1, 1] },
						{ dynamicOffsets: { op: 0 } } as any,
					);
				});
			});
			await device.queue.onSubmittedWorkDone();

			const tokensAfterValue = await definition.resources.tokens.readback({ dropIfBusy: false }) as any;
			expect(Number(tokensAfterValue[definition.capacities.context])).toBe(1);

			const cpuCommitted = state.slice();
			expect(feedGpuConstraintBytes(linked, cpuCommitted, constraintEncoder.encode('"B"'))).toBe(true);
			expect(gpuConstraintComplete(linked, cpuCommitted)).toBe(true);
			const gpuCommitted = await readGpuConstraintState(definition);
			expect(Array.from(gpuCommitted)).toEqual(Array.from(cpuCommitted));

			// Once the root value is complete the exact mask contains EOS only.
			definition.resources.op.write({
				inputOffset: LFM2_ARENA.logits,
				inputDim: 5,
				u0: 2,
				mode: "decode",
			});
			device.queue.writeBuffer(
				definition.resources.arena.gpu,
				LFM2_ARENA.logits * 4,
				new Float32Array([50_000, 40_000, 30_000, 60_000, -1000]),
			);
			dispatchGpuConstraintMask(definition);
			definition.engine.submit((encoder) => {
				encoder.compute({ label: "constraint.argmax.eos.test" }, (pass) => {
					pass.run(
						definition.programs.constraint_argmax,
						{ workgroups: [1, 1, 1] },
						{ dynamicOffsets: { op: 0 } } as any,
					);
				});
			});
			await device.queue.onSubmittedWorkDone();

			const tokensAfterEos = await definition.resources.tokens.readback({ dropIfBusy: false }) as any;
			expect(Number(tokensAfterEos[definition.capacities.context + 1])).toBe(4);
			const runtime = await definition.resources.runtime.readback({ dropIfBusy: false }) as any;
			expect(String(runtime.status)).toBe("eos");
			const stateAfterEos = await readGpuConstraintState(definition);
			expect(Array.from(stateAfterEos)).toEqual(Array.from(cpuCommitted));
		} finally {
			device.destroy();
			void gpu;
		}
	}, 30_000);
});

describe("context checkpoints", () => {
	test("checkpoint + continuation is exactly equivalent to full context", async () => {
		const model = await loadModel(
			"./models/LFM2.5-1.2B-Instruct-WQ4.wq4",
		);
		const engine = model.engine;

		try {
			const a = await engine.putBlock(tokens(BOS, 42));
			const b = await engine.putBlock(tokens(43, 44));
			const c = await engine.putBlock(tokens(45, 46));

			const checkpoint = await engine.checkpoint({
				blocks: [a, b],
			});

			const fromCheckpoint = await collect(
				engine.generate(
					{
						checkpoint,
						blocks: [c],
					},
					GENERATE,
				),
			);

			const fromScratch = await collect(
				engine.generate(
					{
						blocks: [a, b, c],
					},
					GENERATE,
				),
			);

			expect(fromCheckpoint).toEqual(fromScratch);
		} finally {
			await model.dispose();
		}
	});

	test("one checkpoint can branch without mutating its state", async () => {
		const model = await loadModel(
			"./models/LFM2.5-1.2B-Instruct-WQ4.wq4",
		);
		const engine = model.engine;

		try {
			const baseA = await engine.putBlock(tokens(BOS, 10, 11));
			const baseB = await engine.putBlock(tokens(12, 13));

			const branchA = await engine.putBlock(tokens(100, 101));
			const branchB = await engine.putBlock(tokens(200, 201));

			const checkpoint = await engine.checkpoint({
				blocks: [baseA, baseB],
			});

			const a = await collect(
				engine.generate(
					{
						checkpoint,
						blocks: [branchA],
					},
					GENERATE,
				),
			);

			const b = await collect(
				engine.generate(
					{
						checkpoint,
						blocks: [branchB],
					},
					GENERATE,
				),
			);

			const expectedA = await collect(
				engine.generate(
					{
						blocks: [baseA, baseB, branchA],
					},
					GENERATE,
				),
			);

			const expectedB = await collect(
				engine.generate(
					{
						blocks: [baseA, baseB, branchB],
					},
					GENERATE,
				),
			);

			expect(a).toEqual(expectedA);
			expect(b).toEqual(expectedB);

			// Branch A must not have mutated the checkpoint used by branch B.
			const aAgain = await collect(
				engine.generate(
					{
						checkpoint,
						blocks: [branchA],
					},
					GENERATE,
				),
			);

			expect(aAgain).toEqual(a);
		} finally {
			await model.dispose();
		}
	});

	test("checkpoints can be extended into new checkpoints", async () => {
		const model = await loadModel(
			"./models/LFM2.5-1.2B-Instruct-WQ4.wq4",
		);
		const engine = model.engine;

		try {
			const a = await engine.putBlock(tokens(BOS, 20));
			const b = await engine.putBlock(tokens(21, 22));
			const c = await engine.putBlock(tokens(23, 24));
			const d = await engine.putBlock(tokens(25, 26));

			const ab = await engine.checkpoint({
				blocks: [a, b],
			});

			const abc = await engine.checkpoint({
				checkpoint: ab,
				blocks: [c],
			});

			const chained = await collect(
				engine.generate(
					{
						checkpoint: abc,
						blocks: [d],
					},
					GENERATE,
				),
			);

			const direct = await collect(
				engine.generate(
					{
						blocks: [a, b, c, d],
					},
					GENERATE,
				),
			);

			expect(chained).toEqual(direct);
		} finally {
			await model.dispose();
		}
	});

	test("materialized checkpoint survives dropping its source blocks", async () => {
		const model = await loadModel(
			"./models/LFM2.5-1.2B-Instruct-WQ4.wq4",
		);
		const engine = model.engine;

		try {
			const a = await engine.putBlock(tokens(BOS, 30, 31));
			const b = await engine.putBlock(tokens(32, 33));
			const tail = await engine.putBlock(tokens(34));

			const checkpoint = await engine.checkpoint({
				blocks: [a, b],
			});

			const before = await collect(
				engine.generate(
					{
						checkpoint,
						blocks: [tail],
					},
					GENERATE,
				),
			);

			await engine.dropBlock(a);
			await engine.dropBlock(b);

			const after = await collect(
				engine.generate(
					{
						checkpoint,
						blocks: [tail],
					},
					GENERATE,
				),
			);

			expect(after).toEqual(before);
		} finally {
			await model.dispose();
		}
	});

	test("checkpoint at every prefix length matches uninterrupted inference", async () => {
		const model = await loadModel(
			"./models/LFM2.5-1.2B-Instruct-WQ4.wq4",
		);
		const engine = model.engine;

		try {
			const sequence = [
				BOS,
				61,
				62,
				63,
				64,
				65,
				66,
				67,
			];

			for (let split = 1; split < sequence.length; split++) {
				const leftTokens = new Uint32Array(sequence.slice(0, split));
				const rightTokens = new Uint32Array(sequence.slice(split));

				const left = await engine.putBlock(leftTokens);
				const right = await engine.putBlock(rightTokens);

				const checkpoint = await engine.checkpoint({
					blocks: [left],
				});

				const resumed = await collect(
					engine.generate(
						{
							checkpoint,
							blocks: [right],
						},
						GENERATE,
					),
				);

				const whole = await engine.putBlock(
					new Uint32Array(sequence),
				);

				const uninterrupted = await collect(
					engine.generate(
						{
							blocks: [whole],
						},
						GENERATE,
					),
				);

				expect(resumed).toEqual(uninterrupted);

				await engine.dropCheckpoint(checkpoint);
				await engine.dropBlock(left);
				await engine.dropBlock(right);
				await engine.dropBlock(whole);
			}
		} finally {
			await model.dispose();
		}
	});

	test("checkpoint does not prefill its prefix again", async () => {
		const model = await loadModel(
			"./models/LFM2.5-1.2B-Instruct-WQ4.wq4",
		);
		const engine = model.engine;

		try {
			const prefixTokens = tokens(BOS, 40, 41, 42, 43, 44);
			const tailTokens = tokens(50, 51);

			const prefix = await engine.putBlock(prefixTokens);
			const tail = await engine.putBlock(tailTokens);

			const checkpoint = await engine.checkpoint({
				blocks: [prefix],
			});

			engine.debug.resetStats();

			await collect(
				engine.generate(
					{
						checkpoint,
						blocks: [tail],
					},
					{
						maxTokens: 1,
						sampler: "argmax",
					},
				),
			);

			const stats = engine.debug.stats();

			// Critical invariant:
			// prefix already exists as KV/conv state and MUST NOT be recomputed.
			expect(stats.prefillTokens).toBe(tailTokens.length);

			// Useful additional assertions once available:
			expect(stats.checkpointHits).toBe(1);
			expect(stats.checkpointMisses).toBe(0);
		} finally {
			await model.dispose();
		}
	});

	test("checkpoint stores only populated KV prefix", async () => {
		const model = await loadModel(
			"./models/LFM2.5-1.2B-Instruct-WQ4.wq4",
		); 
		const engine = model.engine;

		try {
			const short = await engine.putBlock(
				new Uint32Array([BOS, 10, 11, 12]),
			);

			const long = await engine.putBlock(
				new Uint32Array([
					BOS,
					...Array.from({ length: 63 }, (_, i) => 100 + i),
				]),
			);

			engine.debug.resetStats();

			const shortCp = await engine.checkpoint({
				blocks: [short],
			});

			const shortStats = engine.debug.stats();

			engine.debug.resetStats();

			const longCp = await engine.checkpoint({
				blocks: [long],
			});

			const longStats = engine.debug.stats();

			expect(shortStats.checkpointBytes).toBeGreaterThan(0);
			expect(longStats.checkpointBytes).toBeGreaterThan(
				shortStats.checkpointBytes,
			);

			// Najważniejsze:
			// nie możemy snapshotować całego contextCapacity niezależnie od position.
			expect(longStats.kvBytes).toBeGreaterThan(shortStats.kvBytes);

			await engine.dropCheckpoint(shortCp);
			await engine.dropCheckpoint(longCp);
		} finally {
			await model.dispose();
		}
	});

});
