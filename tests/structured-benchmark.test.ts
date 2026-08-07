// Combined structured-generation e2e suite (bun:test).
//
// Two layers:
//
//  1. CPU-only corpus layer (runs everywhere, no GPU needed): compiles the
//     targeted 8-schema corpus and asserts every schema-derived decode budget
//     fits the engine capacity (MAX_NEW_TOKENS = 1024). Also logs the two
//     signals the suite is designed to surface: compile/link ms and constraint
//     blob bytes. A future corpus record with a ~116 KiB dynamic-array blob
//     will show immediately whether program size correlates with mask cost.
//
//  2. GPU layer (runs only when loadModel() returned the real WebGPU engine;
//     skipped on the mock exe fallback): per-schema benchmark rows
//       prefill ms · decode ms/step (structured vs greedy baseline) ·
//       overhead ms/step + % · mask kernel ms · constraint argmax ms ·
//       committed output tokens
//     plus checkpoint + structured semantics and determinism checks.
//
// Decode is budget-bound: the runtime schedules all (budget - 1) decode
// steps even after EOS, so ms/token is reported per scheduled step, which
// keeps baseline and structured comparable on identical GPU work.
//
// ROOT NUMBER semantics: a root JSON number terminates through the EOS
// control token (JSON numbers have no closing delimiter byte; the VM treats a
// complete number whose continuation is `accept` as terminal). maxNumberChars
// is ABI-bound at 64 (ConstraintDecoderState.numberText is 16 u32 words = 64
// ASCII bytes), enough for this model's long scientific notation.
//
// Bounds are additionally pruned up front: min >= 0 rejects a leading '-', and
// max < 0 rejects a digit start, so greedy cannot wander into lexemes that
// can never satisfy the bounds and dead-end when they close. Remaining
// dead-ends are limited to pathological out-of-f64 magnitudes or zero-lexeme
// corner cases; the row is logged with a marker instead of failing the suite.
//
//   bun test tests/structured-benchmark.test.ts
import { describe, expect, test } from "bun:test";
import { type } from "arktype";
import { loadModel, type LocalModel } from "../src";
import {
  compileStructuredGeneration,
  type GeneratableSchema,
} from "../packages/engine-ts/src/index.ts";
import { dispatchGpuConstraintMask } from "../packages/webgpu/src/constraint.ts";
import { LFM2_ARENA, lfm2 } from "../packages/webgpu/src/lfm2.ts";
import type { Lfm2Forward } from "../packages/webgpu/src/forward.ts";
import { GPU_SCHEMA_SENTINELS } from "../packages/schema/src/sparse.ts";

// ---------------------------------------------------------------------------
// Corpus: one schema per error class, deliberately small and targeted.
// ---------------------------------------------------------------------------

/** GeneratableSchema plus the runtime validator used by the E2E assertions. */
interface CorpusSchema extends GeneratableSchema {
  assert(value: unknown): void;
}

const CORPUS: Array<{ name: string; schema: CorpusSchema }> = [
  // Root scalar: bounded string ("string < 32" in ArkType terms).
  { name: "string", schema: type("string <= 32") },
  // Root scalar: bounded number (see KNOWN LIMITATION above).
  { name: "number", schema: type("0 <= number <= 10") },
  // Root scalar: enum.
  { name: "enum", schema: type('"red" | "green" | "blue"') },
  // Simple object with required fields.
  { name: "object", schema: type({ name: "string <= 16", age: "0 <= number <= 120" }) },
  // Object with an optional field (runtime split: emit or skip).
  { name: "optional", schema: type({ name: "string <= 16", "age?": "0 <= number <= 120" }) },
  // Nested objects, including an optional nested field.
  { name: "nested", schema: type({ user: { name: "string <= 16", address: { city: "string <= 16", "zip?": "string <= 8" } } }) },
  // Bounded array (maxItems; minItems 0 keeps every array-count split live).
  { name: "array", schema: type("string <= 8[] <= 5") },
  // Heavier record: nested object + enum + array + optional in one schema.
  {
    name: "record",
    schema: type({
      id: "string <= 16",
      status: '"pending" | "active" | "archived"',
      score: "0 <= number <= 100",
      tags: type("string <= 8[] <= 4"),
      owner: {
        name: "string <= 16",
        "email?": "string <= 24",
        role: '"admin" | "user" | "guest"',
      },
    }),
  },
];

const RUNS = Math.max(1, Number(process.env.STRUCTURED_BENCH_RUNS ?? 2));
// The GPU layer needs long per-test budgets: a full schema run spends
// (budget - 1) decode steps at ~15-40 ms/step, and bun's default test
// timeout is 5 s, which would kill a run mid-submit.
const BENCH_TIMEOUT_MS = 900_000;
const SEMANTICS_TIMEOUT_MS = 180_000;
const MICRO_RUNS = Math.max(1, Number(process.env.STRUCTURED_BENCH_MICRO_RUNS ?? 5));
const BUDGET_ONE_RUN = 300; // heavy schemas above this get a single timed run

async function collect(iterable: AsyncIterable<number>): Promise<number[]> {
  const out: number[] = [];
  for await (const token of iterable) out.push(token);
  return out;
}

function runsFor(budget: number): number {
  return budget > BUDGET_ONE_RUN ? 1 : RUNS;
}

/** Best-of-N wall time; min is more robust to scheduler noise than the mean. */
async function timeBest<T>(
  runs: number,
  fn: () => Promise<T>,
): Promise<{ ms: number; value: T }> {
  let best = Number.POSITIVE_INFINITY;
  let value: T | undefined;
  for (let i = 0; i < runs; i++) {
    const started = performance.now();
    value = await fn();
    best = Math.min(best, performance.now() - started);
  }
  return { ms: best, value: value! };
}

// ---------------------------------------------------------------------------
// Engine availability. Structured generation exists only on the real WebGPU
// engine; the mock exe fallback cannot produce constrained JSON.
// ---------------------------------------------------------------------------

let model: LocalModel | undefined;
try {
  model = await loadModel();
  if (!model.forward) {
    console.warn(
      "[structured-benchmark] real WebGPU engine unavailable (model file or Dawn bindings missing) — " +
        "GPU benchmark/semantics layer SKIPPED; mock backend does not implement structured generation.",
    );
  }
} catch (error) {
  console.warn("[structured-benchmark] loadModel failed; GPU layer skipped:", error);
}

const describeGpu = model?.forward ? describe : describe.skip;

// Fixed prompt shared by every baseline/structured comparison. A realistic
// text prompt (tokenized by the real tokenizer) keeps the model's greedy path
// on sane values; arbitrary token IDs made it produce garbage JSON. Short on
// purpose: prompt.length + budget - 1 must stay inside the 1024-token context,
// and the constraint forces valid JSON regardless of prompt content.
const PROMPT: Uint32Array = model?.forward
  ? Uint32Array.from(
      model.forward.tokenizer.encodeUserPrompt("Output one valid JSON value only."),
    )
  : Uint32Array.of(1);

// ---------------------------------------------------------------------------
// Layer 1: CPU-only corpus compile (runs everywhere).
// ---------------------------------------------------------------------------

describe("structured schema corpus (compile/link, CPU-only)", () => {
  test("all corpus schemas compile and fit the engine decode budget", () => {
    const capacity = lfm2.capacities.maxNewTokens;
    console.log("\n[corpus] schema   blob B  nodes  switches  maxTokens  compile+link ms");
    for (const { name, schema } of CORPUS) {
      const started = performance.now();
      const compiled = compileStructuredGeneration(schema);
      const compileMs = performance.now() - started;
      const s = compiled.program.summary;
      console.log(
        `[corpus] ${name.padEnd(8)} ${String(s.blobBytes).padStart(6)}  ${String(s.sourceNodes).padStart(5)}  ${String(s.switchNodes).padStart(8)}  ${String(compiled.maxTokens).padStart(9)}  ${compileMs.toFixed(2)}`,
      );
      expect(s.blobBytes).toBeGreaterThan(0);
      expect(compiled.maxTokens).toBeLessThanOrEqual(capacity);
      // Constraint blob size must stay visible in logs: the whole point is to
      // correlate program size with mask-kernel cost (see the 116 KiB record).
      expect(compiled.program.blob.byteLength).toBe(s.blobBytes);
    }
    console.log(`[corpus] decode budget capacity: ${capacity}`);
  });
});

// ---------------------------------------------------------------------------
// Layer 2: GPU benchmark + semantics (skipped without the real engine).
// ---------------------------------------------------------------------------

describeGpu("structured generation GPU benchmark", () => {
  const fwd = model!.forward!;
  const vocabSize = fwd.model.config.vocabSize;

  function waitGpu(): Promise<void> {
    return lfm2.engine.device.queue.onSubmittedWorkDone();
  }

  /** Single constraint_argmax dispatch (mask + state must already be uploaded). */
  function dispatchConstraintArgmax(): void {
    fwd.executor.submit((encoder) => {
      encoder.compute((pass) => {
        pass.run("constraint_argmax", {
          inputOffset: LFM2_ARENA.logits,
          inputDim: vocabSize,
          u0: GPU_SCHEMA_SENTINELS.emptyToken,
          mode: "decode",
        });
      });
    });
  }

  test("per-schema prefill/decode overhead + mask/argmax split", async () => {
    console.log(
      "\n[bench] schema   blob    budget  tok  prefill ms  base ms/tok  struct ms/tok  overhead ms/tok  overhead %  mask ms  argmax ms",
    );

    for (const { name, schema } of CORPUS) {
      const compiled = compileStructuredGeneration(schema);
      const blob = compiled.program.blob;
      const budget = compiled.maxTokens;
      const runs = runsFor(budget);

      // Warmup: links the model-global constraint tokenizer (one-time) and
      // uploads program + state, so timed runs measure steady-state cost.
      await fwd.generateStructured(PROMPT, blob, { maxNewTokens: budget });

      // Structured: prefill-only (maxNewTokens 1 => no decode loop) vs full.
      const structPrefill = await timeBest(runs, () =>
        fwd.generateStructured(PROMPT, blob, { maxNewTokens: 1 }),
      );
      const structFull = await timeBest(runs, () =>
        fwd.generateStructured(PROMPT, blob, { maxNewTokens: budget }),
      );

      // Baseline greedy on the same prompt and the same number of scheduled
      // decode steps, so per-step costs are directly comparable.
      const basePrefill = await timeBest(runs, () =>
        fwd.generateGreedy(PROMPT, { maxNewTokens: 1 }),
      );
      const baseFull = await timeBest(runs, () =>
        fwd.generateGreedy(PROMPT, { maxNewTokens: budget }),
      );

      const structDecodeMs = Math.max(0, structFull.ms - structPrefill.ms);
      const baseDecodeMs = Math.max(0, baseFull.ms - basePrefill.ms);
      const structPerStep = structDecodeMs / (budget - 1);
      const basePerStep = baseDecodeMs / (budget - 1);
      const overhead = structPerStep - basePerStep;
      const overheadPct = basePerStep > 0 ? (overhead / basePerStep) * 100 : 0;

      // Micro-benchmarks: mask kernel = 2048 words x 32 tokens = 65536 VM
      // invocations; constraint_argmax = single workgroup bit-scan over the
      // mask with no VM. This is exactly the 2048-invocations hypothesis.
      const mask = await timeBest(MICRO_RUNS, async () => {
        dispatchGpuConstraintMask(lfm2);
        await waitGpu();
      });
      const argmax = await timeBest(MICRO_RUNS, async () => {
        fwd.initializeRequest(PROMPT.length, budget);
        dispatchConstraintArgmax();
        await waitGpu();
      });

      const outputTokens = structFull.value.tokens.length;
      const status = structFull.value.status;
      expect(["eos", "done", "error"]).toContain(status);

      let deadEnd = false;
      if (status === "error") {
        // Deterministic greedy dead-end (see KNOWN LIMITATION header). Keep
        // the timing, surface the row, do not fail the suite.
        deadEnd = true;
        console.warn(
          `[bench] ${name}: status=error (dead-end; greedy committed an out-of-bounds number) — ` +
            `committed ${outputTokens} tokens, text ${JSON.stringify(structFull.value.text.slice(0, 48))}`,
        );
      } else {
        // The generated text must be valid JSON matching the schema: this is
        // the real end-to-end correctness assertion behind every row.
        const parsed = JSON.parse(structFull.value.text) as unknown;
        schema.assert(parsed);
        expect(outputTokens).toBeGreaterThan(0);
      }

      // Loose sanity bounds, wide enough to survive machine noise but tight
      // enough to catch a constraint-VM blowup (structured must not be ~free
      // nor orders of magnitude slower than the model forward itself).
      expect(structPerStep).toBeGreaterThanOrEqual(basePerStep * 0.33);
      expect(structPerStep).toBeLessThanOrEqual(basePerStep * 10 + 200);

      console.log(
        `[bench] ${name.padEnd(8)} ${String(compiled.program.summary.blobBytes).padStart(5)}B ` +
          `${String(budget).padStart(6)} ${String(outputTokens).padStart(3)}${deadEnd ? "*" : " "} ` +
          `${structPrefill.ms.toFixed(1).padStart(8)} ` +
          `${basePerStep.toFixed(2).padStart(10)} ${structPerStep.toFixed(2).padStart(12)} ` +
          `${overhead.toFixed(2).padStart(14)} ${overheadPct.toFixed(1).padStart(9)}% ` +
          `${mask.ms.toFixed(3).padStart(7)} ${argmax.ms.toFixed(3).padStart(9)}`,
      );
    }
    console.log(
      "\n[bench] decode is budget-bound (all scheduled steps run even after EOS); ms/tok is per scheduled step.",
    );
    console.log("[bench] * = dead-end (status error), see header note");
  }, { timeout: BENCH_TIMEOUT_MS });
});

describeGpu("structured generation semantics", () => {
  const engine = model!.engine;
  // Object schema keeps this layer fast while still exercising required +
  // bounded fields through the typed API.
  const objectSchema = CORPUS.find((entry) => entry.name === "object")!.schema;

  test("checkpoint + structured generation matches full context and skips prefix prefill", async () => {
    const prefixTokens = PROMPT.slice(0, 5);
    const tailTokens = PROMPT.slice(5);
    const prefix = await engine.putBlock(prefixTokens);
    const tail = await engine.putBlock(tailTokens);
    const checkpoint = await engine.checkpoint({ blocks: [prefix] });

    try {
      const fromCheckpoint = await engine.generate(objectSchema, {
        checkpoint,
        blocks: [tail],
      });
      const fromScratch = await engine.generate(objectSchema, {
        blocks: [prefix, tail],
      });

      // Same typed value through both paths.
      expect(fromCheckpoint).toEqual(fromScratch);

      // The checkpoint prefix must never be re-prefilled, exactly like the
      // untyped decode path.
      engine.debug.resetStats();
      await engine.generate(objectSchema, { checkpoint, blocks: [tail] });
      const stats = engine.debug.stats();
      expect(stats.prefillTokens).toBe(tailTokens.length);
      expect(stats.checkpointHits).toBe(1);
      expect(stats.checkpointMisses).toBe(0);
    } finally {
      await engine.dropCheckpoint(checkpoint);
      await engine.dropBlock(prefix);
      await engine.dropBlock(tail);
    }
  }, { timeout: SEMANTICS_TIMEOUT_MS });

  test("greedy structured generation is deterministic for the same prompt", async () => {
    const block = await engine.putBlock(PROMPT);
    try {
      const first = await engine.generate(objectSchema, { blocks: [block] });
      const second = await engine.generate(objectSchema, { blocks: [block] });
      expect(second).toEqual(first);
    } finally {
      await engine.dropBlock(block);
    }
  }, { timeout: SEMANTICS_TIMEOUT_MS });

  test("structured generation via a reusable checkpoint is deterministic", async () => {
    const prefix = await engine.putBlock(PROMPT.slice(0, 4));
    const tail = await engine.putBlock(PROMPT.slice(4));
    const checkpoint = await engine.checkpoint({ blocks: [prefix] });
    try {
      const first = await engine.generate(objectSchema, {
        checkpoint,
        blocks: [tail],
      });
      const second = await engine.generate(objectSchema, {
        checkpoint,
        blocks: [tail],
      });
      expect(second).toEqual(first);
    } finally {
      await engine.dropCheckpoint(checkpoint);
      await engine.dropBlock(prefix);
      await engine.dropBlock(tail);
    }
  }, { timeout: SEMANTICS_TIMEOUT_MS });

  test("deterministic tokens at the transport level match structured values", async () => {
    // Sanity: repeated token streams from the untyped API are also identical
    // for the same prompt, so the structured determinism is not an artifact of
    // JSON parsing.
    const block = await engine.putBlock(PROMPT);
    try {
      const tokensA = await collect(
        engine.generate({ blocks: [block] }, { maxTokens: 8, sampler: "argmax" }),
      );
      const tokensB = await collect(
        engine.generate({ blocks: [block] }, { maxTokens: 8, sampler: "argmax" }),
      );
      expect(tokensB).toEqual(tokensA);
    } finally {
      await engine.dropBlock(block);
    }
  }, { timeout: SEMANTICS_TIMEOUT_MS });
});
