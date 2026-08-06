import { expect, test } from "bun:test";
import { Engine } from "@chomato/engine-ts";
import { createLfm2WebGpuTransport, type Lfm2GenerationRuntime, type Lfm2RuntimeCheckpoint } from "../src/engine-transport";

class FakeCheckpoint implements Lfm2RuntimeCheckpoint {
  destroyed = false;
  constructor(readonly position: number, readonly byteLength = 1234) {}
  destroy(): void { this.destroyed = true; }
}

class FakeRuntime implements Lfm2GenerationRuntime {
  readonly freshPrompts: number[][] = [];
  readonly checkpointTails: number[][] = [];
  readonly created: Array<{ tail: number[]; base?: Lfm2RuntimeCheckpoint }> = [];

  async generateGreedy(promptTokens: Uint32Array | readonly number[]) {
    const prompt = Array.from(promptTokens);
    this.freshPrompts.push(prompt);
    return {
      tokens: [700],
      execution: { prefillTokens: prompt.length, restoredCheckpointBytes: 0 },
    };
  }

  async createCheckpoint(
    tailTokens: Uint32Array | readonly number[],
    base?: Lfm2RuntimeCheckpoint,
  ): Promise<Lfm2RuntimeCheckpoint> {
    const tail = Array.from(tailTokens);
    this.created.push({ tail, base });
    return new FakeCheckpoint((base?.position ?? 0) + tail.length);
  }

  async generateGreedyFromCheckpoint(
    checkpoint: Lfm2RuntimeCheckpoint,
    tailTokens: Uint32Array | readonly number[],
  ) {
    const tail = Array.from(tailTokens);
    this.checkpointTails.push(tail);
    expect(checkpoint.position).toBe(3);
    return {
      tokens: [701],
      execution: { prefillTokens: tail.length, restoredCheckpointBytes: checkpoint.byteLength },
    };
  }
}

async function collect(values: AsyncIterable<number>): Promise<number[]> {
  const result: number[] = [];
  for await (const value of values) result.push(value);
  return result;
}

test("checkpoint generation computes only the appended blocks and reports backend truth", async () => {
  const runtime = new FakeRuntime();
  const engine = new Engine(createLfm2WebGpuTransport(runtime));

  const prefix = await engine.putBlock(Uint32Array.of(1, 2, 3));
  const checkpoint = await engine.checkpoint({ blocks: [prefix] });
  const tail = await engine.putBlock(Uint32Array.of(4, 5));

  engine.debug.resetStats();
  const generated = await collect(engine.generate({ checkpoint, blocks: [tail] }, { maxTokens: 1 }));

  expect(generated).toEqual([701]);
  expect(runtime.freshPrompts).toEqual([]);
  expect(runtime.checkpointTails).toEqual([[4, 5]]);
  expect(engine.debug.stats()).toMatchObject({
    prefillTokens: 2,
    checkpointHits: 1,
    checkpointMisses: 0,
    restoredCheckpointBytes: 1234,
  });

  await engine.close();
});

test("checkpoint chaining extends physical state without replaying the base prefix", async () => {
  const runtime = new FakeRuntime();
  const engine = new Engine(createLfm2WebGpuTransport(runtime));

  const a = await engine.putBlock(Uint32Array.of(1, 2));
  const b = await engine.putBlock(Uint32Array.of(3));
  const ab = await engine.checkpoint({ blocks: [a, b] });
  const c = await engine.putBlock(Uint32Array.of(4, 5));
  const abc = await engine.checkpoint({ checkpoint: ab, blocks: [c] });

  expect(runtime.created).toHaveLength(2);
  expect(runtime.created[0]!.tail).toEqual([1, 2, 3]);
  expect(runtime.created[0]!.base).toBeUndefined();
  expect(runtime.created[1]!.tail).toEqual([4, 5]);
  expect(runtime.created[1]!.base?.position).toBe(3);

  await engine.dropCheckpoint(ab);
  await engine.dropCheckpoint(abc);
  await engine.close();
});


test("WebGPU transport reports runtime execution facts rather than inferring them from context", async () => {
  class ReportingRuntime extends FakeRuntime {
    override async generateGreedyFromCheckpoint(
      checkpoint: Lfm2RuntimeCheckpoint,
      tailTokens: Uint32Array | readonly number[],
    ) {
      this.checkpointTails.push(Array.from(tailTokens));
      return {
        tokens: [702],
        execution: { prefillTokens: 9, restoredCheckpointBytes: 4321 },
      };
    }
  }

  const runtime = new ReportingRuntime();
  const engine = new Engine(createLfm2WebGpuTransport(runtime));
  const prefix = await engine.putBlock(Uint32Array.of(1, 2, 3));
  const checkpoint = await engine.checkpoint({ blocks: [prefix] });
  const tail = await engine.putBlock(Uint32Array.of(4, 5));

  engine.debug.resetStats();
  await collect(engine.generate({ checkpoint, blocks: [tail] }, { maxTokens: 1 }));

  // Tail has only two tokens. Nine proves telemetry came from the runtime's
  // actual execution report instead of being inferred from the request shape.
  expect(engine.debug.stats()).toMatchObject({
    prefillTokens: 9,
    checkpointHits: 1,
    checkpointMisses: 0,
    restoredCheckpointBytes: 4321,
  });
  await engine.close();
});
