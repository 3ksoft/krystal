import { Engine, type BlockId, type CheckpointId } from "../src/transport";

async function collect(values: AsyncIterable<number>): Promise<number[]> {
  const result: number[] = [];
  for await (const value of values) result.push(value);
  return result;
}

function equalTokens(actual: readonly number[], expected: readonly number[], label: string): void {
  if (actual.length !== expected.length) {
    throw new Error(`${label}: length ${actual.length} !== ${expected.length}`);
  }
  for (let i = 0; i < actual.length; i++) {
    if (actual[i] !== expected[i]) {
      throw new Error(`${label}: token ${i}: ${actual[i]} !== ${expected[i]}`);
    }
  }
}

/**
 * Backend-neutral checkpoint contract. It intentionally uses only engine-ts.
 * Run the same function against WebGPU and native/binary transports.
 *
 * A real checkpoint implementation must pass the semantic equality checks AND
 * report only the appended tail as freshly prefetched. A token-prefix replay
 * implementation will fail the prefillTokens assertion when telemetry is honest.
 */
export async function runCheckpointContract(engine: Engine): Promise<{
  readonly checkpoint: CheckpointId;
  readonly resumed: readonly number[];
  readonly stats: ReturnType<typeof engine.debug.stats>;
}> {
  const resources: Array<{ kind: "block"; id: BlockId } | { kind: "checkpoint"; id: CheckpointId }> = [];
  try {
    const prefixTokens = Uint32Array.of(1, 42, 43, 44, 45);
    const tailTokens = Uint32Array.of(46, 47);
    const wholeTokens = new Uint32Array(prefixTokens.length + tailTokens.length);
    wholeTokens.set(prefixTokens);
    wholeTokens.set(tailTokens, prefixTokens.length);

    const prefix = await engine.putBlock(prefixTokens);
    resources.push({ kind: "block", id: prefix });
    const tail = await engine.putBlock(tailTokens);
    resources.push({ kind: "block", id: tail });
    const whole = await engine.putBlock(wholeTokens);
    resources.push({ kind: "block", id: whole });

    const checkpoint = await engine.checkpoint({ blocks: [prefix] });
    resources.push({ kind: "checkpoint", id: checkpoint });

    engine.debug.resetStats();
    const resumed = await collect(engine.generate(
      { checkpoint, blocks: [tail] },
      { maxTokens: 2, sampler: "argmax" },
    ));
    const stats = engine.debug.stats();

    if (stats.prefillTokens !== tailTokens.length) {
      throw new Error(
        `checkpoint replayed its prefix: actual fresh prefill=${stats.prefillTokens}, expected tail=${tailTokens.length}`,
      );
    }
    if (stats.checkpointHits !== 1 || stats.checkpointMisses !== 0) {
      throw new Error(
        `checkpoint telemetry mismatch: hits=${stats.checkpointHits}, misses=${stats.checkpointMisses}`,
      );
    }
    if (stats.restoredCheckpointBytes <= 0) {
      throw new Error("checkpoint reported a hit without restoring physical state");
    }

    const uninterrupted = await collect(engine.generate(
      { blocks: [whole] },
      { maxTokens: 2, sampler: "argmax" },
    ));
    equalTokens(resumed, uninterrupted, "checkpoint + tail vs uninterrupted context");

    // The checkpoint is immutable and can be reused after another generation.
    const resumedAgain = await collect(engine.generate(
      { checkpoint, blocks: [tail] },
      { maxTokens: 2, sampler: "argmax" },
    ));
    equalTokens(resumedAgain, resumed, "checkpoint reuse");

    // Materialization owns its state; source blocks may be released.
    await engine.dropBlock(prefix);
    resources.splice(resources.findIndex((r) => r.kind === "block" && r.id === prefix), 1);
    const afterDrop = await collect(engine.generate(
      { checkpoint, blocks: [tail] },
      { maxTokens: 2, sampler: "argmax" },
    ));
    equalTokens(afterDrop, resumed, "checkpoint after source block release");

    return { checkpoint, resumed, stats };
  } finally {
    for (let i = resources.length - 1; i >= 0; i--) {
      const resource = resources[i]!;
      try {
        if (resource.kind === "checkpoint") await engine.dropCheckpoint(resource.id);
        else await engine.dropBlock(resource.id);
      } catch {
        // Contract failures should not be hidden by best-effort cleanup.
      }
    }
  }
}
