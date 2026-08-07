// Where does a real generation spend time outside the GPU?
//
// decode-profile.test.ts splits a decode step into four passes and reads the
// probe after each one, which forces a CPU/GPU sync per step. Its wall-vs-GPU
// gap is therefore partly the measurement. generateGreedy() does the opposite:
// one submit, one compute pass, every decode step encoded up front. This test
// decomposes that real path into
//
//   encode  - CPU time inside executor.submit() building the command buffer
//   gpu     - the pass duration the GPU reports
//   readback- everything after: queue drain, buffer maps, token decode
//
// so it is clear whether shrinking the per-dispatch cost is worth anything.
import { expect, test } from "bun:test";
import { loadModel } from "../src";
import { lfm2 } from "../packages/webgpu/src/lfm2.ts";

const MODEL = "./models/LFM2.5-1.2B-Instruct-WQ4.wq4";
const TOKEN_COUNTS = [8, 32, 128];
const SAMPLES = 5;

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[sorted.length >> 1] ?? Number.NaN;
}

test("generateGreedy: CPU encode versus GPU execution", async () => {
  const model = await loadModel(MODEL);
  const forward = model.forward;
  if (!forward) throw new Error("real GPU engine required");

  const probe = lfm2.engine.attachSamdoneter({ capacity: 8, label: "decode.overhead" });
  if (!probe) {
    console.log("[decode-overhead] timestamp-query unavailable; skipping");
    await model.dispose();
    return;
  }

  try {
    await forward.prepareAll();
    const prompt = Uint32Array.of(forward.model.config.bosToken, 42, 43, 44);

    // Instrument the encode phase without touching Lfm2Forward: submit() is
    // synchronous CPU work, so wrapping the executor measures exactly it.
    const executor = forward.executor as unknown as { submit: (cb: unknown) => void };
    const originalSubmit = executor.submit.bind(executor);
    let encodeMs = 0;
    executor.submit = (callback: unknown) => {
      const started = performance.now();
      originalSubmit(callback);
      encodeMs = performance.now() - started;
    };

    const rows: Record<string, unknown>[] = [];
    for (const maxNewTokens of TOKEN_COUNTS) {
      const encode: number[] = [];
      const gpu: number[] = [];
      const wall: number[] = [];

      for (let s = 0; s < SAMPLES + 1; s++) {
        const started = performance.now();
        encodeMs = 0;
        await forward.generateGreedy(prompt, { maxNewTokens, resetState: true });
        const totalMs = performance.now() - started;
        const report = await probe.read();
        if (s === 0) continue; // warmup
        encode.push(encodeMs);
        gpu.push(report.gpuNs / 1e6);
        wall.push(totalMs);
      }

      const e = median(encode);
      const g = median(gpu);
      const w = median(wall);
      rows.push({
        maxNewTokens,
        wallMs: Number(w.toFixed(2)),
        encodeMs: Number(e.toFixed(2)),
        gpuMs: Number(g.toFixed(2)),
        otherMs: Number((w - e - g).toFixed(2)),
        encodePctOfWall: Number(((e / w) * 100).toFixed(1)),
        wallMsPerToken: Number((w / maxNewTokens).toFixed(3)),
        encodeUsPerToken: Number(((e * 1000) / maxNewTokens).toFixed(1)),
      });
    }

    executor.submit = originalSubmit;
    console.log("[decode-overhead] " + JSON.stringify({
      dispatchesPerToken: "~250",
      rows,
    }, null, 2));

    expect(rows.length).toBe(TOKEN_COUNTS.length);
    for (const row of rows) expect(row.gpuMs).toBeGreaterThan(0);
  } finally {
    lfm2.engine.detachSamdoneter();
    await model.dispose();
  }
}, 300_000);
