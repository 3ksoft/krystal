// Where does a decode step actually go?
//
// generateGreedy() runs prefill and every decode step inside ONE compute pass,
// and WebGPU stamps timestamps only at pass boundaries — so the shipped
// schedule can only ever report one number. This test re-encodes the same work
// with L1 pass boundaries (embed / 16 blocks / LM head / argmax) purely to
// measure it. The production path is not modified; the split exists only here.
//
// The number under test is the LM head's share, because that is the ceiling on
// what sparse LM-head execution could ever save.
import { expect, test } from "bun:test";
import { loadModel } from "../src";
import { LFM2_ARENA, lfm2 } from "../packages/webgpu/src/lfm2.ts";

const MODEL = "./models/LFM2.5-1.2B-Instruct-WQ4.wq4";
const WARMUP = 3;
const SAMPLES = 10;

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[sorted.length >> 1] ?? Number.NaN;
}

test("decode step: LM head share versus the block stack", async () => {
  const model = await loadModel(MODEL);
  const forward = model.forward;
  if (!forward) throw new Error("real GPU engine required");

  const probe = lfm2.engine.attachSamdoneter({ capacity: 64, label: "lfm2.decode" });
  if (!probe) {
    console.log("[decode-profile] timestamp-query unavailable; skipping");
    await model.dispose();
    return;
  }

  try {
    await forward.prepareAll();
    const blockCount = model.forward!.model.config.blockCount;

    // One decode step, re-encoded with pass boundaries so each stage is timed.
    const step = (): void => {
      forward.executor.submit((encoder) => {
        encoder.compute((pass) => forward.embed(pass, 1, "decode"), { label: "embed" });
        encoder.compute((pass) => forward.layers(pass, 0, blockCount, 1, { mode: "decode" }), { label: "blocks" });
        encoder.compute((pass) => forward.projectLogits(pass, 1, "decode", LFM2_ARENA), { label: "lm_head" });
        encoder.compute((pass) => forward.commitArgmax(pass, "decode"), { label: "argmax" });
      });
    };

    forward.initializeRequest(1, 8);
    forward.writeTokens(Uint32Array.of(model.forward!.model.config.bosToken), 0);

    for (let i = 0; i < WARMUP; i++) {
      step();
      await probe.read();
    }

    const byLabel = new Map<string, number[]>();
    const totals: number[] = [];
    const walls: number[] = [];
    let resolutionNs = 0;
    for (let i = 0; i < SAMPLES; i++) {
      step();
      const report = await probe.read();
      expect(report.skipped).toBe(0);
      for (const pass of report.passes) {
        const list = byLabel.get(pass.label) ?? [];
        list.push(pass.durationNs / 1e6);
        byLabel.set(pass.label, list);
      }
      totals.push(report.gpuNs / 1e6);
      walls.push(report.wallMs);
      resolutionNs = Math.max(resolutionNs, report.resolutionNs);
    }

    const stageMs = [...byLabel].map(([label, samples]) => [label, median(samples)] as const);
    const totalMs = median(totals);
    const lmHeadMs = stageMs.find(([label]) => label === "lm_head")?.[1] ?? 0;
    const blocksMs = stageMs.find(([label]) => label === "blocks")?.[1] ?? 0;

    console.log("[decode-profile] " + JSON.stringify({
      stages: Object.fromEntries(stageMs.map(([l, ms]) => [l, Number(ms.toFixed(3))])),
      gpuTotalMs: Number(totalMs.toFixed(3)),
      wallMs: Number(median(walls).toFixed(3)),
      lmHeadSharePct: Number(((lmHeadMs / totalMs) * 100).toFixed(1)),
      blocksSharePct: Number(((blocksMs / totalMs) * 100).toFixed(1)),
      resolutionNs,
      // 2048x65536 WQ4 at 20 B per 32 values.
      lmHeadMiB: Number(((2048 * 65536 / 32 * 20) / 1048576).toFixed(1)),
      lmHeadGiBs: Number((((2048 * 65536 / 32 * 20) / 1073741824) / (lmHeadMs / 1000)).toFixed(1)),
    }, null, 2));

    // Every stage must have been stamped, otherwise the shares are meaningless.
    expect(byLabel.size).toBe(4);
    expect(totalMs).toBeGreaterThan(0);
    expect(lmHeadMs).toBeGreaterThan(0);
  } finally {
    lfm2.engine.detachSamdoneter();
    await model.dispose();
  }
}, 240_000);
