// WQ4 matmul kernel bake-off, across every shape the model actually uses.
//
// The bandwidth floor (tests/bandwidth-floor.test.ts) showed the memory path
// delivers ~300 GiB/s while the shipped matmul gets ~21.7 GiB/s, so the kernel
// is the bottleneck. This harness iterates kernel shapes without rebuilding the
// LFM2 artifact or touching the model definition.
//
// Two things make the numbers trustworthy:
//
//   - `baseline` is a literal copy of the shipped matmul_wq4 body, so its
//     result can be checked against the in-model measurement (21.7 GiB/s).
//   - Every variant is verified against the baseline on real data before it is
//     timed. Against zeroed buffers a kernel that reads nothing looks fastest.
//
// All four shapes are measured because the block stack is 88% of a decode step
// and its matrices are much smaller than the LM head; a tiling that wins on the
// LM head and loses on 2048-row matrices would be a net regression.
import { expect, test } from "bun:test";
import { scope } from "arktype";
import { wgsl } from "@schema-pop/schema";
import { Sandblaster } from "@sandblaster/core";
import { createWebGpuDevice } from "../packages/webgpu/src/device.ts";
import { installDawn } from "./dawn.ts";

const WG = 64;
const WARMUP = 3;
const SAMPLES = 9;

// Lfm2GpuModel pages weights at 64 MiB and at maxComputeWorkgroupsPerDimension,
// so the 65536-row LM head ships as two dispatches. One page is measured here:
// same kernel, same access pattern, no cross-page write races.
const SHAPES = [
  { name: "lm_head", inputDim: 2048, outputDim: 52428 },
  { name: "attn_q/out", inputDim: 2048, outputDim: 2048 },
  { name: "ffn_gate/up", inputDim: 2048, outputDim: 8192 },
  { name: "ffn_down", inputDim: 8192, outputDim: 2048 },
  // 6144 is the only output width between the two regimes, and it carries 11%
  // of the model's bytes. Picking a rows8/rows16 threshold without it would be
  // an extrapolation across exactly the gap the threshold sits in.
  { name: "conv_in_proj", inputDim: 2048, outputDim: 6144 },
] as const;

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[sorted.length >> 1] ?? Number.NaN;
}

/** Literal copy of packages/webgpu/src/shaders/matmul_wq4.wgsl. */
const BASELINE = `
  let localRow = wid.x;
  let tokenIndex = wid.y;
  if (localRow >= op.rowCount || tokenIndex >= op.tokenCount) { return; }

  let inputBase = op.inputOffset + tokenIndex * op.inputDim;
  let blocksPerRow = op.inputDim / 32u;
  let rowBlockStart = localRow * blocksPerRow;

  var sum: f32 = 0.0;
  var b = lid.x;
  loop {
    if (b >= blocksPerRow) { break; }
    let baseU32 = (rowBlockStart + b) * 5u;
    let scale = exp2(f32(bitcast<i32>(weightRaw[baseU32 + 4u])));
    let kStart = b * 32u;
    for (var w = 0u; w < 4u; w++) {
      let packed = weightRaw[baseU32 + w];
      let kBase = kStart + w * 8u;
      sum += (f32((packed >>  0u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 0u];
      sum += (f32((packed >>  4u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 1u];
      sum += (f32((packed >>  8u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 2u];
      sum += (f32((packed >> 12u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 3u];
      sum += (f32((packed >> 16u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 4u];
      sum += (f32((packed >> 20u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 5u];
      sum += (f32((packed >> 24u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 6u];
      sum += (f32((packed >> 28u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 7u];
    }
    b += ${WG}u;
  }

  reduceF32[lid.x] = sum;
  workgroupBarrier();
  var width = ${WG}u >> 1u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) { reduceF32[lid.x] += reduceF32[lid.x + width]; }
    workgroupBarrier();
    width >>= 1u;
  }
  if (lid.x == 0u) {
    arena[op.outputOffset + tokenIndex * op.outputDim + op.rowStart + localRow] = reduceF32[0];
  }
`;

/**
 * Row tiling: one workgroup owns ROWS output rows.
 *
 * Thread `lid.x` walks the same block index for every row in the tile, so the
 * launch count drops by ROWS and each thread does ROWS times more work between
 * reductions. Written as a loop rather than unrolled — measured at ~5% below
 * the unrolled form, which is a good trade for 40 lines instead of 500.
 */
function tiled(rows: number): string {
  return `
  let rowBase = wid.x * ${rows}u;
  let tokenIndex = wid.y;
  if (rowBase >= op.rowCount || tokenIndex >= op.tokenCount) { return; }

  let inputBase = op.inputOffset + tokenIndex * op.inputDim;
  let blocksPerRow = op.inputDim / 32u;

  var acc: array<f32, ${rows}>;
  for (var r = 0u; r < ${rows}u; r++) { acc[r] = 0.0; }

  var b = lid.x;
  loop {
    if (b >= blocksPerRow) { break; }
    let kStart = b * 32u;
    for (var r = 0u; r < ${rows}u; r++) {
      let base = (rowBase + r) * blocksPerRow * 5u + b * 5u;
      let scale = exp2(f32(bitcast<i32>(weightRaw[base + 4u])));
      var bs: f32 = 0.0;
      for (var w = 0u; w < 4u; w++) {
        let packed = weightRaw[base + w];
        let kBase = kStart + w * 8u;
        bs += (f32((packed >>  0u) & 0x0Fu) - 8.0) * arena[inputBase + kBase + 0u];
        bs += (f32((packed >>  4u) & 0x0Fu) - 8.0) * arena[inputBase + kBase + 1u];
        bs += (f32((packed >>  8u) & 0x0Fu) - 8.0) * arena[inputBase + kBase + 2u];
        bs += (f32((packed >> 12u) & 0x0Fu) - 8.0) * arena[inputBase + kBase + 3u];
        bs += (f32((packed >> 16u) & 0x0Fu) - 8.0) * arena[inputBase + kBase + 4u];
        bs += (f32((packed >> 20u) & 0x0Fu) - 8.0) * arena[inputBase + kBase + 5u];
        bs += (f32((packed >> 24u) & 0x0Fu) - 8.0) * arena[inputBase + kBase + 6u];
        bs += (f32((packed >> 28u) & 0x0Fu) - 8.0) * arena[inputBase + kBase + 7u];
      }
      acc[r] += bs * scale;
    }
    b += ${WG}u;
  }

  for (var r = 0u; r < ${rows}u; r++) {
    reduceF32[lid.x] = acc[r];
    workgroupBarrier();
    var width = ${WG}u >> 1u;
    loop {
      if (width == 0u) { break; }
      if (lid.x < width) { reduceF32[lid.x] += reduceF32[lid.x + width]; }
      workgroupBarrier();
      width >>= 1u;
    }
    if (lid.x == 0u && rowBase + r < op.rowCount) {
      arena[op.outputOffset + tokenIndex * op.outputDim + op.rowStart + rowBase + r] = reduceF32[0];
    }
    workgroupBarrier();
  }
`;
}

const VARIANTS = [
  { name: "baseline", code: BASELINE, rows: 1 },
  { name: "rows4", code: tiled(4), rows: 4 },
  { name: "rows8", code: tiled(8), rows: 8 },
  { name: "rows16", code: tiled(16), rows: 16 },
  { name: "rows32", code: tiled(32), rows: 32 },
] as const;

async function benchShape(
  device: GPUDevice,
  shape: { name: string; inputDim: number; outputDim: number },
): Promise<Record<string, unknown>> {
  const { inputDim, outputDim } = shape;
  const blocksPerRow = inputDim / 32;
  const rowBytes = blocksPerRow * 20;
  const weightWords = outputDim * blocksPerRow * 5;
  const weightBytes = outputDim * rowBytes;
  const arenaElements = inputDim + outputDim + 1024;

  const $ = scope({
    ...wgsl.import(),
    Op: {
      inputOffset: "u32", outputOffset: "u32", tokenCount: "u32", inputDim: "u32",
      outputDim: "u32", rowStart: "u32", rowCount: "u32",
    },
  });
  const engine = Sandblaster.create($);
  const widLid = engine.type({ wid: "workgroup_id", lid: "local_invocation_id" });

  const op = engine.buffer(engine.type("Op"), {
    label: "mm.op",
    value: {
      inputOffset: 0, outputOffset: inputDim, tokenCount: 1, inputDim,
      outputDim, rowStart: 0, rowCount: outputDim,
    },
  });
  const arena = engine.buffer(engine.type(`f32[] == ${arenaElements}`), { label: "mm.arena", readback: true });
  // Placeholder, always overridden — the `lfm2.probe-weight-raw` trick.
  const weightRaw = engine.buffer(engine.type("u32"), { label: "mm.weights.probe", count: 2 });

  const programs = VARIANTS.map((variant) =>
    engine.compute({
      label: `${shape.name}:${variant.name}`,
      // Binding types mirror the shipped kernel: `op` uniform, so its reads are
      // uniform and the early return before a barrier is legal.
      resources: {
        op: { resource: op, buffer: { type: "uniform" } },
        arena,
        weightRaw: { resource: weightRaw, buffer: { type: "read-only-storage" } },
      },
      // Module-scope declarations must arrive as includes; `code` is a body.
      includes: [`var<workgroup> reduceF32: array<f32, ${WG}>;`],
      compute: {
        entryPoint: "main",
        workgroupSize: WG,
        params: widLid,
        code: variant.code,
      },
    } as never)
  );

  const compiled = await engine.compile({ device });
  expect(compiled.failed).toBe(0);

  const weightPage = device.createBuffer({
    label: "mm.weights",
    size: weightBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  let seed = 0x2545f491;
  const nextWord = (): number => {
    seed ^= seed << 13; seed >>>= 0;
    seed ^= seed >>> 17;
    seed ^= seed << 5; seed >>>= 0;
    return seed;
  };
  const words = new Uint32Array(weightWords);
  for (let i = 0; i < words.length; i++) {
    // Every 5th word is the block exponent; keep it small so sums stay finite.
    words[i] = i % 5 === 4 ? ((i % 7) - 3) >>> 0 : nextWord();
  }
  device.queue.writeBuffer(weightPage, 0, words);
  const activations = new Float32Array(arenaElements);
  for (let i = 0; i < inputDim; i++) activations[i] = ((nextWord() % 2000) / 1000) - 1;
  device.queue.writeBuffer(arena.gpu, 0, activations);
  await device.queue.onSubmittedWorkDone();

  const dispatch = (index: number, label: string) => {
    engine.submit((encoder) => {
      encoder.compute({ label }, (pass) => {
        pass.run(programs[index] as never, {
          workgroups: [Math.ceil(outputDim / VARIANTS[index]!.rows), 1, 1],
        }, { resources: { weightRaw: weightPage } } as never);
      });
    });
  };

  // Correctness first, probe detached: readback() submits and maps buffers of
  // its own, which collides with an in-flight timing readback.
  const mismatch = new Map<string, number>();
  let reference: Float32Array | undefined;
  for (let i = 0; i < VARIANTS.length; i++) {
    dispatch(i, `verify:${VARIANTS[i]!.name}`);
    await device.queue.onSubmittedWorkDone();
    const output = (await arena.readback() as Float32Array).slice(inputDim, inputDim + outputDim);
    if (i === 0) {
      reference = output;
      expect(output.some((v) => v !== 0)).toBe(true);
    } else {
      let worst = 0;
      for (let k = 0; k < outputDim; k++) worst = Math.max(worst, Math.abs(output[k]! - reference![k]!));
      mismatch.set(VARIANTS[i]!.name, worst);
    }
  }

  const probe = engine.attachSamdoneter({ capacity: 8, label: "mm" })!;
  const results: Record<string, unknown>[] = [];
  let baselineMs = 0;
  for (let i = 0; i < VARIANTS.length; i++) {
    const variant = VARIANTS[i]!;
    for (let w = 0; w < WARMUP; w++) {
      dispatch(i, variant.name);
      await probe.read();
    }
    const samples: number[] = [];
    for (let s = 0; s < SAMPLES; s++) {
      dispatch(i, variant.name);
      const report = await probe.read();
      const pass = report.passes[0];
      if (pass) samples.push(pass.durationNs / 1e6);
    }
    const ms = median(samples);
    if (i === 0) baselineMs = ms;
    results.push({
      variant: variant.name,
      ms: Number(ms.toFixed(3)),
      giBs: Number(((weightBytes / 1073741824) / (ms / 1000)).toFixed(1)),
      speedup: Number((baselineMs / ms).toFixed(2)),
      workgroups: Math.ceil(outputDim / variant.rows),
      maxAbsDiff: mismatch.get(variant.name) ?? 0,
    });
  }

  engine.detachSamdoneter();
  engine.destroy();
  weightPage.destroy();

  for (const result of results) expect(result.ms).toBeGreaterThan(0);
  // Reassociating a dot product can change rounding; a wrong kernel misses by
  // orders of magnitude, not by an ulp.
  for (const [name, worst] of mismatch) {
    expect(`${shape.name}/${name}: ${worst < 0.01}`).toBe(`${shape.name}/${name}: true`);
  }

  return { shape: `${shape.name} ${inputDim}x${outputDim}`, weightMiB: Number((weightBytes / 1048576).toFixed(1)), results };
}

test("WQ4 matmul variants across every model shape", async () => {
  await installDawn();

  const { device } = await createWebGpuDevice({
    label: "matmul-variants",
    timestampQuery: true,
    requiredLimits: {
      maxBufferSize: 1024 * 1024 * 1024,
      maxStorageBufferBindingSize: 1024 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
    },
  });

  const all: unknown[] = [];
  for (const shape of SHAPES) all.push(await benchShape(device, shape));

  console.log("[matmul-variants] " + JSON.stringify({
    memoryFloorGiBs: 302.8,
    inModelLmHead: { ms: 3.6, giBs: 21.7 },
    shapes: all,
  }, null, 2));
}, 600_000);
