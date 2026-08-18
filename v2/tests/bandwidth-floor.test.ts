// Golden bandwidth diagnostic.
//
// The WQ4 matmul achieves ~23 GiB/s on a 360 GB/s card (~6% of peak). Every
// proposed kernel optimization assumes the ceiling is the kernel. This measures
// the floor underneath it: how fast can this stack read a large storage buffer
// at all, with no arithmetic, no activations and no reduction?
//
//   >200 GiB/s  -> the memory path is fine; the matmul kernel is the problem.
//   ~20-30 GiB/s -> the problem is below the kernel (binding model, dispatch
//                   overhead, driver), and kernel rewrites are aimed wrong.
//
// Deliberately standalone: its own device, its own Sandblaster engine, no
// dependency on the LFM2 definition. Nothing here can perturb the model path.
import { expect, test } from "bun:test";
import { scope } from "arktype";
import { wgsl } from "@schema-pop/schema";
import { Sandblaster } from "@sandblaster/core";
import { createWebGpuDevice } from "../../packages/webgpu/src/device.ts";
import { installDawn } from "./dawn.ts";

// 80 MiB — the size of the LM head weight tensor, so the numbers are directly
// comparable to the measured lm_head pass.
const BYTES = 80 * 1024 * 1024;
const WORDS = BYTES / 4;
const WG = 256;
const SCALAR_GROUPS = 65535;
/** Enough quad-threads to cover every word exactly once. */
const VEC4_GROUPS = Math.ceil(WORDS / 4 / WG);
const WARMUP = 3;
const SAMPLES = 10;

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[sorted.length >> 1] ?? Number.NaN;
}

function gibPerSecond(bytes: number, ms: number): number {
  return (bytes / 1073741824) / (ms / 1000);
}

test("raw storage-buffer read bandwidth floor", async () => {
  await installDawn();

  const { adapter, device } = await createWebGpuDevice({
    label: "bandwidth-floor",
    timestampQuery: true,
    requiredLimits: {
      maxBufferSize: 1024 * 1024 * 1024,
      maxStorageBufferBindingSize: 1024 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
    },
  });

  const $ = scope({ ...wgsl.import(), Word: "u32", Sink: "u32" });
  const engine = Sandblaster.create($);
  const gid = engine.type({ gid: "global_invocation_id" });
  const data = engine.buffer(engine.type("Word"), { count: WORDS, label: "stream.src" });
  const sink = engine.buffer(engine.type("Sink"), { count: 4, label: "stream.sink" });

  // Each thread reads a strided run so the whole buffer is covered by a
  // dispatch that fits the workgroup-per-dimension limit. The guarded write is
  // what stops the compiler from deleting the loads; the condition never fires.
  const scalar = engine.compute({
    label: "stream_scalar",
    resources: { data, sink },
    compute: {
      workgroupSize: WG,
      entryPoint: "stream_scalar",
      code: `
        let stride = ${WG}u * ${SCALAR_GROUPS}u;
        var acc = 0u;
        var i = gid.x;
        loop {
          if (i >= ${WORDS}u) { break; }
          acc = acc ^ data[i];
          i += stride;
        }
        if (acc == 0xDEADBEEFu) { sink[0] = acc; }
      `,
      params: gid,
    },
  } as never);

  // Same traffic, four words per load. Tests the "scalar loads are the problem"
  // hypothesis directly, without touching the model.
  const vector = engine.compute({
    label: "stream_vec4",
    resources: { data, sink },
    compute: {
      workgroupSize: WG,
      entryPoint: "stream_vec4",
      code: `
        // One quad per thread per iteration; the dispatch below is sized so the
        // whole buffer is covered, otherwise the GiB/s figure is inflated.
        let stride = ${WG}u * ${VEC4_GROUPS}u;
        var acc = 0u;
        var q = gid.x;
        loop {
          let b = q * 4u;
          if (b + 3u >= ${WORDS}u) { break; }
          acc = acc ^ data[b] ^ data[b + 1u] ^ data[b + 2u] ^ data[b + 3u];
          q += stride;
        }
        if (acc == 0xDEADBEEFu) { sink[0] = acc; }
      `,
      params: gid,
    },
  } as never);

  const compiled = await engine.compile({ device });
  expect(compiled.failed).toBe(0);

  const probe = engine.attachSamdoneter({ capacity: 8, label: "bandwidth" });
  if (!probe) {
    console.log("[bandwidth-floor] timestamp-query unavailable; skipping");
    engine.destroy();
    return;
  }

  const measure = async (
    program: unknown,
    label: string,
    groups: number,
  ): Promise<number> => {
    const run = () => {
      engine.submit((encoder) => {
        encoder.compute({ label }, (pass) => {
          pass.run(program as never, { workgroups: [groups] });
        });
      });
    };
    for (let i = 0; i < WARMUP; i++) {
      run();
      await probe.read();
    }
    const samples: number[] = [];
    for (let i = 0; i < SAMPLES; i++) {
      run();
      const report = await probe.read();
      const pass = report.passes[0];
      if (pass) samples.push(pass.durationNs / 1e6);
    }
    return median(samples);
  };

  const scalarMs = await measure(scalar, "scalar", SCALAR_GROUPS);
  const vectorMs = await measure(vector, "vec4", VEC4_GROUPS);

  // RTX 3060 = 360 GB/s = 335.3 GiB/s. Reported for context, not asserted:
  // this test must not fail on a different card.
  const peakGiBs = 335.3;
  console.log("[bandwidth-floor] " + JSON.stringify({
    adapter: adapter.info?.description ?? adapter.info?.vendor ?? "unknown",
    bufferMiB: BYTES / 1048576,
    scalar: {
      ms: Number(scalarMs.toFixed(3)),
      giBs: Number(gibPerSecond(BYTES, scalarMs).toFixed(1)),
      pctOfPeak: Number(((gibPerSecond(BYTES, scalarMs) / peakGiBs) * 100).toFixed(1)),
    },
    vec4: {
      ms: Number(vectorMs.toFixed(3)),
      giBs: Number(gibPerSecond(BYTES, vectorMs).toFixed(1)),
      pctOfPeak: Number(((gibPerSecond(BYTES, vectorMs) / peakGiBs) * 100).toFixed(1)),
    },
    // For comparison: the LM head moves the same 80 MiB in ~3.6 ms.
    lmHeadGiBs: Number(gibPerSecond(BYTES, 3.6).toFixed(1)),
  }, null, 2));

  expect(scalarMs).toBeGreaterThan(0);
  engine.detachSamdoneter();
  engine.destroy();
}, 180_000);
