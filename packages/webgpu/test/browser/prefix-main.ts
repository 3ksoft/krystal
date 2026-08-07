import { BlobSource } from "../../../quant/src/gguf/source-web";
import { createWebGpuDevice } from "../../src/device";
import { Lfm2Forward } from "../../src/forward";
import { lfm2 } from "../../src/lfm2";
import { Lfm2GpuModel } from "../../src/model";

const status = document.querySelector<HTMLDivElement>("#status")!;
const output = document.querySelector<HTMLPreElement>("#output")!;
const fileInput = document.querySelector<HTMLInputElement>("#model")!;
const tokenInput = document.querySelector<HTMLInputElement>("#token")!;
const runButton = document.querySelector<HTMLButtonElement>("#run")!;

function log(message: string, value?: unknown) {
  const suffix = value === undefined ? "" : ` ${typeof value === "string" ? value : JSON.stringify(value, null, 2)}`;
  output.textContent += `${message}${suffix}\n`;
  console.log(message, value ?? "");
}

function setStatus(message: string, kind: "idle" | "ok" | "error" = "idle") {
  status.textContent = message;
  status.className = kind;
}

function fail(error: unknown): never {
  const message = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  setStatus("FAILED", "error");
  log("✗", message);
  console.error(error);
  throw error;
}

async function readHidden(device: GPUDevice, tokenCount: number): Promise<Float32Array> {
  const count = tokenCount * lfm2.config.hiddenSize;
  const byteLength = count * 4;
  const staging = device.createBuffer({
    label: "lfm2.prefix.readback",
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder({ label: "lfm2.prefix.readback" });
    encoder.copyBufferToBuffer(
      lfm2.resources.arena.gpu,
      lfm2.arena.hiddenA * 4,
      staging,
      0,
      byteLength,
    );
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ, 0, byteLength);
    return new Float32Array(staging.getMappedRange(0, byteLength).slice(0));
  } finally {
    if (staging.mapState === "mapped") staging.unmap();
    staging.destroy();
  }
}

function summarizeToken(values: Float32Array, tokenIndex: number) {
  const h = lfm2.config.hiddenSize;
  const start = tokenIndex * h;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < h; i++) {
    const value = values[start + i]!;
    if (!Number.isFinite(value)) throw new Error(`prefix produced non-finite value at token ${tokenIndex}, dim ${i}: ${value}`);
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
    sumSq += value * value;
  }
  return {
    min,
    max,
    mean: sum / h,
    rms: Math.sqrt(sumSq / h),
    first8: Array.from(values.slice(start, start + 8)),
  };
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) throw new Error(`length mismatch ${a.length} != ${b.length}`);
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i]! - b[i]!));
  return max;
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[sorted.length >> 1] ?? Number.NaN;
}

let devicePromise: Promise<GPUDevice> | undefined;

async function getDevice(): Promise<GPUDevice> {
  const GIB = 1024 * 1024 * 1024;
  const { adapter, device } = await createWebGpuDevice({
    label: "chomato-prefix2",
    requiredLimits: {
      maxBufferSize: GIB,
      maxStorageBufferBindingSize: GIB,
      maxComputeWorkgroupsPerDimension: 65535,
    },
  });
  device.addEventListener("uncapturederror", (event: GPUUncapturedErrorEvent) => {
    setStatus("FAILED", "error");
    log("✗ WebGPU uncaptured error", event.error.message);
  });
  log("✓ WebGPU device", {
    adapter: adapter.info ? `${adapter.info.vendor} · ${adapter.info.architecture}` : "unknown",
    maxStorageBufferBindingSizeMiB: Math.round(Number(device.limits.maxStorageBufferBindingSize) / 1048576),
  });
  const compiled = await lfm2.engine.compile({ device });
  if (compiled.failed) throw new Error(`LFM2 compile failed ${compiled.failed}/${compiled.total}`);
  log(`✓ LFM2 programs compiled by Dawn (${compiled.ok}/${compiled.total})`);
  return device;
}

async function run(): Promise<void> {
  const file = fileInput.files?.[0];
  if (!file) throw new Error("Choose WQ4 v3 model first");
  runButton.disabled = true;
  output.textContent = "";
  setStatus("loading layers 0..2…");

  let model: Lfm2GpuModel | undefined;
  try {
    const device = await (devicePromise ??= getDevice());
    const startedLoad = performance.now();
    model = await Lfm2GpuModel.open(device, new BlobSource(file), {
      preload: false,
      drainUploads: true,
      onProgress: (progress) => {
        if (progress.tensorIndex % 3 === 0 || progress.tensorIndex + 1 === progress.tensorCount) {
          setStatus(`loading ${progress.tensorIndex + 1}/${progress.tensorCount}…`);
        }
      },
    });
    const forward = new Lfm2Forward(model);
    await forward.preparePrefix(3);
    const loadMs = performance.now() - startedLoad;
    const loadedBytes = [...model.tensors.values()].reduce((sum, tensor) => sum + tensor.byteLength, 0);
    log("✓ production prefix tensor store", {
      loadedTensors: model.tensors.size,
      loadedMiB: Number((loadedBytes / 1048576).toFixed(2)),
      loadMs: Number(loadMs.toFixed(1)),
      layers: model.config.layers.slice(0, 3),
      attentionSlots: model.config.attentionLayerSlots.slice(0, 3),
    });

    const probe = tokenInput.value.trim() === "" ? 42 : Number(tokenInput.value);
    if (!Number.isInteger(probe) || probe < 0 || probe >= model.config.vocabSize) {
      throw new Error(`Invalid token ${tokenInput.value}`);
    }
    const tokens = Uint32Array.of(
      model.config.bosToken,
      probe,
      (probe + 1) % model.config.vocabSize,
      (probe + 2) % model.config.vocabSize,
    );

    const runOnce = async () => {
      const t0 = performance.now();
      forward.prefillPrefix(tokens, 3, { resetState: true });
      await device.queue.onSubmittedWorkDone();
      return performance.now() - t0;
    };

    log("[prefix] embedding → conv0 → conv1 → attention2 (Q/K/V + RoPE + KV + causal attention) → FFN…", {
      tokens: Array.from(tokens),
    });
    const firstMs = await runOnce();
    const first = await readHidden(device, tokens.length);
    const firstStats = Array.from({ length: tokens.length }, (_, token) => summarizeToken(first, token));

    const secondMs = await runOnce();
    const second = await readHidden(device, tokens.length);
    const deterministicDelta = maxAbsDiff(first, second);
    if (deterministicDelta > 1e-7) {
      throw new Error(`layers 0..2 are not deterministic after state reset: maxAbs=${deterministicDelta}`);
    }

    await runOnce();
    const samples: number[] = [];
    for (let i = 0; i < 5; i++) samples.push(await runOnce());

    log("✓ production prefix through first attention layer", {
      tokens: Array.from(tokens),
      outputByToken: firstStats,
      deterministicMaxAbs: deterministicDelta,
      queueWallMs: {
        first: Number(firstMs.toFixed(3)),
        second: Number(secondMs.toFixed(3)),
        median: Number(median(samples).toFixed(3)),
        min: Number(Math.min(...samples).toFixed(3)),
        max: Number(Math.max(...samples).toFixed(3)),
        samples: samples.map((value) => Number(value.toFixed(3))),
      },
    });
    setStatus("PASS · conv0 + conv1 + attention2", "ok");
  } finally {
    model?.destroy();
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", () => run().catch(fail));
window.addEventListener("unhandledrejection", (event) => {
  setStatus("FAILED", "error");
  log("✗ unhandled rejection", String(event.reason));
});
window.addEventListener("error", (event) => {
  setStatus("FAILED", "error");
  log("✗ window error", event.message);
});
