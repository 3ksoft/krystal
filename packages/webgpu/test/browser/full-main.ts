import { BlobSource } from "../../../quant/src/gguf/source";
import { createWebGpuDevice } from "../../src/device";
import { LFM2_GREEDY_SHADER_PATH, Lfm2Forward } from "../../src/forward";
import { LFM2_ARENA, LFM2_SHADER_NAMES, lfm2 } from "../../src/lfm2";
import { Lfm2GpuModel } from "../../src/model";

const status = document.querySelector<HTMLDivElement>("#status")!;
const output = document.querySelector<HTMLPreElement>("#output")!;
const fileInput = document.querySelector<HTMLInputElement>("#model")!;
const tokenInput = document.querySelector<HTMLInputElement>("#token")!;
const maxNewInput = document.querySelector<HTMLInputElement>("#max-new")!;
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

function sameNumbers(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[sorted.length >> 1] ?? Number.NaN;
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) throw new Error(`length mismatch ${a.length} != ${b.length}`);
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i]! - b[i]!));
  return max;
}

async function readArenaRange(
  device: GPUDevice,
  offsetElements: number,
  count: number,
): Promise<Float32Array> {
  const byteLength = count * 4;
  const staging = device.createBuffer({
    label: "lfm2.full.readback",
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder({ label: "lfm2.full.readback" });
    encoder.copyBufferToBuffer(lfm2.resources.arena.gpu, offsetElements * 4, staging, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ, 0, byteLength);
    return new Float32Array(staging.getMappedRange(0, byteLength).slice(0));
  } finally {
    if (staging.mapState === "mapped") staging.unmap();
    staging.destroy();
  }
}

function f32FromBytes(bytes: Uint8Array): Float32Array {
  const copy = bytes.slice();
  return new Float32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4);
}

async function exerciseAlternativeShaders(
  device: GPUDevice,
  model: Lfm2GpuModel,
  forward: Lfm2Forward,
): Promise<{ coverage: readonly string[]; continuationMaxAbs: number; candidateToken: number }> {
  const h = model.config.hiddenSize;
  const pair = Uint32Array.of(0, 1);

  // Reference block-0 prefill. A continuation from an empty recurrent state is
  // mathematically equivalent for this two-token causal-conv window.
  forward.prefillBlock0(pair, { resetState: true });
  await device.queue.onSubmittedWorkDone();
  const reference = await readArenaRange(device, LFM2_ARENA.hiddenA, pair.length * h);

  // One tiny F16 matrix is enough to exercise the legacy F16 embedding/matmul
  // fallback kernels. Values are exact IEEE-f16 encodings.
  const f16Words = new Uint16Array([
    0x3c00, 0x4000, 0x4200, 0x4400, // [1, 2, 3, 4]
    0xbc00, 0x3800, 0x0000, 0x4000, // [-1, .5, 0, 2]
  ]);
  const f16Buffer = device.createBuffer({
    label: "lfm2.coverage.f16",
    size: f16Words.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(f16Buffer, 0, f16Words);

  const conv = model.tensor("blk.0.shortconv.conv.weight");
  if (conv.format !== "f32" || conv.pages.length !== 1) {
    throw new Error("coverage: blk.0.shortconv.conv.weight must be one F32 page");
  }
  const convInfo = model.reader.requireTensor("blk.0.shortconv.conv.weight");
  const convCpu = f32FromBytes(await model.reader.readTensor(convInfo, 0, 4 * 3 * 4));
  const probe3 = new Float32Array([1, 2, 3]);

  forward.executor.clearShaderCoverage();
  forward.initializeRequest(pair.length, 8);
  forward.writeTokens(pair);
  device.queue.writeBuffer(lfm2.resources.arena.gpu, LFM2_ARENA.repair.tmpH * 4, probe3);
  forward.executor.submit((encoder) => {
    forward.clearState(encoder);
    encoder.compute((pass) => {
      pass.run("embedding", {
        outputOffset: LFM2_ARENA.repair.hiddenA,
        tokenCount: 2,
        outputDim: 4,
        rowStart: 0,
        rowCount: 2,
        mode: "prefill",
        u0: 0,
      }, f16Buffer);
      pass.run("matmul_f16", {
        inputOffset: LFM2_ARENA.repair.hiddenA,
        outputOffset: LFM2_ARENA.repair.tmpA,
        tokenCount: 1,
        inputDim: 4,
        outputDim: 2,
        rowStart: 0,
        rowCount: 2,
      }, f16Buffer);
      pass.run("matmul_f32", {
        inputOffset: LFM2_ARENA.repair.tmpH,
        outputOffset: LFM2_ARENA.repair.tmpB,
        tokenCount: 1,
        inputDim: 3,
        outputDim: h,
        rowStart: 0,
        rowCount: h,
      }, conv.pages[0]!);

      forward.embed(pass, pair.length, "prefill", LFM2_ARENA, 0);
      forward.block(pass, 0, pair.length, {
        mode: "continuation",
        positionBase: 0,
        work: LFM2_ARENA,
      });
      pass.run("arena_copy", {
        inputOffset: LFM2_ARENA.hiddenA,
        outputOffset: LFM2_ARENA.repair.hiddenA,
        tokenCount: pair.length,
        inputDim: h,
      });
    }, { label: "lfm2.coverage.alternatives" });
  });
  await device.queue.onSubmittedWorkDone();

  const f16Result = await readArenaRange(device, LFM2_ARENA.repair.tmpA, 2);
  if (Math.abs(f16Result[0]! - 30) > 1e-5 || Math.abs(f16Result[1]! - 8) > 1e-5) {
    throw new Error(`F16 fallback mismatch: [${f16Result[0]}, ${f16Result[1]}] != [30, 8]`);
  }

  const f32Result = await readArenaRange(device, LFM2_ARENA.repair.tmpB, 4);
  for (let row = 0; row < 4; row++) {
    const expected = convCpu[row * 3]! + 2 * convCpu[row * 3 + 1]! + 3 * convCpu[row * 3 + 2]!;
    if (Math.abs(f32Result[row]! - expected) > 1e-5) {
      throw new Error(`F32 fallback mismatch at row ${row}: ${f32Result[row]} != ${expected}`);
    }
  }

  const continued = await readArenaRange(device, LFM2_ARENA.repair.hiddenA, pair.length * h);
  const continuationMaxAbs = maxAbsDiff(reference, continued);
  if (continuationMaxAbs > 1e-6) {
    throw new Error(`shortconv_continue differs from clean prefill: maxAbs=${continuationMaxAbs}`);
  }

  // Candidate argmax runs against real full-model logits. The CPU reads the
  // same 256 KiB logits range and verifies the selected sparse candidate.
  const prompt = Uint32Array.of(model.config.bosToken, 42, 43, 44);
  const candidates = Uint32Array.of(0, 1, 2, 3, 42, model.config.eosToken, model.config.vocabSize - 2, model.config.vocabSize - 1);
  forward.initializeRequest(prompt.length, 8);
  forward.writeTokens(prompt);
  device.queue.writeBuffer(lfm2.resources.candidateTokens.gpu, 0, candidates);
  forward.executor.submit((encoder) => {
    forward.clearState(encoder);
    encoder.compute((pass) => {
      forward.forwardToLogits(pass, prompt.length, "prefill", LFM2_ARENA, 0);
      pass.run("argmax_candidates", {
        inputOffset: LFM2_ARENA.logits,
        inputDim: candidates.length,
        mode: "prefill",
      });
    }, { label: "lfm2.coverage.argmax-candidates" });
  });
  await device.queue.onSubmittedWorkDone();

  const logits = await readArenaRange(device, LFM2_ARENA.logits, model.config.vocabSize);
  let expectedToken = candidates[0]!;
  let best = logits[expectedToken]!;
  for (let i = 1; i < candidates.length; i++) {
    const token = candidates[i]!;
    const value = logits[token]!;
    if (value > best || (value === best && token < expectedToken)) {
      best = value;
      expectedToken = token;
    }
  }
  const tokenReadback = await lfm2.resources.tokens.readback({ dropIfBusy: false }) as any;
  if (!Array.isArray(tokenReadback) || Array.isArray(tokenReadback[0])) throw new Error("Invalid token readback");
  const candidateToken = Number(tokenReadback[lfm2.capacities.context]);
  if (candidateToken !== expectedToken) {
    throw new Error(`argmax_candidates selected ${candidateToken}, CPU expected ${expectedToken}`);
  }

  f16Buffer.destroy();
  return {
    coverage: [...forward.executor.shaderCoverage],
    continuationMaxAbs,
    candidateToken,
  };
}

let devicePromise: Promise<GPUDevice> | undefined;

async function getDevice(): Promise<GPUDevice> {
  const GIB = 1024 * 1024 * 1024;
  const { adapter, device } = await createWebGpuDevice({
    label: "chomato-full-forward",
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
  const maxNewTokens = Number(maxNewInput.value || "2");
  if (!Number.isInteger(maxNewTokens) || maxNewTokens < 2 || maxNewTokens > lfm2.capacities.maxNewTokens) {
    throw new Error(`full shader coverage requires max new tokens 2..${lfm2.capacities.maxNewTokens}`);
  }

  runButton.disabled = true;
  output.textContent = "";
  setStatus("loading full model…");

  let model: Lfm2GpuModel | undefined;
  try {
    const device = await (devicePromise ??= getDevice());
    const startedLoad = performance.now();
    model = await Lfm2GpuModel.open(device, new BlobSource(file), {
      preload: false,
      drainUploads: true,
      onProgress: (progress) => {
        if (progress.tensorIndex % 8 === 0 || progress.tensorIndex + 1 === progress.tensorCount) {
          setStatus(`loading ${progress.tensorIndex + 1}/${progress.tensorCount}…`);
        }
      },
    });
    const forward = new Lfm2Forward(model);
    await forward.prepareAll();
    const loadMs = performance.now() - startedLoad;
    const loadedBytes = [...model.tensors.values()].reduce((sum, tensor) => sum + tensor.byteLength, 0);
    log("✓ full production tensor store", {
      loadedTensors: model.tensors.size,
      loadedMiB: Number((loadedBytes / 1048576).toFixed(2)),
      loadMs: Number(loadMs.toFixed(1)),
      blockCount: model.config.blockCount,
      layers: model.config.layers,
      attentionSlots: model.config.attentionLayerSlots,
    });

    const probe = tokenInput.value.trim() === "" ? 42 : Number(tokenInput.value);
    if (!Number.isInteger(probe) || probe < 0 || probe >= model.config.vocabSize) {
      throw new Error(`Invalid token ${tokenInput.value}`);
    }
    const prompt = Uint32Array.of(
      model.config.bosToken,
      probe,
      (probe + 1) % model.config.vocabSize,
      (probe + 2) % model.config.vocabSize,
    );

    const runOnce = async () => {
      forward.executor.clearShaderCoverage();
      const started = performance.now();
      const result = await forward.generateGreedy(prompt, { maxNewTokens, resetState: true });
      const wallMs = performance.now() - started;
      return { result, wallMs, coverage: [...forward.executor.shaderCoverage] };
    };

    log("[full] embedding → all 16 blocks → final norm/logits → argmax → decode…", {
      prompt: Array.from(prompt),
      maxNewTokens,
    });

    const first = await runOnce();
    const second = await runOnce();
    if (!sameNumbers(first.result.tokens, second.result.tokens)) {
      throw new Error(
        `full generation is not deterministic: ${JSON.stringify(first.result.tokens)} != ${JSON.stringify(second.result.tokens)}`,
      );
    }
    if (first.result.generatedCount !== second.result.generatedCount || first.result.status !== second.result.status) {
      throw new Error("full generation runtime state differs after reset");
    }

    const missingGreedy = LFM2_GREEDY_SHADER_PATH.filter((name) => !first.coverage.includes(name));
    if (missingGreedy.length) {
      throw new Error(`full greedy path did not execute shaders: ${missingGreedy.join(", ")}`);
    }

    const samples: number[] = [];
    for (let i = 0; i < 3; i++) samples.push((await runOnce()).wallMs);

    const alternatives = await exerciseAlternativeShaders(device, model, forward);
    const allCoverage = [...new Set([...first.coverage, ...alternatives.coverage])];
    const missingAll = LFM2_SHADER_NAMES.filter((name) => !allCoverage.includes(name));
    if (missingAll.length) throw new Error(`shader integration coverage missing: ${missingAll.join(", ")}`);

    log("✓ alternative shader paths", {
      continuationMaxAbs: alternatives.continuationMaxAbs,
      candidateToken: alternatives.candidateToken,
      coverage: alternatives.coverage,
    });

    log("✓ full production forward + greedy decode", {
      prompt: Array.from(prompt),
      generated: first.result.tokens,
      generatedCount: first.result.generatedCount,
      status: first.result.status,
      lastToken: first.result.lastToken,
      deterministic: true,
      shaderCoverage: {
        greedy: first.coverage,
        alternatives: alternatives.coverage,
        all: allCoverage,
        usedCount: allCoverage.length,
        totalCompiled: LFM2_SHADER_NAMES.length,
      },
      queueAndReadbackWallMs: {
        first: Number(first.wallMs.toFixed(3)),
        second: Number(second.wallMs.toFixed(3)),
        median: Number(median(samples).toFixed(3)),
        min: Number(Math.min(...samples).toFixed(3)),
        max: Number(Math.max(...samples).toFixed(3)),
        samples: samples.map((value) => Number(value.toFixed(3))),
      },
    });

    setStatus("PASS · full forward + all 17 shaders", "ok");
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
