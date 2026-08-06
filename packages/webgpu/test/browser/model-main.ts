import { BlobSource, GgmlType, GgufReader, type GgufTensorInfo } from "../../../quant/src/gguf";
import { createWebGpuDevice } from "../../src/device";
import { lfm2 } from "../../src/lfm2";
import { Lfm2Executor } from "../../src/pass";

const status = document.querySelector<HTMLDivElement>("#status")!;
const output = document.querySelector<HTMLPreElement>("#output")!;
const fileInput = document.querySelector<HTMLInputElement>("#model")!;
const tokenInput = document.querySelector<HTMLInputElement>("#token")!;
const runButton = document.querySelector<HTMLButtonElement>("#run")!;

function log(message: string, value?: unknown) {
  const suffix = value === undefined
    ? ""
    : ` ${typeof value === "string" ? value : JSON.stringify(value, null, 2)}`;
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

function asNumber(value: unknown, key: string): number {
  if (typeof value === "number") return value;
  if (typeof value === "bigint") return Number(value);
  throw new Error(`${key} must be numeric`);
}

function product(values: readonly number[]): number {
  let result = 1;
  for (const value of values) result *= value;
  return result;
}

function f16ToNumber(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;

  if (exponent === 0) {
    return fraction === 0
      ? sign * 0
      : sign * 2 ** -14 * (fraction / 1024);
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function decodeF16(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength % 2 !== 0) throw new Error(`F16 payload has odd byte length ${bytes.byteLength}`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(bytes.byteLength / 2);
  for (let i = 0; i < out.length; i++) out[i] = f16ToNumber(view.getUint16(i * 2, true));
  return out;
}

function decodeF32(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength % 4 !== 0) throw new Error(`F32 payload is not 4-byte aligned: ${bytes.byteLength}`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(bytes.byteLength / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

function tensorRowShape(tensor: GgufTensorInfo): { width: number; rows: number } {
  const width = tensor.dimensions[0] ?? 1;
  const rows = tensor.dimensions.length > 1 ? product(tensor.dimensions.slice(1)) : 1;
  return { width, rows };
}

async function readF16Row(
  reader: GgufReader,
  tensor: GgufTensorInfo,
  row: number,
): Promise<{ bytes: Uint8Array; values: Float32Array }> {
  if (tensor.type !== GgmlType.F16) {
    throw new Error(`${tensor.name} must be F16 for this smoke, got GGML type ${tensor.type}`);
  }
  const { width, rows } = tensorRowShape(tensor);
  if (row < 0 || row >= rows) throw new RangeError(`${tensor.name}: row ${row} outside 0..${rows - 1}`);
  const rowBytes = width * 2;
  const bytes = await reader.readTensor(tensor, row * rowBytes, rowBytes);
  return { bytes, values: decodeF16(bytes) };
}

async function readF32Tensor(reader: GgufReader, tensor: GgufTensorInfo): Promise<Float32Array> {
  if (tensor.type !== GgmlType.F32) {
    throw new Error(`${tensor.name} must be F32 for this smoke, got GGML type ${tensor.type}`);
  }
  return decodeF32(await reader.readTensor(tensor));
}

function createStorageBuffer(device: GPUDevice, label: string, bytes: Uint8Array): GPUBuffer {
  const size = Math.max(4, Math.ceil(bytes.byteLength / 4) * 4);
  const buffer = device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, bytes);
  return buffer;
}

function createF32StorageBuffer(device: GPUDevice, label: string, values: Float32Array): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: Math.max(4, values.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, values);
  return buffer;
}

async function readArenaF32(
  device: GPUDevice,
  elementOffset: number,
  count: number,
): Promise<Float32Array> {
  const byteLength = count * 4;
  const staging = device.createBuffer({
    label: "chomato-model-smoke.readback",
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  try {
    const encoder = device.createCommandEncoder({ label: "chomato-model-smoke.readback" });
    encoder.copyBufferToBuffer(lfm2.resources.arena.gpu, elementOffset * 4, staging, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ, 0, byteLength);
    return new Float32Array(staging.getMappedRange(0, byteLength).slice(0));
  } finally {
    if (staging.mapState === "mapped") staging.unmap();
    staging.destroy();
  }
}

interface DiffStats {
  maxAbs: number;
  meanAbs: number;
  maxIndex: number;
  actualAtMax: number;
  expectedAtMax: number;
}

function diffStats(actual: Float32Array, expected: Float32Array): DiffStats {
  if (actual.length !== expected.length) {
    throw new Error(`length mismatch: GPU ${actual.length}, CPU ${expected.length}`);
  }
  let maxAbs = 0;
  let maxIndex = 0;
  let sumAbs = 0;
  for (let i = 0; i < actual.length; i++) {
    const delta = Math.abs(actual[i]! - expected[i]!);
    sumAbs += delta;
    if (delta > maxAbs) {
      maxAbs = delta;
      maxIndex = i;
    }
  }
  return {
    maxAbs,
    meanAbs: actual.length ? sumAbs / actual.length : 0,
    maxIndex,
    actualAtMax: actual[maxIndex] ?? 0,
    expectedAtMax: expected[maxIndex] ?? 0,
  };
}

function cpuRmsNorm(input: Float32Array, weights: Float32Array, epsilon: number): Float32Array {
  if (input.length !== weights.length) {
    throw new Error(`RMSNorm shape mismatch: input=${input.length}, weights=${weights.length}`);
  }
  let sumSquares = 0;
  for (let i = 0; i < input.length; i++) sumSquares += input[i]! * input[i]!;
  const invRms = 1 / Math.sqrt(sumSquares / input.length + epsilon);
  const out = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) out[i] = input[i]! * invRms * weights[i]!;
  return out;
}

async function createDeviceAndCompile(): Promise<GPUDevice> {
  const GIB = 1024 * 1024 * 1024;
  const { adapter, device } = await createWebGpuDevice({
    label: "chomato-model-smoke",
    requiredLimits: {
      maxBufferSize: GIB,
      maxStorageBufferBindingSize: GIB,
      maxComputeWorkgroupsPerDimension: 65535,
    },
  });

  device.addEventListener("uncapturederror", (event) => {
    setStatus("FAILED", "error");
    log("✗ WebGPU uncaptured error", event.error.message);
    console.error(event.error);
  });

  log("✓ WebGPU device", {
    adapter: adapter.info
      ? `${adapter.info.vendor} · ${adapter.info.architecture} · ${adapter.info.description}`
      : "unknown",
    maxStorageBufferBindingSizeMiB: Math.round(Number(device.limits.maxStorageBufferBindingSize) / 1048576),
  });

  const compiled = await lfm2.engine.compile({ device });
  if (compiled.failed > 0) {
    throw new Error(`LFM2 compile failed (${compiled.failed}/${compiled.total})`);
  }
  log(`✓ LFM2 programs compiled by Dawn (${compiled.ok}/${compiled.total})`);
  return device;
}

let devicePromise: Promise<GPUDevice> | undefined;

async function run(): Promise<void> {
  const file = fileInput.files?.[0];
  if (!file) throw new Error("Choose the F16 GGUF model first");

  runButton.disabled = true;
  output.textContent = "";
  setStatus("running…");

  try {
    const device = await (devicePromise ??= createDeviceAndCompile());
    const source = new BlobSource(file);

    log("[model] reading GGUF metadata…", { name: file.name, sizeMiB: (file.size / 1048576).toFixed(1) });
    const reader = await GgufReader.open(source);
    const architecture = reader.metadata<string>("general.architecture");
    if (architecture !== "lfm2") throw new Error(`Expected lfm2 GGUF, got '${architecture}'`);

    const hiddenSize = asNumber(reader.metadata("lfm2.embedding_length"), "lfm2.embedding_length");
    const normEpsilon = asNumber(
      reader.metadata("lfm2.attention.layer_norm_rms_epsilon"),
      "lfm2.attention.layer_norm_rms_epsilon",
    );
    const vocabSize = asNumber(reader.metadata("lfm2.vocab_size"), "lfm2.vocab_size");
    if (hiddenSize !== lfm2.config.hiddenSize || vocabSize !== lfm2.config.vocabSize) {
      throw new Error(
        `Model/artifact mismatch: model H=${hiddenSize}, vocab=${vocabSize}; `
        + `artifact H=${lfm2.config.hiddenSize}, vocab=${lfm2.config.vocabSize}`,
      );
    }
    const bosToken = asNumber(reader.metadata("tokenizer.ggml.bos_token_id"), "tokenizer.ggml.bos_token_id");
    const requestedToken = tokenInput.value.trim() === "" ? bosToken : Number(tokenInput.value);
    if (!Number.isInteger(requestedToken) || requestedToken < 0) {
      throw new Error(`Invalid token id '${tokenInput.value}'`);
    }
    const tokenId = requestedToken;

    const embedding = reader.tensor("token_embd.weight");
    const embeddingShape = tensorRowShape(embedding);
    if (embeddingShape.width !== hiddenSize) {
      throw new Error(`token_embd.weight width ${embeddingShape.width} != hidden size ${hiddenSize}`);
    }
    if (tokenId >= embeddingShape.rows) {
      throw new Error(`token ${tokenId} outside embedding rows ${embeddingShape.rows}`);
    }

    const norm = reader.tensor("blk.0.attn_norm.weight");
    if (product(norm.dimensions) !== hiddenSize) {
      throw new Error(`blk.0.attn_norm.weight has ${product(norm.dimensions)} values, expected ${hiddenSize}`);
    }

    log("✓ model probe", {
      tokenId,
      bosToken,
      hiddenSize,
      vocabSize,
      embeddingRows: embeddingShape.rows,
      embeddingType: GgmlType[embedding.type],
      normType: GgmlType[norm.type],
      normEpsilon,
    });

    // Read only one embedding row (~4 KiB at H=2048) plus one norm vector.
    // No full-model upload is involved in this smoke.
    const [{ bytes: embeddingBytes, values: cpuEmbedding }, cpuNormWeights] = await Promise.all([
      readF16Row(reader, embedding, tokenId),
      readF32Tensor(reader, norm),
    ]);
    const cpuNormalized = cpuRmsNorm(cpuEmbedding, cpuNormWeights, normEpsilon);

    const embeddingPage = createStorageBuffer(
      device,
      `chomato-model-smoke.token_embd.weight[${tokenId}]`,
      embeddingBytes,
    );
    const normPage = createF32StorageBuffer(
      device,
      "chomato-model-smoke.blk.0.attn_norm.weight",
      cpuNormWeights,
    );

    try {
      device.queue.writeBuffer(lfm2.resources.tokens.gpu, 0, new Uint32Array([tokenId]));

      const embeddingOffset = lfm2.arena.hiddenA;
      const normalizedOffset = lfm2.arena.tmpH;
      const executor = new Lfm2Executor(lfm2);

      log("[model] GPU embedding → blk.0.attn_norm (one submit)…");
      const started = performance.now();
      executor.submit((encoder) => {
        encoder.compute((pass) => {
          pass.run("embedding", {
            outputOffset: embeddingOffset,
            tokenCount: 1,
            outputDim: hiddenSize,
            rowStart: tokenId,
            rowCount: 1,
            mode: "prefill",
            u0: 0,
          }, embeddingPage);

          pass.run("rms_norm", {
            inputOffset: embeddingOffset,
            outputOffset: normalizedOffset,
            tokenCount: 1,
            inputDim: hiddenSize,
            f0: normEpsilon,
          }, normPage);
        }, { label: "chomato-model-smoke.embedding-norm" });
      });
      await device.queue.onSubmittedWorkDone();
      const elapsedMs = performance.now() - started;

      const [gpuEmbedding, gpuNormalized] = await Promise.all([
        readArenaF32(device, embeddingOffset, hiddenSize),
        readArenaF32(device, normalizedOffset, hiddenSize),
      ]);

      const embeddingDiff = diffStats(gpuEmbedding, cpuEmbedding);
      const normDiff = diffStats(gpuNormalized, cpuNormalized);
      if (embeddingDiff.maxAbs > 1e-7) {
        throw new Error(`embedding mismatch: maxAbs=${embeddingDiff.maxAbs} at ${embeddingDiff.maxIndex}`);
      }
      // Reduction order differs between JS and the 64-lane WGSL RMS kernel, so
      // the norm comparison intentionally allows a small floating-point delta.
      if (normDiff.maxAbs > 5e-4) {
        throw new Error(`RMSNorm mismatch: maxAbs=${normDiff.maxAbs} at ${normDiff.maxIndex}`);
      }

      log("✓ real embedding", {
        diff: embeddingDiff,
        first8: Array.from(gpuEmbedding.slice(0, 8)),
      });
      log("✓ real blk.0.attn_norm", {
        diff: normDiff,
        first8: Array.from(gpuNormalized.slice(0, 8)),
      });
      log("✓ model execution smoke", {
        dispatches: 2,
        submits: 1,
        elapsedMs: Number(elapsedMs.toFixed(3)),
        uploadedWeightBytes: embeddingBytes.byteLength + cpuNormWeights.byteLength,
      });

      setStatus("PASS · real embedding + blk.0.attn_norm", "ok");
    } finally {
      embeddingPage.destroy();
      normPage.destroy();
    }
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", () => void run().catch(fail));
fileInput.addEventListener("change", () => {
  if (fileInput.files?.[0]) setStatus("ready · click Run model smoke");
});

setStatus("choose LFM2 F16 GGUF");
