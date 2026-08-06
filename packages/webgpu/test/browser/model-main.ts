import { BlobSource, GgmlType } from "../../../quant/src/gguf";
import {
  WQ4_BLOCK_SIZE,
  WQ4_BYTES_PER_BLOCK,
  Wq4Reader,
  type Wq4TensorInfo,
} from "../../../quant/src/wq4/reader";
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

function tensorRowShape(tensor: { dimensions: readonly number[] }): { width: number; rows: number } {
  const width = tensor.dimensions[0] ?? 1;
  const rows = tensor.dimensions.length > 1 ? product(tensor.dimensions.slice(1)) : 1;
  return { width, rows };
}

async function readRawF32Tensor(reader: Wq4Reader, tensor: Wq4TensorInfo): Promise<Float32Array> {
  if (tensor.encoding !== "raw" || tensor.sourceType !== GgmlType.F32) {
    throw new Error(
      `${tensor.name} must be raw F32 in WQ4 v3, got encoding=${tensor.encoding}, sourceType=${String(tensor.sourceType)}`,
    );
  }
  return decodeF32(await reader.readTensor(tensor));
}

function wq4RowBytes(width: number): number {
  if (width % WQ4_BLOCK_SIZE !== 0) {
    throw new Error(`WQ4 row width ${width} is not divisible by ${WQ4_BLOCK_SIZE}`);
  }
  return (width / WQ4_BLOCK_SIZE) * WQ4_BYTES_PER_BLOCK;
}

function decodeWq4Row(bytes: Uint8Array, width: number): Float32Array {
  const expectedBytes = wq4RowBytes(width);
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(`WQ4 row has ${bytes.byteLength} B, expected ${expectedBytes} B for width ${width}`);
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(width);
  const blocks = width / WQ4_BLOCK_SIZE;
  for (let block = 0; block < blocks; block++) {
    const byteBase = block * WQ4_BYTES_PER_BLOCK;
    const exp = view.getInt32(byteBase + 16, true);
    const scale = 2 ** exp;
    const outBase = block * WQ4_BLOCK_SIZE;

    for (let wordIndex = 0; wordIndex < 4; wordIndex++) {
      const packed = view.getUint32(byteBase + wordIndex * 4, true);
      for (let lane = 0; lane < 8; lane++) {
        const q = ((packed >>> (lane * 4)) & 0x0f) - 8;
        out[outBase + wordIndex * 8 + lane] = q * scale;
      }
    }
  }
  return out;
}

async function readWq4Row(
  reader: Wq4Reader,
  tensor: Wq4TensorInfo,
  row: number,
): Promise<{ bytes: Uint8Array; values: Float32Array }> {
  const { width, rows } = tensorRowShape(tensor);
  if (row < 0 || row >= rows) throw new RangeError(`${tensor.name}: row ${row} outside 0..${rows - 1}`);
  const rowBytes = wq4RowBytes(width);
  const bytes = await reader.readTensor(tensor, row * rowBytes, rowBytes);
  return { bytes, values: decodeWq4Row(bytes, width) };
}

function decodeWq4TensorRow(bytes: Uint8Array, width: number, row: number): Float32Array {
  const rowBytes = wq4RowBytes(width);
  const start = row * rowBytes;
  const end = start + rowBytes;
  if (start < 0 || end > bytes.byteLength) {
    throw new RangeError(`WQ4 row ${row} byte range [${start}, ${end}) outside ${bytes.byteLength} B tensor`);
  }
  return decodeWq4Row(bytes.subarray(start, end), width);
}

function median(values: readonly number[]): number {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 === 1
    ? sorted[middle]!
    : (sorted[middle - 1]! + sorted[middle]!) / 2;
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

/** Mirrors matmul_wq4's 64-lane block assignment + tree reduction. */
function cpuMatmulWq4(input: Float32Array, weights: Float32Array): number {
  if (input.length !== weights.length || input.length % WQ4_BLOCK_SIZE !== 0) {
    throw new Error(`WQ4 matmul shape mismatch: input=${input.length}, weights=${weights.length}`);
  }
  const WG = 64;
  const blocks = input.length / WQ4_BLOCK_SIZE;
  const lanes = new Float32Array(WG);

  for (let lane = 0; lane < WG; lane++) {
    let sum = 0;
    for (let block = lane; block < blocks; block += WG) {
      const base = block * WQ4_BLOCK_SIZE;
      for (let k = 0; k < WQ4_BLOCK_SIZE; k++) {
        const productF32 = Math.fround(input[base + k]! * weights[base + k]!);
        sum = Math.fround(sum + productF32);
      }
    }
    lanes[lane] = sum;
  }

  for (let width = WG >> 1; width > 0; width >>= 1) {
    for (let lane = 0; lane < width; lane++) {
      lanes[lane] = Math.fround(lanes[lane]! + lanes[lane + width]!);
    }
  }
  return lanes[0]!;
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
  if (!file) throw new Error("Choose the self-contained WQ4 v3 model first");

  runButton.disabled = true;
  output.textContent = "";
  setStatus("running…");

  try {
    const device = await (devicePromise ??= createDeviceAndCompile());
    const wq4 = await Wq4Reader.open(new BlobSource(file));
    if (!wq4.selfContained) {
      throw new Error(`This smoke requires self-contained WQ4 v3, got v${wq4.version}`);
    }
    const meta = (key: string) => wq4.metadataValue(key);

    const rawCount = [...wq4.tensors.values()].filter((tensor) => tensor.encoding === "raw").length;
    const quantizedCount = wq4.tensorCount - rawCount;
    log("[model] reading self-contained WQ4 metadata…", {
      file: { name: file.name, sizeMiB: (file.size / 1048576).toFixed(1) },
      version: wq4.version,
      tensors: wq4.tensorCount,
      quantized: quantizedCount,
      raw: rawCount,
      tokenizerTokens: Array.isArray(meta("tokenizer.ggml.tokens"))
        ? (meta("tokenizer.ggml.tokens") as unknown[]).length
        : null,
    });

    const architecture = meta("general.architecture");
    if (architecture !== "lfm2") throw new Error(`Expected lfm2 metadata, got '${String(architecture)}'`);

    const hiddenSize = asNumber(meta("lfm2.embedding_length"), "lfm2.embedding_length");
    const normEpsilon = asNumber(
      meta("lfm2.attention.layer_norm_rms_epsilon"),
      "lfm2.attention.layer_norm_rms_epsilon",
    );
    const vocabSize = asNumber(meta("lfm2.vocab_size"), "lfm2.vocab_size");
    if (hiddenSize !== lfm2.config.hiddenSize || vocabSize !== lfm2.config.vocabSize) {
      throw new Error(
        `Model/artifact mismatch: model H=${hiddenSize}, vocab=${vocabSize}; `
        + `artifact H=${lfm2.config.hiddenSize}, vocab=${lfm2.config.vocabSize}`,
      );
    }

    const bosToken = asNumber(meta("tokenizer.ggml.bos_token_id"), "tokenizer.ggml.bos_token_id");
    const requestedToken = tokenInput.value.trim() === "" ? bosToken : Number(tokenInput.value);
    if (!Number.isInteger(requestedToken) || requestedToken < 0) {
      throw new Error(`Invalid token id '${tokenInput.value}'`);
    }
    const tokenId = requestedToken;

    const embedding = wq4.requireTensor("token_embd.weight");
    if (embedding.encoding !== "wq4") throw new Error("token_embd.weight must be WQ4 encoded");
    const embeddingShape = tensorRowShape(embedding);
    if (embeddingShape.width !== hiddenSize) {
      throw new Error(`token_embd.weight width ${embeddingShape.width} != hidden size ${hiddenSize}`);
    }
    if (tokenId >= embeddingShape.rows) {
      throw new Error(`token ${tokenId} outside embedding rows ${embeddingShape.rows}`);
    }
    const embeddingWq4 = embedding;

    const norm = wq4.requireTensor("blk.0.attn_norm.weight");
    if (product(norm.dimensions) !== hiddenSize) {
      throw new Error(`blk.0.attn_norm.weight has ${product(norm.dimensions)} values, expected ${hiddenSize}`);
    }

    const matmulName = "blk.0.shortconv.in_proj.weight";
    const matmul = wq4.requireTensor(matmulName);
    if (matmul.encoding !== "wq4") throw new Error(`${matmulName} must be WQ4 encoded`);
    const matmulShape = tensorRowShape(matmul);
    if (matmulShape.width !== hiddenSize) {
      throw new Error(`${matmulName} width ${matmulShape.width} != hidden size ${hiddenSize}`);
    }
    const matmulWq4 = matmul;
    const matmulRowBytes = wq4RowBytes(matmulShape.width);
    const expectedMatmulBytes = matmulRowBytes * matmulShape.rows;
    if (matmulWq4.size !== expectedMatmulBytes) {
      throw new Error(
        `${matmulName}: WQ4 tensor has ${matmulWq4.size} B, expected ${expectedMatmulBytes} B `
        + `(${matmulShape.rows} × ${matmulRowBytes} B)`,
      );
    }

    log("✓ model probe", {
      tokenId,
      bosToken,
      hiddenSize,
      vocabSize,
      embeddingRows: embeddingShape.rows,
      embeddingSourceType: embedding.sourceType === undefined ? "unknown" : GgmlType[embedding.sourceType],
      embeddingWq4BytesPerRow: wq4RowBytes(hiddenSize),
      normType: norm.sourceType === undefined ? "unknown" : GgmlType[norm.sourceType],
      matmul: {
        name: matmulName,
        sourceType: matmul.sourceType === undefined ? "unknown" : GgmlType[matmul.sourceType],
        width: matmulShape.width,
        rows: matmulShape.rows,
        wq4Bytes: matmulWq4.size,
        wq4MiB: Number((matmulWq4.size / 1048576).toFixed(3)),
      },
      normEpsilon,
    });

    // The full in_proj WQ4 matrix is only ~7.5 MiB at H=2048, N=6144.
    // Upload it as one weight page and validate a handful of rows on CPU rather
    // than expanding all 12.5M quantized weights into a ~50 MiB Float32Array.
    const [
      wq4EmbeddingRow,
      cpuNormWeights,
      wq4MatmulBytes,
    ] = await Promise.all([
      readWq4Row(wq4, embeddingWq4, tokenId),
      readRawF32Tensor(wq4, norm),
      wq4.readTensor(matmulWq4),
    ]);

    const cpuEmbedding = wq4EmbeddingRow.values;
    const cpuNormalized = cpuRmsNorm(cpuEmbedding, cpuNormWeights, normEpsilon);
    const sampleRows = Array.from(new Set([
      0,
      1,
      17,
      63,
      Math.floor(matmulShape.rows / 4),
      Math.floor(matmulShape.rows / 2),
      matmulShape.rows - 1,
    ])).filter((row) => row >= 0 && row < matmulShape.rows);
    const cpuMatmulSamples = new Map<number, number>();
    for (const row of sampleRows) {
      cpuMatmulSamples.set(
        row,
        cpuMatmulWq4(cpuNormalized, decodeWq4TensorRow(wq4MatmulBytes, hiddenSize, row)),
      );
    }

    const embeddingPage = createStorageBuffer(
      device,
      `chomato-model-smoke.wq4.token_embd.weight[${tokenId}]`,
      wq4EmbeddingRow.bytes,
    );
    const normPage = createF32StorageBuffer(
      device,
      "chomato-model-smoke.blk.0.attn_norm.weight",
      cpuNormWeights,
    );
    const matmulPage = createStorageBuffer(
      device,
      `chomato-model-smoke.wq4.${matmulName}`,
      wq4MatmulBytes,
    );

    try {
      device.queue.writeBuffer(lfm2.resources.tokens.gpu, 0, new Uint32Array([tokenId]));

      const embeddingOffset = lfm2.arena.hiddenA;
      const normalizedOffset = lfm2.arena.tmpH;
      const matmulOffset = lfm2.arena.tmpA;
      const executor = new Lfm2Executor(lfm2);

      log(`[model] GPU embedding_wq4 → norm → full ${matmulShape.rows}-row matmul_wq4 (one submit)…`);
      const started = performance.now();
      executor.submit((encoder) => {
        // Separate compute passes make the full GEMV an independent scheduling
        // unit while retaining one queue submit for the whole dependency chain.
        encoder.compute((pass) => {
          pass.run("embedding_wq4", {
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
        }, { label: "chomato-model-smoke.prepare" });

        encoder.compute((pass) => {
          pass.run("matmul_wq4", {
            inputOffset: normalizedOffset,
            outputOffset: matmulOffset,
            tokenCount: 1,
            inputDim: hiddenSize,
            outputDim: matmulShape.rows,
            rowStart: 0,
            rowCount: matmulShape.rows,
          }, matmulPage);
        }, { label: "chomato-model-smoke.full-gemv" });
      });
      await device.queue.onSubmittedWorkDone();
      const chainElapsedMs = performance.now() - started;

      const [gpuEmbedding, gpuNormalized, gpuMatmul] = await Promise.all([
        readArenaF32(device, embeddingOffset, hiddenSize),
        readArenaF32(device, normalizedOffset, hiddenSize),
        readArenaF32(device, matmulOffset, matmulShape.rows),
      ]);

      const embeddingDiff = diffStats(gpuEmbedding, cpuEmbedding);
      const normDiff = diffStats(gpuNormalized, cpuNormalized);
      if (embeddingDiff.maxAbs > 1e-7) {
        throw new Error(`WQ4 embedding mismatch: maxAbs=${embeddingDiff.maxAbs} at ${embeddingDiff.maxIndex}`);
      }
      if (normDiff.maxAbs > 5e-4) {
        throw new Error(`RMSNorm mismatch: maxAbs=${normDiff.maxAbs} at ${normDiff.maxIndex}`);
      }

      const sampleResults = sampleRows.map((row) => {
        const cpu = cpuMatmulSamples.get(row)!;
        const gpu = gpuMatmul[row]!;
        const delta = Math.abs(gpu - cpu);
        const tolerance = Math.max(5e-4, Math.abs(cpu) * 2e-4);
        if (!Number.isFinite(gpu)) throw new Error(`${matmulName}[${row}] produced non-finite GPU output ${gpu}`);
        if (delta > tolerance) {
          throw new Error(
            `WQ4 full GEMV row ${row} mismatch: GPU=${gpu}, CPU=${cpu}, `
            + `delta=${delta}, tolerance=${tolerance}`,
          );
        }
        return { row, gpu, cpu, delta, tolerance };
      });

      for (let row = 0; row < gpuMatmul.length; row++) {
        if (!Number.isFinite(gpuMatmul[row]!)) {
          throw new Error(`${matmulName}[${row}] produced non-finite GPU output ${gpuMatmul[row]}`);
        }
      }

      log("✓ real WQ4 embedding", {
        gpuVsWq4: embeddingDiff,
        first8: Array.from(gpuEmbedding.slice(0, 8)),
      });
      log("✓ real blk.0.attn_norm", {
        diff: normDiff,
        first8: Array.from(gpuNormalized.slice(0, 8)),
      });
      log(`✓ full WQ4 GEMV (${matmulShape.rows} rows)`, {
        tensor: matmulName,
        outputFirst8: Array.from(gpuMatmul.slice(0, 8)),
        samples: sampleResults,
      });

      // Warm the exact full-GEMV scheduling path once, then report queue-wall
      // timings separately from file I/O, weight upload and result readback.
      const runGemvOnly = async () => {
        const t0 = performance.now();
        executor.submit((encoder) => {
          encoder.compute((pass) => {
            pass.run("matmul_wq4", {
              inputOffset: normalizedOffset,
              outputOffset: matmulOffset,
              tokenCount: 1,
              inputDim: hiddenSize,
              outputDim: matmulShape.rows,
              rowStart: 0,
              rowCount: matmulShape.rows,
            }, matmulPage);
          }, { label: "chomato-model-smoke.full-gemv-bench" });
        });
        await device.queue.onSubmittedWorkDone();
        return performance.now() - t0;
      };

      await runGemvOnly();
      const gemvSamplesMs: number[] = [];
      for (let i = 0; i < 7; i++) gemvSamplesMs.push(await runGemvOnly());

      log("✓ WQ4 full-matrix execution smoke", {
        rows: matmulShape.rows,
        inputDim: hiddenSize,
        dispatchWorkgroups: matmulShape.rows,
        correctnessChain: { dispatches: 3, submits: 1, wallMs: Number(chainElapsedMs.toFixed(3)) },
        gemvQueueWallMs: {
          median: Number(median(gemvSamplesMs).toFixed(3)),
          min: Number(Math.min(...gemvSamplesMs).toFixed(3)),
          max: Number(Math.max(...gemvSamplesMs).toFixed(3)),
          samples: gemvSamplesMs.map((value) => Number(value.toFixed(3))),
        },
        uploadedWeightBytes:
          wq4EmbeddingRow.bytes.byteLength
          + cpuNormWeights.byteLength
          + wq4MatmulBytes.byteLength,
      });

      setStatus(`PASS · full WQ4 GEMV ${matmulShape.rows} rows`, "ok");
    } finally {
      embeddingPage.destroy();
      normPage.destroy();
      matmulPage.destroy();
    }
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", () => {
  run().catch(fail);
});

window.addEventListener("unhandledrejection", (event) => {
  setStatus("FAILED", "error");
  log("✗ unhandled rejection", String(event.reason));
});

window.addEventListener("error", (event) => {
  setStatus("FAILED", "error");
  log("✗ window error", event.message);
});

setStatus("choose self-contained WQ4 v3 model");
