import type { RandomAccessSource } from "../../quant/src/gguf/source";
import { GgmlType } from "../../quant/src/gguf/types";
import {
  WQ4_BLOCK_SIZE,
  WQ4_BYTES_PER_BLOCK,
  Wq4Reader,
  type Wq4TensorInfo,
} from "../../quant/src/wq4/reader";
import { lfm2 } from "./lfm2";

const MIB = 1024 * 1024;
const DEFAULT_PAGE_BYTES = 64 * MIB;
const UPLOAD_CHUNK_BYTES = 16 * MIB;

function asNumber(value: unknown, key: string): number {
  if (typeof value === "number") return value;
  if (typeof value === "bigint") return Number(value);
  throw new Error(`${key} must be numeric`);
}

function asNumberArray(value: unknown, key: string): number[] {
  if (!Array.isArray(value)) throw new Error(`${key} must be an array`);
  return value.map((value) => asNumber(value, key));
}

function product(values: readonly number[]): number {
  let result = 1;
  for (const value of values) result *= value;
  return result;
}

function sameNumbers(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function scalarBytes(type: GgmlType): number | undefined {
  if (type === GgmlType.F16 || type === GgmlType.BF16) return 2;
  if (type === GgmlType.F32) return 4;
  return undefined;
}

export type Lfm2GpuWeightFormat = "f16" | "f32" | "wq4";
export type Lfm2LayerKind = "conv" | "attention";

export interface Lfm2ModelConfig {
  architecture: "lfm2";
  blockCount: number;
  contextLength: number;
  hiddenSize: number;
  feedForwardSize: number;
  attentionHeads: number;
  kvHeadsByLayer: number[];
  headDim: number;
  ropeTheta: number;
  vocabSize: number;
  convCacheLength: number;
  normEpsilon: number;
  bosToken: number;
  eosToken: number;
  addBosToken: boolean;
  addEosToken: boolean;
  layers: Lfm2LayerKind[];
  attentionLayerSlots: number[];
}

export interface Lfm2GpuTensorPage {
  readonly buffer: GPUBuffer;
  readonly rowStart: number;
  readonly rowCount: number;
  readonly byteLength: number;
}

export interface Lfm2GpuTensor {
  readonly name: string;
  readonly sourceType: GgmlType;
  readonly format: Lfm2GpuWeightFormat;
  readonly dimensions: readonly number[];
  readonly pages: readonly Lfm2GpuTensorPage[];
  readonly byteLength: number;
}

export interface Lfm2GpuModelProgress {
  readonly tensor: string;
  readonly tensorIndex: number;
  readonly tensorCount: number;
  readonly uploadedBytes: number;
  readonly totalBytes: number;
  readonly allocatedBytes: number;
}

export interface Lfm2GpuModelOptions {
  /** Upload all tensors by default. Pass false for an explicitly lazy model. */
  readonly preload?: "all" | Iterable<string> | false;
  readonly maxPageBytes?: number;
  /** Flush queue.writeBuffer staging after every tensor. Safer for large browser loads. */
  readonly drainUploads?: boolean;
  readonly onProgress?: (progress: Lfm2GpuModelProgress) => void;
}

export function lfm2BlockTensorNames(layer: number, kind: Lfm2LayerKind): string[] {
  const names = [
    `blk.${layer}.attn_norm.weight`,
    `blk.${layer}.ffn_norm.weight`,
    `blk.${layer}.ffn_gate.weight`,
    `blk.${layer}.ffn_up.weight`,
    `blk.${layer}.ffn_down.weight`,
  ];
  if (kind === "conv") {
    names.push(
      `blk.${layer}.shortconv.in_proj.weight`,
      `blk.${layer}.shortconv.conv.weight`,
      `blk.${layer}.shortconv.out_proj.weight`,
    );
  } else {
    names.push(
      `blk.${layer}.attn_q.weight`,
      `blk.${layer}.attn_k.weight`,
      `blk.${layer}.attn_v.weight`,
      `blk.${layer}.attn_q_norm.weight`,
      `blk.${layer}.attn_k_norm.weight`,
      `blk.${layer}.attn_output.weight`,
    );
  }
  return names;
}

export function lfm2Block0TensorNames(): string[] {
  return ["token_embd.weight", ...lfm2BlockTensorNames(0, "conv")];
}

function readConfig(reader: Wq4Reader): Lfm2ModelConfig {
  const meta = (key: string) => reader.metadataValue(key);
  const optionalMeta = (key: string) => (reader.hasMetadata(key) ? reader.metadataValue(key) : undefined);
  const architecture = meta("general.architecture");
  if (architecture !== "lfm2") {
    throw new Error(`Expected lfm2 WQ4 model, got '${String(architecture)}'`);
  }

  const blockCount = asNumber(meta("lfm2.block_count"), "lfm2.block_count");
  const hiddenSize = asNumber(meta("lfm2.embedding_length"), "lfm2.embedding_length");
  const attentionHeads = asNumber(meta("lfm2.attention.head_count"), "lfm2.attention.head_count");
  const kvHeadsByLayer = asNumberArray(
    meta("lfm2.attention.head_count_kv"),
    "lfm2.attention.head_count_kv",
  );
  if (kvHeadsByLayer.length !== blockCount) {
    throw new Error(`Expected ${blockCount} KV-head entries, got ${kvHeadsByLayer.length}`);
  }

  const layers: Lfm2LayerKind[] = kvHeadsByLayer.map((heads) => heads === 0 ? "conv" : "attention");
  const attentionLayerSlots = new Array<number>(blockCount).fill(-1);
  let slot = 0;
  for (let layer = 0; layer < blockCount; layer++) {
    if (layers[layer] === "attention") attentionLayerSlots[layer] = slot++;
  }

  const config: Lfm2ModelConfig = {
    architecture: "lfm2",
    blockCount,
    contextLength: asNumber(meta("lfm2.context_length"), "lfm2.context_length"),
    hiddenSize,
    feedForwardSize: asNumber(meta("lfm2.feed_forward_length"), "lfm2.feed_forward_length"),
    attentionHeads,
    kvHeadsByLayer,
    headDim: hiddenSize / attentionHeads,
    ropeTheta: asNumber(meta("lfm2.rope.freq_base"), "lfm2.rope.freq_base"),
    vocabSize: asNumber(meta("lfm2.vocab_size"), "lfm2.vocab_size"),
    convCacheLength: asNumber(meta("lfm2.shortconv.l_cache"), "lfm2.shortconv.l_cache"),
    normEpsilon: asNumber(
      meta("lfm2.attention.layer_norm_rms_epsilon"),
      "lfm2.attention.layer_norm_rms_epsilon",
    ),
    bosToken: asNumber(meta("tokenizer.ggml.bos_token_id"), "tokenizer.ggml.bos_token_id"),
    eosToken: asNumber(meta("tokenizer.ggml.eos_token_id"), "tokenizer.ggml.eos_token_id"),
    // Optional flags: the VL GGUF conversion omits add_eos_token, and llama.cpp
    // tolerates both being absent. Missing means false, not an error.
    addBosToken: Boolean(optionalMeta("tokenizer.ggml.add_bos_token")),
    addEosToken: Boolean(optionalMeta("tokenizer.ggml.add_eos_token")),
    layers,
    attentionLayerSlots,
  };

  if (config.headDim !== 64) throw new Error(`LFM2 kernels require headDim=64, got ${config.headDim}`);
  if (config.convCacheLength !== 3) {
    throw new Error(`LFM2 shortconv kernels require cache length 3, got ${config.convCacheLength}`);
  }
  if (config.hiddenSize !== lfm2.config.hiddenSize) {
    throw new Error(`WQ4 hidden size ${config.hiddenSize} != AOT artifact ${lfm2.config.hiddenSize}`);
  }
  if (config.feedForwardSize !== lfm2.config.feedForwardSize) {
    throw new Error(`WQ4 FF size ${config.feedForwardSize} != AOT artifact ${lfm2.config.feedForwardSize}`);
  }
  if (config.vocabSize !== lfm2.config.vocabSize) {
    throw new Error(`WQ4 vocab ${config.vocabSize} != AOT artifact ${lfm2.config.vocabSize}`);
  }
  if (config.blockCount !== lfm2.config.blockCount) {
    throw new Error(`WQ4 block count ${config.blockCount} != AOT artifact ${lfm2.config.blockCount}`);
  }
  if (config.attentionHeads !== lfm2.config.attentionHeads) {
    throw new Error(`WQ4 attention heads ${config.attentionHeads} != AOT artifact ${lfm2.config.attentionHeads}`);
  }
  if (!sameNumbers(config.kvHeadsByLayer, lfm2.config.kvHeadsByLayer)) {
    throw new Error("WQ4 per-layer KV head layout does not match the AOT artifact");
  }
  const shaderKvHeads = Math.max(...lfm2.config.kvHeadsByLayer);
  for (let layer = 0; layer < config.blockCount; layer++) {
    if (config.layers[layer] === "attention" && config.kvHeadsByLayer[layer] !== shaderKvHeads) {
      throw new Error(
        `Layer ${layer} has ${config.kvHeadsByLayer[layer]} KV heads; current AOT attention kernels require ${shaderKvHeads}`,
      );
    }
  }
  if (config.contextLength < lfm2.capacities.context) {
    throw new Error(
      `WQ4 context ${config.contextLength} is below runtime capacity ${lfm2.capacities.context}`,
    );
  }

  return config;
}

/**
 * GPU-resident tensor store backed exclusively by the self-contained WQ4 v3
 * container. No GGUF reader participates in the runtime path.
 */
export class Lfm2GpuModel {
  readonly tensors = new Map<string, Lfm2GpuTensor>();
  readonly config: Lfm2ModelConfig;
  private readonly loading = new Map<string, Promise<Lfm2GpuTensor>>();
  private destroyed = false;

  private constructor(
    readonly device: GPUDevice,
    readonly reader: Wq4Reader,
    private readonly options: Lfm2GpuModelOptions,
  ) {
    this.config = readConfig(reader);
  }

  static async open(
    device: GPUDevice,
    source: RandomAccessSource,
    options: Lfm2GpuModelOptions = {},
  ): Promise<Lfm2GpuModel> {
    const reader = await Wq4Reader.open(source);
    if (!reader.selfContained) {
      throw new Error(`Lfm2GpuModel requires self-contained WQ4 v3; got v${reader.version}`);
    }
    const model = new Lfm2GpuModel(device, reader, options);
    const preload = options.preload ?? "all";
    if (preload === "all") {
      await model.preload(reader.tensors.keys());
    } else if (preload !== false) {
      await model.preload(preload);
    }
    return model;
  }

  metadata<T = unknown>(key: string): T {
    return this.reader.metadataValue<T>(key);
  }

  hasTensor(name: string): boolean {
    return this.tensors.has(name);
  }

  tensor(name: string): Lfm2GpuTensor {
    const tensor = this.tensors.get(name);
    if (!tensor) throw new Error(`GPU tensor is not loaded: ${name}`);
    return tensor;
  }

  async preload(names: Iterable<string>): Promise<void> {
    const unique = [...new Set(names)];
    const totalBytes = unique.reduce((sum, name) => sum + this.reader.requireTensor(name).size, 0);
    let uploadedBytes = 0;
    let allocatedBytes = 0;

    for (let tensorIndex = 0; tensorIndex < unique.length; tensorIndex++) {
      const name = unique[tensorIndex]!;
      const tensor = await this.loadTensor(name);
      uploadedBytes += tensor.byteLength;
      allocatedBytes += tensor.pages.reduce((sum, page) => sum + page.byteLength, 0);
      this.options.onProgress?.({
        tensor: `${name} [${tensor.format}]`,
        tensorIndex,
        tensorCount: unique.length,
        uploadedBytes,
        totalBytes,
        allocatedBytes,
      });
    }
  }

  loadTensor(name: string): Promise<Lfm2GpuTensor> {
    if (this.destroyed) return Promise.reject(new Error("Lfm2GpuModel is destroyed"));
    const existing = this.tensors.get(name);
    if (existing) return Promise.resolve(existing);
    const inFlight = this.loading.get(name);
    if (inFlight) return inFlight;

    const promise = this.uploadTensor(this.reader.requireTensor(name))
      .then((tensor) => {
        this.tensors.set(name, tensor);
        this.loading.delete(name);
        return tensor;
      }, (error) => {
        this.loading.delete(name);
        throw error;
      });
    this.loading.set(name, promise);
    return promise;
  }

  async preloadBlock(layer: number): Promise<void> {
    const kind = this.config.layers[layer];
    if (!kind) throw new RangeError(`LFM2 layer ${layer} does not exist`);
    await this.preload(lfm2BlockTensorNames(layer, kind));
  }

  private async uploadTensor(info: Wq4TensorInfo): Promise<Lfm2GpuTensor> {
    const sourceType = info.sourceType;
    if (sourceType === undefined) throw new Error(`${info.name}: WQ4 v3 sourceType is missing`);

    let format: Lfm2GpuWeightFormat;
    let rowBytes: number;
    const width = info.dimensions[0] ?? 1;
    const rows = info.dimensions.length > 1 ? product(info.dimensions.slice(1)) : 1;

    if (info.encoding === "wq4") {
      if (info.dimensions.length < 2) throw new Error(`${info.name}: WQ4 encoding requires a matrix`);
      if (width % WQ4_BLOCK_SIZE !== 0) {
        throw new Error(`${info.name}: WQ4 row width ${width} is not divisible by ${WQ4_BLOCK_SIZE}`);
      }
      format = "wq4";
      rowBytes = (width / WQ4_BLOCK_SIZE) * WQ4_BYTES_PER_BLOCK;
      const expectedBytes = rows * rowBytes;
      if (expectedBytes !== info.size) {
        throw new Error(`${info.name}: index has ${info.size} B, expected ${expectedBytes} B from WQ4 shape`);
      }
    } else {
      const bytes = scalarBytes(sourceType);
      if (!bytes || (sourceType !== GgmlType.F16 && sourceType !== GgmlType.F32)) {
        throw new Error(
          `${info.name}: raw runtime tensor type ${GgmlType[sourceType] ?? sourceType} is unsupported`,
        );
      }
      format = sourceType === GgmlType.F16 ? "f16" : "f32";
      rowBytes = width * bytes;
      const expectedBytes = rows * rowBytes;
      if (expectedBytes !== info.size) {
        throw new Error(`${info.name}: raw index has ${info.size} B, expected ${expectedBytes} B from shape`);
      }
    }

    const pages = await this.allocatePages(info, rows, rowBytes);
    return {
      name: info.name,
      sourceType,
      format,
      dimensions: [...info.dimensions],
      pages,
      byteLength: info.size,
    };
  }

  private async allocatePages(
    info: Wq4TensorInfo,
    rows: number,
    rowBytes: number,
  ): Promise<Lfm2GpuTensorPage[]> {
    const storageLimit = Number(this.device.limits.maxStorageBufferBindingSize);
    const bufferLimit = Number(this.device.limits.maxBufferSize);
    const requested = this.options.maxPageBytes ?? DEFAULT_PAGE_BYTES;
    const maxPageBytes = Math.min(storageLimit, bufferLimit, requested);
    if (rowBytes > maxPageBytes) {
      throw new Error(`${info.name}: one row is ${rowBytes} B, max page is ${maxPageBytes} B`);
    }

    const rowsPerPage = Math.max(1, Math.min(
      Math.floor(maxPageBytes / rowBytes),
      Number(this.device.limits.maxComputeWorkgroupsPerDimension),
    ));
    const pages: Lfm2GpuTensorPage[] = [];
    let tensorOffset = 0;

    for (let rowStart = 0; rowStart < rows; rowStart += rowsPerPage) {
      const rowCount = Math.min(rowsPerPage, rows - rowStart);
      const byteLength = rowCount * rowBytes;
      const gpuSize = Math.max(4, Math.ceil(byteLength / 4) * 4);
      this.device.pushErrorScope("out-of-memory");
      this.device.pushErrorScope("validation");
      const buffer = this.device.createBuffer({
        label: `lfm2.weight:${info.name}[${rowStart}:${rowStart + rowCount}]`,
        size: gpuSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const validationError = await this.device.popErrorScope();
      const oomError = await this.device.popErrorScope();
      if (validationError || oomError) {
        buffer.destroy();
        const error = validationError ?? oomError;
        throw new Error(
          `${info.name}[${rowStart}:${rowStart + rowCount}]: GPU allocation failed for `
          + `${(gpuSize / MIB).toFixed(1)} MiB: ${error?.message ?? "unknown error"}`,
        );
      }

      try {
        for (let pageOffset = 0; pageOffset < byteLength; pageOffset += UPLOAD_CHUNK_BYTES) {
          const length = Math.min(UPLOAD_CHUNK_BYTES, byteLength - pageOffset);
          const bytes = await this.reader.readTensor(info, tensorOffset + pageOffset, length);
          if ((bytes.byteLength & 3) === 0) {
            this.device.queue.writeBuffer(buffer, pageOffset, bytes);
          } else {
            const padded = new Uint8Array((bytes.byteLength + 3) & ~3);
            padded.set(bytes);
            this.device.queue.writeBuffer(buffer, pageOffset, padded);
          }
        }
        if (this.options.drainUploads ?? true) await this.device.queue.onSubmittedWorkDone();
      } catch (error) {
        buffer.destroy();
        throw error;
      }

      pages.push({ buffer, rowStart, rowCount, byteLength });
      tensorOffset += byteLength;
    }

    return pages;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const tensor of this.tensors.values()) {
      for (const page of tensor.pages) page.buffer.destroy();
    }
    this.tensors.clear();
    this.reader.source.close?.();
  }
}
