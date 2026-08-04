import { DenoFileSource, type RandomAccessSource, GgufReader, GgmlType, type GgufTensorInfo, type GgufValue } from "../../quant/src/gguf";

import { WQ4_BLOCK_SIZE, WQ4_BYTES_PER_BLOCK, Wq4Reader, type Wq4TensorInfo } from "../../quant/src/wq4/reader";

const MIB = 1024 * 1024;
const UPLOAD_CHUNK_BYTES = 16 * MIB;

function asNumber(value: GgufValue, key: string): number {
  if (typeof value === "number") return value;
  if (typeof value === "bigint") return Number(value);
  throw new Error(`${key} must be numeric`);
}

function asNumberArray(value: GgufValue, key: string): number[] {
  if (!Array.isArray(value)) throw new Error(`${key} must be an array`);
  return value.map((v) => {
    if (typeof v === "number") return v;
    if (typeof v === "bigint") return Number(v);
    throw new Error(`${key} contains a non-numeric value`);
  });
}

function elementSize(type: GgmlType): number | null {
  switch (type) {
    case GgmlType.F16:
    case GgmlType.BF16: return 2;
    case GgmlType.F32: return 4;
    default: return null;
  }
}

export type Lfm2LayerKind = "conv" | "attention";
export type GpuWeightFormat = "f16" | "f32" | "wq4";

export interface Lfm2Config {
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

export interface GpuTensorPage {
  buffer: GPUBuffer;
  /** First logical row stored by this page (dimension 1 in GGUF matrices). */
  rowStart: number;
  rowCount: number;
  byteLength: number;
}

export interface GpuTensor {
  name: string;
  /** Original GGUF scalar type. Kept for non-matmul kernels such as RMSNorm. */
  type: GgmlType;
  /** GPU storage format consumed by the selected matmul kernel. */
  format: GpuWeightFormat;
  dimensions: number[];
  pages: GpuTensorPage[];
  byteLength: number;
}

export interface Lfm2LoadProgress {
  tensor: string;
  tensorIndex: number;
  tensorCount: number;
  uploadedBytes: number;
  totalBytes: number;
  allocatedBytes: number;
}

export interface Lfm2LoadOptions {
  onProgress?: (progress: Lfm2LoadProgress) => void;
  /** Optional WQ4 v2 sidecar. GGUF remains the metadata/tokenizer/fallback source. */
  wq4Source?: RandomAccessSource;
  /** Deno convenience used by loadPath(). */
  wq4Path?: string;
  /**
   * Keep pages comfortably below WebGPU's advertised storage-binding ceiling.
   * Native wgpu/Vulkan stacks can be surprisingly unhappy with allocations
   * exactly at the limit, even when plenty of physical VRAM is free.
   */
  maxPageBytes?: number;
  /** Drain queue.writeBuffer staging after each tensor. Slow, but deterministic for bring-up. */
  drainUploads?: boolean;
}

export class Lfm2Model {
  readonly tensors = new Map<string, GpuTensor>();

  private constructor(
    readonly device: GPUDevice,
    readonly reader: GgufReader,
    readonly config: Lfm2Config,
    readonly wq4Reader?: Wq4Reader,
  ) {}

  static async loadPath(
    device: GPUDevice,
    path: string,
    options: Lfm2LoadOptions = {},
  ): Promise<Lfm2Model> {
    const source = await DenoFileSource.open(path);
    const wq4Source = options.wq4Source ?? (options.wq4Path ? await DenoFileSource.open(options.wq4Path) : undefined);
    try {
      return await Lfm2Model.load(device, source, { ...options, wq4Source });
    } catch (error) {
      source.close();
      wq4Source?.close?.();
      throw error;
    }
  }

  static async load(
    device: GPUDevice,
    source: RandomAccessSource,
    options: Lfm2LoadOptions = {},
  ): Promise<Lfm2Model> {
    const reader = await GgufReader.open(source);
    const wq4Reader = options.wq4Source ? await Wq4Reader.open(options.wq4Source) : undefined;
    const architecture = reader.metadata<string>("general.architecture");
    if (architecture !== "lfm2") throw new Error(`Expected lfm2 GGUF, got '${architecture}'`);

    const kvHeadsByLayer = asNumberArray(
      reader.metadata("lfm2.attention.head_count_kv"),
      "lfm2.attention.head_count_kv",
    );
    const blockCount = asNumber(reader.metadata("lfm2.block_count"), "lfm2.block_count");
    if (kvHeadsByLayer.length !== blockCount) {
      throw new Error(`Expected ${blockCount} per-layer KV head entries, got ${kvHeadsByLayer.length}`);
    }

    const hiddenSize = asNumber(reader.metadata("lfm2.embedding_length"), "lfm2.embedding_length");
    const attentionHeads = asNumber(reader.metadata("lfm2.attention.head_count"), "lfm2.attention.head_count");
    const layers: Lfm2LayerKind[] = kvHeadsByLayer.map((n) => n === 0 ? "conv" : "attention");
    const attentionLayerSlots = new Array(blockCount).fill(-1);
    let attentionSlot = 0;
    for (let i = 0; i < blockCount; i++) {
      if (layers[i] === "attention") attentionLayerSlots[i] = attentionSlot++;
    }

    const config: Lfm2Config = {
      architecture: "lfm2",
      blockCount,
      contextLength: asNumber(reader.metadata("lfm2.context_length"), "lfm2.context_length"),
      hiddenSize,
      feedForwardSize: asNumber(reader.metadata("lfm2.feed_forward_length"), "lfm2.feed_forward_length"),
      attentionHeads,
      kvHeadsByLayer,
      headDim: hiddenSize / attentionHeads,
      ropeTheta: asNumber(reader.metadata("lfm2.rope.freq_base"), "lfm2.rope.freq_base"),
      vocabSize: asNumber(reader.metadata("lfm2.vocab_size"), "lfm2.vocab_size"),
      convCacheLength: asNumber(reader.metadata("lfm2.shortconv.l_cache"), "lfm2.shortconv.l_cache"),
      normEpsilon: asNumber(reader.metadata("lfm2.attention.layer_norm_rms_epsilon"), "lfm2.attention.layer_norm_rms_epsilon"),
      bosToken: asNumber(reader.metadata("tokenizer.ggml.bos_token_id"), "tokenizer.ggml.bos_token_id"),
      eosToken: asNumber(reader.metadata("tokenizer.ggml.eos_token_id"), "tokenizer.ggml.eos_token_id"),
      addBosToken: Boolean(reader.metadata("tokenizer.ggml.add_bos_token")),
      addEosToken: Boolean(reader.metadata("tokenizer.ggml.add_eos_token")),
      layers,
      attentionLayerSlots,
    };

    if (config.headDim !== 64) throw new Error(`Initial kernels assume headDim=64, got ${config.headDim}`);
    if (config.convCacheLength !== 3) throw new Error(`Initial kernels assume conv cache length=3, got ${config.convCacheLength}`);

    const model = new Lfm2Model(device, reader, config, wq4Reader);
    await model.uploadAll(options);
    model.validateRequiredTensors();
    return model;
  }

  private async uploadAll(options: Lfm2LoadOptions): Promise<void> {
    const infos = [...this.reader.info.tensors.values()];
    const totalBytes = infos.reduce(
      (sum, info) => sum + (this.wq4Reader?.tensor(info.name)?.size ?? info.byteLength),
      0,
    );
    let uploadedBytes = 0;
    let allocatedBytes = 0;

    for (let tensorIndex = 0; tensorIndex < infos.length; tensorIndex++) {
      const info = infos[tensorIndex]!;
      if (info.type !== GgmlType.F16 && info.type !== GgmlType.F32) {
        throw new Error(
          `${info.name}: initial runtime accepts only F16/F32 GGUF sources, got GGML type ${info.type}. ` +
          `Use LFM2.5-1.2B-Instruct-F16.gguf as the metadata/fallback model.`,
        );
      }

      const wq4 = this.wq4Reader?.tensor(info.name);
      const tensor = wq4
        ? await this.uploadWq4Tensor(info, wq4, options)
        : await this.uploadTensor(info, options);
      this.tensors.set(info.name, tensor);

      uploadedBytes += wq4?.size ?? info.byteLength;
      allocatedBytes += tensor.pages.reduce((sum, page) => sum + page.byteLength, 0);
      options.onProgress?.({
        tensor: `${info.name} [${tensor.format}]`,
        tensorIndex,
        tensorCount: infos.length,
        uploadedBytes,
        totalBytes,
        allocatedBytes,
      });
    }
  }

  private async allocateAndUploadPages(
    name: string,
    rows: number,
    rowBytes: number,
    sourceRead: (byteOffset: number, byteLength: number) => Promise<Uint8Array>,
    options: Lfm2LoadOptions,
  ): Promise<GpuTensorPage[]> {
    const bindingLimit = Number(this.device.limits.maxStorageBufferBindingSize);
    const bufferLimit = Number(this.device.limits.maxBufferSize);
    const requestedPageBytes = options.maxPageBytes ?? (64 * MIB);
    const maxPageBytes = Math.min(bindingLimit, bufferLimit, requestedPageBytes);
    const rowsPerPage = Math.max(1, Math.min(
      Math.floor(maxPageBytes / rowBytes),
      Number(this.device.limits.maxComputeWorkgroupsPerDimension),
    ));

    if (rowBytes > maxPageBytes) {
      throw new Error(`${name}: one logical row is ${rowBytes} B, above storage binding limit ${maxPageBytes} B`);
    }

    const pages: GpuTensorPage[] = [];
    let tensorByteOffset = 0;
    for (let rowStart = 0; rowStart < rows; rowStart += rowsPerPage) {
      const rowCount = Math.min(rowsPerPage, rows - rowStart);
      const byteLength = rowCount * rowBytes;
      const paddedSize = Math.ceil(byteLength / 4) * 4;
      const label = `${name}[${rowStart}:${rowStart + rowCount}]`;

      this.device.pushErrorScope("out-of-memory");
      this.device.pushErrorScope("validation");
      const buffer = this.device.createBuffer({
        label,
        size: paddedSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const validationError = await this.device.popErrorScope();
      const oomError = await this.device.popErrorScope();
      if (validationError || oomError) {
        buffer.destroy();
        const error = validationError ?? oomError;
        throw new Error(
          `GPU allocation failed for ${label}: ${(paddedSize / MIB).toFixed(1)} MiB. ` +
          `device limits: storage=${(bindingLimit / MIB).toFixed(0)} MiB, ` +
          `buffer=${(bufferLimit / MIB).toFixed(0)} MiB. Cause: ${error?.message ?? "unknown"}`,
        );
      }

      for (let pageOffset = 0; pageOffset < byteLength; pageOffset += UPLOAD_CHUNK_BYTES) {
        const chunkSize = Math.min(UPLOAD_CHUNK_BYTES, byteLength - pageOffset);
        const bytes = await sourceRead(tensorByteOffset + pageOffset, chunkSize);
        this.device.queue.writeBuffer(buffer, pageOffset, bytes);
      }

      if (options.drainUploads ?? true) await this.device.queue.onSubmittedWorkDone();

      pages.push({ buffer, rowStart, rowCount, byteLength });
      tensorByteOffset += byteLength;
    }
    return pages;
  }

  private async uploadTensor(info: GgufTensorInfo, options: Lfm2LoadOptions): Promise<GpuTensor> {
    const scalarBytes = elementSize(info.type);
    if (!scalarBytes) throw new Error(`${info.name}: unsupported tensor type ${info.type}`);

    const rowElements = info.dimensions[0] ?? 1;
    const rows = info.dimensions.length >= 2 ? info.dimensions.slice(1).reduce((a, b) => a * b, 1) : 1;
    const rowBytes = rowElements * scalarBytes;
    const pages = await this.allocateAndUploadPages(
      info.name,
      rows,
      rowBytes,
      (offset, length) => this.reader.readTensor(info, offset, length),
      options,
    );

    return {
      name: info.name,
      type: info.type,
      format: info.type === GgmlType.F16 ? "f16" : "f32",
      dimensions: info.dimensions,
      pages,
      byteLength: info.byteLength,
    };
  }

  private async uploadWq4Tensor(
    info: GgufTensorInfo,
    wq4: Wq4TensorInfo,
    options: Lfm2LoadOptions,
  ): Promise<GpuTensor> {
    if (!this.wq4Reader) throw new Error("Internal error: WQ4 tensor without WQ4 reader");
    if (info.dimensions.length < 2) {
      throw new Error(`${info.name}: WQ4 sidecar unexpectedly contains a non-matrix tensor`);
    }
    if (wq4.dimensions.length !== info.dimensions.length || !wq4.dimensions.every((v, i) => v === info.dimensions[i])) {
      throw new Error(`${info.name}: WQ4 shape [${wq4.dimensions.join(", ")}] != GGUF shape [${info.dimensions.join(", ")}]`);
    }
    if (wq4.sourceBytes !== 0 && wq4.sourceBytes !== info.byteLength) {
      throw new Error(`${info.name}: WQ4 was built from ${wq4.sourceBytes} source bytes, GGUF tensor has ${info.byteLength}`);
    }

    const rowElements = info.dimensions[0] ?? 1;
    if (rowElements % WQ4_BLOCK_SIZE !== 0) {
      throw new Error(`${info.name}: WQ4 row width ${rowElements} is not divisible by ${WQ4_BLOCK_SIZE}`);
    }
    const rows = info.dimensions.slice(1).reduce((a, b) => a * b, 1);
    const rowBytes = (rowElements / WQ4_BLOCK_SIZE) * WQ4_BYTES_PER_BLOCK;
    const expectedBytes = rows * rowBytes;
    if (wq4.size !== expectedBytes) {
      throw new Error(`${info.name}: WQ4 index says ${wq4.size} B, expected ${expectedBytes} B from GGUF shape`);
    }

    const pages = await this.allocateAndUploadPages(
      info.name,
      rows,
      rowBytes,
      (offset, length) => this.wq4Reader!.readTensor(wq4, offset, length),
      options,
    );

    return {
      name: info.name,
      type: info.type,
      format: "wq4",
      dimensions: info.dimensions,
      pages,
      byteLength: wq4.size,
    };
  }

  tensor(name: string): GpuTensor {
    const tensor = this.tensors.get(name);
    if (!tensor) throw new Error(`GPU tensor not loaded: ${name}`);
    return tensor;
  }

  private validateRequiredTensors(): void {
    const required = ["token_embd.weight", "token_embd_norm.weight"];
    for (let i = 0; i < this.config.blockCount; i++) {
      required.push(`blk.${i}.attn_norm.weight`, `blk.${i}.ffn_norm.weight`);
      required.push(`blk.${i}.ffn_gate.weight`, `blk.${i}.ffn_up.weight`, `blk.${i}.ffn_down.weight`);
      if (this.config.layers[i] === "conv") {
        required.push(
          `blk.${i}.shortconv.in_proj.weight`,
          `blk.${i}.shortconv.conv.weight`,
          `blk.${i}.shortconv.out_proj.weight`,
        );
      } else {
        required.push(
          `blk.${i}.attn_q.weight`, `blk.${i}.attn_k.weight`, `blk.${i}.attn_v.weight`,
          `blk.${i}.attn_q_norm.weight`, `blk.${i}.attn_k_norm.weight`, `blk.${i}.attn_output.weight`,
        );
      }
    }
    for (const name of required) this.tensor(name);
  }

  destroy(): void {
    for (const tensor of this.tensors.values()) {
      for (const page of tensor.pages) page.buffer.destroy();
    }
    this.tensors.clear();
    this.reader.source.close?.();
    this.wq4Reader?.source.close?.();
  }
}
