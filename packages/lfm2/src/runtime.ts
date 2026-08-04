import { type Sandblaster } from "@sandblaster/core";
import { type GpuTensor, type GpuWeightFormat, Lfm2Model } from "./model.ts";
import {
  createInitialRuntimeState,
  deserializeLlmRuntime,
  LLM_RUNTIME_BYTES,
  serializeLlmRuntime,
  type LlmRuntimeState,
} from "../../schema/src/schema.ts";

const PARAM_BYTES = 64;
const PARAM_BUFFER_BYTES = 8 * 1024 * 1024;
const HEAD_DIM = 64;
const KV_HEADS = 8;
const QUERY_HEADS = 32;
const KV_DIM = KV_HEADS * HEAD_DIM;
const MAX_BRINGUP_CONTEXT = 1024;
const BLOCK_EXACT_DEPTH = 2;
const BLOCK_DEFAULT_CACHE_DEPTH = 2;
const BLOCK_CACHE_MAX_TOKENS = 256;
const BLOCK_BOUNDARY_HISTORY = 4;
const BLOCK_BOUNDARY_REPAIR = 4;
const BLOCK_TAIL_REBUILD_TOKENS = 5;
const BLOCK_REPAIR_CAPACITY = BLOCK_BOUNDARY_HISTORY + BLOCK_BOUNDARY_REPAIR;

const CORE_PIPELINES = [
  "embedding",
  "embedding_wq4",
  "rms_norm",
  "residual_add",
  "silu_mul",
  "shortconv_prefill",
  "shortconv_continue",
  "shortconv_decode",
  "qk_norm_rope",
  "kv_store",
  "attention",
  "arena_copy",
  "argmax",
] as const;

type PipelineName = typeof CORE_PIPELINES[number] | (string & {});

interface OpParams {
  inputOffset?: number;
  outputOffset?: number;
  auxOffset?: number;
  aux2Offset?: number;
  tokenCount?: number;
  inputDim?: number;
  outputDim?: number;
  rowStart?: number;
  rowCount?: number;
  layerIndex?: number;
  attentionSlot?: number;
  mode?: number;
  f0?: number;
  f1?: number;
  u0?: number;
  u1?: number;
}

interface WorkLayout {
  hiddenA: number;
  hiddenB: number;
  tmpH: number;
  tmpA: number;
  tmpB: number;
}

interface ArenaLayout extends WorkLayout {
  repair: WorkLayout;
  logits: number;
  elements: number;
}

export interface MatmulDispatchArgs {
  rowCount: number;
  tokenCount: number;
  inputDim: number;
  outputDim: number;
}

export interface MatmulKernelSpec {
  /** WGSL entry point compiled into the shared LFM runtime pipeline layout. */
  entryPoint: string;
  /** Optional WGSL appended to runtime.wgsl, useful for drop-in kernel experiments. */
  wgsl?: string;
  /** Override dispatch geometry if a kernel is not [row, token]. */
  workgroups?: (args: MatmulDispatchArgs) => [number, number?, number?];
}

export interface Lfm2RuntimeOptions {
  contextCapacity?: number;
  maxNewTokens?: number;
  /** Swap only matmul implementations without touching layer scheduling/cache code. */
  matmulKernels?: Partial<Record<GpuWeightFormat, MatmulKernelSpec>>;
}

export interface GenerateOptions {
  maxNewTokens?: number;
  /**
   * Diagnostic mode: submit prefill and decode separately and synchronize once
   * between them so wall-clock timings can be reported independently. There is
   * still no token-by-token CPU/GPU roundtrip.
   */
  profile?: boolean;
}

export interface GenerateTimings {
  prefillMs: number;
  decodeMs: number;
  readbackMs: number;
  totalMs: number;
  promptTokens: number;
  scheduledDecodeSteps: number;
  /** Present for generateFromBlocks(): cache depth used by the prefill. Depth > 2 is approximate. */
  cacheDepth?: number;
  cachedBlocks?: number;
  cachedTokens?: number;
  liveQueryTokens?: number;
  repairedTokens?: number;
}

export interface GenerateResult {
  tokenIds: number[];
  state: LlmRuntimeState;
  timings?: GenerateTimings;
}

export interface CacheBlockOptions {
  /** Maximum standalone prefix depth retained for this block. */
  depth?: number;
}

export class Lfm2CachedBlock {
  readonly tokenBytes: number;

  constructor(
    readonly id: number,
    readonly tokenIds: readonly number[],
    readonly tokenBuffer: GPUBuffer,
    readonly cacheDepth: number,
    /** Hidden-state checkpoints at every reusable cache frontier up to cacheDepth. */
    readonly stateBuffers: ReadonlyMap<number, GPUBuffer>,
    /** Conv layers whose standalone decode tail is retained in convStateBuffer. */
    readonly convLayers: readonly number[],
    readonly convStateBuffer: GPUBuffer,
    readonly convLayerBytes: number,
    readonly cacheMs: number,
  ) {
    this.tokenBytes = tokenIds.length * 4;
  }

  get tokenCount(): number {
    return this.tokenIds.length;
  }

  stateAtDepth(depth: number): GPUBuffer {
    const buffer = this.stateBuffers.get(depth);
    if (!buffer) {
      throw new Error(`Cached block ${this.id} has no checkpoint at depth ${depth} (max ${this.cacheDepth})`);
    }
    return buffer;
  }

  /** Compatibility aliases for the original depth-5 experiment. */
  get exactHiddenBuffer(): GPUBuffer { return this.stateAtDepth(BLOCK_EXACT_DEPTH); }
  get hiddenBuffer(): GPUBuffer { return this.stateAtDepth(this.cacheDepth); }

  convStateOffset(layer: number): number {
    const index = this.convLayers.indexOf(layer);
    if (index < 0) throw new Error(`Cached block ${this.id} has no conv tail for layer ${layer}`);
    return index * this.convLayerBytes;
  }

  get gpuBytes(): number {
    let bytes = this.tokenBuffer.size + this.convStateBuffer.size;
    for (const buffer of this.stateBuffers.values()) bytes += buffer.size;
    return bytes;
  }

  destroy(): void {
    this.tokenBuffer.destroy();
    for (const buffer of this.stateBuffers.values()) buffer.destroy();
    this.convStateBuffer.destroy();
  }
}

function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

function ceilDiv(value: number, divisor: number): number {
  return Math.ceil(value / divisor);
}

function gpuBufferSize(bytes: number): number {
  return Math.max(4, align(bytes, 4));
}

class ParamWriter {
  private readonly data: ArrayBuffer;
  private readonly view: DataView;
  private cursor = 0;

  constructor(
    private readonly stride: number,
    capacityBytes = PARAM_BUFFER_BYTES,
  ) {
    this.data = new ArrayBuffer(capacityBytes);
    this.view = new DataView(this.data);
  }

  reset(): void {
    this.cursor = 0;
  }

  alloc(params: OpParams): number {
    if (this.cursor + this.stride > this.data.byteLength) {
      throw new Error(`Op parameter buffer exhausted at ${this.cursor} B; increase PARAM_BUFFER_BYTES`);
    }
    const offset = this.cursor;
    let p = offset;
    const u32 = (v = 0) => { this.view.setUint32(p, v >>> 0, true); p += 4; };
    const f32 = (v = 0) => { this.view.setFloat32(p, v, true); p += 4; };

    u32(params.inputOffset);
    u32(params.outputOffset);
    u32(params.auxOffset);
    u32(params.aux2Offset);
    u32(params.tokenCount);
    u32(params.inputDim);
    u32(params.outputDim);
    u32(params.rowStart);
    u32(params.rowCount);
    u32(params.layerIndex);
    u32(params.attentionSlot);
    u32(params.mode);
    f32(params.f0);
    f32(params.f1);
    u32(params.u0);
    u32(params.u1);

    this.cursor += this.stride;
    return offset;
  }

  get usedBytes(): number {
    return this.cursor;
  }

  usedView(): Uint8Array {
    return new Uint8Array(this.data, 0, this.cursor);
  }
}

export class Lfm2Runtime {
  readonly device: GPUDevice;
  readonly contextCapacity: number;
  readonly defaultMaxNewTokens: number;
  readonly blockCacheDepth = BLOCK_DEFAULT_CACHE_DEPTH;
  readonly blockCacheMaxTokens = BLOCK_CACHE_MAX_TOKENS;
  readonly blockCacheDepths: readonly number[];

  readonly runtimeBuffer: GPUBuffer;
  readonly tokenBuffer: GPUBuffer;
  readonly arenaBuffer: GPUBuffer;
  readonly kvBuffer: GPUBuffer;
  readonly convBuffer: GPUBuffer;

  private readonly arena: ArenaLayout;
  private readonly runtimeByteSize: number;
  private readonly runtimeCpu: ArrayBuffer;
  private readonly runtimeView: DataView;

  private readonly paramStride: number;
  private readonly paramBuffer: GPUBuffer;
  private readonly paramWriter: ParamWriter;
  private readonly group0: GPUBindGroup;
  private readonly group0Layout: GPUBindGroupLayout;
  private readonly group1Layout: GPUBindGroupLayout;
  private readonly pipelineLayout: GPUPipelineLayout;
  private readonly pipelines = new Map<PipelineName, GPUComputePipeline>();
  private readonly matmulKernels: Record<GpuWeightFormat, MatmulKernelSpec>;

  private readonly dummyWeightRaw: GPUBuffer;
  private readonly dummyWeight32: GPUBuffer;
  private readonly weightGroups = new Map<string, GPUBindGroup>();

  private readonly telemetryStaging: GPUBuffer;
  private telemetryInFlight = false;
  private nextCachedBlockId = 0;
  private readonly cachedBlocks = new Set<Lfm2CachedBlock>();

  private constructor(
    private readonly engine: Sandblaster<any>,
    readonly model: Lfm2Model,
    options: {
      contextCapacity: number;
      maxNewTokens: number;
      matmulKernels: Record<GpuWeightFormat, MatmulKernelSpec>;
    },
    runtimeWgsl: string,
  ) {
    this.device = engine.device;
    this.contextCapacity = options.contextCapacity;
    this.defaultMaxNewTokens = options.maxNewTokens;
    this.matmulKernels = options.matmulKernels;

    if (model.device !== this.device) throw new Error("Lfm2Model and Lfm2Runtime must use the same GPUDevice");
    if (this.contextCapacity > MAX_BRINGUP_CONTEXT) {
      throw new Error(`Bring-up attention kernel is capped at ${MAX_BRINGUP_CONTEXT} tokens; got ${this.contextCapacity}`);
    }
    if (this.contextCapacity > model.config.contextLength) {
      throw new Error(`contextCapacity ${this.contextCapacity} exceeds model context ${model.config.contextLength}`);
    }
    if (model.config.headDim !== HEAD_DIM || model.config.attentionHeads !== QUERY_HEADS) {
      throw new Error("Initial WGSL is specialized for LFM2.5-1.2B (32 Q heads, headDim=64)");
    }
    if (model.config.layers.slice(0, BLOCK_EXACT_DEPTH).some((kind) => kind !== "conv")) {
      throw new Error(`Exact block prefix requires the first ${BLOCK_EXACT_DEPTH} layers to be conv`);
    }
    if (model.config.layers[BLOCK_EXACT_DEPTH] !== "attention") {
      throw new Error(`Block cache expects layer ${BLOCK_EXACT_DEPTH} to be the first attention layer`);
    }
    this.blockCacheDepths = this.computeBlockCacheDepths();
    if (!this.blockCacheDepths.includes(BLOCK_DEFAULT_CACHE_DEPTH)) {
      throw new Error(`Default cache depth ${BLOCK_DEFAULT_CACHE_DEPTH} is not a reusable frontier: ${this.blockCacheDepths.join(", ")}`);
    }
    const attentionLayers = model.config.layers.filter((x) => x === "attention").length;
    if (model.config.kvHeadsByLayer.some((n, i) => model.config.layers[i] === "attention" && n !== KV_HEADS)) {
      throw new Error("Initial WGSL is specialized for 8 KV heads");
    }

    this.runtimeByteSize = LLM_RUNTIME_BYTES;
    this.runtimeCpu = new ArrayBuffer(this.runtimeByteSize);
    this.runtimeView = new DataView(this.runtimeCpu);

    this.runtimeBuffer = this.device.createBuffer({
      label: "lfm2.runtime",
      size: gpuBufferSize(this.runtimeByteSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.tokenBuffer = this.device.createBuffer({
      label: "lfm2.tokens",
      size: gpuBufferSize((this.contextCapacity + this.defaultMaxNewTokens) * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.arena = this.createArenaLayout();
    const arenaBytes = this.arena.elements * 4;
    if (arenaBytes > Number(this.device.limits.maxStorageBufferBindingSize)) {
      throw new Error(`Activation arena is ${(arenaBytes / 1048576).toFixed(1)} MiB, above maxStorageBufferBindingSize`);
    }
    this.arenaBuffer = this.device.createBuffer({
      label: "lfm2.arena.f32",
      size: gpuBufferSize(arenaBytes),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const kvElements = attentionLayers * 2 * this.contextCapacity * KV_DIM;
    this.kvBuffer = this.device.createBuffer({
      label: "lfm2.kv.f32",
      size: gpuBufferSize(kvElements * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const convElements = model.config.blockCount * model.config.hiddenSize * model.config.convCacheLength;
    this.convBuffer = this.device.createBuffer({
      label: "lfm2.conv-cache.f32",
      size: gpuBufferSize(convElements * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.paramStride = align(PARAM_BYTES, Number(this.device.limits.minUniformBufferOffsetAlignment));
    this.paramBuffer = this.device.createBuffer({
      label: "lfm2.op-params",
      size: PARAM_BUFFER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.paramWriter = new ParamWriter(this.paramStride);

    this.group0Layout = this.device.createBindGroupLayout({
      label: "lfm2.runtime-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: PARAM_BYTES } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.group1Layout = this.device.createBindGroupLayout({
      label: "lfm2.weight-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });
    this.pipelineLayout = this.device.createPipelineLayout({
      label: "lfm2.pipeline-layout",
      bindGroupLayouts: [this.group0Layout, this.group1Layout],
    });

    this.group0 = this.device.createBindGroup({
      label: "lfm2.runtime-bg",
      layout: this.group0Layout,
      entries: [
        { binding: 0, resource: { buffer: this.paramBuffer, offset: 0, size: PARAM_BYTES } },
        { binding: 1, resource: { buffer: this.runtimeBuffer } },
        { binding: 2, resource: { buffer: this.tokenBuffer } },
        { binding: 3, resource: { buffer: this.arenaBuffer } },
        { binding: 4, resource: { buffer: this.kvBuffer } },
        { binding: 5, resource: { buffer: this.convBuffer } },
      ],
    });

    this.dummyWeightRaw = this.device.createBuffer({
      label: "lfm2.dummy-weight-raw",
      size: 4,
      usage: GPUBufferUsage.STORAGE,
    });
    this.dummyWeight32 = this.device.createBuffer({
      label: "lfm2.dummy-weight32",
      size: 4,
      usage: GPUBufferUsage.STORAGE,
    });

    this.telemetryStaging = this.device.createBuffer({
      label: "lfm2.telemetry-staging",
      size: gpuBufferSize(this.runtimeByteSize),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Compiled asynchronously by create(); argument retained so constructor owns all resource setup.
    void runtimeWgsl;
  }

  private computeBlockCacheDepths(): number[] {
    const depths = [BLOCK_EXACT_DEPTH];
    for (let layer = BLOCK_EXACT_DEPTH + 1; layer < this.model.config.blockCount; layer++) {
      if (this.model.config.layers[layer] === "attention") depths.push(layer);
    }
    if (depths[depths.length - 1] !== this.model.config.blockCount) depths.push(this.model.config.blockCount);
    return depths;
  }

  private resolveBlockCacheDepth(depth = BLOCK_DEFAULT_CACHE_DEPTH): number {
    if (!Number.isInteger(depth) || !this.blockCacheDepths.includes(depth)) {
      throw new Error(`Cache depth ${depth} is not supported; choose one of: ${this.blockCacheDepths.join(", ")}`);
    }
    return depth;
  }

  static async create(
    engine: Sandblaster<any>,
    model: Lfm2Model,
    options: Lfm2RuntimeOptions = {},
  ): Promise<Lfm2Runtime> {
    const defaults: Record<GpuWeightFormat, MatmulKernelSpec> = {
      f16: { entryPoint: "matmul_f16" },
      f32: { entryPoint: "matmul_f32" },
      wq4: { entryPoint: "matmul_wq4" },
    };
    const resolved = {
      contextCapacity: options.contextCapacity ?? 1024,
      maxNewTokens: options.maxNewTokens ?? 128,
      matmulKernels: {
        f16: options.matmulKernels?.f16 ?? defaults.f16,
        f32: options.matmulKernels?.f32 ?? defaults.f32,
        wq4: options.matmulKernels?.wq4 ?? defaults.wq4,
      },
    };
    if (resolved.maxNewTokens < 1) throw new Error("maxNewTokens must be >= 1");

    const shaderUrl = new URL("./shaders/runtime.wgsl", import.meta.url);
    const deno = (globalThis as any).Deno;
    let body: string;
    if (shaderUrl.protocol === "file:" && deno?.readTextFile) {
      body = await deno.readTextFile(shaderUrl);
    } else {
      const response = await fetch(shaderUrl);
      if (!response.ok) {
        throw new Error(`Could not load LFM2 WGSL (${response.status}) from ${shaderUrl}`);
      }
      body = await response.text();
    }
    const extensions = Object.values(resolved.matmulKernels)
      .map((kernel) => kernel.wgsl?.trim())
      .filter((source): source is string => Boolean(source));
    if (extensions.length > 0) body += `\n\n${extensions.join("\n\n")}\n`;

    const runtime = new Lfm2Runtime(engine, model, resolved, body);
    await runtime.compile(body, Object.values(resolved.matmulKernels).map((kernel) => kernel.entryPoint));
    return runtime;
  }

  private createArenaLayout(): ArenaLayout {
    const c = this.contextCapacity;
    const h = this.model.config.hiddenSize;
    const ff = this.model.config.feedForwardSize;
    const scratchWidth = Math.max(ff, 3 * h);
    let cursor = 0;
    const take = (elements: number) => {
      const out = cursor;
      cursor += elements;
      return out;
    };
    const work = (tokens: number): WorkLayout => ({
      hiddenA: take(tokens * h),
      hiddenB: take(tokens * h),
      tmpH: take(tokens * h),
      tmpA: take(tokens * scratchWidth),
      tmpB: take(tokens * scratchWidth),
    });
    const main = work(c);
    const repair = work(BLOCK_REPAIR_CAPACITY);
    const logits = take(this.model.config.vocabSize);
    return { ...main, repair, logits, elements: cursor };
  }

  private async compile(code: string, matmulEntryPoints: readonly string[]): Promise<void> {
    const module = this.device.createShaderModule({ label: "lfm2.runtime.wgsl", code });
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter((m) => m.type === "error");
    if (errors.length) {
      throw new Error(errors.map((m) => `[${m.lineNum}:${m.linePos}] ${m.message}`).join("\n"));
    }
    const entryPoints = [...new Set<string>([...CORE_PIPELINES, ...matmulEntryPoints])];
    await Promise.all(entryPoints.map(async (name) => {
      const pipeline = await this.device.createComputePipelineAsync({
        label: `lfm2.${name}`,
        layout: this.pipelineLayout,
        compute: { module, entryPoint: name },
      });
      this.pipelines.set(name, pipeline);
    }));
  }

  private weightGroup(tensor: GpuTensor, pageIndex = 0): GPUBindGroup {
    const page = tensor.pages[pageIndex];
    if (!page) throw new Error(`${tensor.name}: page ${pageIndex} missing`);
    const key = `${tensor.name}:${tensor.format}:${pageIndex}`;
    let group = this.weightGroups.get(key);
    if (group) return group;

    const isRaw = tensor.format === "f16" || tensor.format === "wq4";
    const isF32 = tensor.format === "f32";
    group = this.device.createBindGroup({
      label: `lfm2.weight-bg:${key}`,
      layout: this.group1Layout,
      entries: [
        { binding: 0, resource: { buffer: isRaw ? page.buffer : this.dummyWeightRaw } },
        { binding: 1, resource: { buffer: isF32 ? page.buffer : this.dummyWeight32 } },
      ],
    });
    this.weightGroups.set(key, group);
    return group;
  }

  private dummyWeightGroup(): GPUBindGroup {
    const key = "__dummy__";
    let group = this.weightGroups.get(key);
    if (!group) {
      group = this.device.createBindGroup({
        label: "lfm2.weight-bg:dummy",
        layout: this.group1Layout,
        entries: [
          { binding: 0, resource: { buffer: this.dummyWeightRaw } },
          { binding: 1, resource: { buffer: this.dummyWeight32 } },
        ],
      });
      this.weightGroups.set(key, group);
    }
    return group;
  }

  private dispatch(
    pass: GPUComputePassEncoder,
    pipeline: PipelineName,
    params: OpParams,
    workgroups: [number, number?, number?],
    weights: GPUBindGroup = this.dummyWeightGroup(),
  ): void {
    const p = this.paramWriter.alloc(params);
    pass.setPipeline(this.pipelines.get(pipeline)!);
    pass.setBindGroup(0, this.group0, [p]);
    pass.setBindGroup(1, weights);
    pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1);
  }

  private matrix(
    pass: GPUComputePassEncoder,
    tensorName: string,
    inputOffset: number,
    outputOffset: number,
    tokenCount: number,
    inputDim: number,
    outputDim: number,
  ): void {
    const tensor = this.model.tensor(tensorName);
    if ((tensor.dimensions[0] ?? 0) !== inputDim) {
      throw new Error(`${tensorName}: expected input dimension ${inputDim}, tensor is [${tensor.dimensions.join(", ")}]`);
    }
    const rows = tensor.dimensions.length > 1 ? tensor.dimensions.slice(1).reduce((a, b) => a * b, 1) : 1;
    if (rows !== outputDim) {
      throw new Error(`${tensorName}: expected ${outputDim} output rows, tensor has ${rows}`);
    }
    const kernel = this.matmulKernels[tensor.format];
    if (!kernel) throw new Error(`${tensorName}: no matmul kernel registered for ${tensor.format}`);
    if (tensor.format === "wq4" && inputDim % 32 !== 0) {
      throw new Error(`${tensorName}: WQ4 input dimension ${inputDim} must be divisible by 32`);
    }
    for (let pageIndex = 0; pageIndex < tensor.pages.length; pageIndex++) {
      const page = tensor.pages[pageIndex]!;
      const workgroups = kernel.workgroups?.({
        rowCount: page.rowCount,
        tokenCount,
        inputDim,
        outputDim,
      }) ?? [page.rowCount, tokenCount];
      this.dispatch(pass, kernel.entryPoint, {
        inputOffset,
        outputOffset,
        tokenCount,
        inputDim,
        outputDim,
        rowStart: page.rowStart,
        rowCount: page.rowCount,
      }, workgroups, this.weightGroup(tensor, pageIndex));
    }
  }

  private norm(
    pass: GPUComputePassEncoder,
    tensorName: string,
    inputOffset: number,
    outputOffset: number,
    tokenCount: number,
    dim: number,
  ): void {
    const tensor = this.model.tensor(tensorName);
    if (tensor.format !== "f32" || tensor.pages.length !== 1) {
      throw new Error(`${tensorName}: bring-up RMSNorm expects one F32 tensor page`);
    }
    this.dispatch(pass, "rms_norm", {
      inputOffset,
      outputOffset,
      tokenCount,
      inputDim: dim,
      f0: this.model.config.normEpsilon,
    }, [tokenCount], this.weightGroup(tensor));
  }

  private residual(
    pass: GPUComputePassEncoder,
    left: number,
    right: number,
    output: number,
    tokenCount: number,
    dim: number,
  ): void {
    this.dispatch(pass, "residual_add", {
      inputOffset: left,
      auxOffset: right,
      outputOffset: output,
      tokenCount,
      inputDim: dim,
    }, [ceilDiv(tokenCount * dim, 256)]);
  }

  private embed(
    pass: GPUComputePassEncoder,
    tokenCount: number,
    decode: boolean,
    work: WorkLayout = this.arena,
    tokenOffset = 0,
  ): void {
    const tensor = this.model.tensor("token_embd.weight");
    if (tensor.format !== "f16" && tensor.format !== "wq4") {
      throw new Error(`Embedding does not support ${tensor.format} token_embd.weight`);
    }
    const kernel: PipelineName = tensor.format === "wq4" ? "embedding_wq4" : "embedding";
    const h = this.model.config.hiddenSize;
    if (tensor.format === "wq4" && h % 32 !== 0) {
      throw new Error(`WQ4 embedding dimension ${h} must be divisible by 32`);
    }
    for (let pageIndex = 0; pageIndex < tensor.pages.length; pageIndex++) {
      const page = tensor.pages[pageIndex]!;
      this.dispatch(pass, kernel, {
        outputOffset: work.hiddenA,
        tokenCount,
        outputDim: h,
        rowStart: page.rowStart,
        rowCount: page.rowCount,
        mode: decode ? 1 : 0,
        u0: tokenOffset,
      }, [ceilDiv(tokenCount * h, 256)], this.weightGroup(tensor, pageIndex));
    }
  }

  private operatorLayer(
    pass: GPUComputePassEncoder,
    layer: number,
    tokenCount: number,
    decode: boolean,
    work: WorkLayout = this.arena,
  ): void {
    const h = this.model.config.hiddenSize;
    const kind = this.model.config.layers[layer]!;
    this.norm(pass, `blk.${layer}.attn_norm.weight`, work.hiddenA, work.tmpH, tokenCount, h);

    if (kind === "conv") {
      this.matrix(pass, `blk.${layer}.shortconv.in_proj.weight`, work.tmpH, work.tmpA, tokenCount, h, 3 * h);
      const convWeight = this.model.tensor(`blk.${layer}.shortconv.conv.weight`);
      if (convWeight.format !== "f32" || convWeight.pages.length !== 1) {
        throw new Error(`blk.${layer}.shortconv.conv.weight must be one F32 page`);
      }
      this.dispatch(pass, decode ? "shortconv_decode" : "shortconv_prefill", {
        inputOffset: work.tmpA,
        outputOffset: work.tmpH,
        tokenCount,
        inputDim: h,
        layerIndex: layer,
      }, [ceilDiv((decode ? 1 : tokenCount) * h, 256)], this.weightGroup(convWeight));
      this.matrix(pass, `blk.${layer}.shortconv.out_proj.weight`, work.tmpH, work.tmpA, tokenCount, h, h);
      return;
    }

    const kvDim = this.model.config.kvHeadsByLayer[layer]! * HEAD_DIM;
    // Attention is never executed in the small repair work area. Its tmpB
    // allocation therefore keeps the main-context stride used by the original
    // bring-up kernels.
    const vOffset = work.tmpB + this.contextCapacity * kvDim;
    this.matrix(pass, `blk.${layer}.attn_q.weight`, work.tmpH, work.tmpA, tokenCount, h, h);
    this.matrix(pass, `blk.${layer}.attn_k.weight`, work.tmpH, work.tmpB, tokenCount, h, kvDim);
    this.matrix(pass, `blk.${layer}.attn_v.weight`, work.tmpH, vOffset, tokenCount, h, kvDim);

    const qNorm = this.model.tensor(`blk.${layer}.attn_q_norm.weight`);
    const kNorm = this.model.tensor(`blk.${layer}.attn_k_norm.weight`);
    if (qNorm.format !== "f32" || qNorm.pages.length !== 1 || kNorm.format !== "f32" || kNorm.pages.length !== 1) {
      throw new Error(`blk.${layer}: Q/K norm weights must be one F32 page each`);
    }
    const common = {
      inputOffset: work.tmpA,
      auxOffset: work.tmpB,
      tokenCount,
      mode: decode ? 1 : 0,
      f0: this.model.config.normEpsilon,
      f1: this.model.config.ropeTheta,
    };
    this.dispatch(pass, "qk_norm_rope", { ...common, u0: 0 }, [QUERY_HEADS, tokenCount], this.weightGroup(qNorm));
    this.dispatch(pass, "qk_norm_rope", { ...common, u0: 1 }, [KV_HEADS, tokenCount], this.weightGroup(kNorm));

    const attentionSlot = this.model.config.attentionLayerSlots[layer]!;
    this.dispatch(pass, "kv_store", {
      inputOffset: work.tmpB,
      auxOffset: vOffset,
      tokenCount,
      attentionSlot,
      mode: decode ? 1 : 0,
    }, [ceilDiv(tokenCount * kvDim, 256)]);
    this.dispatch(pass, "attention", {
      inputOffset: work.tmpA,
      outputOffset: work.tmpH,
      tokenCount,
      attentionSlot,
      mode: decode ? 1 : 0,
    }, [QUERY_HEADS, tokenCount]);
    this.matrix(pass, `blk.${layer}.attn_output.weight`, work.tmpH, work.tmpA, tokenCount, h, h);
  }

  /**
   * Continue from an already resident prefix. mode=2 means token positions are
   * absolute (positionBase + local token index), while token IDs still come
   * from tokenBuffer. Conv layers consume/update the resident conv tail and
   * attention appends K/V after the cached context.
   */
  private operatorLayerContinuation(
    pass: GPUComputePassEncoder,
    layer: number,
    tokenCount: number,
    positionBase: number,
    work: WorkLayout = this.arena,
  ): void {
    const h = this.model.config.hiddenSize;
    const kind = this.model.config.layers[layer]!;
    this.norm(pass, `blk.${layer}.attn_norm.weight`, work.hiddenA, work.tmpH, tokenCount, h);

    if (kind === "conv") {
      this.matrix(pass, `blk.${layer}.shortconv.in_proj.weight`, work.tmpH, work.tmpA, tokenCount, h, 3 * h);
      const convWeight = this.model.tensor(`blk.${layer}.shortconv.conv.weight`);
      if (convWeight.format !== "f32" || convWeight.pages.length !== 1) {
        throw new Error(`blk.${layer}.shortconv.conv.weight must be one F32 page`);
      }
      this.dispatch(pass, "shortconv_continue", {
        inputOffset: work.tmpA,
        outputOffset: work.tmpH,
        tokenCount,
        inputDim: h,
        layerIndex: layer,
        mode: 2,
        u1: positionBase,
      }, [h], this.weightGroup(convWeight));
      this.matrix(pass, `blk.${layer}.shortconv.out_proj.weight`, work.tmpH, work.tmpA, tokenCount, h, h);
      return;
    }

    const kvDim = this.model.config.kvHeadsByLayer[layer]! * HEAD_DIM;
    const vOffset = work.tmpB + this.contextCapacity * kvDim;
    this.matrix(pass, `blk.${layer}.attn_q.weight`, work.tmpH, work.tmpA, tokenCount, h, h);
    this.matrix(pass, `blk.${layer}.attn_k.weight`, work.tmpH, work.tmpB, tokenCount, h, kvDim);
    this.matrix(pass, `blk.${layer}.attn_v.weight`, work.tmpH, vOffset, tokenCount, h, kvDim);

    const qNorm = this.model.tensor(`blk.${layer}.attn_q_norm.weight`);
    const kNorm = this.model.tensor(`blk.${layer}.attn_k_norm.weight`);
    if (qNorm.format !== "f32" || qNorm.pages.length !== 1 || kNorm.format !== "f32" || kNorm.pages.length !== 1) {
      throw new Error(`blk.${layer}: Q/K norm weights must be one F32 page each`);
    }
    const common = {
      inputOffset: work.tmpA,
      auxOffset: work.tmpB,
      tokenCount,
      mode: 2,
      f0: this.model.config.normEpsilon,
      f1: this.model.config.ropeTheta,
      u1: positionBase,
    };
    this.dispatch(pass, "qk_norm_rope", { ...common, u0: 0 }, [QUERY_HEADS, tokenCount], this.weightGroup(qNorm));
    this.dispatch(pass, "qk_norm_rope", { ...common, u0: 1 }, [KV_HEADS, tokenCount], this.weightGroup(kNorm));

    const attentionSlot = this.model.config.attentionLayerSlots[layer]!;
    this.dispatch(pass, "kv_store", {
      inputOffset: work.tmpB,
      auxOffset: vOffset,
      tokenCount,
      attentionSlot,
      mode: 2,
      u1: positionBase,
    }, [ceilDiv(tokenCount * kvDim, 256)]);
    this.dispatch(pass, "attention", {
      inputOffset: work.tmpA,
      outputOffset: work.tmpH,
      tokenCount,
      attentionSlot,
      mode: 2,
      u1: positionBase,
    }, [QUERY_HEADS, tokenCount]);
    this.matrix(pass, `blk.${layer}.attn_output.weight`, work.tmpH, work.tmpA, tokenCount, h, h);
  }

  private ffnLayer(
    pass: GPUComputePassEncoder,
    layer: number,
    tokenCount: number,
    work: WorkLayout = this.arena,
  ): void {
    const h = this.model.config.hiddenSize;
    const ff = this.model.config.feedForwardSize;
    this.norm(pass, `blk.${layer}.ffn_norm.weight`, work.hiddenB, work.tmpH, tokenCount, h);
    this.matrix(pass, `blk.${layer}.ffn_gate.weight`, work.tmpH, work.tmpA, tokenCount, h, ff);
    this.matrix(pass, `blk.${layer}.ffn_up.weight`, work.tmpH, work.tmpB, tokenCount, h, ff);
    this.dispatch(pass, "silu_mul", {
      inputOffset: work.tmpA,
      auxOffset: work.tmpB,
      outputOffset: work.tmpA,
      tokenCount,
      inputDim: ff,
    }, [ceilDiv(tokenCount * ff, 256)]);
    this.matrix(pass, `blk.${layer}.ffn_down.weight`, work.tmpA, work.tmpH, tokenCount, ff, h);
    this.residual(pass, work.hiddenB, work.tmpH, work.hiddenA, tokenCount, h);
  }

  private forwardLayers(
    pass: GPUComputePassEncoder,
    startLayer: number,
    endLayer: number,
    tokenCount: number,
    decode: boolean,
    work: WorkLayout = this.arena,
  ): void {
    const h = this.model.config.hiddenSize;
    for (let layer = startLayer; layer < endLayer; layer++) {
      this.operatorLayer(pass, layer, tokenCount, decode, work);
      this.residual(pass, work.hiddenA, work.tmpA, work.hiddenB, tokenCount, h);
      this.ffnLayer(pass, layer, tokenCount, work);
    }
  }

  private forwardLayersContinuation(
    pass: GPUComputePassEncoder,
    startLayer: number,
    endLayer: number,
    tokenCount: number,
    positionBase: number,
    work: WorkLayout = this.arena,
  ): void {
    const h = this.model.config.hiddenSize;
    for (let layer = startLayer; layer < endLayer; layer++) {
      this.operatorLayerContinuation(pass, layer, tokenCount, positionBase, work);
      this.residual(pass, work.hiddenA, work.tmpA, work.hiddenB, tokenCount, h);
      this.ffnLayer(pass, layer, tokenCount, work);
    }
  }

  /** Exact context-independent prefix for LFM2.5-1.2B: embedding + conv layers 0..1. */
  private forwardBlockExactPrefix(
    pass: GPUComputePassEncoder,
    tokenCount: number,
    work: WorkLayout = this.arena,
    tokenOffset = 0,
  ): void {
    this.embed(pass, tokenCount, false, work, tokenOffset);
    this.forwardLayers(pass, 0, BLOCK_EXACT_DEPTH, tokenCount, false, work);
  }

  /**
   * Populate prompt K/V for a cached-through attention layer without
   * recomputing its Q/attention/output/FFN. hiddenA must contain that layer's
   * cached input frontier in final prompt order; K/V is rebuilt with the
   * request's real token positions before generation continues.
   */
  private bootstrapAttentionKv(pass: GPUComputePassEncoder, layer: number, tokenCount: number): void {
    const h = this.model.config.hiddenSize;
    const kvDim = this.model.config.kvHeadsByLayer[layer]! * HEAD_DIM;
    if (this.model.config.layers[layer] !== "attention") throw new Error(`Layer ${layer} is not attention`);

    this.norm(pass, `blk.${layer}.attn_norm.weight`, this.arena.hiddenA, this.arena.tmpH, tokenCount, h);
    const vOffset = this.arena.tmpB + this.contextCapacity * kvDim;
    this.matrix(pass, `blk.${layer}.attn_k.weight`, this.arena.tmpH, this.arena.tmpB, tokenCount, h, kvDim);
    this.matrix(pass, `blk.${layer}.attn_v.weight`, this.arena.tmpH, vOffset, tokenCount, h, kvDim);

    const kNorm = this.model.tensor(`blk.${layer}.attn_k_norm.weight`);
    if (kNorm.format !== "f32" || kNorm.pages.length !== 1) {
      throw new Error(`blk.${layer}: K norm weight must be one F32 page`);
    }
    this.dispatch(pass, "qk_norm_rope", {
      inputOffset: this.arena.tmpA,
      auxOffset: this.arena.tmpB,
      tokenCount,
      mode: 0,
      f0: this.model.config.normEpsilon,
      f1: this.model.config.ropeTheta,
      u0: 1,
    }, [KV_HEADS, tokenCount], this.weightGroup(kNorm));

    const attentionSlot = this.model.config.attentionLayerSlots[layer]!;
    this.dispatch(pass, "kv_store", {
      inputOffset: this.arena.tmpB,
      auxOffset: vOffset,
      tokenCount,
      attentionSlot,
      mode: 0,
    }, [ceilDiv(tokenCount * kvDim, 256)]);
  }

  private sampleFromHidden(pass: GPUComputePassEncoder, tokenCount: number, decode: boolean): void {
    const h = this.model.config.hiddenSize;
    const finalInput = decode
      ? this.arena.hiddenA
      : this.arena.hiddenA + (tokenCount - 1) * h;
    this.norm(pass, "token_embd_norm.weight", finalInput, this.arena.tmpH, 1, h);
    this.matrix(pass, "token_embd.weight", this.arena.tmpH, this.arena.logits, 1, h, this.model.config.vocabSize);
    this.dispatch(pass, "argmax", {
      inputOffset: this.arena.logits,
      inputDim: this.model.config.vocabSize,
      mode: decode ? 1 : 0,
    }, [1]);
  }

  private forwardAndSample(pass: GPUComputePassEncoder, tokenCount: number, decode: boolean): void {
    this.embed(pass, tokenCount, decode);
    this.forwardLayers(pass, 0, this.model.config.blockCount, tokenCount, decode);
    this.sampleFromHidden(pass, tokenCount, decode);
  }

  private copyArena(
    pass: GPUComputePassEncoder,
    inputOffset: number,
    outputOffset: number,
    tokenCount: number,
    dim: number,
  ): void {
    this.dispatch(pass, "arena_copy", {
      inputOffset,
      outputOffset,
      tokenCount,
      inputDim: dim,
    }, [ceilDiv(tokenCount * dim, 256)]);
  }

  /**
   * Repair the only context-dependent part of the exact depth-2 prefix.
   * Two causal conv3 layers have receptive field 5, so at most four tokens at
   * every block boundary differ from a standalone block precompute.
   */
  private repairBlockBoundary(
    pass: GPUComputePassEncoder,
    boundary: number,
    blockTokenCount: number,
  ): number {
    const h = this.model.config.hiddenSize;
    const repairCount = Math.min(BLOCK_BOUNDARY_REPAIR, blockTokenCount);
    if (repairCount <= 0) return 0;
    const windowStart = Math.max(0, boundary - BLOCK_BOUNDARY_HISTORY);
    const windowEnd = boundary + repairCount;
    const windowCount = windowEnd - windowStart;
    if (windowCount > BLOCK_REPAIR_CAPACITY) {
      throw new Error(`Internal block repair window ${windowCount} exceeds ${BLOCK_REPAIR_CAPACITY}`);
    }

    this.forwardBlockExactPrefix(pass, windowCount, this.arena.repair, windowStart);
    const repairedStart = boundary - windowStart;
    this.copyArena(
      pass,
      this.arena.repair.hiddenA + repairedStart * h,
      this.arena.hiddenA + boundary * h,
      repairCount,
      h,
    );
    return repairCount;
  }

  /** Rebuild exact layer-0/1 short-conv decode state from the final prompt tail. */
  private rebuildCachedPrefixConvTail(pass: GPUComputePassEncoder, promptTokenCount: number): void {
    const windowCount = Math.min(BLOCK_TAIL_REBUILD_TOKENS, promptTokenCount);
    const windowStart = promptTokenCount - windowCount;
    this.forwardBlockExactPrefix(pass, windowCount, this.arena.repair, windowStart);
  }

  private writeRuntime(state: LlmRuntimeState): void {
    serializeLlmRuntime(this.runtimeView, state);
    this.device.queue.writeBuffer(this.runtimeBuffer, 0, this.runtimeCpu);
  }

  /**
   * Precompute one immutable block up to a reusable model frontier.
   *
   * Besides the requested final depth, retain hidden checkpoints at every
   * earlier attention frontier. A deeper block can therefore participate in a
   * shallower mixed-depth request without being recomputed. Conv decode tails
   * inside the cached prefix are packed separately and restored from the final
   * block in a composition.
   */
  async cacheBlock(
    tokenIds: readonly number[],
    options: CacheBlockOptions = {},
  ): Promise<Lfm2CachedBlock> {
    if (tokenIds.length < 1) throw new Error("Cached block must contain at least one token");
    if (tokenIds.length > BLOCK_CACHE_MAX_TOKENS) {
      throw new Error(`Cached block has ${tokenIds.length} tokens; PoC limit is ${BLOCK_CACHE_MAX_TOKENS}`);
    }

    const cacheDepth = this.resolveBlockCacheDepth(options.depth);
    const checkpoints = this.blockCacheDepths.filter((depth) => depth <= cacheDepth);
    const convLayers = this.model.config.layers
      .map((kind, layer) => ({ kind, layer }))
      .filter(({ kind, layer }) => kind === "conv" && layer >= BLOCK_EXACT_DEPTH && layer < cacheDepth)
      .map(({ layer }) => layer);

    const h = this.model.config.hiddenSize;
    const id = this.nextCachedBlockId++;
    const hiddenBytes = tokenIds.length * h * 4;
    const convLayerBytes = h * this.model.config.convCacheLength * 4;
    const tokenBuffer = this.device.createBuffer({
      label: `lfm2.block.${id}.tokens`,
      size: gpuBufferSize(tokenIds.length * 4),
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const stateBuffers = new Map<number, GPUBuffer>();
    for (const depth of checkpoints) {
      stateBuffers.set(depth, this.device.createBuffer({
        label: `lfm2.block.${id}.depth${depth}.hidden.f32`,
        size: gpuBufferSize(hiddenBytes),
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }));
    }
    const convStateBuffer = this.device.createBuffer({
      label: `lfm2.block.${id}.depth${cacheDepth}.conv-tails`,
      size: gpuBufferSize(convLayers.length * convLayerBytes),
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(tokenBuffer, 0, new Uint32Array(tokenIds));

    this.paramWriter.reset();
    const encoder = this.device.createCommandEncoder({ label: `lfm2.cache-block.${id}.depth${cacheDepth}` });
    encoder.clearBuffer(this.kvBuffer);
    encoder.clearBuffer(this.convBuffer);
    encoder.copyBufferToBuffer(tokenBuffer, 0, this.tokenBuffer, 0, tokenIds.length * 4);

    const exactPass = encoder.beginComputePass({ label: `lfm2.cache-block.${id}.depth${BLOCK_EXACT_DEPTH}` });
    this.forwardBlockExactPrefix(exactPass, tokenIds.length);
    exactPass.end();
    encoder.copyBufferToBuffer(
      this.arenaBuffer,
      this.arena.hiddenA * 4,
      stateBuffers.get(BLOCK_EXACT_DEPTH)!,
      0,
      hiddenBytes,
    );

    let currentDepth = BLOCK_EXACT_DEPTH;
    for (const targetDepth of checkpoints) {
      if (targetDepth <= currentDepth) continue;
      const pass = encoder.beginComputePass({ label: `lfm2.cache-block.${id}.${currentDepth}-to-${targetDepth}` });
      this.forwardLayers(pass, currentDepth, targetDepth, tokenIds.length, false);
      pass.end();
      encoder.copyBufferToBuffer(
        this.arenaBuffer,
        this.arena.hiddenA * 4,
        stateBuffers.get(targetDepth)!,
        0,
        hiddenBytes,
      );
      currentDepth = targetDepth;
    }

    for (let i = 0; i < convLayers.length; i++) {
      const layer = convLayers[i]!;
      encoder.copyBufferToBuffer(
        this.convBuffer,
        layer * convLayerBytes,
        convStateBuffer,
        i * convLayerBytes,
        convLayerBytes,
      );
    }

    if (this.paramWriter.usedBytes > 0) {
      this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
    }
    const started = performance.now();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const cacheMs = performance.now() - started;

    const block = new Lfm2CachedBlock(
      id,
      [...tokenIds],
      tokenBuffer,
      cacheDepth,
      stateBuffers,
      convLayers,
      convStateBuffer,
      convLayerBytes,
      cacheMs,
    );
    this.cachedBlocks.add(block);
    return block;
  }

  destroyCachedBlock(block: Lfm2CachedBlock): void {
    if (!this.cachedBlocks.delete(block)) return;
    block.destroy();
  }

  destroyAllCachedBlocks(): void {
    for (const block of this.cachedBlocks) block.destroy();
    this.cachedBlocks.clear();
  }

  private validateCachedBlocks(blocks: readonly Lfm2CachedBlock[]): { tokenCount: number; cacheDepth: number } {
    if (blocks.length < 1) throw new Error("generateFromBlocks requires at least one cached block");
    let total = 0;
    let cacheDepth = this.model.config.blockCount;
    for (const block of blocks) {
      if (!this.cachedBlocks.has(block)) throw new Error(`Cached block ${block.id} does not belong to this runtime or was destroyed`);
      total += block.tokenCount;
      cacheDepth = Math.min(cacheDepth, block.cacheDepth);
    }
    if (total > this.contextCapacity) {
      throw new Error(`Cached blocks contain ${total} tokens, contextCapacity is ${this.contextCapacity}`);
    }
    this.resolveBlockCacheDepth(cacheDepth);
    for (const block of blocks) block.stateAtDepth(cacheDepth);
    return { tokenCount: total, cacheDepth };
  }

  private copyCachedStatesAtDepth(
    encoder: GPUCommandEncoder,
    blocks: readonly Lfm2CachedBlock[],
    depth: number,
  ): void {
    const h = this.model.config.hiddenSize;
    let tokenCursor = 0;
    for (const block of blocks) {
      encoder.copyBufferToBuffer(
        block.stateAtDepth(depth),
        0,
        this.arenaBuffer,
        (this.arena.hiddenA + tokenCursor * h) * 4,
        block.tokenCount * h * 4,
      );
      tokenCursor += block.tokenCount;
    }
  }

  /**
   * Materialize cached context to the final layer. Mixed-depth blocks compose
   * at the shallowest selected frontier: a block cached to depth 8 can be used
   * at depth 5 because it retained its depth-5 checkpoint, while a depth-5
   * block intentionally prevents the request from skipping layers 5..7.
   */
  private recordCachedContext(
    encoder: GPUCommandEncoder,
    blocks: readonly Lfm2CachedBlock[],
    contextTokenCount: number,
    cacheDepth: number,
  ): { repairedTokens: number; cacheDepth: number } {
    const h = this.model.config.hiddenSize;
    const convLayerBytes = h * this.model.config.convCacheLength * 4;
    encoder.clearBuffer(this.kvBuffer);
    encoder.clearBuffer(this.convBuffer);

    let tokenCursor = 0;
    for (const block of blocks) {
      encoder.copyBufferToBuffer(block.tokenBuffer, 0, this.tokenBuffer, tokenCursor * 4, block.tokenCount * 4);
      tokenCursor += block.tokenCount;
    }

    // Depth 2 is the exact composable prefix. Load it in final token order and
    // repair the finite conv boundary before using it as the first attention
    // frontier.
    this.copyCachedStatesAtDepth(encoder, blocks, BLOCK_EXACT_DEPTH);
    const exactPass = encoder.beginComputePass({ label: `lfm2.block-cache.depth${cacheDepth}.exact-hydrate` });
    let boundary = blocks[0]!.tokenCount;
    let repairedTokens = 0;
    for (let i = 1; i < blocks.length; i++) {
      const block = blocks[i]!;
      repairedTokens += this.repairBlockBoundary(exactPass, boundary, block.tokenCount);
      boundary += block.tokenCount;
    }
    this.rebuildCachedPrefixConvTail(exactPass, contextTokenCount);
    if (cacheDepth > BLOCK_EXACT_DEPTH) {
      this.bootstrapAttentionKv(exactPass, BLOCK_EXACT_DEPTH, contextTokenCount);
    }
    exactPass.end();

    // Every deeper cache frontier starts immediately before an attention layer.
    // Restore that standalone checkpoint in composed order and rebuild K/V with
    // the request's actual global token positions. No Q/attention/output work is
    // needed for the frozen prefix itself.
    for (const layer of this.blockCacheDepths) {
      if (layer <= BLOCK_EXACT_DEPTH || layer >= cacheDepth || layer >= this.model.config.blockCount) continue;
      if (this.model.config.layers[layer] !== "attention") continue;
      this.copyCachedStatesAtDepth(encoder, blocks, layer);
      const pass = encoder.beginComputePass({ label: `lfm2.block-cache.bootstrap-attn-${layer}` });
      this.bootstrapAttentionKv(pass, layer, contextTokenCount);
      pass.end();
    }

    // Conv layers skipped by the cached prefix need the standalone tail of the
    // final block so a live query can continue causally from that block.
    const last = blocks[blocks.length - 1]!;
    for (const layer of last.convLayers) {
      if (layer >= cacheDepth) continue;
      encoder.copyBufferToBuffer(
        last.convStateBuffer,
        last.convStateOffset(layer),
        this.convBuffer,
        layer * convLayerBytes,
        convLayerBytes,
      );
    }

    // At exact depth 2 keep the repaired state already resident. Deeper
    // frontiers deliberately replace it with each block's standalone cached
    // representation, then global interaction resumes from effective depth.
    if (cacheDepth > BLOCK_EXACT_DEPTH) {
      this.copyCachedStatesAtDepth(encoder, blocks, cacheDepth);
    }

    if (cacheDepth < this.model.config.blockCount) {
      const contextPass = encoder.beginComputePass({ label: `lfm2.block-cache.context-from-${cacheDepth}` });
      this.forwardLayers(contextPass, cacheDepth, this.model.config.blockCount, contextTokenCount, false);
      contextPass.end();
    }
    return { repairedTokens, cacheDepth };
  }

  private recordCachedPrefill(
    encoder: GPUCommandEncoder,
    blocks: readonly Lfm2CachedBlock[],
    promptTokenCount: number,
    cacheDepth: number,
  ): { repairedTokens: number; cacheDepth: number } {
    const cache = this.recordCachedContext(encoder, blocks, promptTokenCount, cacheDepth);
    const samplePass = encoder.beginComputePass({ label: "lfm2.block-cache.sample-context" });
    this.sampleFromHidden(samplePass, promptTokenCount, false);
    samplePass.end();
    return cache;
  }

  private recordCachedQueryPrefill(
    encoder: GPUCommandEncoder,
    blocks: readonly Lfm2CachedBlock[],
    contextTokenCount: number,
    queryTokenCount: number,
    cacheDepth: number,
  ): { repairedTokens: number; cacheDepth: number } {
    const cache = this.recordCachedContext(encoder, blocks, contextTokenCount, cacheDepth);

    // Reuse the main arena from token zero. The cached context has already been
    // distilled into KV/conv state, so its final hidden vectors are no longer
    // needed while the live query advances through all layers.
    const queryPass = encoder.beginComputePass({ label: "lfm2.block-cache.live-query" });
    this.embed(queryPass, queryTokenCount, false, this.arena, contextTokenCount);
    this.forwardLayersContinuation(
      queryPass,
      0,
      this.model.config.blockCount,
      queryTokenCount,
      contextTokenCount,
    );
    this.sampleFromHidden(queryPass, queryTokenCount, false);
    queryPass.end();
    return cache;
  }

  async generateFromBlocks(
    blocks: readonly Lfm2CachedBlock[],
    options: GenerateOptions = {},
  ): Promise<GenerateResult> {
    const validated = this.validateCachedBlocks(blocks);
    const promptTokenCount = validated.tokenCount;
    const cacheDepth = validated.cacheDepth;
    const maxNewTokens = options.maxNewTokens ?? this.defaultMaxNewTokens;
    if (maxNewTokens < 1 || maxNewTokens > this.defaultMaxNewTokens) {
      throw new Error(`maxNewTokens must be 1..${this.defaultMaxNewTokens} for this allocated runtime`);
    }
    if (promptTokenCount + maxNewTokens - 1 > this.contextCapacity) {
      throw new Error(`Cached prompt + decode positions exceed contextCapacity ${this.contextCapacity}`);
    }

    const state = createInitialRuntimeState({
      contextCapacity: this.contextCapacity,
      maxNewTokens,
      eosToken: this.model.config.eosToken,
      promptTokenCount,
    });
    this.writeRuntime(state);

    if (options.profile) {
      const totalStarted = performance.now();
      this.paramWriter.reset();
      const prefillEncoder = this.device.createCommandEncoder({ label: "lfm2.block-cache.profile.prefill" });
      const cache = this.recordCachedPrefill(prefillEncoder, blocks, promptTokenCount, cacheDepth);
      if (this.paramWriter.usedBytes > 0) {
        this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
      }
      const prefillStarted = performance.now();
      this.device.queue.submit([prefillEncoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const prefillMs = performance.now() - prefillStarted;

      let decodeMs = 0;
      const scheduledDecodeSteps = Math.max(0, maxNewTokens - 1);
      if (scheduledDecodeSteps > 0) {
        this.paramWriter.reset();
        const decodeEncoder = this.device.createCommandEncoder({ label: "lfm2.block-cache.profile.decode" });
        const decodePass = decodeEncoder.beginComputePass({ label: "lfm2.block-cache.decode" });
        for (let step = 0; step < scheduledDecodeSteps; step++) this.forwardAndSample(decodePass, 1, true);
        decodePass.end();
        if (this.paramWriter.usedBytes > 0) {
          this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
        }
        const decodeStarted = performance.now();
        this.device.queue.submit([decodeEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        decodeMs = performance.now() - decodeStarted;
      }

      const readbackStarted = performance.now();
      const result = await this.readResult(maxNewTokens);
      const readbackMs = performance.now() - readbackStarted;
      return {
        ...result,
        timings: {
          prefillMs,
          decodeMs,
          readbackMs,
          totalMs: performance.now() - totalStarted,
          promptTokens: promptTokenCount,
          scheduledDecodeSteps,
          cacheDepth: cache.cacheDepth,
          cachedBlocks: blocks.length,
          cachedTokens: promptTokenCount,
          repairedTokens: cache.repairedTokens,
        },
      };
    }

    this.paramWriter.reset();
    const encoder = this.device.createCommandEncoder({ label: "lfm2.generate-from-blocks" });
    this.recordCachedPrefill(encoder, blocks, promptTokenCount, cacheDepth);
    const decodePass = encoder.beginComputePass({ label: "lfm2.block-cache.decode" });
    for (let step = 1; step < maxNewTokens; step++) this.forwardAndSample(decodePass, 1, true);
    decodePass.end();
    if (this.paramWriter.usedBytes > 0) {
      this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
    }
    this.device.queue.submit([encoder.finish()]);
    return await this.readResult(maxNewTokens);
  }

  /**
   * Cached blocks are persistent context; queryTokenIds are a live causal
   * frontier evaluated after them. The live query sees cached context already
   * at the first attention layer instead of being precached standalone.
   */
  async generateFromBlocksWithQuery(
    blocks: readonly Lfm2CachedBlock[],
    queryTokenIds: readonly number[],
    options: GenerateOptions = {},
  ): Promise<GenerateResult> {
    const validated = this.validateCachedBlocks(blocks);
    const contextTokenCount = validated.tokenCount;
    const cacheDepth = validated.cacheDepth;
    if (queryTokenIds.length < 1) throw new Error("Live query must contain at least one token");
    const promptTokenCount = contextTokenCount + queryTokenIds.length;
    const maxNewTokens = options.maxNewTokens ?? this.defaultMaxNewTokens;
    if (maxNewTokens < 1 || maxNewTokens > this.defaultMaxNewTokens) {
      throw new Error(`maxNewTokens must be 1..${this.defaultMaxNewTokens} for this allocated runtime`);
    }
    if (promptTokenCount + maxNewTokens - 1 > this.contextCapacity) {
      throw new Error(`Cached context + live query + decode positions exceed contextCapacity ${this.contextCapacity}`);
    }

    const state = createInitialRuntimeState({
      contextCapacity: this.contextCapacity,
      maxNewTokens,
      eosToken: this.model.config.eosToken,
      promptTokenCount,
    });
    this.writeRuntime(state);
    this.device.queue.writeBuffer(
      this.tokenBuffer,
      contextTokenCount * 4,
      new Uint32Array(queryTokenIds),
    );

    if (options.profile) {
      const totalStarted = performance.now();
      this.paramWriter.reset();
      const prefillEncoder = this.device.createCommandEncoder({ label: "lfm2.block-cache-query.profile.prefill" });
      const cache = this.recordCachedQueryPrefill(
        prefillEncoder, blocks, contextTokenCount, queryTokenIds.length, cacheDepth,
      );
      if (this.paramWriter.usedBytes > 0) {
        this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
      }
      const prefillStarted = performance.now();
      this.device.queue.submit([prefillEncoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const prefillMs = performance.now() - prefillStarted;

      let decodeMs = 0;
      const scheduledDecodeSteps = Math.max(0, maxNewTokens - 1);
      if (scheduledDecodeSteps > 0) {
        this.paramWriter.reset();
        const decodeEncoder = this.device.createCommandEncoder({ label: "lfm2.block-cache-query.profile.decode" });
        const decodePass = decodeEncoder.beginComputePass({ label: "lfm2.block-cache-query.decode" });
        for (let step = 0; step < scheduledDecodeSteps; step++) this.forwardAndSample(decodePass, 1, true);
        decodePass.end();
        if (this.paramWriter.usedBytes > 0) {
          this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
        }
        const decodeStarted = performance.now();
        this.device.queue.submit([decodeEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        decodeMs = performance.now() - decodeStarted;
      }

      const readbackStarted = performance.now();
      const result = await this.readResult(maxNewTokens);
      const readbackMs = performance.now() - readbackStarted;
      return {
        ...result,
        timings: {
          prefillMs,
          decodeMs,
          readbackMs,
          totalMs: performance.now() - totalStarted,
          promptTokens: promptTokenCount,
          scheduledDecodeSteps,
          cacheDepth: cache.cacheDepth,
          cachedBlocks: blocks.length,
          cachedTokens: contextTokenCount,
          liveQueryTokens: queryTokenIds.length,
          repairedTokens: cache.repairedTokens,
        },
      };
    }

    this.paramWriter.reset();
    const encoder = this.device.createCommandEncoder({ label: "lfm2.generate-from-blocks-with-query" });
    this.recordCachedQueryPrefill(encoder, blocks, contextTokenCount, queryTokenIds.length, cacheDepth);
    const decodePass = encoder.beginComputePass({ label: "lfm2.block-cache-query.decode" });
    for (let step = 1; step < maxNewTokens; step++) this.forwardAndSample(decodePass, 1, true);
    decodePass.end();
    if (this.paramWriter.usedBytes > 0) {
      this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
    }
    this.device.queue.submit([encoder.finish()]);
    return await this.readResult(maxNewTokens);
  }

  async generateTokens(promptTokenIds: readonly number[], options: GenerateOptions = {}): Promise<GenerateResult> {
    const maxNewTokens = options.maxNewTokens ?? this.defaultMaxNewTokens;
    if (promptTokenIds.length < 1) throw new Error("Prompt must contain at least one token");
    if (promptTokenIds.length > this.contextCapacity) {
      throw new Error(`Prompt has ${promptTokenIds.length} tokens, contextCapacity is ${this.contextCapacity}`);
    }
    if (maxNewTokens < 1 || maxNewTokens > this.defaultMaxNewTokens) {
      throw new Error(`maxNewTokens must be 1..${this.defaultMaxNewTokens} for this allocated runtime`);
    }
    if (promptTokenIds.length + maxNewTokens - 1 > this.contextCapacity) {
      throw new Error(`Prompt + decode positions exceed bring-up contextCapacity ${this.contextCapacity}`);
    }

    const state = createInitialRuntimeState({
      contextCapacity: this.contextCapacity,
      maxNewTokens,
      eosToken: this.model.config.eosToken,
      promptTokenCount: promptTokenIds.length,
    });
    this.writeRuntime(state);
    this.device.queue.writeBuffer(this.tokenBuffer, 0, new Uint32Array(promptTokenIds));

    if (options.profile) {
      return await this.generateProfiled(promptTokenIds.length, maxNewTokens);
    }

    this.paramWriter.reset();
    const encoder = this.device.createCommandEncoder({ label: "lfm2.generate" });
    encoder.clearBuffer(this.kvBuffer);
    encoder.clearBuffer(this.convBuffer);
    const pass = encoder.beginComputePass({ label: "lfm2.inference" });

    this.forwardAndSample(pass, promptTokenIds.length, false);
    for (let step = 1; step < maxNewTokens; step++) {
      this.forwardAndSample(pass, 1, true);
    }
    pass.end();

    if (this.paramWriter.usedBytes > 0) {
      this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
    }
    this.device.queue.submit([encoder.finish()]);
    return await this.readResult(maxNewTokens);
  }

  private async generateProfiled(promptTokenCount: number, maxNewTokens: number): Promise<GenerateResult> {
    const totalStarted = performance.now();

    // Profiling intentionally introduces one synchronization point between
    // prefill and decode. It is diagnostic-only; the normal path above remains
    // one monolithic GPU submission with no CPU dependency between phases.
    this.paramWriter.reset();
    const prefillEncoder = this.device.createCommandEncoder({ label: "lfm2.profile.prefill" });
    prefillEncoder.clearBuffer(this.kvBuffer);
    prefillEncoder.clearBuffer(this.convBuffer);
    const prefillPass = prefillEncoder.beginComputePass({ label: "lfm2.prefill" });
    this.forwardAndSample(prefillPass, promptTokenCount, false);
    prefillPass.end();
    if (this.paramWriter.usedBytes > 0) {
      this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
    }

    const prefillStarted = performance.now();
    this.device.queue.submit([prefillEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const prefillMs = performance.now() - prefillStarted;

    let decodeMs = 0;
    const scheduledDecodeSteps = Math.max(0, maxNewTokens - 1);
    if (scheduledDecodeSteps > 0) {
      this.paramWriter.reset();
      const decodeEncoder = this.device.createCommandEncoder({ label: "lfm2.profile.decode" });
      const decodePass = decodeEncoder.beginComputePass({ label: "lfm2.decode" });
      for (let step = 0; step < scheduledDecodeSteps; step++) {
        this.forwardAndSample(decodePass, 1, true);
      }
      decodePass.end();
      if (this.paramWriter.usedBytes > 0) {
        this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramWriter.usedView());
      }

      const decodeStarted = performance.now();
      this.device.queue.submit([decodeEncoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      decodeMs = performance.now() - decodeStarted;
    }

    const readbackStarted = performance.now();
    const result = await this.readResult(maxNewTokens);
    const readbackMs = performance.now() - readbackStarted;
    return {
      ...result,
      timings: {
        prefillMs,
        decodeMs,
        readbackMs,
        totalMs: performance.now() - totalStarted,
        promptTokens: promptTokenCount,
        scheduledDecodeSteps,
      },
    };
  }

  private async readResult(maxNewTokens: number): Promise<GenerateResult> {
    const runtimeStaging = this.device.createBuffer({
      label: "lfm2.result-runtime",
      size: gpuBufferSize(this.runtimeByteSize),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const tokenStaging = this.device.createBuffer({
      label: "lfm2.result-tokens",
      size: gpuBufferSize(maxNewTokens * 4),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "lfm2.result-readback" });
    encoder.copyBufferToBuffer(this.runtimeBuffer, 0, runtimeStaging, 0, this.runtimeByteSize);
    encoder.copyBufferToBuffer(this.tokenBuffer, this.contextCapacity * 4, tokenStaging, 0, maxNewTokens * 4);
    this.device.queue.submit([encoder.finish()]);

    try {
      await Promise.all([
        runtimeStaging.mapAsync(GPUMapMode.READ),
        tokenStaging.mapAsync(GPUMapMode.READ),
      ]);
      const runtimeBytes = new Uint8Array(runtimeStaging.getMappedRange());
      new Uint8Array(this.runtimeCpu).set(runtimeBytes.subarray(0, this.runtimeByteSize));
      const state = deserializeLlmRuntime(this.runtimeView);
      const mappedTokens = new Uint32Array(tokenStaging.getMappedRange());
      const tokenIds = Array.from(mappedTokens.subarray(0, state.generatedCount));
      return { tokenIds, state };
    } finally {
      if (runtimeStaging.mapState === "mapped") runtimeStaging.unmap();
      if (tokenStaging.mapState === "mapped") tokenStaging.unmap();
      runtimeStaging.destroy();
      tokenStaging.destroy();
    }
  }

  /**
   * Drop-if-busy telemetry readback. With the v0 monolithic command buffer this
   * observes request completion; later token-level command chaining can reuse
   * the same API for live telemetry without changing callers.
   */
  async readTelemetry(): Promise<LlmRuntimeState | null> {
    if (this.telemetryInFlight) return null;
    this.telemetryInFlight = true;
    const encoder = this.device.createCommandEncoder({ label: "lfm2.telemetry" });
    encoder.copyBufferToBuffer(this.runtimeBuffer, 0, this.telemetryStaging, 0, this.runtimeByteSize);
    this.device.queue.submit([encoder.finish()]);
    try {
      await this.telemetryStaging.mapAsync(GPUMapMode.READ);
      new Uint8Array(this.runtimeCpu).set(
        new Uint8Array(this.telemetryStaging.getMappedRange()).subarray(0, this.runtimeByteSize),
      );
      return deserializeLlmRuntime(this.runtimeView);
    } finally {
      if (this.telemetryStaging.mapState === "mapped") this.telemetryStaging.unmap();
      this.telemetryInFlight = false;
    }
  }

  destroy(): void {
    this.destroyAllCachedBlocks();
    this.runtimeBuffer.destroy();
    this.tokenBuffer.destroy();
    this.arenaBuffer.destroy();
    this.kvBuffer.destroy();
    this.convBuffer.destroy();
    this.paramBuffer.destroy();
    this.telemetryStaging.destroy();
    this.dummyWeightRaw.destroy();
    this.dummyWeight32.destroy();
  }
}
