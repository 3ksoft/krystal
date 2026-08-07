import type { Sandblaster } from "@sandblaster/core";
import { type GpuTensor, type GpuWeightFormat, Lfm2Model } from "../model.ts";
import { LLM_RUNTIME_BYTES, serializeLlmRuntime, type LlmRuntimeState } from "../../../schema/src/schema.ts";
import { SIZEOF_DecodeTelemetry } from "../../../schema/dist/util/codec.ts";
import {
  BLOCK_BOUNDARY_HISTORY,
  BLOCK_BOUNDARY_REPAIR,
  BLOCK_CACHE_MAX_TOKENS,
  BLOCK_DEFAULT_CACHE_DEPTH,
  BLOCK_EXACT_DEPTH,
  BLOCK_REPAIR_CAPACITY,
  BLOCK_TAIL_REBUILD_TOKENS,
  HEAD_DIM,
  KV_DIM,
  KV_HEADS,
  MAX_BRINGUP_CONTEXT,
  PARAM_BUFFER_BYTES,
  PARAM_BYTES,
  QUERY_HEADS,
} from "./constants.ts";
import type {
  Lfm2RuntimeOptions,
  MatmulDispatchArgs,
  MatmulKernelSpec,
  ResolvedLfm2RuntimeOptions,
} from "./types.ts";

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
  "argmax_candidates",
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

export interface WorkLayout {
  hiddenA: number;
  hiddenB: number;
  tmpH: number;
  tmpA: number;
  tmpB: number;
}

export interface ArenaLayout extends WorkLayout {
  repair: WorkLayout;
  logits: number;
  elements: number;
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

function cpuArgmax(logits: Float32Array): number {
  let bestToken = 0;
  let bestValue = -Number.MAX_VALUE;
  for (let i = 0; i < logits.length; i++) {
    const value = logits[i]!;
    if (value > bestValue || (value === bestValue && i < bestToken)) {
      bestValue = value;
      bestToken = i;
    }
  }
  return bestToken;
}

export class ParamWriter {
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


export class Lfm2GpuRuntime {
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
  readonly decodeTelemetryBuffer: GPUBuffer;

  readonly arena: ArenaLayout;
  readonly runtimeByteSize: number;
  readonly runtimeCpu: ArrayBuffer;
  readonly runtimeView: DataView;

  private readonly paramStride: number;
  readonly paramBuffer: GPUBuffer;
  readonly paramWriter: ParamWriter;
  private readonly group0: GPUBindGroup;
  private readonly group0Layout: GPUBindGroupLayout;
  private readonly group1Layout: GPUBindGroupLayout;
  private readonly pipelineLayout: GPUPipelineLayout;
  private readonly pipelines = new Map<PipelineName, GPUComputePipeline>();
  private readonly matmulKernels: Record<GpuWeightFormat, MatmulKernelSpec>;

  private readonly dummyWeightRaw: GPUBuffer;
  private readonly dummyWeight32: GPUBuffer;
  private readonly weightGroups = new Map<string, GPUBindGroup>();

  readonly telemetryStaging: GPUBuffer;
  readonly decodeTelemetryStaging: GPUBuffer;
  readonly logitsStaging: GPUBuffer;
  readonly candidateTokenBuffer: GPUBuffer;
  readonly selectedTokenStaging: GPUBuffer;

  private constructor(
    private readonly engine: Sandblaster<any>,
    readonly model: Lfm2Model,
    options: ResolvedLfm2RuntimeOptions,
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
    this.decodeTelemetryBuffer = this.device.createBuffer({
      label: "lfm2.decode-telemetry",
      size: gpuBufferSize(SIZEOF_DecodeTelemetry),
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
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
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

    this.candidateTokenBuffer = this.device.createBuffer({
      label: "lfm2.candidate-tokens",
      size: gpuBufferSize(this.model.config.vocabSize * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.selectedTokenStaging = this.device.createBuffer({
      label: "lfm2.selected-token-staging",
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
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
        { binding: 6, resource: { buffer: this.candidateTokenBuffer } },
        { binding: 7, resource: { buffer: this.decodeTelemetryBuffer } },
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
    this.decodeTelemetryStaging = this.device.createBuffer({
      label: "lfm2.decode-telemetry-staging",
      size: gpuBufferSize(SIZEOF_DecodeTelemetry),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.logitsStaging = this.device.createBuffer({
      label: "lfm2.logits-staging",
      size: gpuBufferSize(this.model.config.vocabSize * 4),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

  }

  computeBlockCacheDepths(): number[] {
    const depths = [BLOCK_EXACT_DEPTH];
    for (let layer = BLOCK_EXACT_DEPTH + 1; layer < this.model.config.blockCount; layer++) {
      if (this.model.config.layers[layer] === "attention") depths.push(layer);
    }
    if (depths[depths.length - 1] !== this.model.config.blockCount) depths.push(this.model.config.blockCount);
    return depths;
  }

  resolveBlockCacheDepth(depth = BLOCK_DEFAULT_CACHE_DEPTH): number {
    if (!Number.isInteger(depth) || !this.blockCacheDepths.includes(depth)) {

    }
    return depth;
  }

  static async create(
    engine: Sandblaster<any>,
    model: Lfm2Model,
    options: Lfm2RuntimeOptions = {},
  ): Promise<Lfm2GpuRuntime> {
    const defaults: Record<GpuWeightFormat, MatmulKernelSpec> = {
      f16: { entryPoint: "matmul_f16" },
      f32: { entryPoint: "matmul_f32" },
      wq4: { entryPoint: "matmul_wq4" },
    };
    const resolved = {
      contextCapacity: options.contextCapacity ?? 1024,
      maxNewTokens: options.maxNewTokens ?? 1024,
      matmulKernels: {
        f16: options.matmulKernels?.f16 ?? defaults.f16,
        f32: options.matmulKernels?.f32 ?? defaults.f32,
        wq4: options.matmulKernels?.wq4 ?? defaults.wq4,
      },
    };
    if (resolved.maxNewTokens < 1) throw new Error("maxNewTokens must be >= 1");

    const deno = (globalThis as any).Deno;
    const loadText = async (url: URL, label: string): Promise<string> => {
      if (url.protocol === "file:" && deno?.readTextFile) return await deno.readTextFile(url);
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Could not load ${label} (${response.status}) from ${url}`);
      return await response.text();
    };
    const [runtimeBody, schemaWgsl, codecWgsl] = await Promise.all([
      loadText(new URL("./shaders/runtime.wgsl", import.meta.url), "LFM2 WGSL"),
      loadText(new URL("../../schema/dist/shaders/includes/schema.wgsl", import.meta.url), "schema WGSL"),
      loadText(new URL("../../schema/dist/shaders/includes/codec.wgsl", import.meta.url), "schema codec WGSL"),
    ]);
    let body = `${schemaWgsl}\n\n${codecWgsl}\n\n${runtimeBody}`;
    const extensions = Object.values(resolved.matmulKernels)
      .map((kernel) => kernel.wgsl?.trim())
      .filter((source): source is string => Boolean(source));
    if (extensions.length > 0) body += `\n\n${extensions.join("\n\n")}\n`;

    const runtime = new Lfm2GpuRuntime(engine, model, resolved);
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

  dispatch(
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

  embed(
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

  forwardLayers(
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

  forwardLayersContinuation(
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
  forwardBlockExactPrefix(
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
  bootstrapAttentionKv(pass: GPUComputePassEncoder, layer: number, tokenCount: number): void {
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

  projectLogitsFromHidden(pass: GPUComputePassEncoder, tokenCount: number, decode: boolean): void {
    const h = this.model.config.hiddenSize;
    const finalInput = decode
      ? this.arena.hiddenA
      : this.arena.hiddenA + (tokenCount - 1) * h;
    this.norm(pass, "token_embd_norm.weight", finalInput, this.arena.tmpH, 1, h);
    this.matrix(pass, "token_embd.weight", this.arena.tmpH, this.arena.logits, 1, h, this.model.config.vocabSize);
  }

  commitLogitsArgmax(pass: GPUComputePassEncoder, decode: boolean): void {
    this.dispatch(pass, "argmax", {
      inputOffset: this.arena.logits,
      inputDim: this.model.config.vocabSize,
      mode: decode ? 1 : 0,
    }, [1]);
  }

  sampleFromHidden(pass: GPUComputePassEncoder, tokenCount: number, decode: boolean): void {
    this.projectLogitsFromHidden(pass, tokenCount, decode);
    this.commitLogitsArgmax(pass, decode);
  }

  forwardToLogits(pass: GPUComputePassEncoder, tokenCount: number, decode: boolean): void {
    this.embed(pass, tokenCount, decode);
    this.forwardLayers(pass, 0, this.model.config.blockCount, tokenCount, decode);
    this.projectLogitsFromHidden(pass, tokenCount, decode);
  }

  forwardAndSample(pass: GPUComputePassEncoder, tokenCount: number, decode: boolean): void {
    this.forwardToLogits(pass, tokenCount, decode);
    this.commitLogitsArgmax(pass, decode);
  }

  copyArena(
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
  repairBlockBoundary(
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
  rebuildCachedPrefixConvTail(pass: GPUComputePassEncoder, promptTokenCount: number): void {
    const windowCount = Math.min(BLOCK_TAIL_REBUILD_TOKENS, promptTokenCount);
    const windowStart = promptTokenCount - windowCount;
    this.forwardBlockExactPrefix(pass, windowCount, this.arena.repair, windowStart);
  }

  writeRuntime(state: LlmRuntimeState): void {
    serializeLlmRuntime(this.runtimeView, state);
    this.device.queue.writeBuffer(this.runtimeBuffer, 0, this.runtimeCpu);
    // Header is { enabled: u32, cursor: atomic<u32> }. Entries do not need clearing:
    // readback trusts cursor and ignores stale tail records.
    this.device.queue.writeBuffer(this.decodeTelemetryBuffer, 0, new Uint32Array([1, 0]));
  }

  destroy(): void {
    this.runtimeBuffer.destroy();
    this.tokenBuffer.destroy();
    this.arenaBuffer.destroy();
    this.kvBuffer.destroy();
    this.convBuffer.destroy();
    this.decodeTelemetryBuffer.destroy();
    this.paramBuffer.destroy();
    this.telemetryStaging.destroy();
    this.decodeTelemetryStaging.destroy();
    this.logitsStaging.destroy();
    this.candidateTokenBuffer.destroy();
    this.selectedTokenStaging.destroy();
    this.dummyWeightRaw.destroy();
    this.dummyWeight32.destroy();
  }
}
