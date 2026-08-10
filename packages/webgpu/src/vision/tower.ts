/**
 * GPU vision tower (ADA-0009, M2 — f32 correctness milestone).
 *
 * Self-contained WebGPU implementation of the SigLIP2-style tower + projector
 * from packages/quant/src/vision/reference.ts. It deliberately does NOT go
 * through the Sandblaster AOT machinery: the tower is stateless, one-shot and
 * small (<= maxPatches tokens), and M2's goal is differential correctness
 * against the CPU oracle, not scheduler integration. The shaders are
 * self-contained WGSL files in ./shaders (embedded by
 * scripts/build-vision-shaders.ts).
 *
 * Layout:
 *   group 0 { 0: params (storage read, dynamic offset, one 256-B record per
 *                    dispatch — storage, not uniform: 27 blocks * 12 passes
 *                    exceed the 64 KiB uniform binding size limit),
 *              1: arena (storage read_write) }
 *   group 1 { 0: weights (storage read_write) — the per-layer block buffer
 *                    for block passes, or the misc buffer (patch emb / post
 *                    LN / projector) for the tower head and tail }
 *
 * The arena packs every activation (patches, posEmb, padMask, hidden, normed,
 * qkv, attn, ff, h, embeddings) into one storage buffer; per-pass offsets come
 * from the params record. All weights are uploaded as exact F32 (host-side
 * F16 decode by the shared vision loader), so kernel error is isolated from
 * quantization error. The WQ4 sidecar path lands in M3.
 *
 * Exactness notes (all mirrored from the oracle):
 *   - LayerNorm with weight+bias, population variance,
 *   - bidirectional attention, scale = headDim^-0.5, padding mask,
 *   - MLP activation tanh-GELU, projector activation exact erf (A&S),
 *   - mmproj/llama.cpp unshuffle channel order in the projector tail
 *     (u = c + dim*i + dim*factor*j, blocked sub-pixel bits — NOT torch
 *     PixelUnshuffle; verified against llama.cpp ground truth).
 */

import type { VisionConfig } from "../../../quant/src/vision/config.ts";
import type { VisionReferenceWeights } from "../../../quant/src/vision/reference.ts";
import type { ProcessedImage } from "./processor.ts";
import { visionShaders } from "./shaders.generated.ts";

const PARAMS_BYTES = 80;
const DISPATCH_SLOTS = 336; // 1 patch_embed + 27*12 block passes + 1 post_ln + 1 projector


export interface ArenaLayout {
  patches: number;
  posEmb: number;
  padMask: number;
  hidden: number;
  normed: number;
  qkv: number;
  attn: number;
  ff: number;
  h: number;
  embeddings: number;
  totalF32: number;
}

interface LayerWeightLayout {
  ln1W: number; ln1B: number; ln2W: number; ln2B: number;
  qW: number; qB: number; kW: number; kB: number; vW: number; vB: number;
  oW: number; oB: number; upW: number; upB: number; downW: number; downB: number;
  totalF32: number;
}

interface MiscWeightLayout {
  patchEmb: number; patchEmbB: number; postLnW: number; postLnB: number;
  mm1: number; mm1B: number; mm2: number; mm2B: number;
  totalF32: number;
}

function arenaLayout(config: VisionConfig): ArenaLayout {
  const P = config.maxPatches;
  const patchDim = 3 * config.patchSize * config.patchSize;
  const dim = config.hiddenSize;
  const ff = config.feedForwardSize;
  const tokens = Math.floor(P / (config.projectorScaleFactor * config.projectorScaleFactor));
  let off = 0;
  const next = (n: number): number => {
    const o = off;
    off += n;
    return o;
  };
  return {
    patches: next(P * patchDim),
    posEmb: next(P * dim),
    padMask: next(P),
    hidden: next(P * dim),
    normed: next(P * dim),
    qkv: next(3 * P * dim),
    attn: next(P * dim),
    ff: next(P * ff),
    h: next(P * dim),
    embeddings: next(tokens * config.projectorHiddenSize),
    totalF32: off,
  };
}

function layerWeightLayout(hidden: number, ff: number): LayerWeightLayout {
  let off = 0;
  const v = (n: number): number => {
    const o = off;
    off += n;
    return o;
  };
  return {
    ln1W: v(hidden), ln1B: v(hidden),
    ln2W: v(hidden), ln2B: v(hidden),
    qW: v(hidden * hidden), qB: v(hidden),
    kW: v(hidden * hidden), kB: v(hidden),
    vW: v(hidden * hidden), vB: v(hidden),
    oW: v(hidden * hidden), oB: v(hidden),
    upW: v(hidden * ff), upB: v(ff),
    downW: v(ff * hidden), downB: v(hidden),
    totalF32: off,
  };
}

function miscWeightLayout(config: VisionConfig): MiscWeightLayout {
  const dim = config.hiddenSize;
  const patchDim = 3 * config.patchSize * config.patchSize;
  const unsh = dim * config.projectorScaleFactor * config.projectorScaleFactor;
  const proj = config.projectorHiddenSize;
  let off = 0;
  const v = (n: number): number => {
    const o = off;
    off += n;
    return o;
  };
  return {
    patchEmb: v(dim * patchDim),
    patchEmbB: v(dim),
    postLnW: v(dim), postLnB: v(dim),
    mm1: v(unsh * proj), mm1B: v(proj),
    mm2: v(proj * proj), mm2B: v(proj),
    totalF32: off,
  };
}

function packLayerWeights(
  block: VisionReferenceWeights["blocks"][number],
  layout: LayerWeightLayout,
): Float32Array {
  const out = new Float32Array(layout.totalF32);
  const put = (data: Float32Array, off: number): void => {
    out.set(data, off);
  };
  put(block.ln1.weight, layout.ln1W);
  put(block.ln1.bias, layout.ln1B);
  put(block.ln2.weight, layout.ln2W);
  put(block.ln2.bias, layout.ln2B);
  put(block.q, layout.qW);
  put(block.qBias, layout.qB);
  put(block.k, layout.kW);
  put(block.kBias, layout.kB);
  put(block.v, layout.vW);
  put(block.vBias, layout.vB);
  put(block.o, layout.oW);
  put(block.oBias, layout.oB);
  put(block.up, layout.upW);
  put(block.upBias, layout.upB);
  put(block.down, layout.downW);
  put(block.downBias, layout.downB);
  return out;
}

function packMiscWeights(
  w: VisionReferenceWeights,
  config: VisionConfig,
  layout: MiscWeightLayout,
): Float32Array {
  const out = new Float32Array(layout.totalF32);
  out.set(w.patchEmb, layout.patchEmb);
  out.set(w.patchEmbBias, layout.patchEmbB);
  out.set(w.postLn.weight, layout.postLnW);
  out.set(w.postLn.bias, layout.postLnB);
  out.set(w.projector.mm1, layout.mm1);
  out.set(w.projector.mm1Bias, layout.mm1B);
  out.set(w.projector.mm2, layout.mm2);
  out.set(w.projector.mm2Bias, layout.mm2B);
  return out;
}

function alignTo(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

export interface VisionTowerOptions {
  readonly device: GPUDevice;
  readonly config: VisionConfig;
  /** Exact F32 weights (from the shared vision loader, F16 GGUF source). */
  readonly weights: VisionReferenceWeights;
}

export class VisionTower {
  readonly config: VisionConfig;

  private readonly device: GPUDevice;
  private readonly arena: GPUBuffer;
  private readonly arenaLayout: ArenaLayout;
  private readonly paramsRing: GPUBuffer;
  private readonly staging: GPUBuffer;
  private readonly layerBuffers: GPUBuffer[];
  private readonly layerLayout: LayerWeightLayout;
  private readonly miscLayout: MiscWeightLayout;
  private readonly pipelines: Record<
    "patch_embed" | "layernorm" | "matmul" | "gelu_tanh" | "attention" | "residual_add" | "unshuffle_project",
    GPUComputePipeline
  >;
  private readonly bg0: GPUBindGroup;
  private readonly bg1Misc: GPUBindGroup;
  private readonly bg1Layer: GPUBindGroup[];
  private readonly stride: number;
  private destroyed = false;

  private constructor(
    options: VisionTowerOptions,
    arena: GPUBuffer,
    arenaLayout: ArenaLayout,
    paramsRing: GPUBuffer,
    staging: GPUBuffer,
    layerBuffers: GPUBuffer[],
    miscBuffer: GPUBuffer,
    layerLayout: LayerWeightLayout,
    miscLayout: MiscWeightLayout,
    pipelines: VisionTower["pipelines"],
    bg0: GPUBindGroup,
    bg1Misc: GPUBindGroup,
    bg1Layer: GPUBindGroup[],
    stride: number,
  ) {
    this.device = options.device;
    this.config = options.config;
    this.arena = arena;
    this.arenaLayout = arenaLayout;
    this.paramsRing = paramsRing;
    this.staging = staging;
    this.layerBuffers = layerBuffers;
    this.layerLayout = layerLayout;
    this.miscLayout = miscLayout;
    this.pipelines = pipelines;
    this.bg0 = bg0;
    this.bg1Misc = bg1Misc;
    this.bg1Layer = bg1Layer;
    this.stride = stride;
  }

  static async create(options: VisionTowerOptions): Promise<VisionTower> {
    const { device, config, weights } = options;
    const dim = config.hiddenSize;
    const ff = config.feedForwardSize;
    if (config.maxPatches > 1024) {
      throw new Error(`VisionTower: shaders cap the attention grid at 1024 patches, got ${config.maxPatches}`);
    }
    if (config.headDim > 256) {
      throw new Error(
        `VisionTower: the attention shader writes only the first 256 head dims, got headDim ${config.headDim}`,
      );
    }

    const layout = arenaLayout(config);
    const arena = await createStorage(device, "vision.arena", layout.totalF32 * 4, true);
    // One slot per dispatch plus one spare: with a dynamic offset, the bound
    // range must not reach the end of the buffer (offset + size <= bufferSize),
    // so the buffer is one stride larger than the bound range.
    const paramsRing = device.createBuffer({
      label: "vision.params",
      size: (DISPATCH_SLOTS + 1) * 256,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const maxTokens = Math.floor(config.maxPatches / (config.projectorScaleFactor ** 2));
    // Large enough for the largest single readback: embeddings (256*2048*4)
    // and the debug qkv dump (3*1024*1152*4 = 14 MiB).
    const staging = device.createBuffer({
      label: "vision.staging",
      size: Math.max(maxTokens * config.projectorHiddenSize * 4, 3 * config.maxPatches * dim * 4),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // weight buffers (all F32, exact)
    const layerLayout = layerWeightLayout(dim, ff);
    const layerBuffers = await Promise.all(
      weights.blocks.map(async (block, i) => {
        const data = packLayerWeights(block, layerLayout);
        const buffer = await createStorage(device, `vision.weights.blk.${i}`, data.byteLength, false);
        device.queue.writeBuffer(buffer, 0, data);
        return buffer;
      }),
    );
    const miscLayout = miscWeightLayout(config);
    const miscData = packMiscWeights(weights, config, miscLayout);
    const miscBuffer = await createStorage(device, "vision.weights.misc", miscData.byteLength, false);
    device.queue.writeBuffer(miscBuffer, 0, miscData);

    // pipelines — all shaders share one layout so one bind group shape fits all
    const group0Layout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage", hasDynamicOffset: true, minBindingSize: PARAMS_BYTES },
        },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const group1Layout = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
    });
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [group0Layout, group1Layout],
    });

    const pipelines = {} as VisionTower["pipelines"];
    for (const [name, source] of Object.entries(visionShaders) as [keyof typeof visionShaders, string][]) {
      const module = device.createShaderModule({ label: `vision.${name}`, code: source });
      const info = await module.getCompilationInfo();
      const error = info.messages.find((m) => m.type === "error");
      if (error) throw new Error(`Vision shader ${name}: ${error.message}`);
      pipelines[name] = device.createComputePipeline({
        label: `vision.${name}`,
        layout: pipelineLayout,
        compute: { module, entryPoint: name },
      });
    }

    const stride = alignTo(PARAMS_BYTES, Number(device.limits.minStorageBufferOffsetAlignment ?? 256));
    if (stride * DISPATCH_SLOTS > 256 * DISPATCH_SLOTS) {
      throw new Error("VisionTower: params ring stride exceeds the 256-B slot budget");
    }
    const bg0 = device.createBindGroup({
      label: "vision.bg0",
      layout: group0Layout,
      entries: [
        // Bind one stride: with a dynamic offset the bound range (offset +
        // size) must fit the buffer for EVERY dispatch, so the entry size is a
        // single record, not the whole ring.
        { binding: 0, resource: { buffer: paramsRing, size: 256 } },
        { binding: 1, resource: { buffer: arena } },
      ],
    });
    const bg1Misc = device.createBindGroup({
      label: "vision.bg1.misc",
      layout: group1Layout,
      entries: [{ binding: 0, resource: { buffer: miscBuffer } }],
    });
    const bg1Layer = layerBuffers.map((buffer, i) =>
      device.createBindGroup({
        label: `vision.bg1.blk.${i}`,
        layout: group1Layout,
        entries: [{ binding: 0, resource: { buffer } }],
      }),
    );

    return new VisionTower(
      options,
      arena,
      layout,
      paramsRing,
      staging,
      layerBuffers,
      miscBuffer,
      layerLayout,
      miscLayout,
      pipelines,
      bg0,
      bg1Misc,
      bg1Layer,
      stride,
    );
  }

  /**
   * Encode the full tower forward for one processed image and read back the
   * image embeddings [tokens, projectorHiddenSize]. GPU kernels are the exact
   * f32 mirror of the CPU oracle (see module doc), so a differential test can
   * measure pure kernel error.
   *
   * Debug affordance: `stopAtLayer` runs only the head + blocks [0, n) and
   * returns the hidden buffer [patchCount * hiddenSize] instead of the image
   * embeddings. Used by the differential bisect; keep it out of the hot path.
   */
  /** Arena f32 offsets (debug bisect helper). */
  get arenaOffsets(): ArenaLayout {
    return this.arenaLayout;
  }

  /**
   * Read back a raw arena region in f32 (debug bisect helper).
   * Must not run concurrently with run()/readArena(): both share the staging
   * buffer. Debug-only — the differential harness calls it sequentially.
   */
  async readArena(offsetF32: number, count: number): Promise<Float32Array> {
    if (this.destroyed) throw new Error("VisionTower is destroyed");
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.arena, offsetF32 * 4, this.staging, 0, count * 4);
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    await this.staging.mapAsync(GPUMapMode.READ);
    try {
      return new Float32Array(this.staging.getMappedRange()).slice(0, count);
    } finally {
      this.staging.unmap();
    }
  }

  async run(image: ProcessedImage, options: { readonly stopAtLayer?: number } = {}): Promise<Float32Array> {
    if (this.destroyed) throw new Error("VisionTower is destroyed");
    const { config, arenaLayout } = this;
    const dim = config.hiddenSize;
    const ff = config.feedForwardSize;
    const patchDim = 3 * config.patchSize * config.patchSize;
    const factor = config.projectorScaleFactor;
    const P = image.patchCount;
    const gridW = image.gridW;
    const gridH = image.gridH;
    const tokens = Math.floor(gridH / factor) * Math.floor(gridW / factor);
    if (gridH % factor !== 0 || gridW % factor !== 0) {
      throw new Error(`VisionTower: grid ${gridH}x${gridW} not divisible by unshuffle factor ${factor}`);
    }
    if (gridH * gridW !== P) {
      throw new Error(`VisionTower: grid ${gridH}x${gridW} does not match patchCount ${P}`);
    }
    if (P > config.maxPatches) throw new Error(`VisionTower: ${P} patches exceed maxPatches ${config.maxPatches}`);

    // arena inputs (padded arrays are maxPatches-sized; write only the valid prefix)
    this.device.queue.writeBuffer(this.arena, arenaLayout.patches * 4, image.patches.subarray(0, P * patchDim));
    this.device.queue.writeBuffer(this.arena, arenaLayout.posEmb * 4, image.posEmb.subarray(0, P * dim));
    const mask = new Float32Array(config.maxPatches);
    for (let i = 0; i < config.maxPatches; i++) mask[i] = image.paddingMask[i] ?? 0;
    this.device.queue.writeBuffer(this.arena, arenaLayout.padMask * 4, mask);

    // params records, one per dispatch, selected by dynamic offset
    const paramsBytes = new Uint8Array(DISPATCH_SLOTS * this.stride);
    const params = new DataView(paramsBytes.buffer);
    let slot = 0;
    const writeParams = (values: {
      tokenCount: number; inputDim: number; outputDim: number; headDim: number; heads: number;
      gridW: number; gridH: number; inOff: number; outOff: number; wOff: number;
      bOff: number; auxOff: number; aux2Off: number; factor: number; projectorDim: number;
      eps: number; scale: number;
    }): number => {
      if (slot >= DISPATCH_SLOTS) throw new Error("VisionTower: dispatch slots exhausted");
      const base = slot * this.stride;
      const u32 = (i: number, v: number): void => params.setUint32(base + i * 4, v >>> 0, true);
      const f32 = (i: number, v: number): void => params.setFloat32(base + i * 4, v, true);
      u32(0, values.tokenCount); u32(1, values.inputDim); u32(2, values.outputDim);
      u32(3, values.headDim); u32(4, values.heads);
      u32(5, values.gridW); u32(6, values.gridH);
      u32(7, values.inOff); u32(8, values.outOff); u32(9, values.wOff);
      u32(10, values.bOff); u32(11, values.auxOff); u32(12, values.aux2Off);
      u32(13, values.factor); u32(14, values.projectorDim);
      f32(15, values.eps); f32(16, values.scale);
      return slot++;
    };

    const encoder = this.device.createCommandEncoder();
    const L = arenaLayout;
    const dispatch = (
      pass: keyof VisionTower["pipelines"],
      workgroups: [number, number, number],
      layer: "misc" | number,
      p: Parameters<typeof writeParams>[0],
    ): void => {
      const offset = writeParams(p);
      const compute = encoder.beginComputePass();
      compute.setPipeline(this.pipelines[pass]);
      compute.setBindGroup(0, this.bg0, [offset * this.stride]);
      compute.setBindGroup(1, layer === "misc" ? this.bg1Misc : this.bg1Layer[layer]!);
      compute.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
      compute.end();
    };

    const block = config.hiddenSize;
    const eps = config.layerNormEpsilon;
    const scale = Math.pow(config.headDim, -0.5);
    const base = {
      gridW,
      gridH,
      factor,
      projectorDim: config.projectorHiddenSize,
      eps,
      scale,
      headDim: config.headDim,
      heads: config.attentionHeads,
      auxOff: 0,
      aux2Off: 0,
    };

    // head: patch embedding + position embeddings (one workgroup per
    // (output row, token), like the matmul kernel)
    dispatch("patch_embed", [block, P, 1], "misc", {
      ...base,
      tokenCount: P, inputDim: patchDim, outputDim: block,
      inOff: L.patches, outOff: L.hidden, wOff: this.miscLayout.patchEmb,
      bOff: L.posEmb, auxOff: this.miscLayout.patchEmbB,
    });

    for (let layer = 0; layer < (options.stopAtLayer ?? config.blockCount); layer++) {
      const w = this.layerLayout;
      const qkv = L.qkv;
      // pre-LN + q/k/v projections
      dispatch("layernorm", [P, 1, 1], layer, {
        ...base, tokenCount: P, inputDim: block, outputDim: block,
        inOff: L.hidden, outOff: L.normed, wOff: w.ln1W, bOff: w.ln1B,
      });
      dispatch("matmul", [block, P, 1], layer, {
        ...base, tokenCount: P, inputDim: block, outputDim: block,
        inOff: L.normed, outOff: qkv, wOff: w.qW, bOff: w.qB,
      });
      dispatch("matmul", [block, P, 1], layer, {
        ...base, tokenCount: P, inputDim: block, outputDim: block,
        inOff: L.normed, outOff: qkv + P * block, wOff: w.kW, bOff: w.kB,
      });
      dispatch("matmul", [block, P, 1], layer, {
        ...base, tokenCount: P, inputDim: block, outputDim: block,
        inOff: L.normed, outOff: qkv + 2 * P * block, wOff: w.vW, bOff: w.vB,
      });
      // bidirectional attention (padding mask) -> attn
      dispatch("attention", [config.attentionHeads, P, 1], layer, {
        ...base, tokenCount: P, inputDim: block, outputDim: block,
        inOff: qkv, outOff: L.attn, wOff: 0, bOff: L.padMask,
      });
      // attention output projection + residual
      dispatch("matmul", [block, P, 1], layer, {
        ...base, tokenCount: P, inputDim: block, outputDim: block,
        inOff: L.attn, outOff: L.h, wOff: w.oW, bOff: w.oB,
      });
      dispatch("residual_add", [ceilDiv(P * block, 256), 1, 1], layer, {
        ...base, tokenCount: P, inputDim: block, outputDim: block,
        inOff: L.h, outOff: L.hidden, wOff: 0, bOff: 0,
      });
      // post-LN -> MLP (up, tanh-GELU, down) -> residual
      dispatch("layernorm", [P, 1, 1], layer, {
        ...base, tokenCount: P, inputDim: block, outputDim: block,
        inOff: L.hidden, outOff: L.normed, wOff: w.ln2W, bOff: w.ln2B,
      });
      dispatch("matmul", [ff, P, 1], layer, {
        ...base, tokenCount: P, inputDim: block, outputDim: ff,
        inOff: L.normed, outOff: L.ff, wOff: w.upW, bOff: w.upB,
      });
      dispatch("gelu_tanh", [ceilDiv(P * ff, 256), 1, 1], layer, {
        ...base, tokenCount: P, inputDim: block, outputDim: ff,
        inOff: L.ff, outOff: L.ff, wOff: 0, bOff: 0,
      });
      dispatch("matmul", [block, P, 1], layer, {
        ...base, tokenCount: P, inputDim: ff, outputDim: block,
        inOff: L.ff, outOff: L.h, wOff: w.downW, bOff: w.downB,
      });
      dispatch("residual_add", [ceilDiv(P * block, 256), 1, 1], layer, {
        ...base, tokenCount: P, inputDim: block, outputDim: block,
        inOff: L.h, outOff: L.hidden, wOff: 0, bOff: 0,
      });
    }

    // post LayerNorm -> unshuffle + projector -> image embeddings
    if (options.stopAtLayer !== undefined) {
      encoder.copyBufferToBuffer(this.arena, L.hidden * 4, this.staging, 0, P * block * 4);
    } else {
      dispatch("layernorm", [P, 1, 1], "misc", {
        ...base, tokenCount: P, inputDim: block, outputDim: block,
        inOff: L.hidden, outOff: L.normed, wOff: this.miscLayout.postLnW, bOff: this.miscLayout.postLnB,
      });
      dispatch("unshuffle_project", [tokens, 1, 1], "misc", {
        ...base, tokenCount: tokens, inputDim: block, outputDim: config.projectorHiddenSize,
        inOff: L.normed, outOff: L.embeddings,
        wOff: this.miscLayout.mm1, bOff: this.miscLayout.mm1B,
        auxOff: this.miscLayout.mm2, aux2Off: this.miscLayout.mm2B,
      });
      encoder.copyBufferToBuffer(this.arena, L.embeddings * 4, this.staging, 0, tokens * config.projectorHiddenSize * 4);
    }
    const commandBuffer = encoder.finish();

    this.device.queue.writeBuffer(this.paramsRing, 0, paramsBytes);
    this.device.queue.submit([commandBuffer]);
    await this.device.queue.onSubmittedWorkDone();
    await this.staging.mapAsync(GPUMapMode.READ);
    try {
      const mapped = new Float32Array(this.staging.getMappedRange());
      const count = options.stopAtLayer !== undefined ? P * block : tokens * config.projectorHiddenSize;
      return mapped.slice(0, count);
    } finally {
      this.staging.unmap();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.arena.destroy();
    this.paramsRing.destroy();
    this.staging.destroy();
    for (const buffer of this.layerBuffers) buffer.destroy();
  }
}

async function createStorage(device: GPUDevice, label: string, size: number, copySrc: boolean): Promise<GPUBuffer> {
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | (copySrc ? GPUBufferUsage.COPY_SRC : 0);
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  const buffer = device.createBuffer({ label, size, usage });
  const validationError = await device.popErrorScope();
  const oomError = await device.popErrorScope();
  if (validationError || oomError) {
    buffer.destroy();
    throw new Error(`${label}: GPU allocation failed: ${(validationError ?? oomError)?.message ?? "unknown"}`);
  }
  return buffer;
}

function ceilDiv(a: number, b: number): number {
  return Math.ceil(a / b);
}
