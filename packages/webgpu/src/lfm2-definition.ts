import {
  Sandblaster,
  type AnyComputeHandle,
  type BufferResource,
  type BufferResourceUse,
} from "@sandblaster/core-next";
import { $ } from "../../schema/src/schema";
import { LFM2_CONFIG } from "./lfm2-init";
export const CONTEXT_CAPACITY = 1024;
export const MAX_NEW_TOKENS = 128;

// block boundary history 4 + repair 4
export const REPAIR_CAPACITY = 8;

export const H = LFM2_CONFIG.hiddenSize;          // 2048
export const FF = LFM2_CONFIG.feedForwardSize;    // 8192
export const VOCAB = LFM2_CONFIG.vocabSize;       // 65536

export const SCRATCH_WIDTH = Math.max(FF, 3 * H);

export interface Lfm2WorkLayout {
  hiddenA: number;
  hiddenB: number;
  tmpH: number;
  tmpA: number;
  tmpB: number;
}

export interface Lfm2ArenaLayout extends Lfm2WorkLayout {
  repair: Lfm2WorkLayout;
  logits: number;
  elements: number;
}

function createArenaLayout(): Lfm2ArenaLayout {
  let cursor = 0;
  const take = (elements: number) => {
    const offset = cursor;
    cursor += elements;
    return offset;
  };
  const work = (tokenCapacity: number): Lfm2WorkLayout => ({
    hiddenA: take(tokenCapacity * H),
    hiddenB: take(tokenCapacity * H),
    tmpH: take(tokenCapacity * H),
    tmpA: take(tokenCapacity * SCRATCH_WIDTH),
    tmpB: take(tokenCapacity * SCRATCH_WIDTH),
  });

  const main = work(CONTEXT_CAPACITY);
  const repair = work(REPAIR_CAPACITY);
  const logits = take(VOCAB);
  return { ...main, repair, logits, elements: cursor };
}

/** Canonical activation offsets shared by all runtime scheduling code. */
export const LFM2_ARENA = createArenaLayout();
export const ARENA_ELEMENTS = LFM2_ARENA.elements;

// Generalniejsze niż attentionLayerCount * 8:
// obsłuży też ewentualnie różną liczbę KV heads per layer.
export const KV_ELEMENTS =
  2 * // K + V
  CONTEXT_CAPACITY *
  LFM2_CONFIG.headDim *
  LFM2_CONFIG.kvHeadsByLayer.reduce((sum, heads) => sum + heads, 0);

export const CONV_ELEMENTS =
  LFM2_CONFIG.blockCount *
  H *
  LFM2_CONFIG.convCacheLength;

export const TOKEN_CAPACITY =
  CONTEXT_CAPACITY + MAX_NEW_TOKENS;

export const TELEMETRY_CAPACITY = 256;
export const OP_PARAM_BUFFER_BYTES = 8 * 1024 * 1024;

export const LFM2_SHADER_NAMES = [
  "embedding",
  "embedding_wq4",
  "rms_norm",
  "matmul_f16",
  "matmul_f32",
  "matmul_wq4",
  "residual_add",
  "silu_mul",
  "shortconv_prefill",
  "shortconv_continue",
  "shortconv_decode",
  "qk_norm_rope",
  "kv_store",
  "attention",
  "arena_copy",
  "argmax_candidates",
  "argmax",
] as const;

export type Lfm2ShaderName = (typeof LFM2_SHADER_NAMES)[number];

export type Lfm2Mode = "prefill" | "decode" | "continuation";

/** Host-side shape of the per-dispatch OpParams schema. */
export interface Lfm2OpParams {
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
  mode?: Lfm2Mode;
  f0?: number;
  f1?: number;
  u0?: number;
  u1?: number;
}

export type Lfm2Workgroups = readonly [x: number, y: number, z: number];
export type Lfm2WeightBinding = "none" | "raw" | "f32";

/**
 * A pass is the stable execution-level description of one shader entry point.
 * It owns dispatch geometry; the runtime only supplies OpParams and, where
 * required, the concrete tensor page bound to the weight resource.
 */
export interface Lfm2PassSpec {
  readonly program: AnyComputeHandle;
  readonly weight: Lfm2WeightBinding;
  workgroups(op: Readonly<Lfm2OpParams>): Lfm2Workgroups;
}

const QUERY_HEADS = LFM2_CONFIG.attentionHeads;
const KV_HEADS = Math.max(...LFM2_CONFIG.kvHeadsByLayer);
const KV_DIM = KV_HEADS * LFM2_CONFIG.headDim;

function required(value: number | undefined, field: keyof Lfm2OpParams): number {
  if (value === undefined) throw new Error(`LFM2 pass requires op.${field}`);
  return value;
}

function linear(value: number, workgroupSize: number): Lfm2Workgroups {
  return [Math.ceil(value / workgroupSize), 1, 1];
}

function definePass(
  program: AnyComputeHandle,
  weight: Lfm2WeightBinding,
  workgroups: Lfm2PassSpec["workgroups"],
): Lfm2PassSpec {
  return { program, weight, workgroups };
}


export const LFM2_INCLUDE_NAMES = [
  "arena",
  "attention-scores",
  "common",
  "reduce-f32",
  "reduce-u32",
  "runtime",
  "telemetry",
  "weights",
] as const;

export type Lfm2IncludeName = (typeof LFM2_INCLUDE_NAMES)[number];

export interface Lfm2ShaderBundle {
  readonly sources: Record<Lfm2ShaderName, string>;
  readonly includes: Record<Lfm2IncludeName, string>;
}

function emptyRecord<const K extends readonly string[]>(keys: K): Record<K[number], string> {
  return Object.fromEntries(keys.map((key) => [key, ""])) as Record<K[number], string>;
}

export function emptyLfm2ShaderBundle(): Lfm2ShaderBundle {
  return {
    sources: emptyRecord(LFM2_SHADER_NAMES),
    includes: emptyRecord(LFM2_INCLUDE_NAMES),
  };
}

export function defineLfm2(bundle: Lfm2ShaderBundle = emptyLfm2ShaderBundle()) {
  const engine = Sandblaster.create($, {
    codec: "jit",
    schema: { autoSort: true },
  });
  const sources = bundle.sources;
  const shaderIncludes = bundle.includes;

  const OpParams = engine.type("OpParams");
  const LlmRuntime = engine.type("LlmRuntime");

  const op = engine.buffer(OpParams, {
    label: "lfm2.op",
    size: OP_PARAM_BUFFER_BYTES,
    value: OpParams.assert({}),
  });
  const runtime = engine.buffer(LlmRuntime, { label: "lfm2.runtime", value: LlmRuntime.assert({}), readback: true });

  const tokens = engine.buffer(engine.type(`u32[] == ${TOKEN_CAPACITY}`), { label: "lfm2.tokens", readback: true });
  const arena = engine.buffer(engine.type(`f32[] == ${ARENA_ELEMENTS}`), { label: "lfm2.arena", readback: true });
  // Checkpoints snapshot these buffers with copyBufferToBuffer. readback=true
  // adds COPY_SRC without forcing any staging allocation until readback() is used.
  const kvCache = engine.buffer(engine.type(`f32[] == ${KV_ELEMENTS}`), { label: "lfm2.kv-cache", readback: true });
  const convCache = engine.buffer(engine.type(`f32[] == ${CONV_ELEMENTS}`), { label: "lfm2.conv-cache", readback: true });
  const candidateTokens = engine.buffer(engine.type(`u32[] == ${VOCAB}`), { label: "lfm2.candidate-tokens" });
  const decodeTelemetry = engine.buffer(engine.type(`u32[] == ${TELEMETRY_CAPACITY}`), { label: "lfm2.decode-telemetry" });
  const weightRaw = engine.buffer(engine.type("u32"), { label: "lfm2.probe-weight-raw", count: 2 });
  const weight32 = engine.buffer(engine.type("f32"), { label: "lfm2.probe-weight32", count: 2 });


  type Resource = BufferResource<any>;
  function nativeRead(resource: Resource, group = 0): BufferResourceUse {
    return {
      resource,
      group,
      buffer: { type: "read-only-storage" },
      representation: "native",
    };
  }
  function nativeWrite(resource: Resource, group = 0): BufferResourceUse {
    return {
      resource,
      group,
      buffer: { type: "storage" },
      representation: "native",
    };
  }

  /**
   * Canonical WGSL views. `op` is one 256-byte uniform record selected by a
   * dynamic offset; tensor pages live in group 1 so they can be overridden per
   * dispatch without rebuilding the long-lived runtime group.
   */
  function lfm2ResourceViews() {
    return {
      op: {
        resource: op,
        group: 0,
        buffer: {
          type: "uniform",
          hasDynamicOffset: true,
          minBindingSize: 256,
        },
        offset: 0,
        size: 256,
        representation: "native",
      } satisfies BufferResourceUse,
      runtime: nativeWrite(runtime),
      tokens: nativeWrite(tokens),
      arena: nativeWrite(arena),
      kvCache: nativeWrite(kvCache),
      convCache: nativeWrite(convCache),
      candidateTokens: nativeRead(candidateTokens),
      decodeTelemetry: nativeWrite(decodeTelemetry),
      weightRaw: nativeRead(weightRaw, 1),
      weight32: nativeRead(weight32, 1),
    } as const;
  }

  const r = lfm2ResourceViews();

  /**
   * Define all current compute entry points against the per-program resource
   * subsets. This is sufficient for engine.link() and shader validation/Dawn.
   *
   * It is intentionally NOT yet the inference call scheduler: the legacy runtime
   * changes OpParams and concrete weight pages per dispatch, while Sandblaster's
   * current ComputePassRunner binds a program's resources statically.
   */
  const gid = engine.type({ gid: "global_invocation_id" })
  const wid = engine.type({ wid: "workgroup_id" });
  const lid = engine.type({ lid: "local_invocation_id" });
  const widLid = engine.type({
    wid: "workgroup_id",
    lid: "local_invocation_id",
  });

  const include = (name: keyof typeof shaderIncludes) => shaderIncludes[name];

  const commonIncludes = [include("common")];
  const weightIncludes = [...commonIncludes, include("weights")];
  const reduceF32Includes = [...commonIncludes, include("reduce-f32")];
  const weightReduceF32Includes = [...weightIncludes, include("reduce-f32")];
  const runtimeIncludes = [...commonIncludes, include("runtime")];
  const ropeIncludes = [...runtimeIncludes, include("arena"), include("reduce-f32")];
  const attentionIncludes = [...runtimeIncludes, include("attention-scores")];
  const argmaxIncludes = [
    ...commonIncludes,
    include("telemetry"),
    include("reduce-f32"),
    include("reduce-u32"),
  ];

  const programs = {
    embedding: engine.compute({
      label: "embedding",
      resources: { op: r.op, runtime: r.runtime, tokens: r.tokens, arena: r.arena, weightRaw: r.weightRaw },
      includes: weightIncludes,
      compute: { entryPoint: "embedding", params: gid, workgroupSize: 256, code: sources.embedding },
    }),

    embedding_wq4: engine.compute({
      label: "embedding_wq4",
      resources: { op: r.op, runtime: r.runtime, tokens: r.tokens, arena: r.arena, weightRaw: r.weightRaw },
      includes: weightIncludes,
      compute: { entryPoint: "embedding_wq4", params: gid, workgroupSize: 256, code: sources.embedding_wq4 },
    }),

    rms_norm: engine.compute({
      label: "rms_norm",
      resources: { op: r.op, arena: r.arena, weight32: r.weight32 },
      includes: reduceF32Includes,
      compute: { entryPoint: "rms_norm", params: widLid, workgroupSize: 64, code: sources.rms_norm },
    }),

    matmul_f16: engine.compute({
      label: "matmul_f16",
      resources: { op: r.op, arena: r.arena, weightRaw: r.weightRaw },
      includes: weightReduceF32Includes,
      compute: { entryPoint: "matmul_f16", params: widLid, workgroupSize: 64, code: sources.matmul_f16 },
    }),

    matmul_f32: engine.compute({
      label: "matmul_f32",
      resources: { op: r.op, arena: r.arena, weight32: r.weight32 },
      includes: reduceF32Includes,
      compute: { entryPoint: "matmul_f32", params: widLid, workgroupSize: 64, code: sources.matmul_f32 },
    }),

    matmul_wq4: engine.compute({
      label: "matmul_wq4",
      resources: { op: r.op, arena: r.arena, weightRaw: r.weightRaw },
      includes: reduceF32Includes,
      compute: { entryPoint: "matmul_wq4", params: widLid, workgroupSize: 64, code: sources.matmul_wq4 },
    }),

    residual_add: engine.compute({
      label: "residual_add",
      resources: { op: r.op, arena: r.arena },
      compute: { entryPoint: "residual_add", params: gid, workgroupSize: 256, code: sources.residual_add },
    }),

    silu_mul: engine.compute({
      label: "silu_mul",
      resources: { op: r.op, arena: r.arena },
      compute: { entryPoint: "silu_mul", params: gid, workgroupSize: 256, code: sources.silu_mul },
    }),

    shortconv_prefill: engine.compute({
      label: "shortconv_prefill",
      resources: { op: r.op, arena: r.arena, convCache: r.convCache, weight32: r.weight32 },
      compute: { entryPoint: "shortconv_prefill", params: gid, workgroupSize: 256, code: sources.shortconv_prefill },
    }),

    shortconv_continue: engine.compute({
      label: "shortconv_continue",
      resources: { op: r.op, arena: r.arena, convCache: r.convCache, weight32: r.weight32 },
      compute: { entryPoint: "shortconv_continue", params: wid, workgroupSize: 1, code: sources.shortconv_continue },
    }),

    shortconv_decode: engine.compute({
      label: "shortconv_decode",
      resources: { op: r.op, arena: r.arena, convCache: r.convCache, weight32: r.weight32 },
      compute: { entryPoint: "shortconv_decode", params: gid, workgroupSize: 256, code: sources.shortconv_decode },
    }),

    qk_norm_rope: engine.compute({
      label: "qk_norm_rope",
      resources: { op: r.op, runtime: r.runtime, arena: r.arena, weight32: r.weight32 },
      includes: ropeIncludes,
      compute: { entryPoint: "qk_norm_rope", params: widLid, workgroupSize: 64, code: sources.qk_norm_rope },
    }),

    kv_store: engine.compute({
      label: "kv_store",
      resources: { op: r.op, runtime: r.runtime, arena: r.arena, kvCache: r.kvCache },
      includes: runtimeIncludes,
      compute: { entryPoint: "kv_store", params: gid, workgroupSize: 256, code: sources.kv_store },
    }),

    attention: engine.compute({
      label: "attention",
      resources: { op: r.op, runtime: r.runtime, arena: r.arena, kvCache: r.kvCache },
      includes: attentionIncludes,
      compute: { entryPoint: "attention", params: widLid, workgroupSize: 64, code: sources.attention },
    }),

    arena_copy: engine.compute({
      label: "arena_copy",
      resources: { op: r.op, arena: r.arena },
      compute: { entryPoint: "arena_copy", params: gid, workgroupSize: 256, code: sources.arena_copy },
    }),

    argmax_candidates: engine.compute({
      label: "argmax_candidates",
      resources: {
        op: r.op,
        runtime: r.runtime,
        tokens: r.tokens,
        arena: r.arena,
        candidateTokens: r.candidateTokens,
        decodeTelemetry: r.decodeTelemetry,
      },
      codecs: [engine.type("DecodeTelemetryEntry")],
      includes: argmaxIncludes,
      compute: { entryPoint: "argmax_candidates", params: lid, workgroupSize: 256, code: sources.argmax_candidates },
    }),

    argmax: engine.compute({
      label: "argmax",
      resources: { op: r.op, runtime: r.runtime, tokens: r.tokens, arena: r.arena, decodeTelemetry: r.decodeTelemetry },
      codecs: [engine.type("DecodeTelemetryEntry")],
      includes: argmaxIncludes,
      compute: { entryPoint: "argmax", params: lid, workgroupSize: 256, code: sources.argmax },
    }),
  } satisfies Record<Lfm2ShaderName, AnyComputeHandle>;


  /**
   * All shader-specific dispatch rules live here, on the GPU-definition side of
   * the module boundary. Runtime orchestration should never duplicate these.
   */
  const passes = {
    embedding: definePass(programs.embedding, "raw", (op) =>
      linear(required(op.tokenCount, "tokenCount") * required(op.outputDim, "outputDim"), 256)),

    embedding_wq4: definePass(programs.embedding_wq4, "raw", (op) =>
      linear(required(op.tokenCount, "tokenCount") * required(op.outputDim, "outputDim"), 256)),

    rms_norm: definePass(programs.rms_norm, "f32", (op) =>
      [required(op.tokenCount, "tokenCount"), 1, 1]),

    matmul_f16: definePass(programs.matmul_f16, "raw", (op) =>
      [required(op.rowCount, "rowCount"), required(op.tokenCount, "tokenCount"), 1]),

    matmul_f32: definePass(programs.matmul_f32, "f32", (op) =>
      [required(op.rowCount, "rowCount"), required(op.tokenCount, "tokenCount"), 1]),

    matmul_wq4: definePass(programs.matmul_wq4, "raw", (op) =>
      [required(op.rowCount, "rowCount"), required(op.tokenCount, "tokenCount"), 1]),

    residual_add: definePass(programs.residual_add, "none", (op) =>
      linear(required(op.tokenCount, "tokenCount") * required(op.inputDim, "inputDim"), 256)),

    silu_mul: definePass(programs.silu_mul, "none", (op) =>
      linear(required(op.tokenCount, "tokenCount") * required(op.inputDim, "inputDim"), 256)),

    shortconv_prefill: definePass(programs.shortconv_prefill, "f32", (op) =>
      linear(required(op.tokenCount, "tokenCount") * required(op.inputDim, "inputDim"), 256)),

    shortconv_continue: definePass(programs.shortconv_continue, "f32", (op) =>
      [required(op.inputDim, "inputDim"), 1, 1]),

    shortconv_decode: definePass(programs.shortconv_decode, "f32", (op) =>
      linear(required(op.inputDim, "inputDim"), 256)),

    qk_norm_rope: definePass(programs.qk_norm_rope, "f32", (op) =>
      [op.u0 === 0 ? QUERY_HEADS : KV_HEADS, required(op.tokenCount, "tokenCount"), 1]),

    kv_store: definePass(programs.kv_store, "none", (op) =>
      linear(required(op.tokenCount, "tokenCount") * KV_DIM, 256)),

    attention: definePass(programs.attention, "none", (op) =>
      [QUERY_HEADS, required(op.tokenCount, "tokenCount"), 1]),

    arena_copy: definePass(programs.arena_copy, "none", (op) =>
      linear(required(op.tokenCount, "tokenCount") * required(op.inputDim, "inputDim"), 256)),

    argmax_candidates: definePass(programs.argmax_candidates, "none", () => [1, 1, 1]),
    argmax: definePass(programs.argmax, "none", () => [1, 1, 1]),
  } satisfies Record<Lfm2ShaderName, Lfm2PassSpec>;

  const resources = {
    op,
    runtime,
    tokens,
    arena,
    kvCache,
    convCache,
    candidateTokens,
    decodeTelemetry,
    weightRaw,
    weight32,
  } as const;


  return {
    config: LFM2_CONFIG,
    capacities: {
      context: CONTEXT_CAPACITY,
      maxNewTokens: MAX_NEW_TOKENS,
      repair: REPAIR_CAPACITY,
      tokens: TOKEN_CAPACITY,
      arena: ARENA_ELEMENTS,
      kv: KV_ELEMENTS,
      conv: CONV_ELEMENTS,
      telemetry: TELEMETRY_CAPACITY,
    },
    arena: LFM2_ARENA,
    engine,
    resources,
    programs,
    passes,
  } as const;
}

export type Lfm2Definition = ReturnType<typeof defineLfm2>;
