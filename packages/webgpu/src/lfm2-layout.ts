// Pure-TS LFM2 layout and dispatch spec.
//
// Everything here is static: constants, the activation arena layout, the
// schema-independent host types and the per-pass dispatch geometry. It is the
// part of lfm2-definition.ts that the runtime actually needs, and it is shared
// by the two definition builders:
//
//   - defineLfm2(bundle)            (lfm2-definition.ts) — arktype-backed DSL,
//                                     used only by the AOT build/validate scripts
//   - defineLfm2FromArtifact()      (lfm2-artifact.ts)   — handle creation from
//                                     lfm2.artifact.generated.ts, used at runtime
//
// The runtime entry (lfm2.ts) imports only this module and lfm2-artifact.ts,
// so the DSL, `$` and arktype never enter a scriptc-compiled graph.
import { LFM2_CONFIG } from "./lfm2-init";
import type { AnyComputeHandle } from "@sandblaster/core";

export const CONTEXT_CAPACITY = 1024;
// Capacity guard for the schema-derived decode budget. The parameter itself is
// scheduled to disappear once buffer sizes are derived from the schema; for now
// 1024 is the maximum the current arena/token layout can support.
export const MAX_NEW_TOKENS = 1024;

// block boundary history 4 + repair 4
export const REPAIR_CAPACITY = 8;

export const H = LFM2_CONFIG.hiddenSize;          // 2048
export const FF = LFM2_CONFIG.feedForwardSize;    // 8192
export const VOCAB = LFM2_CONFIG.vocabSize;       // 65536

export const SCRATCH_WIDTH = Math.max(FF, 3 * H);

/**
 * Sampling lanes and per-lane candidate slots.
 *
 * The argmax kernel runs as ONE workgroup of ARGMAX_WG(256) invocations, so a
 * top-k selection cannot be spread over more parallelism than that. Each lane
 * owns the tokens `lane, lane + 256, lane + 512, ...` and keeps its own sorted
 * top-k of them; the union of those lists provably contains the global top-k,
 * which a k-round tournament over the list heads then merges exactly.
 *
 * The lists live in the arena rather than workgroup memory: 256 lanes * 64
 * slots * (value + token) is 128 KiB, far past the 16 KiB workgroup storage
 * limit, and each lane only ever reads its own slots so no cross-lane storage
 * barrier is needed. 64 is the cap on topK because of this allocation; raising
 * it costs 2 KiB of arena per extra slot.
 */
export const SAMPLE_LANES = 256;
export const SAMPLE_TOP_K_MAX = 64;
export const SAMPLE_SCRATCH_ELEMENTS = 2 * SAMPLE_LANES * SAMPLE_TOP_K_MAX;

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
  /** Per-lane top-k candidate lists, written and read only by the argmax pass. */
  sampleScratch: number;
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
  const sampleScratch = take(SAMPLE_SCRATCH_ELEMENTS);
  return { ...main, repair, logits, sampleScratch, elements: cursor };
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
// One 256-byte (aligned) OpParams record per dispatch. A full decode step of
// the 16-block model spends ~250 dispatches, so the buffer must fit the whole
// schema-derived budget: MAX_NEW_TOKENS(1024) * ~250 * 256 B ~= 64 MiB. 128 MiB
// leaves headroom for paged matmuls and the prefill of a long prompt.
export const OP_PARAM_BUFFER_BYTES = 128 * 1024 * 1024;

/**
 * Output rows computed per matmul_wq4 workgroup, narrow and wide.
 *
 * Must equal MATMUL_ROWS in shaders/includes/matmul-rows.wgsl and
 * matmul-rows-wide.wgsl. Tiling rows is what takes the kernel off its
 * launch/reduction bound; which tiling wins depends on the output width, so
 * both are linked and chosen per call (see matmulWq4Program).
 */
export const MATMUL_WQ4_ROWS = 8;
export const MATMUL_WQ4_ROWS_WIDE = 16;

/**
 * The measured crossover. Wide tiling wins at every output width the model uses
 * above 2048 (6144 +8%, 8192 +18%, 52428 +32%) and loses at 2048 itself
 * (attention projections -10%, ffn_down -3%). Numbers and method are in
 * shaders/includes/matmul-rows-wide.wgsl.
 */
export function matmulWq4Program(outputDim: number): "matmul_wq4" | "matmul_wq4_wide" {
  return outputDim > 2048 ? "matmul_wq4_wide" : "matmul_wq4";
}

// Structured-generation VM. These are fixed AOT binding capacities, not
// semantic schema limits. Actual program/tokenizer blobs carry their lengths.
export const CONSTRAINT_PROGRAM_WORD_CAPACITY = 1 << 18;   // 1 MiB
export const CONSTRAINT_TOKENIZER_WORD_CAPACITY = 1 << 20; // 4 MiB
export const CONSTRAINT_STATE_WORDS = 24;                   // 96 B
export const CONSTRAINT_MASK_WORDS = Math.ceil(VOCAB / 32);
export const CONSTRAINT_MASK_WORKGROUP_SIZE = 64;
export const CONSTRAINT_MASK_WORKGROUPS = Math.ceil(
  CONSTRAINT_MASK_WORDS / CONSTRAINT_MASK_WORKGROUP_SIZE,
);

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
  "constraint_mask",
  "constraint_argmax",
] as const;

export type Lfm2ShaderName = (typeof LFM2_SHADER_NAMES)[number];

/**
 * Programs are no longer 1:1 with shader files: matmul_wq4 is compiled twice
 * from one body, once per row tiling. Names above index the .wgsl files on
 * disk; names here index the linked programs.
 */
export const LFM2_PROGRAM_NAMES = [...LFM2_SHADER_NAMES, "matmul_wq4_wide"] as const;
export type Lfm2ProgramName = (typeof LFM2_PROGRAM_NAMES)[number];

export type Lfm2PassName = Exclude<Lfm2ProgramName, "constraint_mask">;

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

/**
 * Definition-level plain fields shared by both builders (lfm2-definition.ts
 * DSL and lfm2-artifact.ts). Keeping them in one place means a capacity or
 * constraint change cannot silently diverge between the two paths.
 */
export const LFM2_DEFINITION_PLAIN = {
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
  constraint: {
    programWordCapacity: CONSTRAINT_PROGRAM_WORD_CAPACITY,
    tokenizerWordCapacity: CONSTRAINT_TOKENIZER_WORD_CAPACITY,
    stateWords: CONSTRAINT_STATE_WORDS,
    maskWords: CONSTRAINT_MASK_WORDS,
    maskWorkgroups: CONSTRAINT_MASK_WORKGROUPS,
  },
} as const;

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

/**
 * Per-shader dispatch rules, parameterized by the program handles so the same
 * geometry drives both the DSL-built and the artifact-built definition.
 */
export function defineLfm2Passes(
  programs: Record<Lfm2ProgramName, AnyComputeHandle>,
): Record<Lfm2PassName, Lfm2PassSpec> {
  return {
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

    // Each matmul_wq4 workgroup computes MATMUL_ROWS output rows, so the launch
    // count is divided accordingly — and the divisor has to match the variant's
    // include, or the tail rows are silently never written.
    matmul_wq4: definePass(programs.matmul_wq4, "raw", (op) =>
      [
        Math.ceil(required(op.rowCount, "rowCount") / MATMUL_WQ4_ROWS),
        required(op.tokenCount, "tokenCount"),
        1,
      ]),

    matmul_wq4_wide: definePass(programs.matmul_wq4_wide, "raw", (op) =>
      [
        Math.ceil(required(op.rowCount, "rowCount") / MATMUL_WQ4_ROWS_WIDE),
        required(op.tokenCount, "tokenCount"),
        1,
      ]),

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
    constraint_argmax: definePass(programs.constraint_argmax, "none", () => [1, 1, 1]),
  } satisfies Record<Lfm2PassName, Lfm2PassSpec>;
}
