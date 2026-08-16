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
import type { AnyComputeHandle } from "@sandblaster/core";

export const CONTEXT_CAPACITY = 1024;
// Capacity guard for the schema-derived decode budget. The parameter itself is
// scheduled to disappear once buffer sizes are derived from the schema; for now
// 1024 is the maximum the current arena/token layout can support.
export const MAX_NEW_TOKENS = 1024;

// --- Training arena (M1 tiny f32 training vertical slice) --------------------
//
// Fixed capacity constants, not ABI limits: shaders read actual dims from
// OpParams. The host packs regions for the concrete (M, V, H) of each
// trainStep and validates against these capacities before dispatching.

export const TRAINING_MAX_M = 64;
export const TRAINING_MAX_V = 4096;
export const TRAINING_MAX_H = 128;
/** Krystal first profile: 4 full attention heads × 32 dims (answer 11). */
export const TRAINING_MAX_HEADS = 4;

export interface Lfm2TrainingArenaLayout {
  hidden: number;      // [M,H]
  logits: number;      // [M,V]
  lossRows: number;    // [M]
  scalarLoss: number;  // [1]
  dLogits: number;     // [M,V]
  dHidden: number;     // [M,H]
  dClassifier: number; // [V,H]
  dEmbedding: number;  // [V,H]

  // Attention encoder block (§17 item 6 wiring). Q/K/V and the attention
  // output are [M,H] (head h at columns [h*headDim, (h+1)*headDim)); P and
  // dScores are [heads, M, M]; mask is a host-compiled [M,M].
  q: number;           // [M,H]
  k: number;           // [M,H]
  v: number;           // [M,H]
  out: number;         // [M,H]
  p: number;           // [heads,M,M]
  mask: number;        // [M,M]
  dOut: number;        // [M,H]
  dScores: number;     // [heads,M,M]
  dQ: number;          // [M,H]
  dK: number;          // [M,H]
  dV: number;          // [M,H]
  dHiddenQ: number;    // [M,H] (accumulated into dHidden via residual_add)
  dHiddenK: number;    // [M,H]
  dHiddenV: number;    // [M,H]
  dWq: number;         // [H,H]
  dWk: number;         // [H,H]
  dWv: number;         // [H,H]
  elements: number;
}

function createTrainingArenaLayout(): Lfm2TrainingArenaLayout {
  let cursor = 0;
  const take = (elements: number) => {
    const offset = cursor;
    cursor += elements;
    return offset;
  };
  const mh = TRAINING_MAX_M * TRAINING_MAX_H;
  const hh = TRAINING_MAX_H * TRAINING_MAX_H;
  const hm = TRAINING_MAX_HEADS * TRAINING_MAX_M * TRAINING_MAX_M;
  return {
    hidden: take(mh),
    logits: take(TRAINING_MAX_M * TRAINING_MAX_V),
    lossRows: take(TRAINING_MAX_M),
    scalarLoss: take(1),
    dLogits: take(TRAINING_MAX_M * TRAINING_MAX_V),
    dHidden: take(mh),
    dClassifier: take(TRAINING_MAX_V * TRAINING_MAX_H),
    dEmbedding: take(TRAINING_MAX_V * TRAINING_MAX_H),

    q: take(mh),
    k: take(mh),
    v: take(mh),
    out: take(mh),
    p: take(hm),
    mask: take(TRAINING_MAX_M * TRAINING_MAX_M),
    dOut: take(mh),
    dScores: take(hm),
    dQ: take(mh),
    dK: take(mh),
    dV: take(mh),
    dHiddenQ: take(mh),
    dHiddenK: take(mh),
    dHiddenV: take(mh),
    dWq: take(hh),
    dWk: take(hh),
    dWv: take(hh),
    elements: cursor,
  };
}

/** Training regions are appended after the LFM2 regions inside the one arena. */
// The legacy LFM2 activation regions were removed with the LFM2 runtime; the
// training regions now lead the shared arena.
export const TRAINING_ARENA_BASE = 0;
export const LFM2_TRAINING_ARENA = createTrainingArenaLayout();
export const TRAINING_ARENA_ELEMENTS = LFM2_TRAINING_ARENA.elements;

/**
 * Debug readback staging capacity: enough for the largest training region
 * (a full V*H parameter page at TRAINING_MAX_V * TRAINING_MAX_H).
 */
export const TRAINING_READBACK_ELEMENTS = TRAINING_MAX_V * TRAINING_MAX_H;

// --- Krystal forward arena (M2b: record/query encoder + mixer forward) ------
//
// Fixed capacity constants, not ABI limits: shaders read actual dims from
// OpParams. The SoA frame (tokenIds/fieldRoles/...) is uploaded into the arena
// as u32 payloads and bitcast in WGSL; the host-compiled active lists and
// record masks follow concerns answer 15/16 (host-compiled masks, dispatch
// over active records/tokens only).

export const KRYSTAL_MAX_TOKENS = 1024; // frameTokens (hard v2 capacity)
export const KRYSTAL_MAX_RECORDS = 128; // frameRecordSlots
export const KRYSTAL_MAX_H = 128; // first profile hidden size (answer 9)
export const KRYSTAL_MAX_FFN = 384; // first profile FFN size (answer 9)
export const KRYSTAL_MAX_HEADS = 4; // first profile full attention heads (answer 11)
export const KRYSTAL_MAX_QUERIES = 8; // maxQueries
export const KRYSTAL_MAX_ROUTE_KINDS = 8; // typed decision-head classes (capacity, not ABI)
export const KRYSTAL_MAX_BLOCKS = 2; // shared capacity for encoder + mixer block stacks

export interface KrystalForwardArenaLayout {
  // SoA frame inputs (u32 payloads bitcast in shaders) + host-compiled lists.
  tokenIds: number; // [maxTokens]
  fieldRoles: number; // [maxTokens]
  schemaIds: number; // [maxRecords]
  bandIds: number; // [maxRecords]
  streamIds: number; // [maxRecords]
  activeTokens: number; // [maxTokens]
  recordCompactOffset: number; // [maxRecords]
  recordCompactCount: number; // [maxRecords]
  bankIndices: number; // [maxRecords]
  queryIndices: number; // [maxQueries]

  fieldStates: number; // [maxTokens, H]
  encQ: number; // [maxTokens, H]
  encK: number; // [maxTokens, H]
  encV: number; // [maxTokens, H]
  encOut: number; // [maxTokens, H]
  encH1: number; // [maxTokens, FFN]
  encMask: number; // [maxTokens, maxTokens]
  encP: number; // [maxHeads, maxTokens, maxTokens] persisted attention probs

  bankKeys: number; // [maxRecords, H]
  bankValues: number; // [maxRecords, H]
  queryKeys: number; // [maxQueries, H]
  queryValues: number; // [maxQueries, H]

  mixerQ: number; // [maxQueries, H]
  mixerK: number; // [maxQueries, H]
  mixerV: number; // [maxQueries, H]
  mixerH1: number; // [maxQueries, FFN]
  mixed: number; // [maxQueries, H]
  mixerMask: number; // [maxQueries, maxRecords]
  mixerP: number; // [maxHeads, maxQueries, maxRecords] persisted mixer probs

  // Catalog selection + soft gather (architecture v2 §7, answer 26).
  selectorQ: number; // [maxQueries, H]  shared query/key projections
  selectorK: number; // [maxRecords, H]
  intentMask: number; // [maxQueries, maxRecords]
  argMask: number; // [maxQueries, maxRecords]
  intentP: number; // [maxQueries, maxRecords]
  intentGather: number; // [maxQueries, H]
  intentIndices: number; // [maxQueries]
  argP: number; // [maxQueries, maxRecords]
  argGather: number; // [maxQueries, H]
  argIndices: number; // [maxQueries]
  decisionLogits: number; // [maxQueries, routeKinds] (typed decision head)

  // Per-block saved activations for the composed backward runner (M3 close,
  // §17 item 10). The forward mutates fieldStates/queryValues in place and
  // overwrites the Q/K/V/P/H1 scratch per block, so backward needs block b's
  // inputs and intermediates. Stacked [maxBlocks, ...] regions; each block's
  // slice is written by the forward when saving is enabled.
  encSavedIn: number; // [maxBlocks, maxTokens, H]    block input (fieldStates before block)
  encSavedFfnIn: number; // [maxBlocks, maxTokens, H]  x1 = fieldStates after attention residual
  encSavedQ: number; // [maxBlocks, maxTokens, H]
  encSavedK: number; // [maxBlocks, maxTokens, H]
  encSavedV: number; // [maxBlocks, maxTokens, H]
  encSavedP: number; // [maxBlocks, maxHeads, maxTokens, maxTokens]
  encSavedH1: number; // [maxBlocks, maxTokens, FFN]  post-ReLU
  mixerSavedIn: number; // [maxBlocks, maxQueries, H]
  mixerSavedFfnIn: number; // [maxBlocks, maxQueries, H]
  mixerSavedQ: number; // [maxBlocks, maxQueries, H]
  mixerSavedK: number; // [maxBlocks, maxRecords, H]
  mixerSavedV: number; // [maxBlocks, maxRecords, H]
  mixerSavedP: number; // [maxBlocks, maxHeads, maxQueries, maxRecords]
  mixerSavedH1: number; // [maxBlocks, maxQueries, FFN]  post-ReLU
  elements: number;
}

function createKrystalForwardArenaLayout(): KrystalForwardArenaLayout {
  let cursor = 0;
  const take = (elements: number) => {
    const offset = cursor;
    cursor += elements;
    return offset;
  };
  const th = KRYSTAL_MAX_TOKENS * KRYSTAL_MAX_H;
  const rh = KRYSTAL_MAX_RECORDS * KRYSTAL_MAX_H;
  const qh = KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_H;
  const tf = KRYSTAL_MAX_TOKENS * KRYSTAL_MAX_FFN;
  return {
    tokenIds: take(KRYSTAL_MAX_TOKENS),
    fieldRoles: take(KRYSTAL_MAX_TOKENS),
    schemaIds: take(KRYSTAL_MAX_RECORDS),
    bandIds: take(KRYSTAL_MAX_RECORDS),
    streamIds: take(KRYSTAL_MAX_RECORDS),
    activeTokens: take(KRYSTAL_MAX_TOKENS),
    recordCompactOffset: take(KRYSTAL_MAX_RECORDS),
    recordCompactCount: take(KRYSTAL_MAX_RECORDS),
    bankIndices: take(KRYSTAL_MAX_RECORDS),
    queryIndices: take(KRYSTAL_MAX_QUERIES),

    fieldStates: take(th),
    encQ: take(th),
    encK: take(th),
    encV: take(th),
    encOut: take(th),
    encH1: take(tf),
    encMask: take(KRYSTAL_MAX_TOKENS * KRYSTAL_MAX_TOKENS),
    encP: take(KRYSTAL_MAX_HEADS * KRYSTAL_MAX_TOKENS * KRYSTAL_MAX_TOKENS),

    bankKeys: take(rh),
    bankValues: take(rh),
    queryKeys: take(qh),
    queryValues: take(qh),

    mixerQ: take(qh),
    mixerK: take(qh),
    mixerV: take(qh),
    mixerH1: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_FFN),
    mixed: take(qh),
    mixerMask: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_RECORDS),
    mixerP: take(KRYSTAL_MAX_HEADS * KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_RECORDS),

    selectorQ: take(qh),
    selectorK: take(rh),
    intentMask: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_RECORDS),
    argMask: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_RECORDS),
    intentP: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_RECORDS),
    intentGather: take(qh),
    intentIndices: take(KRYSTAL_MAX_QUERIES),
    argP: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_RECORDS),
    argGather: take(qh),
    argIndices: take(KRYSTAL_MAX_QUERIES),
    decisionLogits: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_ROUTE_KINDS),

    // Per-block saved activations (composed backward runner).
    encSavedIn: take(KRYSTAL_MAX_BLOCKS * th),
    encSavedFfnIn: take(KRYSTAL_MAX_BLOCKS * th),
    encSavedQ: take(KRYSTAL_MAX_BLOCKS * th),
    encSavedK: take(KRYSTAL_MAX_BLOCKS * th),
    encSavedV: take(KRYSTAL_MAX_BLOCKS * th),
    encSavedP: take(KRYSTAL_MAX_BLOCKS * KRYSTAL_MAX_HEADS * KRYSTAL_MAX_TOKENS * KRYSTAL_MAX_TOKENS),
    encSavedH1: take(KRYSTAL_MAX_BLOCKS * tf),
    mixerSavedIn: take(KRYSTAL_MAX_BLOCKS * qh),
    mixerSavedFfnIn: take(KRYSTAL_MAX_BLOCKS * qh),
    mixerSavedQ: take(KRYSTAL_MAX_BLOCKS * qh),
    mixerSavedK: take(KRYSTAL_MAX_BLOCKS * rh),
    mixerSavedV: take(KRYSTAL_MAX_BLOCKS * rh),
    mixerSavedP: take(KRYSTAL_MAX_BLOCKS * KRYSTAL_MAX_HEADS * KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_RECORDS),
    mixerSavedH1: take(KRYSTAL_MAX_BLOCKS * KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_FFN),
    elements: cursor,
  };
}

/** Krystal forward regions are appended after the training regions. */
export const KRYSTAL_FORWARD_ARENA_BASE = TRAINING_ARENA_ELEMENTS;
export const KRYSTAL_FORWARD_ARENA = createKrystalForwardArenaLayout();
export const KRYSTAL_FORWARD_ARENA_ELEMENTS = KRYSTAL_FORWARD_ARENA.elements;

// --- Krystal backward arena (M3: backward ops for the M2b forward graph) -----
//
// Fixed capacity constants, not ABI limits: shaders read actual dims from
// OpParams. Gradients live here so a later composed KrystalBackward runner
// mirrors KrystalForward's one-submit structure (WEBGPU_BACKWARD_PLAN.md §17
// items 6-8).

export interface KrystalBackwardArenaLayout {
  // Field embedding backward (scatter-add into the six concatenated tables).
  dEmbedding: number; // [embeddingsPageElements]
  dFieldStates: number; // [maxTokens, H]

  // Encoder-block attention backward (cross-capable, mirrored per block).
  dEncQ: number; // [maxTokens, H]
  dEncK: number; // [maxTokens, H]
  dEncV: number; // [maxTokens, H]
  dEncOut: number; // [maxTokens, H]
  dScoresEnc: number; // [maxHeads, maxTokens, maxTokens]
  dHiddenQ: number; // [maxTokens, H] (projection-input grads)
  dHiddenK: number; // [maxTokens, H]
  dHiddenV: number; // [maxTokens, H]
  dWq: number; // [H, H] (block weight grads, reused across blocks)
  dWk: number; // [H, H]
  dWv: number; // [H, H]
  dH1: number; // [maxTokens, FFN] (FFN pre-activation grads)
  dW1: number; // [FFN, H]
  dW2: number; // [H, FFN]

  // Learned-query pooling backward (§17 item 7, second half): upstream
  // gradients of the pooled bank/query keys and values, the per-record dPool
  // partials and the final dPool [2, H]. dPoolPartial rows are disjoint per
  // record, so the backward writes them directly and krystal_pool_dpool
  // reduces them into dPool.
  dBankKeys: number; // [maxRecords, H]
  dBankValues: number; // [maxRecords, H]
  dQueryKeys: number; // [maxQueries, H]
  dQueryValues: number; // [maxQueries, H]
  dPoolPartial: number; // [maxRecords, 2, H]
  dPool: number; // [2, H]

  // Selector backward (§17 item 8): soft-gather gradients of the selector
  // projections and bank values plus the pointer-loss-aware dScore. dScore
  // and dQProj are row-owned (written by the scores pass); dKProj/dValue are
  // written by the companion qkv pass. gold [Q] holds optional pointer-loss
  // targets as u32 payloads (0xffffffff = none).
  dSelectorScores: number; // [maxQueries, maxRecords]
  dSelectorQProj: number; // [maxQueries, H]
  dSelectorKProj: number; // [maxRecords, H]
  dSelectorValue: number; // [maxRecords, H]
  selectorGold: number; // [maxQueries]

  // Composed runner additions (M3 close): selector weight gradients, the
  // optional pointer-loss targets for the argument slot, and the query-side
  // pool-gradient accumulator (the shared dPool is reduced per dispatch, so
  // the second pool needs its own target before residual accumulation).
  dSelectorWq: number; // [H, H]
  dSelectorWk: number; // [H, H]
  argGold: number; // [maxQueries]
  dPool2: number; // [2, H]

  // Typed decision head backward (§17 item 9): the final linear head over the
  // gathered context (query output + intent gather + arg gather, HIN = 3H)
  // producing route-kind logits. dLogits [Q, C] is the upstream cross-entropy
  // gradient; the pass writes dCtx split into its three gathered-context parts
  // plus the head-weight gradient dWh [C, 3H].
  dDecisionLogits: number; // [maxQueries, routeKinds]
  dDecisionQuery: number; // [maxQueries, H]  (d of queryOutput)
  dDecisionIntent: number; // [maxQueries, H]  (d of intentGather)
  dDecisionArg: number; // [maxQueries, H]  (d of argGather)
  dDecisionWh: number; // [routeKinds, 3H]
  elements: number;
}

function createKrystalBackwardArenaLayout(): KrystalBackwardArenaLayout {
  let cursor = 0;
  const take = (elements: number) => {
    const offset = cursor;
    cursor += elements;
    return offset;
  };
  const th = KRYSTAL_MAX_TOKENS * KRYSTAL_MAX_H;
  const rh = KRYSTAL_MAX_RECORDS * KRYSTAL_MAX_H;
  const hh = KRYSTAL_MAX_H * KRYSTAL_MAX_H;
  const tf = KRYSTAL_MAX_TOKENS * KRYSTAL_MAX_FFN;
  const fh = KRYSTAL_MAX_FFN * KRYSTAL_MAX_H;
  const hf = KRYSTAL_MAX_H * KRYSTAL_MAX_FFN;
  const emb =
    (0x1000 + 0x1000 + 0x100 + 11 + 2 + 8) * KRYSTAL_MAX_H; // EMBEDDING_TABLES rows
  return {
    dEmbedding: take(emb),
    dFieldStates: take(th),
    dEncQ: take(th),
    dEncK: take(th),
    dEncV: take(th),
    dEncOut: take(th),
    dScoresEnc: take(KRYSTAL_MAX_HEADS * KRYSTAL_MAX_TOKENS * KRYSTAL_MAX_TOKENS),
    dHiddenQ: take(th),
    dHiddenK: take(th),
    dHiddenV: take(th),
    dWq: take(hh),
    dWk: take(hh),
    dWv: take(hh),
    dH1: take(tf),
    dW1: take(fh),
    dW2: take(hf),
    dBankKeys: take(rh),
    dBankValues: take(rh),
    dQueryKeys: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_H),
    dQueryValues: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_H),
    dPoolPartial: take(KRYSTAL_MAX_RECORDS * 2 * KRYSTAL_MAX_H),
    dPool: take(2 * KRYSTAL_MAX_H),
    dSelectorScores: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_RECORDS),
    dSelectorQProj: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_H),
    dSelectorKProj: take(rh),
    dSelectorValue: take(rh),
    selectorGold: take(KRYSTAL_MAX_QUERIES),
    dSelectorWq: take(hh),
    dSelectorWk: take(hh),
    argGold: take(KRYSTAL_MAX_QUERIES),
    dPool2: take(2 * KRYSTAL_MAX_H),
    dDecisionLogits: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_ROUTE_KINDS),
    dDecisionQuery: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_H),
    dDecisionIntent: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_H),
    dDecisionArg: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_H),
    dDecisionWh: take(KRYSTAL_MAX_ROUTE_KINDS * 3 * KRYSTAL_MAX_H),
    elements: cursor,
  };
}

export const KRYSTAL_BACKWARD_ARENA_BASE =
  TRAINING_ARENA_ELEMENTS + KRYSTAL_FORWARD_ARENA_ELEMENTS;
export const KRYSTAL_BACKWARD_ARENA = createKrystalBackwardArenaLayout();
export const KRYSTAL_BACKWARD_ARENA_ELEMENTS = KRYSTAL_BACKWARD_ARENA.elements;

export const ARENA_ELEMENTS =
  TRAINING_ARENA_ELEMENTS +
  KRYSTAL_FORWARD_ARENA_ELEMENTS +
  KRYSTAL_BACKWARD_ARENA_ELEMENTS;

export const TOKEN_CAPACITY =
  CONTEXT_CAPACITY + MAX_NEW_TOKENS;

// One 256-byte (aligned) OpParams record per dispatch. The Krystal trainStep
// and M1 trainer submit every pass of a step in ONE submit (~150 dispatches
// worst case), so a single MiB is more than enough; the old 128 MiB budget was
// sized for the LFM2 decode path (1024 tokens * ~250 dispatches).
export const OP_PARAM_BUFFER_BYTES = 1024 * 1024;

// Shared elementwise/utility programs. The LFM2 model-inference shaders were
// removed with the legacy runtime; only the programs the Krystal and M1
// training paths dispatch remain (the training/*.wgsl shaders follow below).
export const LFM2_SHADER_NAMES = [
  "matmul_f32",
  "residual_add",
  "arena_copy",
] as const;

export type Lfm2ShaderName = (typeof LFM2_SHADER_NAMES)[number];

/**
 * M1 training shaders, one file per entry point under src/shaders/training/.
 * They share the LFM2 OpParams/arena conventions and link into the same
 * artifact so trainStep reuses the existing pass.run orchestration.
 */
export const TRAINING_SHADER_NAMES = [
  "embedding_f32",
  "zero_f32",
  "cross_entropy_forward_backward",
  "loss_reduce",
  "matmul_backward_input",
  "matmul_backward_weight",
  "embedding_backward",
  "sgd_step",
  // Attention (§17 item 6): forward saves probs, backward splits into the
  // softmax-score gradient and the Q/K/V gradients.
  "attention_forward",
  "attention_backward_scores",
  "attention_backward_qkv",
] as const;

export type TrainingShaderName = (typeof TRAINING_SHADER_NAMES)[number];
export type TrainingPassName = TrainingShaderName;

/**
 * M2b Krystal forward shaders (record/query encoder + mixer): one file per
 * entry point under src/shaders/training/. They share the OpParams/arena
 * conventions and link into the same artifact.
 */
export const KRYSTAL_FORWARD_SHADER_NAMES = [
  "krystal_field_embed",
  "krystal_attention_forward",
  "relu",
  "krystal_pool",
  "krystal_selector",
  "krystal_decision_head",
] as const;

export type KrystalForwardShaderName = (typeof KRYSTAL_FORWARD_SHADER_NAMES)[number];

/**
 * M3 Krystal backward shaders (record encoder + attention backward, §17
 * order). One file per entry point under src/shaders/training/; they share
 * the OpParams/arena conventions and link into the same artifact.
 */
export const KRYSTAL_BACKWARD_SHADER_NAMES = [
  "relu_backward",
  "krystal_attention_backward_scores",
  "krystal_attention_backward_qkv",
  "krystal_field_embed_backward",
  "krystal_pool_backward",
  "krystal_pool_dpool",
  "krystal_selector_backward_scores",
  "krystal_selector_backward_qkv",
  "krystal_decision_head_backward",
] as const;

export type KrystalBackwardShaderName = (typeof KRYSTAL_BACKWARD_SHADER_NAMES)[number];

/**
 * All linked program names. Names above index the .wgsl files on disk; names
 * here index the linked programs (currently 1:1 with the shader files).
 */
export const LFM2_PROGRAM_NAMES = [
  ...LFM2_SHADER_NAMES,
  ...TRAINING_SHADER_NAMES,
  ...KRYSTAL_FORWARD_SHADER_NAMES,
  ...KRYSTAL_BACKWARD_SHADER_NAMES,
] as const;
export type Lfm2ProgramName = (typeof LFM2_PROGRAM_NAMES)[number];

export type Lfm2PassName = Exclude<Lfm2ProgramName, "constraint_mask">;

export type Lfm2Mode = "prefill" | "decode" | "continuation";

/** Host-side shape of the per-dispatch OpParams schema. */
export interface Lfm2OpParams {
  inputOffset?: number;
  outputOffset?: number;
  auxOffset?: number;
  aux2Offset?: number;
  aux3Offset?: number;
  aux4Offset?: number;
  aux5Offset?: number;
  aux6Offset?: number;
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
  u2?: number;
  u3?: number;
  u4?: number;
  u5?: number;
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
  capacities: {
    context: CONTEXT_CAPACITY,
    maxNewTokens: MAX_NEW_TOKENS,
    tokens: TOKEN_CAPACITY,
    arena: ARENA_ELEMENTS,
    training: {
      maxM: TRAINING_MAX_M,
      maxV: TRAINING_MAX_V,
      maxH: TRAINING_MAX_H,
      arena: TRAINING_ARENA_ELEMENTS,
      readback: TRAINING_READBACK_ELEMENTS,
    },
  },
  trainingArena: LFM2_TRAINING_ARENA,
} as const;

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
    matmul_f32: definePass(programs.matmul_f32, "f32", (op) =>
      [required(op.rowCount, "rowCount"), required(op.tokenCount, "tokenCount"), 1]),

    residual_add: definePass(programs.residual_add, "none", (op) =>
      linear(required(op.tokenCount, "tokenCount") * required(op.inputDim, "inputDim"), 256)),

    arena_copy: definePass(programs.arena_copy, "none", (op) =>
      linear(required(op.tokenCount, "tokenCount") * required(op.inputDim, "inputDim"), 256)),

    // --- M1 training passes ---
    // Shader contracts are documented in each training/*.wgsl file; workgroup
    // geometry mirrors the existing per-op conventions (gid-linear for
    // elementwise, one workgroup per row for reductions).
    embedding_f32: definePass(programs.embedding_f32, "f32", (op) =>
      linear(required(op.tokenCount, "tokenCount") * required(op.outputDim, "outputDim"), 256)),

    zero_f32: definePass(programs.zero_f32, "none", (op) =>
      linear(required(op.tokenCount, "tokenCount"), 256)),

    cross_entropy_forward_backward: definePass(programs.cross_entropy_forward_backward, "none", (op) =>
      [required(op.tokenCount, "tokenCount"), 1, 1]),

    loss_reduce: definePass(programs.loss_reduce, "none", () => [1, 1, 1]),

    matmul_backward_input: definePass(programs.matmul_backward_input, "f32", (op) =>
      linear(required(op.tokenCount, "tokenCount") * required(op.outputDim, "outputDim"), 256)),

    matmul_backward_weight: definePass(programs.matmul_backward_weight, "none", (op) =>
      linear(required(op.inputDim, "inputDim") * required(op.outputDim, "outputDim"), 256)),

    embedding_backward: definePass(programs.embedding_backward, "none", (op) =>
      linear(required(op.inputDim, "inputDim") * required(op.outputDim, "outputDim"), 256)),

    sgd_step: definePass(programs.sgd_step, "f32", (op) =>
      linear(required(op.tokenCount, "tokenCount"), 256)),

    attention_forward: definePass(programs.attention_forward, "none", (op) =>
      [required(op.u0, "u0"), required(op.tokenCount, "tokenCount"), 1]),

    attention_backward_scores: definePass(programs.attention_backward_scores, "none", (op) =>
      [required(op.u0, "u0"), required(op.tokenCount, "tokenCount"), 1]),

    attention_backward_qkv: definePass(programs.attention_backward_qkv, "none", (op) =>
      linear(3 * required(op.tokenCount, "tokenCount") * required(op.inputDim, "inputDim"), 256)),

    // --- M2b Krystal forward passes ---
    // Shader contracts are documented in each training/*.wgsl file.
    krystal_field_embed: definePass(programs.krystal_field_embed, "f32", (op) =>
      linear(required(op.tokenCount, "tokenCount") * required(op.inputDim, "inputDim"), 256)),

    krystal_attention_forward: definePass(programs.krystal_attention_forward, "none", (op) =>
      [required(op.u1, "u1"), required(op.tokenCount, "tokenCount"), 1]),

    relu: definePass(programs.relu, "none", (op) =>
      linear(required(op.tokenCount, "tokenCount"), 256)),

    krystal_pool: definePass(programs.krystal_pool, "f32", (op) =>
      [required(op.tokenCount, "tokenCount"), 1, 1]),

    krystal_selector: definePass(programs.krystal_selector, "none", (op) =>
      [1, required(op.tokenCount, "tokenCount"), 1]),

    krystal_decision_head: definePass(programs.krystal_decision_head, "f32", (op) =>
      [1, required(op.tokenCount, "tokenCount"), 1]),

    // --- M3 Krystal backward passes ---
    // Contracts are documented in each training/*.wgsl file; geometry mirrors
    // the corresponding forward pass (gid-linear for elementwise, one
    // workgroup per (head, row) for the attention score gradient).
    relu_backward: definePass(programs.relu_backward, "none", (op) =>
      linear(required(op.tokenCount, "tokenCount"), 256)),

    krystal_attention_backward_scores: definePass(programs.krystal_attention_backward_scores, "none", (op) =>
      [required(op.u1, "u1"), required(op.tokenCount, "tokenCount"), 1]),

    krystal_attention_backward_qkv: definePass(programs.krystal_attention_backward_qkv, "none", (op) =>
      linear(required(op.tokenCount, "tokenCount") * required(op.inputDim, "inputDim") +
        2 * required(op.u0, "u0") * required(op.inputDim, "inputDim"), 256)),

    krystal_field_embed_backward: definePass(programs.krystal_field_embed_backward, "none", (op) =>
      linear(required(op.tokenCount, "tokenCount") * required(op.inputDim, "inputDim"), 256)),

    krystal_pool_backward: definePass(programs.krystal_pool_backward, "f32", (op) =>
      [required(op.tokenCount, "tokenCount"), 1, 1]),

    krystal_pool_dpool: definePass(programs.krystal_pool_dpool, "none", (op) =>
      linear(2 * required(op.inputDim, "inputDim"), 256)),

    krystal_selector_backward_scores: definePass(programs.krystal_selector_backward_scores, "none", (op) =>
      [1, required(op.tokenCount, "tokenCount"), 1]),

    krystal_selector_backward_qkv: definePass(programs.krystal_selector_backward_qkv, "none", (op) =>
      linear(required(op.tokenCount, "tokenCount") * required(op.inputDim, "inputDim") +
        2 * required(op.u0, "u0") * required(op.inputDim, "inputDim"), 256)),

    // Fused dCtx (three gathered-context parts) + dWh, gid-linear split:
    // [0, 3*QH) are the dCtx parts, [3*QH, 3*QH + C*3H) is dWh.
    krystal_decision_head_backward: definePass(programs.krystal_decision_head_backward, "f32", (op) =>
      linear(3 * required(op.tokenCount, "tokenCount") * required(op.inputDim, "inputDim") +
        required(op.outputDim, "outputDim") * 3 * required(op.inputDim, "inputDim"), 256)),
  } satisfies Record<Lfm2PassName, Lfm2PassSpec>;
}
