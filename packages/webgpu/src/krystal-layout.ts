// Pure-TS Krystal layout and dispatch spec.
//
// Everything here is static: constants, the activation arena layout, the
// schema-independent host types and the per-pass dispatch geometry. It is the
// part of krystal-definition.ts that the runtime actually needs, and it is shared
// by the two definition builders:
//
//   - defineKrystal(bundle)            (krystal-definition.ts) — arktype-backed DSL,
//                                     used only by the AOT build/validate scripts
//   - defineKrystalFromArtifact()      (krystal-artifact.ts)   — handle creation from
//                                     krystal.artifact.generated.ts, used at runtime
//
// The runtime entry (krystal.ts) imports only this module and krystal-artifact.ts,
// so the DSL, `$` and arktype never enter a scriptc-compiled graph.
import type { AnyComputeHandle } from "@sandblaster/core";
import { BRAIN_LIMITS } from "../../schema/src/krystal-engine-schema.ts";
import { EMBEDDING_TABLES } from "../../krystal/src/forward/model.ts";

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

export interface KrystalTrainingArenaLayout {
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

function createTrainingArenaLayout(): KrystalTrainingArenaLayout {
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

/** The training regions lead the shared arena. */
// The legacy LFM2 activation regions were removed with the legacy runtime; the
// training regions now lead the shared arena.
export const TRAINING_ARENA_BASE = 0;
export const KRYSTAL_TRAINING_ARENA = createTrainingArenaLayout();
export const TRAINING_ARENA_ELEMENTS = KRYSTAL_TRAINING_ARENA.elements;

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

// PROCESSING capacity, deliberately NOT the frame's slot geometry.
//
// The arena preallocates encoder attention scores as [heads, T, T], so token
// capacity costs quadratic memory while record capacity costs linear. The
// frame may therefore hold many more slots than the model ever processes at
// once: a band exists to give real records somewhere to live, and only the
// OCCUPIED ones reach the encoder. Sizing this from BRAIN_LIMITS.frameTokens
// would make every band enlargement quadruple the arena — at 288 slots that
// is a ~400 MB allocation, past the default 256 MB maxBufferSize, which
// surfaces as silent garbage rather than an allocation error.
//
// A frame whose occupancy exceeds this budget fails loudly in prepare() with
// "active tokens exceed capacity"; the kaleidoscope noise budget in
// training/policy.ts is what keeps occupancy well inside it.
export const KRYSTAL_MAX_TOKENS = 1536; // ≈192 occupied records of 8 tokens
export const KRYSTAL_MAX_RECORDS = BRAIN_LIMITS.frameRecordSlots;

/**
 * Storage for the SoA frame arrays that are addressed by FRAME token index
 * (slot * recordWidth + local), not by compacted active-token index. They must
 * span the whole frame even when only part of it is occupied, so they follow
 * the ABI geometry rather than the processing budget above.
 */
export const KRYSTAL_FRAME_TOKENS = BRAIN_LIMITS.frameTokens;
export const KRYSTAL_MAX_H = 128; // first profile hidden size (answer 9)
export const KRYSTAL_MAX_FFN = 384; // first profile FFN size (answer 9)
export const KRYSTAL_MAX_HEADS = 4; // first profile full attention heads (answer 11)
export const KRYSTAL_MAX_QUERIES = 8; // maxQueries
export const KRYSTAL_MAX_ROUTE_KINDS = 8; // typed decision-head classes (capacity, not ABI)
export const KRYSTAL_MAX_BLOCKS = 2; // shared capacity for encoder + mixer block stacks

export interface KrystalForwardArenaLayout {
  // SoA frame inputs (u32 payloads bitcast in shaders) + host-compiled lists.
  // tokenIds/fieldRoles are indexed by FRAME token index, so they span the
  // whole frame; every [maxTokens, …] array below is compacted-active indexed.
  tokenIds: number; // [frameTokens]
  fieldRoles: number; // [frameTokens]
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
  mixerK: number; // [maxRecords, H]
  mixerV: number; // [maxRecords, H]
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
  /**
   * A [maxQueries, H] block of zeros, and it stays zero.
   *
   * The "what is on offer here" context is the mean bank value over the
   * records a question is ALLOWED to choose — and a selector whose query
   * projection is zero computes exactly that: every score is 0 + mask, so the
   * softmax is uniform over the open positions and the gather is their mean.
   * An entirely blocked row falls to the shader's all-blocked path and gathers
   * zero, which is what the CPU oracle does with an empty allowed set.
   *
   * So the third context block needs no kernel of its own; it needs a region
   * that is reliably zero.
   */
  zeroQuery: number; // [maxQueries, H]
  /**
   * The mean bank value over what a question is allowed to choose, and the
   * uniform distribution it came from. Only the value head reads them: the
   * critic must not be conditioned on WHICH action was drawn (it is the
   * baseline that draw is measured against), so its third context block is a
   * state feature — "what is on offer here" — rather than a soft gather under
   * the chosen slot.
   */
  availableGather: number; // [maxQueries, H]
  availableP: number; // [maxQueries, maxRecords]
  /** Value-head prediction [maxQueries] (a decision head with one class). */
  valuePrediction: number; // [maxQueries]

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
    tokenIds: take(KRYSTAL_FRAME_TOKENS),
    fieldRoles: take(KRYSTAL_FRAME_TOKENS),
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
    mixerK: take(rh),
    mixerV: take(rh),
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
    zeroQuery: take(qh),
    availableGather: take(qh),
    availableP: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_RECORDS),
    valuePrediction: take(KRYSTAL_MAX_QUERIES),

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

  // Value head backward. Structurally a decision head with a single class, so
  // it reuses that pass exactly; only the loss differs — squared error against
  // an observed number rather than cross-entropy against a label. Its three
  // context gradients land in their own regions and are then added into the
  // decision head's, because the two heads read the SAME context and their
  // gradients sum there (which is how the value signal shapes the trunk and
  // not just its own head).
  dValuePrediction: number; // [maxQueries]
  valueLossRows: number; // [maxQueries] (per-row 0.5*err^2, reduced for telemetry)
  dValueQuery: number; // [maxQueries, H]
  dValueIntent: number; // [maxQueries, H]
  dValueArg: number; // [maxQueries, H]
  dValueWv: number; // [1, 3H]
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
  // Summed from EMBEDDING_TABLES, not restated. The literal that used to stand
  // here (0x1000 + 0x1000 + 0x100 + 11 + 2 + 8) was a copy of those row counts
  // and had already drifted: it still assumed 12-bit token/field tables and 11
  // bands. Nothing related the two, so `dEmbedding` was simply reserved too
  // small and the scatter-add wrote past it into the next region.
  const emb = EMBEDDING_TABLES.reduce((sum, table) => sum + table.rows, 0) * KRYSTAL_MAX_H;
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
    dValuePrediction: take(KRYSTAL_MAX_QUERIES),
    valueLossRows: take(KRYSTAL_MAX_QUERIES),
    dValueQuery: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_H),
    dValueIntent: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_H),
    dValueArg: take(KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_H),
    dValueWv: take(3 * KRYSTAL_MAX_H),
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
// sized for the removed LFM2 decode path (1024 tokens * ~250 dispatches).
export const OP_PARAM_BUFFER_BYTES = 1024 * 1024;

// Shared elementwise/utility programs. The LFM2 model-inference shaders were
// removed with the legacy runtime; only the programs the Krystal and M1
// training paths dispatch remain (the training/*.wgsl shaders follow below).
export const KRYSTAL_SHADER_NAMES = [
  "matmul_f32",
  "residual_add",
  "arena_copy",
] as const;

export type KrystalShaderName = (typeof KRYSTAL_SHADER_NAMES)[number];

/**
 * M1 training shaders, one file per entry point under src/shaders/training/.
 * They share the Krystal OpParams/arena conventions and link into the same
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
  "krystal_field_embed_sgd",
  "krystal_pool_backward",
  "krystal_pool_dpool",
  "krystal_selector_backward_scores",
  "krystal_selector_backward_qkv",
  "krystal_decision_head_backward",
  "krystal_value_head_loss",
] as const;

export type KrystalBackwardShaderName = (typeof KRYSTAL_BACKWARD_SHADER_NAMES)[number];

/**
 * All linked program names. Names above index the .wgsl files on disk; names
 * here index the linked programs (currently 1:1 with the shader files).
 */
export const KRYSTAL_PROGRAM_NAMES = [
  ...KRYSTAL_SHADER_NAMES,
  ...TRAINING_SHADER_NAMES,
  ...KRYSTAL_FORWARD_SHADER_NAMES,
  ...KRYSTAL_BACKWARD_SHADER_NAMES,
] as const;
export type KrystalProgramName = (typeof KRYSTAL_PROGRAM_NAMES)[number];

export type KrystalPassName = Exclude<KrystalProgramName, "constraint_mask">;

export type KrystalMode = "prefill" | "decode" | "continuation";

/** Host-side shape of the per-dispatch OpParams schema. */
export interface KrystalOpParams {
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
  mode?: KrystalMode;
  f0?: number;
  f1?: number;
  u0?: number;
  u1?: number;
  u2?: number;
  u3?: number;
  u4?: number;
  u5?: number;
}

export type KrystalWorkgroups = readonly [x: number, y: number, z: number];
export type KrystalWeightBinding = "none" | "raw" | "f32";

/**
 * A pass is the stable execution-level description of one shader entry point.
 * It owns dispatch geometry; the runtime only supplies OpParams and, where
 * required, the concrete tensor page bound to the weight resource.
 */
export interface KrystalPassSpec {
  readonly program: AnyComputeHandle;
  readonly weight: KrystalWeightBinding;
  workgroups(op: Readonly<KrystalOpParams>): KrystalWorkgroups;
}

/**
 * Definition-level plain fields shared by both builders (krystal-definition.ts
 * DSL and krystal-artifact.ts). Keeping them in one place means a capacity or
 * constraint change cannot silently diverge between the two paths.
 */
export const KRYSTAL_DEFINITION_PLAIN = {
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
  trainingArena: KRYSTAL_TRAINING_ARENA,
} as const;

function required(value: number | undefined, field: keyof KrystalOpParams): number {
  if (value === undefined) throw new Error(`Krystal pass requires op.${field}`);
  return value;
}

function linear(value: number, workgroupSize: number): KrystalWorkgroups {
  return [Math.ceil(value / workgroupSize), 1, 1];
}

function definePass(
  program: AnyComputeHandle,
  weight: KrystalWeightBinding,
  workgroups: KrystalPassSpec["workgroups"],
): KrystalPassSpec {
  return { program, weight, workgroups };
}

/**
 * Per-shader dispatch rules, parameterized by the program handles so the same
 * geometry drives both the DSL-built and the artifact-built definition.
 */
export function defineKrystalPasses(
  programs: Record<KrystalProgramName, AnyComputeHandle>,
): Record<KrystalPassName, KrystalPassSpec> {
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

    krystal_field_embed_sgd: definePass(programs.krystal_field_embed_sgd, "f32", (op) =>
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

    krystal_value_head_loss: definePass(programs.krystal_value_head_loss, "none", (op) =>
      linear(required(op.tokenCount, "tokenCount"), 256)),
  } satisfies Record<KrystalPassName, KrystalPassSpec>;
}
