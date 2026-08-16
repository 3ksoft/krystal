import {} from "./env";
import { scope } from "arktype";
import { wgsl } from "@schema-pop/schema";

/**
 * Krystal brain-engine contracts — architecture v2, first concrete draft.
 *
 * Scope of this file:
 *   - logical token ABI and vocabulary manifests,
 *   - typed record format,
 *   - stable BrainFrame geometry,
 *   - runtime reference sidecars,
 *   - homeostatic queries, working memory and focus,
 *   - ActionIntent catalog, soft-gather results and concurrent IntentSet,
 *   - generic tutorial orchestration,
 *   - engine/runtime state and telemetry.
 *
 * Explicitly out of scope:
 *   - world/entity/component schemas,
 *   - sensory simulation and feature extraction,
 *   - body physics and motor lowering,
 *   - comfort dynamics,
 *   - exact world mutation/history implementation,
 *   - the ALU beyond opaque route/effect tokens,
 *   - neural weights and model-specific hidden tensors.
 *
 * There are deliberately two layouts, but only the first is normative here.
 *
 * 1. TokenLayoutPlan (defined below)
 *
 *    The model sees 128 fixed record slots, eight logical token positions per
 *    record, for a 1024-token BrainFrame. Band and selected structural record
 *    positions are stable. Dynamic records may be shuffled only inside their
 *    band's fixed slot range. Record-local token positions are interpreted by
 *    schema metadata rather than by a universal flat grammar.
 *
 * 2. BinaryLayoutPlan (intentionally not frozen yet)
 *
 *    The GPU implementation will probably lower the logical AoS records into
 *    SoA buffers resembling:
 *
 *      tokenIds[recordSlot][localToken]
 *      schemaIds[recordSlot]
 *      bandIds[recordSlot]
 *      fieldRoles[recordSlot][localToken]
 *      runtimeRefs[recordSlot][localReference]
 *      recordFlags[recordSlot]
 *
 *    Token IDs may initially remain u32-aligned and later pack as two u16 per
 *    word. Hidden record keys/values and model-specific f16 tensors belong to
 *    that second plan. Logical token positions, record slots and handles must
 *    not change merely because physical packing changes.
 */

// ---------------------------------------------------------------------------
// Frozen/provisional constants
// ---------------------------------------------------------------------------

export const KRYSTAL_ABI = {
  /** Frozen semantic/token ABI from KRYSTAL_ABI_V0.md. */
  tokenAbiVersion: 0,
  /** Neural/frame architecture described by KRYSTAL_BRAIN_ARCHITECTURE_V2. */
  architectureVersion: 2,
  /** First concrete geometry in this file; independent from both above. */
  frameLayoutVersion: 1,
  tokenBits: 12,
  vocabSize: 0x1000,
  tokenClassBits: 4,
  tokenClassSize: 0x100,
  dynamicRefStart: 0xe00,
  dynamicRefEnd: 0xeff,
  dynamicRefCount: 0x100,
} as const;

/**
 * Logical limits. The first binary plan may use wider u32 storage even when a
 * value has a smaller semantic range. A compiler must reject overflows before
 * frame materialization.
 */
/**
 * SoA BinaryLayoutPlan version (M2a freeze).
 *
 * This is the version of the GPU-facing SoA lowering of `BrainFrame` (the
 * `BrainFrameGpu` buffers below). It is independent from the logical
 * TokenLayoutPlan (`frameLayoutVersion`) because physical packing may change
 * without moving logical record slots or token positions. Bump this version
 * and the plan hash whenever a buffer's element type, count or meaning
 * changes; a compiled frame then fails the compatibility check instead of
 * being misread by an older runtime.
 */
export const BINARY_LAYOUT_PLAN_VERSION = 1;

export const BRAIN_LIMITS = {
  recordWidth: 8,
  frameRecordSlots: 128,
  frameTokens: 128 * 8,
  frameBands: 11,
  fixedRecordBindings: 32,

  maxRecordSchemas: 0x100,
  maxRecordFields: 0x800,
  // A record can legally contain a reference in every logical token position.
  maxReferencesPerRecord: 8,

  maxActionIntents: 0x100,
  maxActionArguments: 4,
  maxIntentProposals: 8,
  maxActiveIntents: 16,

  maxQueries: 8,
  maxMemorySlots: 32,

  maxTutorialBeats: 0x100,
  maxTutorialProbes: 0x40,
  maxTutorialTokens: 0x1000,
} as const;

export const KRYSTAL_TOKEN_RANGES = {
  system: [0x000, 0x0ff],
  structure: [0x100, 0x1ff],
  operation: [0x200, 0x2ff],
  object: [0x300, 0x3ff],
  property: [0x400, 0x4ff],
  quantity: [0x500, 0x5ff],
  action: [0x600, 0x6ff],
  reference: [0x700, 0x7ff],
  relation: [0x800, 0x8ff],
  logic: [0x900, 0x9ff],
  domain: [0xa00, 0xdff],
  context: [0xe00, 0xeff],
  experimental: [0xf00, 0xfff],
} as const;

/**
 * Logical frame geometry. Every band owns whole record slots; every record is
 * exactly eight logical token positions. Consequently tokenOffset is always
 * recordOffset * recordWidth and the complete layout is exactly 1024 tokens.
 *
 * Placement semantics:
 *   fixed              exact record roles stay at exact indices,
 *   shuffled_records   records may permute only inside this range,
 *   stable_resident    a record keeps its slot until eviction/replacement.
 */
export const BRAIN_FRAME_BANDS = [
  { kind: "system", recordOffset: 0, recordCapacity: 4, tokenOffset: 0, tokenCapacity: 32, placement: "fixed", overflow: "error" },
  { kind: "homeostasis", recordOffset: 4, recordCapacity: 8, tokenOffset: 32, tokenCapacity: 64, placement: "fixed", overflow: "error" },
  { kind: "body", recordOffset: 12, recordCapacity: 12, tokenOffset: 96, tokenCapacity: 96, placement: "fixed", overflow: "error" },
  { kind: "vision", recordOffset: 24, recordCapacity: 32, tokenOffset: 192, tokenCapacity: 256, placement: "shuffled_records", overflow: "truncate_low_salience" },
  { kind: "audio", recordOffset: 56, recordCapacity: 12, tokenOffset: 448, tokenCapacity: 96, placement: "shuffled_records", overflow: "truncate_low_salience" },
  { kind: "olfaction", recordOffset: 68, recordCapacity: 6, tokenOffset: 544, tokenCapacity: 48, placement: "shuffled_records", overflow: "truncate_low_salience" },
  { kind: "taste", recordOffset: 74, recordCapacity: 4, tokenOffset: 592, tokenCapacity: 32, placement: "shuffled_records", overflow: "truncate_low_salience" },
  { kind: "touch", recordOffset: 78, recordCapacity: 12, tokenOffset: 624, tokenCapacity: 96, placement: "shuffled_records", overflow: "truncate_low_salience" },
  { kind: "memory", recordOffset: 90, recordCapacity: 26, tokenOffset: 720, tokenCapacity: 208, placement: "stable_resident", overflow: "evict_low_priority" },
  { kind: "focus", recordOffset: 116, recordCapacity: 6, tokenOffset: 928, tokenCapacity: 48, placement: "stable_resident", overflow: "drop_oldest" },
  { kind: "query", recordOffset: 122, recordCapacity: 6, tokenOffset: 976, tokenCapacity: 48, placement: "stable_resident", overflow: "drop_oldest" },
] as const;

/**
 * Suggested structural record bindings for the first morphology/profile.
 * They are compiler defaults, not a claim that every future body has exactly
 * these parts. A missing part keeps its record and reports unavailable/missing;
 * later records never shift left to fill the hole.
 */
export const BRAIN_FIXED_RECORDS = {
  frame: 0,
  clock: 1,
  actor: 2,
  runtimeFeedback: 3,

  homeostasisSummary: 4,
  primaryNeed: 5,

  self: 12,
  head: 13,
  leftHand: 14,
  rightHand: 15,
  mouth: 16,
  locomotion: 17,
  torsoBalance: 18,
  internalFocus: 19,
  vocalizer: 20,

  perceptualFocus: 116,
  thoughtFocus: 117,
  speechTopic: 118,

  primaryQuery: 122,
} as const;

export const INVALID_U32 = 0xffff_ffff;

export const TOKEN_FLAGS = {
  padding: 1 << 0,
  structural: 1 << 1,
  reference: 1 << 2,
  query: 1 << 3,
  candidate: 1 << 4,
  truncated: 1 << 5,
  hiddenFromModel: 1 << 6,
  creatorAuthored: 1 << 7,
} as const;

export const RECORD_FLAGS = {
  occupied: 1 << 0,
  fixed: 1 << 1,
  shuffled: 1 << 2,
  remembered: 1 << 3,
  focused: 1 << 4,
  query: 1 << 5,
  stale: 1 << 6,
  truncated: 1 << 7,
  candidate: 1 << 8,
  creatorAuthored: 1 << 9,
  unavailable: 1 << 10,
} as const;

export const REFERENCE_FLAGS = {
  primary: 1 << 0,
  historical: 1 << 1,
  live: 1 << 2,
  stale: 1 << 3,
  candidate: 1 << 4,
} as const;

export const ACTION_INTENT_FLAGS = {
  durative: 1 << 0,
  internal: 1 << 1,
  perceptual: 1 << 2,
  communicative: 1 << 3,
  motor: 1 << 4,
  mayFail: 1 << 5,
  allowsOverlap: 1 << 6,
  creatorOnly: 1 << 7,
} as const;

export const MEMORY_FLAGS = {
  pinned: 1 << 0,
  autobiographical: 1 << 1,
  goal: 1 << 2,
  intent: 1 << 3,
  staleObservation: 1 << 4,
  eligibleForEviction: 1 << 5,
} as const;

// Guard the hand-written first layout. This is also expected to become a
// schema-pop analyzer diagnostic once TokenLayoutPlan is a first-class plan.
let expectedRecordOffset = 0;
let expectedTokenOffset = 0;
for (const band of BRAIN_FRAME_BANDS) {
  if (
    band.recordOffset !== expectedRecordOffset ||
    band.tokenOffset !== expectedTokenOffset ||
    band.tokenCapacity !== band.recordCapacity * BRAIN_LIMITS.recordWidth
  ) {
    throw new Error(`Invalid BrainFrame band geometry: ${band.kind}`);
  }
  expectedRecordOffset += band.recordCapacity;
  expectedTokenOffset += band.tokenCapacity;
}
if (
  BRAIN_FRAME_BANDS.length !== BRAIN_LIMITS.frameBands ||
  expectedRecordOffset !== BRAIN_LIMITS.frameRecordSlots ||
  expectedTokenOffset !== BRAIN_LIMITS.frameTokens
) {
  throw new Error("Invalid BrainFrame aggregate geometry");
}

export const schema = scope({
  ...wgsl.import(),

  // -----------------------------------------------------------------------
  // Logical IDs, enums and exact runtime references
  // -----------------------------------------------------------------------

  /** Lower 12 bits are the normative Krystal token ID. Stored as u32 in v0. */
  KrystalTokenId: "u32",
  SchemaId: "u32",
  FieldId: "u32",
  IntentId: "u32",
  RecordIndex: "u32",
  LocalTokenIndex: "u32",
  BandMask: "u32",

  KrystalTokenClass:
    "'system' | 'structure' | 'operation' | 'object' | 'property' | 'quantity' | 'action' | 'reference' | 'relation' | 'logic' | 'domain' | 'context' | 'experimental'",

  BrainBandKind:
    "'system' | 'homeostasis' | 'body' | 'vision' | 'audio' | 'olfaction' | 'taste' | 'touch' | 'memory' | 'focus' | 'query'",

  BandPlacementPolicy: "'fixed' | 'shuffled_records' | 'stable_resident'",
  BandOverflowPolicy: "'error' | 'truncate_low_salience' | 'evict_low_priority' | 'drop_oldest'",

  RecordSource:
    "'runtime' | 'sensor' | 'body' | 'homeostasis' | 'memory' | 'focus' | 'query' | 'creator' | 'intent_feedback'",

  RuntimeRefKind:
    "'none' | 'entity' | 'value' | 'memory' | 'event' | 'goal' | 'intent' | 'snapshot' | 'controller' | 'topic'",

  RuntimeRefStatus: "'invalid' | 'live' | 'stale' | 'historical' | 'destroyed'",

  /**
   * Exact runtime identity sidecar.
   *
   * tokenId is the brain-visible 0xExx local symbol. generation prevents ABA
   * reuse while history can still address an older binding epoch. kind and
   * status are exact runtime metadata; only compiler-selected projections are
   * exposed to the model as tokens/features.
   */
  RuntimeRefHandle: {
    tokenId: "KrystalTokenId",
    generation: "u32",
    kind: "RuntimeRefKind",
    status: "RuntimeRefStatus",
  },

  // -----------------------------------------------------------------------
  // Vocabulary and record-schema compiler artifacts
  // -----------------------------------------------------------------------

  VocabManifestHeader: {
    tokenAbiVersion: `u32 = ${KRYSTAL_ABI.tokenAbiVersion}`,
    manifestVersion: "u32 = 0",
    vocabSize: `u32 = ${KRYSTAL_ABI.vocabSize}`,
    activeTokenCount: "u32 = 0",
    manifestHashLo: "u32 = 0",
    manifestHashHi: "u32 = 0",
    reserved0: "u32 = 0",
    reserved1: "u32 = 0",
  },

  /** Device/compiler representation; human symbol strings stay host-only. */
  VocabManifestEntry: {
    tokenId: "KrystalTokenId",
    tokenClass: "KrystalTokenClass",
    flags: "u32 = 0",
    arity: "u32 = 0",
    semanticTypeToken: "KrystalTokenId = 0",
    inverseToken: "KrystalTokenId = 0",
    reserved0: "u32 = 0",
    reserved1: "u32 = 0",
  },

  BrainValueKind:
    "'none' | 'token' | 'context_ref' | 'record_ref' | 'boolean_class' | 'scalar_band' | 'quantity_projection' | 'opaque_payload'",

  RecordSchemaManifestHeader: {
    version: "u32 = 0",
    schemaCount: "u32 = 0",
    fieldCount: "u32 = 0",
    maxRecordTokens: `u32 = ${BRAIN_LIMITS.recordWidth}`,
    schemaHashLo: "u32 = 0",
    schemaHashHi: "u32 = 0",
    reserved0: "u32 = 0",
    reserved1: "u32 = 0",
  },

  /**
   * One learned record family. `familyToken` may be embedded even when it is
   * not serialized inside the eight-token payload; schemaId remains exact
   * structural metadata supplied by the compiler.
   */
  RecordSchemaEntry: {
    schemaId: "SchemaId",
    familyToken: "KrystalTokenId",
    defaultBand: "BrainBandKind",
    tokenCount: "u32",
    fieldOffset: "u32",
    fieldCount: "u32",
    flags: "u32 = 0",
    reserved0: "u32 = 0",
  },

  /**
   * Field roles are schema metadata. They need not consume token positions.
   * This is the explicit structural signal that the old flat tape lacked.
   */
  RecordFieldEntry: {
    schemaId: "SchemaId",
    fieldId: "FieldId",
    localTokenIndex: "LocalTokenIndex",
    tokenWidth: "u32 = 1",
    roleToken: "KrystalTokenId",
    valueKind: "BrainValueKind",
    acceptedSchemaId: "SchemaId = 0",
    allowedBandMask: "BandMask = 0",
    flags: "u32 = 0",
    reserved0: "u32 = 0",
  },

  // Host/compiler authoring forms. These are deliberately excluded from the
  // portable binary layout build because names/docs and variable arrays are
  // build-time data, not GPU state.
  TokenAuthoringSpec: {
    id: "number",
    symbol: "string",
    tokenClass: "KrystalTokenClass",
    "semanticType?": "string",
    "arity?": "number",
    "doc?": "string",
  },

  RecordFieldAuthoringSpec: {
    name: "string",
    localTokenIndex: "number",
    roleToken: "number",
    valueKind: "BrainValueKind",
    "acceptedSchema?": "string",
    "allowedBands?": "BrainBandKind[]",
    "required?": "boolean",
    "exactRuntime?": "boolean",
    "doc?": "string",
  },

  RecordSchemaAuthoringSpec: {
    name: "string",
    familyToken: "number",
    defaultBand: "BrainBandKind",
    fields: "RecordFieldAuthoringSpec[]",
    "doc?": "string",
  },

  /**
   * Illustrative lowering; these are examples, not frozen domain schemas:
   *
   *   VisionObject:
   *     [APPLE, #17, RED, ROUND, SHINY, SMALL, PAD, PAD]
   *
   *   HomeostasisQuery:
   *     [FEEL_BAD, NEED, SATIATED, PAD, PAD, PAD, PAD, PAD]
   *
   *   MemoryObject:
   *     [REMEMBER, #43, STICK, LAST_ACTION, HOLD, PAD, PAD, PAD]
   *
   * APPLE/#17/etc. all consume ordinary logical token positions. In contrast,
   * schemaId, fieldId, field roles and exact reference generations are sidecar
   * structure and do not consume the eight-token payload.
   */

  // -----------------------------------------------------------------------
  // BrainFrame geometry and logical record slots
  // -----------------------------------------------------------------------

  BrainBandLayout: {
    kind: "BrainBandKind",
    recordOffset: "u32",
    recordCapacity: "u32",
    tokenOffset: "u32",
    tokenCapacity: "u32",
    placement: "BandPlacementPolicy",
    overflow: "BandOverflowPolicy",
    flags: "u32 = 0",
    reserved0: "u32 = 0",
  },

  FixedRecordBinding: {
    roleToken: "KrystalTokenId",
    recordIndex: "RecordIndex",
    expectedSchemaId: "SchemaId",
    flags: "u32 = 0",
  },

  BrainFrameLayoutHeader: {
    tokenAbiVersion: `u32 = ${KRYSTAL_ABI.tokenAbiVersion}`,
    architectureVersion: `u32 = ${KRYSTAL_ABI.architectureVersion}`,
    layoutVersion: `u32 = ${KRYSTAL_ABI.frameLayoutVersion}`,
    recordWidth: `u32 = ${BRAIN_LIMITS.recordWidth}`,
    recordSlots: `u32 = ${BRAIN_LIMITS.frameRecordSlots}`,
    tokenCapacity: `u32 = ${BRAIN_LIMITS.frameTokens}`,
    bandCount: `u32 = ${BRAIN_LIMITS.frameBands}`,
    fixedRecordCount: "u32 = 0",
    flags: "u32 = 0",
    layoutHashLo: "u32 = 0",
    layoutHashHi: "u32 = 0",
  },

  BrainFrameLayout: {
    header: "BrainFrameLayoutHeader",
    bands: `BrainBandLayout[] == ${BRAIN_LIMITS.frameBands}`,
    fixedRecords: `FixedRecordBinding[] == ${BRAIN_LIMITS.fixedRecordBindings}`,
  },

  /** Metadata added to the corresponding token embedding. */
  BrainTokenMeta: {
    fieldId: "FieldId",
    roleToken: "KrystalTokenId",
    flags: "u32 = 0",
    referenceBinding: `u32 = ${INVALID_U32}`,
  },

  /**
   * Maps one brain-visible reference token occurrence to exact runtime
   * identity. Equality attention bias is derived from equal handle epochs, not
   * from permanent semantics of token slot #17.
   */
  BrainReferenceBinding: {
    localTokenIndex: `LocalTokenIndex = ${INVALID_U32}`,
    fieldId: "FieldId = 0",
    flags: "u32 = 0",
    reserved0: "u32 = 0",
    handle: "RuntimeRefHandle",
  },

  BrainRecordHeader: {
    schemaId: "SchemaId",
    band: "BrainBandKind",
    source: "RecordSource",
    flags: "u32 = 0",

    tokenCount: "u32 = 0",
    referenceCount: "u32 = 0",
    observedAt: "u32 = 0",
    revision: "u32 = 0",

    primaryReference: `u32 = ${INVALID_U32}`,
    continuationRecord: `RecordIndex = ${INVALID_U32}`,
    salience: "f32 = 0",
    freshness: "f32 = 0",
  },

  /**
   * Canonical logical record. Record boundaries and field roles live in exact
   * metadata, so BEGIN/END and field-name tokens are not required merely to
   * rediscover structure. Every unused token position is PAD and masked.
   */
  BrainRecordSlot: {
    header: "BrainRecordHeader",
    tokens: `KrystalTokenId[] == ${BRAIN_LIMITS.recordWidth}`,
    tokenMeta: `BrainTokenMeta[] == ${BRAIN_LIMITS.recordWidth}`,
    references: `BrainReferenceBinding[] == ${BRAIN_LIMITS.maxReferencesPerRecord}`,
  },

  BrainBandState: {
    kind: "BrainBandKind",
    activeRecords: "u32 = 0",
    activeTokens: "u32 = 0",
    overflowRecords: "u32 = 0",
    truncatedRecords: "u32 = 0",
    revision: "u32 = 0",
    flags: "u32 = 0",
    reserved0: "u32 = 0",
  },

  BrainFrameHeader: {
    tokenAbiVersion: `u32 = ${KRYSTAL_ABI.tokenAbiVersion}`,
    architectureVersion: `u32 = ${KRYSTAL_ABI.architectureVersion}`,
    layoutVersion: `u32 = ${KRYSTAL_ABI.frameLayoutVersion}`,
    tick: "u32 = 0",
    snapshot: "u32 = 0",

    activeRecordCount: "u32 = 0",
    activeTokenCount: "u32 = 0",
    activeQueryRecord: `RecordIndex = ${INVALID_U32}`,
    actorRecord: `RecordIndex = ${BRAIN_FIXED_RECORDS.actor}`,

    frameRevision: "u32 = 0",
    memoryRevision: "u32 = 0",
    intentRevision: "u32 = 0",
    flags: "u32 = 0",
  },

  /**
   * Logical model input. The compiler may lower this AoS aggregate to the SoA
   * BinaryLayoutPlan described at the top of the file. Until that second plan
   * is frozen, do not treat this aggregate as the final WGSL storage buffer;
   * it is the canonical schema from which `BrainFrameGpu` should be generated.
   */
  BrainFrame: {
    header: "BrainFrameHeader",
    bands: `BrainBandState[] == ${BRAIN_LIMITS.frameBands}`,
    records: `BrainRecordSlot[] == ${BRAIN_LIMITS.frameRecordSlots}`,
  },

  // -----------------------------------------------------------------------
  // SoA BinaryLayoutPlan (M2a freeze) — the GPU-facing frame lowering
  // -----------------------------------------------------------------------

  /**
   * Versioned plan metadata. `planVersion` identifies this SoA packing;
   * `planHashLo/Hi` cover the buffer descriptor list below. A compiled frame
   * carries the header and the runtime checks it against its own compiled
   * plan before reading any buffer.
   */
  BinaryLayoutPlanHeader: {
    planVersion: `u32 = ${BINARY_LAYOUT_PLAN_VERSION}`,
    layoutVersion: `u32 = ${KRYSTAL_ABI.frameLayoutVersion}`,
    bufferCount: "u32 = 0",

    recordSlots: `u32 = ${BRAIN_LIMITS.frameRecordSlots}`,
    recordWidth: `u32 = ${BRAIN_LIMITS.recordWidth}`,
    tokenCapacity: `u32 = ${BRAIN_LIMITS.frameTokens}`,
    maxReferencesPerRecord: `u32 = ${BRAIN_LIMITS.maxReferencesPerRecord}`,

    planHashLo: "u32 = 0",
    planHashHi: "u32 = 0",
    flags: "u32 = 0",
    reserved0: "u32 = 0",
  },

  /**
   * One SoA buffer in the plan. `bufferId` is a stable enum per plan version
   * (see BINARY_LAYOUT_BUFFER_IDS). `elementCount` is the number of elements
   * in the buffer; all buffers are u32 in v1, so byteSize = 4 * elementCount.
   */
  BinaryLayoutBufferDesc: {
    bufferId: "u32",
    elementCount: "u32",
    byteSize: "u32",
    flags: "u32 = 0",
  },

  BinaryLayoutPlan: {
    header: "BinaryLayoutPlanHeader",
    buffers: "BinaryLayoutBufferDesc[]",
  },

  /**
   * The GPU-facing SoA lowering of `BrainFrame`.
   *
   * tokenIds/fieldRoles are indexed `[recordSlot * recordWidth + localToken]`;
   * schemaIds/bandIds/recordFlags are indexed `[recordSlot]`; runtimeRefs is
   * indexed `[recordSlot * maxReferencesPerRecord + localReference]`;
   * activeRecordIndices lists occupied record slots in ascending order up to
   * activeRecordCount. Every unused token position is PAD and masked.
   */
  BrainFrameGpu: {
    header: "BinaryLayoutPlanHeader",
    tokenIds: `u32[] == ${BRAIN_LIMITS.frameTokens}`,
    fieldRoles: `u32[] == ${BRAIN_LIMITS.frameTokens}`,
    schemaIds: `u32[] == ${BRAIN_LIMITS.frameRecordSlots}`,
    bandIds: `u32[] == ${BRAIN_LIMITS.frameRecordSlots}`,
    runtimeRefs: `u32[] == ${BRAIN_LIMITS.frameRecordSlots * BRAIN_LIMITS.maxReferencesPerRecord}`,
    recordFlags: `u32[] == ${BRAIN_LIMITS.frameRecordSlots}`,
    activeRecordIndices: `u32[] == ${BRAIN_LIMITS.frameRecordSlots}`,
  },

  // -----------------------------------------------------------------------
  // Homeostasis and active query interface
  // -----------------------------------------------------------------------

  /**
   * Simulation-owned comfort dynamics lower into this generic engine signal.
   * No concrete comfort channel (satiation, pain, curiosity...) is frozen here;
   * channelToken and desiredStateToken come from the compiled domain manifest.
   */
  HomeostasisSignal: {
    channelToken: "KrystalTokenId",
    currentStateToken: "KrystalTokenId",
    desiredStateToken: "KrystalTokenId",
    flags: "u32 = 0",
    currentValue: "f32 = 0",
    targetValue: "f32 = 0",
    urgency: "f32 = 0",
    delta: "f32 = 0",
    source: "RuntimeRefHandle",
  },

  BrainQueryKind:
    "'none' | 'homeostasis' | 'tutorial' | 'external' | 'internal' | 'continuation' | 'runtime_feedback'",

  BrainQueryState: {
    queryRef: "RuntimeRefHandle",
    kind: "BrainQueryKind",
    routeToken: "KrystalTokenId",
    predicateToken: "KrystalTokenId",
    subject: "RuntimeRefHandle",
    objectToken: "KrystalTokenId = 0",
    objectRef: "RuntimeRefHandle",
    urgency: "f32 = 0",
    createdAt: "u32 = 0",
    expiresAt: `u32 = ${INVALID_U32}`,
    flags: "u32 = 0",
  },

  BrainQuerySet: {
    count: "u32 = 0",
    primary: `u32 = ${INVALID_U32}`,
    revision: "u32 = 0",
    reserved0: "u32 = 0",
    queries: `BrainQueryState[] == ${BRAIN_LIMITS.maxQueries}`,
  },

  // -----------------------------------------------------------------------
  // Working memory — bounded, exact storage; relevance remains learned
  // -----------------------------------------------------------------------

  MemoryTraceKind: "'none' | 'entity' | 'event' | 'goal' | 'intent' | 'topic' | 'observation'",
  MemorySlotState: "'empty' | 'active' | 'evictable' | 'evicted'",
  MemoryUpdateReason:
    "'observation' | 'look' | 'interaction' | 'comfort_delta' | 'retrieval' | 'rehearsal' | 'goal' | 'intent' | 'decay'",

  MemoryConfig: {
    slotCount: `u32 = ${BRAIN_LIMITS.maxMemorySlots}`,
    activationDecay: "f32 = 0.98",
    familiarityDecay: "f32 = 0.999",
    familiarityGain: "f32 = 0.05",
    evictionHysteresis: "f32 = 0.05",
    minimumResidenceTicks: "u32 = 2",
    flags: "u32 = 0",
    reserved0: "u32 = 0",
  },

  /**
   * A trace stores what was observed, not omniscient current world truth.
   * rememberedRecord therefore retains observedAt/source metadata and may be
   * stale even while its subject handle still denotes a live world object.
   */
  MemoryTrace: {
    memoryRef: "RuntimeRefHandle",
    subject: "RuntimeRefHandle",
    kind: "MemoryTraceKind",
    state: "MemorySlotState",
    flags: "u32 = 0",

    createdAt: "u32 = 0",
    lastObservedAt: "u32 = 0",
    lastAccessedAt: "u32 = 0",
    interactionCount: "u32 = 0",

    activation: "f32 = 0",
    familiarity: "f32 = 0",
    affectMagnitude: "f32 = 0",
    reserved0: "f32 = 0",

    rememberedRecord: "BrainRecordSlot",
  },

  /** Generic update emitted by perception, intent feedback or homeostasis. */
  MemoryUpdate: {
    subject: "RuntimeRefHandle",
    reason: "MemoryUpdateReason",
    interactionToken: "KrystalTokenId = 0",
    flags: "u32 = 0",
    activationDelta: "f32 = 0",
    familiarityDelta: "f32 = 0",
    affectMagnitude: "f32 = 0",
    tick: "u32 = 0",
  },

  WorkingMemoryState: {
    revision: "u32 = 0",
    activeCount: "u32 = 0",
    evictedCount: "u32 = 0",
    flags: "u32 = 0",
    slots: `MemoryTrace[] == ${BRAIN_LIMITS.maxMemorySlots}`,
  },

  // -----------------------------------------------------------------------
  // Static ActionIntent catalog
  // -----------------------------------------------------------------------

  ActionIntentDomain: "'external' | 'perceptual' | 'internal' | 'communicative' | 'postural'",
  ActionIntentCatalogHeader: {
    version: "u32 = 0",
    intentCount: "u32 = 0",
    argumentCount: "u32 = 0",
    flags: "u32 = 0",
    catalogHashLo: "u32 = 0",
    catalogHashHi: "u32 = 0",
    reserved0: "u32 = 0",
    reserved1: "u32 = 0",
  },

  /**
   * Capability/precondition classes are descriptive conditioning and runtime
   * diagnostics, never exclusive resource locks. The compiler masks malformed
   * types and inaccessible handles; physical overload, awkwardness and likely
   * failure remain attemptable and are resolved by the simulation.
   */
  ActionIntentDescriptor: {
    intentId: "IntentId",
    actionToken: "KrystalTokenId",
    semanticIntentToken: "KrystalTokenId",
    domain: "ActionIntentDomain",

    actorSchemaId: "SchemaId",
    argumentOffset: "u32",
    argumentCount: "u32",
    flags: "u32 = 0",

    effectClassToken: "KrystalTokenId = 0",
    capabilityClassToken: "KrystalTokenId = 0",
    preconditionClassToken: "KrystalTokenId = 0",
    preferredControllerRole: "KrystalTokenId = 0",
  },

  ActionArgumentDescriptor: {
    intentId: "IntentId",
    argumentIndex: "u32",
    roleToken: "KrystalTokenId",
    valueKind: "BrainValueKind",
    acceptedSchemaId: "SchemaId = 0",
    candidateBandMask: "BandMask = 0",
    flags: "u32 = 0",
    reserved0: "u32 = 0",
  },

  ActionArgumentAuthoringSpec: {
    name: "string",
    roleToken: "number",
    valueKind: "BrainValueKind",
    "acceptedSchema?": "string",
    "candidateBands?": "BrainBandKind[]",
    "required?": "boolean",
    "doc?": "string",
  },

  ActionIntentAuthoringSpec: {
    name: "string",
    actionToken: "number",
    semanticIntentToken: "number",
    domain: "ActionIntentDomain",
    arguments: "ActionArgumentAuthoringSpec[]",
    "effectClassToken?": "number",
    "capabilityClassToken?": "number",
    "preconditionClassToken?": "number",
    "preferredControllerRole?": "number",
    "durative?": "boolean",
    "doc?": "string",
  },

  // -----------------------------------------------------------------------
  // Learned selection and concurrent intent output
  // -----------------------------------------------------------------------

  SoftGatherStatus: "'empty' | 'selected' | 'masked' | 'ambiguous' | 'error'",

  /**
   * Diagnostic learned selection plus exact selected location. The runtime
   * resolves the handle from the chosen record/field sidecar; it never trusts
   * the network to recreate an arbitrary exact reference value.
   */
  SoftGatherResult: {
    status: "SoftGatherStatus",
    selectedRecord: `RecordIndex = ${INVALID_U32}`,
    selectedField: `FieldId = ${INVALID_U32}`,
    selectedReference: `u32 = ${INVALID_U32}`,
    candidateCount: "u32 = 0",
    probability: "f32 = 0",
    entropy: "f32 = 0",
    reserved0: "u32 = 0",
  },

  TypedArgumentValue: {
    kind: "BrainValueKind",
    token: "KrystalTokenId = 0",
    flags: "u32 = 0",
    reserved0: "u32 = 0",
    handle: "RuntimeRefHandle",
    selector: "SoftGatherResult",
  },

  IntentLifecycle: "'empty' | 'start' | 'maintain' | 'stop' | 'resume'",
  IntentExecutionStatus:
    "'empty' | 'proposed' | 'accepted' | 'active' | 'succeeded' | 'partial' | 'failed' | 'cancelled' | 'forgotten'",

  /**
   * One desired effect, not a guaranteed exclusive controller command.
   * Multiple proposals may overlap in body/controller preferences. The motor
   * layer combines them into soft goals and physics decides the outcome.
   */
  IntentProposal: {
    proposalSlot: "u32",
    lifecycle: "IntentLifecycle",
    intentId: "IntentId",
    flags: "u32 = 0",

    intentRef: "RuntimeRefHandle",
    purposeGoal: "RuntimeRefHandle",
    controllerHint: "RuntimeRefHandle",
    topic: "RuntimeRefHandle",

    activation: "f32 = 0",
    priority: "f32 = 0",
    persistence: "f32 = 0",
    confidence: "f32 = 0",

    arguments: `TypedArgumentValue[] == ${BRAIN_LIMITS.maxActionArguments}`,
  },

  /**
   * Unordered learned set. Slots are transport capacity, not semantic lanes;
   * training should shuffle non-active proposal slots. A durative accepted
   * intent receives an exact intentRef and may later be MAINTAINed or STOPped.
   */
  IntentSet: {
    tick: "u32 = 0",
    count: "u32 = 0",
    revision: "u32 = 0",
    flags: "u32 = 0",
    proposals: `IntentProposal[] == ${BRAIN_LIMITS.maxIntentProposals}`,
  },

  ActiveIntentState: {
    intentRef: "RuntimeRefHandle",
    purposeGoal: "RuntimeRefHandle",
    intentId: "IntentId",
    status: "IntentExecutionStatus",
    flags: "u32 = 0",
    startedAt: "u32 = 0",
    lastMaintainedAt: "u32 = 0",
    completedAt: `u32 = ${INVALID_U32}`,
    activation: "f32 = 0",
    progress: "f32 = 0",
    outcomeMagnitude: "f32 = 0",
    reserved0: "f32 = 0",
  },

  ActiveIntentTable: {
    revision: "u32 = 0",
    activeCount: "u32 = 0",
    completedCount: "u32 = 0",
    flags: "u32 = 0",
    intents: `ActiveIntentState[] == ${BRAIN_LIMITS.maxActiveIntents}`,
  },

  /** Generic runtime feedback; concrete world/body effects stay in simulation. */
  IntentFeedback: {
    intentRef: "RuntimeRefHandle",
    status: "IntentExecutionStatus",
    effectClassToken: "KrystalTokenId = 0",
    resultToken: "KrystalTokenId = 0",
    progress: "f32 = 0",
    outcomeMagnitude: "f32 = 0",
    comfortMagnitude: "f32 = 0",
    tick: "u32 = 0",
    feedbackRecord: `RecordIndex = ${INVALID_U32}`,
    flags: "u32 = 0",
  },

  // -----------------------------------------------------------------------
  // Generic Creator/tutorial orchestration
  // -----------------------------------------------------------------------

  TutorialBeatKind:
    "'narrate' | 'ask' | 'present' | 'focus' | 'demonstrate' | 'show_reaction' | 'assess' | 'wait' | 'reset'",
  TutorialProbeKind:
    "'token' | 'boolean' | 'pointer' | 'property' | 'intent' | 'consequence' | 'counterexample' | 'unknown'",

  TutorialProgramHeader: {
    version: "u32 = 0",
    lessonToken: "KrystalTokenId",
    beatOffset: "u32 = 0",
    beatCount: "u32 = 0",
    probeOffset: "u32 = 0",
    probeCount: "u32 = 0",
    creatorTokenOffset: "u32 = 0",
    creatorTokenCount: "u32 = 0",
    flags: "u32 = 0",
    reserved0: "u32 = 0",
  },

  /**
   * sceneCue is an opaque hook into the separate simulation/tutorial scene
   * compiler. This engine never defines what spawning an apple means.
   */
  TutorialBeat: {
    kind: "TutorialBeatKind",
    sceneCue: "u32 = 0",
    utteranceOffset: "u32 = 0",
    utteranceCount: "u32 = 0",
    holdFrames: "u32 = 0",
    probeIndex: `u32 = ${INVALID_U32}`,
    expectedIntentId: `IntentId = ${INVALID_U32}`,
    flags: "u32 = 0",
  },

  /** Gold fields are selected according to kind; unused values stay invalid. */
  TutorialProbe: {
    kind: "TutorialProbeKind",
    querySchemaId: "SchemaId = 0",
    expectedToken: "KrystalTokenId = 0",
    expectedIntentId: `IntentId = ${INVALID_U32}`,
    expectedRecord: `RecordIndex = ${INVALID_U32}`,
    expectedField: `FieldId = ${INVALID_U32}`,
    oracleBinding: `u32 = ${INVALID_U32}`,
    flags: "u32 = 0",
    reserved0: "u32 = 0",
  },

  TutorialRuntimeStatus: "'idle' | 'demonstrating' | 'waiting' | 'assessing' | 'passed' | 'failed' | 'done'",

  TutorialRuntimeState: {
    program: "u32 = 0",
    beat: "u32 = 0",
    probe: `u32 = ${INVALID_U32}`,
    status: "TutorialRuntimeStatus",
    frameInBeat: "u32 = 0",
    attempts: "u32 = 0",
    correct: "u32 = 0",
    incorrect: "u32 = 0",
  },

  // Host-only authoring forms. `sceneCue` belongs to the independent scene
  // compiler; narration is already semantic Krystal token IDs, not UTF-8.
  TutorialBeatAuthoringSpec: {
    kind: "TutorialBeatKind",
    "sceneCue?": "string",
    "utterance?": "number[]",
    "holdFrames?": "number",
    "expectedIntent?": "string",
    "probe?": "TutorialProbeAuthoringSpec",
    "doc?": "string",
  },

  TutorialProbeAuthoringSpec: {
    kind: "TutorialProbeKind",
    "querySchema?": "string",
    "expectedToken?": "number",
    "expectedIntent?": "string",
    "oracleBinding?": "string",
    "doc?": "string",
  },

  TutorialAuthoringSpec: {
    name: "string",
    lessonTokens: "number[]",
    "prerequisites?": "string[]",
    beats: "TutorialBeatAuthoringSpec[]",
    "doc?": "string",
  },

  // -----------------------------------------------------------------------
  // Brain engine runtime
  // -----------------------------------------------------------------------

  BrainRuntimeStatus: "'idle' | 'assembling_frame' | 'running' | 'executing' | 'done' | 'error'",

  BrainModelConfig: {
    vocabSize: `u32 = ${KRYSTAL_ABI.vocabSize}`,
    contextTokens: `u32 = ${BRAIN_LIMITS.frameTokens}`,
    recordWidth: `u32 = ${BRAIN_LIMITS.recordWidth}`,
    recordSlots: `u32 = ${BRAIN_LIMITS.frameRecordSlots}`,
    hiddenSize: "u32",
    recordSize: "u32",
    layerCount: "u32",
    attentionHeads: "u32",
    maxIntentProposals: `u32 = ${BRAIN_LIMITS.maxIntentProposals}`,
    flags: "u32 = 0",
  },

  BrainRuntimeConfig: {
    tokenAbiVersion: `u32 = ${KRYSTAL_ABI.tokenAbiVersion}`,
    architectureVersion: `u32 = ${KRYSTAL_ABI.architectureVersion}`,
    frameLayoutVersion: `u32 = ${KRYSTAL_ABI.frameLayoutVersion}`,
    vocabManifestVersion: "u32 = 0",
    recordManifestVersion: "u32 = 0",
    actionCatalogVersion: "u32 = 0",
    tutorialVersion: "u32 = 0",
    flags: "u32 = 0",
    reserved0: "u32 = 0",
    model: "BrainModelConfig",
    memory: "MemoryConfig",
  },

  BrainRuntimeState: {
    status: "BrainRuntimeStatus",
    tick: "u32 = 0",
    snapshot: "u32 = 0",
    frameRevision: "u32 = 0",
    memoryRevision: "u32 = 0",
    queryRevision: "u32 = 0",
    intentRevision: "u32 = 0",
    errorCode: "u32 = 0",
  },

  BrainStepTelemetry: {
    tick: "u32 = 0",
    activeRecords: "u32 = 0",
    activeTokens: "u32 = 0",
    truncatedRecords: "u32 = 0",
    intentCount: "u32 = 0",
    activeIntentCount: "u32 = 0",
    memoryCount: "u32 = 0",
    queryCount: "u32 = 0",

    frameBuildMs: "f32 = 0",
    localEncodeMs: "f32 = 0",
    recordMixMs: "f32 = 0",
    gatherMs: "f32 = 0",
    decideMs: "f32 = 0",
    runtimeMs: "f32 = 0",

    meanGatherEntropy: "f32 = 0",
    minGatherProbability: "f32 = 0",
    flags: "u32 = 0",
    errorCode: "u32 = 0",
  },

  BrainStepResult: {
    state: "BrainRuntimeState",
    intents: "IntentSet",
    telemetry: "BrainStepTelemetry",
  },
});
