import { } from "./env";
import { scope } from "arktype";
import { wgsl } from "@schema-pop/schema";

/**
 * Krystal brain-engine contracts — architecture v3.
 *
 * Core enhancements in v3:
 *   - Language of Thought (Mentalese): Sentence modalities ('!', '?', '->', '.')
 *   - Temporal-Sensory Fusion: Temporal features integrated directly into sensory records;
 *     standalone temporal band eliminated; memory band expanded.
 *   - NSM Semantic Primes & Relational Foundations (IS_A, PART_OF, NOT).
 *   - Multi-effector parallel action proposals.
 */

// ---------------------------------------------------------------------------
// Frozen/provisional constants
// ---------------------------------------------------------------------------

export const KRYSTAL_ABI = {
  tokenAbiVersion: 2,
  architectureVersion: 3,
  frameLayoutVersion: 3,

  tokenBits: 16,
  tokenSpaceSize: 0x10000,

  semanticStart: 0x0000,
  semanticEnd: 0x7fff,
  semanticVocabSize: 0x8000,

  refSpaceStart: 0x8000,
  refSpaceEnd: 0xfffe,
  refSpaceSize: 0x7fff,
  reservedEmptyToken: 0xffff,
  refEmbeddingRows: 0x100,

  semanticEmbeddingRows: 0x1000,

  tokenClassBits: 4,
  tokenClassSize: 0x800,
} as const;

export const KRYSTAL_SENTINEL_TOKENS = {
  pad: 0x0000,
  bos: 0x0001,
  eos: 0x0002,
  boolTrue: 0x0003,
  boolFalse: 0x0004,
  unknown: 0x0005,
  begin: 0x0006,
  end: 0x0007,
  void: 0x0008,
  unavailable: 0x0009,
  something: 0x000a,
  // NSM / Logic core primitives
  not: 0x000b,
  isA: 0x000c,
  partOf: 0x000d,
} as const;

export const BINARY_LAYOUT_PLAN_VERSION = 3;

export const BRAIN_LIMITS = {
  recordWidth: 8,
  frameRecordSlots: 304,
  frameTokens: 304 * 8,
  // 12 bands: 'temporal' merged into sensory + memory expanded (52 -> 68)
  frameBands: 12,
  fixedRecordBindings: 32,

  maxRecordSchemas: 0x100,
  maxRecordFields: 0x800,
  maxRoleAcceptedTokens: 16,
  maxReferencesPerRecord: 8,

  maxActionIntents: 0x100,
  relationArity: 2,
  maxIntentProposals: 8,
  maxActiveIntents: 16,

  maxQueries: 8,
  maxMemorySlots: 48,

  maxTutorialBeats: 0x100,
  maxTutorialProbes: 0x40,
  maxTutorialTokens: 0x1000,
} as const;

export const KRYSTAL_TOKEN_RANGES = {
  system: [0x0000, 0x07ff],
  structure: [0x0800, 0x0fff],
  operation: [0x1000, 0x17ff],
  object: [0x1800, 0x1fff],
  property: [0x2000, 0x27ff],
  quantity: [0x2800, 0x2fff],
  action: [0x3000, 0x37ff],
  reference: [0x3800, 0x3fff],
  relation: [0x4000, 0x47ff],
  logic: [0x4800, 0x4fff],
  temporal: [0x5000, 0x57ff],
  domain: [0x5800, 0x77ff],
  experimental: [0x7800, 0x7fff],
  context: [0x8000, 0xfffe],
} as const;

export const KRYSTAL_TOKEN_CLASS_ORDER = Object.keys(
  KRYSTAL_TOKEN_RANGES,
) as readonly (keyof typeof KRYSTAL_TOKEN_RANGES)[];

export function tokenClassIndex(tokenId: number): number {
  for (let index = 0; index < KRYSTAL_TOKEN_CLASS_ORDER.length; index++) {
    const [lo, hi] = KRYSTAL_TOKEN_RANGES[KRYSTAL_TOKEN_CLASS_ORDER[index]!];
    if (tokenId >= lo && tokenId <= hi) return index;
  }
  return -1;
}

/**
 * Logical frame geometry: 12 bands, 304 slots total.
 * The 16 slots formerly dedicated to the standalone temporal band have been
 * absorbed directly into `memory` (52 -> 68 slots).
 */
export const BRAIN_FRAME_BANDS = [
  { kind: "system", recordOffset: 0, recordCapacity: 8, tokenOffset: 0, tokenCapacity: 64, placement: "fixed", overflow: "error" },
  { kind: "homeostasis", recordOffset: 8, recordCapacity: 16, tokenOffset: 64, tokenCapacity: 128, placement: "fixed", overflow: "error" },
  { kind: "body", recordOffset: 24, recordCapacity: 24, tokenOffset: 192, tokenCapacity: 192, placement: "fixed", overflow: "error" },
  { kind: "vision", recordOffset: 48, recordCapacity: 64, tokenOffset: 384, tokenCapacity: 512, placement: "shuffled_records", overflow: "truncate_low_salience" },
  { kind: "audio", recordOffset: 112, recordCapacity: 24, tokenOffset: 896, tokenCapacity: 192, placement: "shuffled_records", overflow: "truncate_low_salience" },
  { kind: "olfaction", recordOffset: 136, recordCapacity: 12, tokenOffset: 1088, tokenCapacity: 96, placement: "shuffled_records", overflow: "truncate_low_salience" },
  { kind: "taste", recordOffset: 148, recordCapacity: 8, tokenOffset: 1184, tokenCapacity: 64, placement: "shuffled_records", overflow: "truncate_low_salience" },
  { kind: "touch", recordOffset: 156, recordCapacity: 24, tokenOffset: 1248, tokenCapacity: 192, placement: "shuffled_records", overflow: "truncate_low_salience" },
  // Expanded from 52 to 68 slots (absorbed 16 slots from temporal):
  { kind: "memory", recordOffset: 180, recordCapacity: 68, tokenOffset: 1440, tokenCapacity: 544, placement: "stable_resident", overflow: "evict_low_priority" },
  { kind: "focus", recordOffset: 248, recordCapacity: 12, tokenOffset: 1984, tokenCapacity: 96, placement: "stable_resident", overflow: "drop_oldest" },
  { kind: "query", recordOffset: 260, recordCapacity: 12, tokenOffset: 2080, tokenCapacity: 96, placement: "stable_resident", overflow: "drop_oldest" },
  { kind: "catalog", recordOffset: 272, recordCapacity: 32, tokenOffset: 2176, tokenCapacity: 256, placement: "fixed", overflow: "error" },
] as const;

export const BRAIN_FIXED_RECORDS = {
  frame: 0,
  clock: 1,
  actor: 2,
  runtimeFeedback: 3,

  homeostasisSummary: 8,
  primaryNeed: 9,

  self: 24,
  head: 25,
  leftHand: 26,
  rightHand: 27,
  mouth: 28,
  locomotion: 29,
  torsoBalance: 30,
  internalFocus: 31,
  vocalizer: 32,

  perceptualFocus: 248,
  thoughtFocus: 249,
  speechTopic: 250,

  primaryQuery: 260,
  catalogBase: 272,
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
  // Modality & logic bits
  negated: 1 << 8,
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
  // Modality bits
  modalityImperative: 1 << 11,    // '!'
  modalityInterrogative: 1 << 12,  // '?'
  modalityImplicative: 1 << 13,    // '->'
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
  canonicallyReflexive: 1 << 8,
} as const;

export const RELATION_FLAGS = {
  transitive: 1 << 8,
  symmetric: 1 << 9,
  antisymmetric: 1 << 10,
  functional: 1 << 11,
  negated: 1 << 12, // Operator NOT on relations
} as const;

export const QUANTIFIER_FLAGS = {
  restrictorUpward: 1 << 12,
  restrictorDownward: 1 << 13,
  scopeUpward: 1 << 14,
  scopeDownward: 1 << 15,
} as const;

export const QUANTITY_BANDS = {
  signedDeadzone: 0.05,
  signedMagnitude: [0.25, 0.5, 0.75],
  unipolar: [0.25, 0.5, 0.75],
  count: [1, 3],
} as const;

export const RELATION_ROLE_FLAGS = {
  acceptsAny: 1 << 0,
  required: 1 << 1,
} as const;

export const INTENT_PROPOSAL_FLAGS = {
  objectFromSubject: 1 << 0,
  objectUnknown: 1 << 1,
  objectExistential: 1 << 2,
} as const;

export const MEMORY_FLAGS = {
  pinned: 1 << 0,
  autobiographical: 1 << 1,
  goal: 1 << 2,
  intent: 1 << 3,
  staleObservation: 1 << 4,
  eligibleForEviction: 1 << 5,
} as const;

// Layout geometry verification
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

  KrystalTokenId: "u32",
  SchemaId: "u32",
  FieldId: "u32",
  IntentId: "u32",
  RecordIndex: "u32",
  LocalTokenIndex: "u32",
  BandMask: "u32",

  KrystalTokenClass:
    "'system' | 'structure' | 'operation' | 'object' | 'property' | 'quantity' | 'action' | 'reference' | 'relation' | 'logic' | 'temporal' | 'domain' | 'experimental' | 'context'",

  BrainBandKind:
    "'system' | 'homeostasis' | 'body' | 'vision' | 'audio' | 'olfaction' | 'taste' | 'touch' | 'memory' | 'focus' | 'query' | 'catalog'",

  /**
   * Sentence modality in the Language of Thought (Mentalese):
   *   imperative     (!) - Executable action command to a physical effector
   *   interrogative  (?) - Internal query / epistemic probe across bands
   *   implicative    (->) - Causal rule / prediction / hypothesis in memory
   *   declarative    (.) - Factual state or observation
   */
  PropositionModality: "'declarative' | 'imperative' | 'interrogative' | 'implicative'",

  BandPlacementPolicy: "'fixed' | 'shuffled_records' | 'stable_resident'",
  BandOverflowPolicy: "'error' | 'truncate_low_salience' | 'evict_low_priority' | 'drop_oldest'",

  RecordSource:
    "'runtime' | 'sensor' | 'body' | 'homeostasis' | 'memory' | 'focus' | 'query' | 'creator' | 'intent_feedback'",

  RuntimeRefKind:
    "'none' | 'entity' | 'value' | 'memory' | 'event' | 'goal' | 'intent' | 'snapshot' | 'controller' | 'topic'",

  RuntimeRefStatus: "'invalid' | 'live' | 'stale' | 'historical' | 'destroyed'",

  RuntimeRefHandle: {
    tokenId: "KrystalTokenId",
    generation: "u32",
    kind: "RuntimeRefKind",
    status: "RuntimeRefStatus",
  },

  VocabManifestHeader: {
    tokenAbiVersion: `u32 = ${KRYSTAL_ABI.tokenAbiVersion}`,
    manifestVersion: "u32 = 0",
    vocabSize: `u32 = ${KRYSTAL_ABI.semanticVocabSize}`,
    activeTokenCount: "u32 = 0",
    embeddingRows: `u32 = ${KRYSTAL_ABI.semanticEmbeddingRows}`,
    manifestHashLo: "u32 = 0",
    manifestHashHi: "u32 = 0",
    reserved0: "u32 = 0",
    reserved1: "u32 = 0",
  },

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

  QuantityKind: "'signed' | 'unipolar' | 'count' | 'proportion'",

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

  RecordFieldEntry: {
    schemaId: "SchemaId",
    fieldId: "FieldId",
    localTokenIndex: "LocalTokenIndex",
    tokenWidth: "u32 = 1",
    roleToken: "KrystalTokenId",
    valueKind: "BrainValueKind",
    quantityKind: "QuantityKind = 'unipolar'",
    acceptedSchemaId: "SchemaId = 0",
    allowedBandMask: "BandMask = 0",
    flags: "u32 = 0",
    reserved0: "u32 = 0",
  },

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

  BrainTokenMeta: {
    fieldId: "FieldId",
    roleToken: "KrystalTokenId",
    flags: "u32 = 0",
    referenceBinding: `u32 = ${INVALID_U32}`,
  },

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
    modality: "PropositionModality = 'declarative'",
    flags: "u32 = 0",

    tokenCount: "u32 = 0",
    referenceCount: "u32 = 0",
    observedAt: "u32 = 0",
    revision: "u32 = 0",

    primaryReference: `u32 = ${INVALID_U32}`,
    continuationRecord: `RecordIndex = ${INVALID_U32}`,
    salience: "f32 = 0",
    freshness: "f32 = 0",

    previousObservedAt: `u32 = ${INVALID_U32}`,
    changeMagnitude: "f32 = 0",
    reserved0: "u32 = 0",
    reserved1: "u32 = 0",
  },

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
    deltaMillis: "f32 = 0",

    activeRecordCount: "u32 = 0",
    activeTokenCount: "u32 = 0",
    activeQueryRecord: `RecordIndex = ${INVALID_U32}`,
    actorRecord: `RecordIndex = ${BRAIN_FIXED_RECORDS.actor}`,

    frameRevision: "u32 = 0",
    memoryRevision: "u32 = 0",
    intentRevision: "u32 = 0",
    flags: "u32 = 0",
  },

  BrainFrame: {
    header: "BrainFrameHeader",
    bands: `BrainBandState[] == ${BRAIN_LIMITS.frameBands}`,
    records: `BrainRecordSlot[] == ${BRAIN_LIMITS.frameRecordSlots}`,
  },

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

  BrainFrameGpu: {
    header: "BinaryLayoutPlanHeader",
    tokenIds: `u32[] == ${BRAIN_LIMITS.frameTokens}`,
    fieldRoles: `u32[] == ${BRAIN_LIMITS.frameTokens}`,
    attentionMask: `u32[] == ${BRAIN_LIMITS.frameTokens}`,
    schemaIds: `u32[] == ${BRAIN_LIMITS.frameRecordSlots}`,
    bandIds: `u32[] == ${BRAIN_LIMITS.frameRecordSlots}`,
    runtimeRefs: `u32[] == ${BRAIN_LIMITS.frameRecordSlots * BRAIN_LIMITS.maxReferencesPerRecord}`,
    recordFlags: `u32[] == ${BRAIN_LIMITS.frameRecordSlots}`,
    activeRecordIndices: `u32[] == ${BRAIN_LIMITS.frameRecordSlots}`,
  },

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
    modality: "PropositionModality = 'interrogative'",
    routeToken: "KrystalTokenId",
    predicateToken: "KrystalTokenId",
    subject: "ConceptRef",
    object: "ConceptRef",
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

  MemoryTraceKind: "'none' | 'entity' | 'event' | 'goal' | 'intent' | 'rule' | 'topic' | 'observation'",
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

  MemoryTrace: {
    memoryRef: "RuntimeRefHandle",
    subject: "RuntimeRefHandle",
    kind: "MemoryTraceKind",
    modality: "PropositionModality = 'declarative'",
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

  ActionIntentDomain: "'external' | 'perceptual' | 'internal' | 'communicative' | 'postural'",
  ActionIntentCatalogHeader: {
    version: "u32 = 0",
    intentCount: "u32 = 0",
    relationArity: `u32 = ${BRAIN_LIMITS.relationArity}`,
    flags: "u32 = 0",
    catalogHashLo: "u32 = 0",
    catalogHashHi: "u32 = 0",
    reserved0: "u32 = 0",
    reserved1: "u32 = 0",
  },

  RelationRoleDescriptor: {
    roleToken: "KrystalTokenId = 0",
    valueKind: "BrainValueKind",
    acceptedTokens: `u32[] == ${BRAIN_LIMITS.maxRoleAcceptedTokens}`,
    candidateBandMask: "BandMask = 0",
    flags: "u32 = 0",
    reserved0: "u32 = 0",
  },

  ActionIntentDescriptor: {
    intentId: "IntentId",
    actionToken: "KrystalTokenId",
    semanticIntentToken: "KrystalTokenId",
    domain: "ActionIntentDomain",

    subjectSchemaId: "SchemaId",
    flags: "u32 = 0",
    effectClassToken: "KrystalTokenId = 0",
    capabilityClassToken: "KrystalTokenId = 0",

    preconditionClassToken: "KrystalTokenId = 0",
    preferredControllerRole: "KrystalTokenId = 0",
    reserved0: "u32 = 0",
    reserved1: "u32 = 0",

    subjectRole: "RelationRoleDescriptor",
    objectRole: "RelationRoleDescriptor",
  },

  RelationRoleAuthoringSpec: {
    name: "string",
    roleToken: "number",
    valueKind: "BrainValueKind",
    "acceptedSchema?": "string",
    "candidateBands?": "BrainBandKind[]",
    "doc?": "string",
  },

  ActionIntentAuthoringSpec: {
    name: "string",
    actionToken: "number",
    semanticIntentToken: "number",
    domain: "ActionIntentDomain",
    subject: "RelationRoleAuthoringSpec",
    "object?": "RelationRoleAuthoringSpec",
    "effectClassToken?": "number",
    "capabilityClassToken?": "number",
    "preconditionClassToken?": "number",
    "preferredControllerRole?": "number",
    "durative?": "boolean",
    "doc?": "string",
  },

  SoftGatherStatus: "'empty' | 'selected' | 'masked' | 'ambiguous' | 'error'",

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

  ConceptRef: {
    kind: "BrainValueKind",
    token: "KrystalTokenId = 0",
    flags: "u32 = 0",
    reserved0: "u32 = 0",
    handle: "RuntimeRefHandle",
  },

  SelectedConceptRef: {
    concept: "ConceptRef",
    selector: "SoftGatherResult",
  },

  IntentLifecycle: "'empty' | 'start' | 'maintain' | 'stop' | 'resume'",
  IntentExecutionStatus:
    "'empty' | 'proposed' | 'accepted' | 'active' | 'succeeded' | 'partial' | 'failed' | 'cancelled' | 'forgotten'",

  IntentProposal: {
    proposalSlot: "u32",
    lifecycle: "IntentLifecycle",
    modality: "PropositionModality = 'imperative'",
    intentId: "IntentId",
    flags: "u32 = 0",

    intentRef: "RuntimeRefHandle",
    purposeGoal: "RuntimeRefHandle",
    controllerHint: "RuntimeRefHandle",
    topic: "RuntimeRefHandle",

    activation: "f32 = 0",
    priority: "f32 = 0",
    commitment: "f32 = 0.5",
    intensity: "f32 = 0",
    persistence: "f32 = 0",
    confidence: "f32 = 0",

    subject: "SelectedConceptRef",
    object: "SelectedConceptRef",
  },

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

  BrainRuntimeStatus: "'idle' | 'assembling_frame' | 'running' | 'executing' | 'done' | 'error'",

  BrainModelConfig: {
    vocabSize: `u32 = ${KRYSTAL_ABI.semanticEmbeddingRows}`,
    refEmbeddingRows: `u32 = ${KRYSTAL_ABI.refEmbeddingRows}`,
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