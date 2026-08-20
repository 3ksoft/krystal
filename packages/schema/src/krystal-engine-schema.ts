import {} from "./env";
import { scope } from "arktype";
import { wgsl } from "@schema-pop/schema";

/**
 * Krystal brain-engine contracts — architecture v3.
 *
 * Core architectural pillars:
 *   1. Language of Thought (Mentalese):
 *      Modalities for relations: Imperative ('!'), Interrogative ('?'),
 *      Implicative / Temporal Transition ('->'), Declarative ('.').
 *   2. Unified Temporal Semantics (before -> then):
 *      Time and motion are not specialized sensor channels. They are modeled as
 *      relational state transitions (before -> then) and causal rules in working memory.
 *      Standalone 'temporal' band is merged into 'memory' (expanded to 68 slots).
 *   3. NSM Semantic Primes & Relational Algebra:
 *      Core primitives (NOT, IS_A, PART_OF, BEFORE, NOW, THEN) ground the ontology
 *      and enable deterministic goal backward-chaining in the ALU/attention layers.
 *   4. Multi-Effector Parallel Intent Output:
 *      IntentSet supports concurrent action proposals per free body controller.
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

/**
 * Fundamental system, logical and NSM sentinels.
 * The runtime and compiler branch on these tokens directly.
 */
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

  // NSM Logic & Structural Relational Primes
  not: 0x000b,
  isA: 0x000c,
  partOf: 0x000d,

  // NSM Temporal Reference Primes (Transitions: BEFORE -> THEN / NOW)
  before: 0x000e,
  now: 0x000f,
  then: 0x0010,

  /**
   * WANT — an operator, not a relation the creature declares.
   *
   * A creature never generates this about itself. It proposes `EAT(self,
   * stone)` like any other act, and the volitive reading is arrived at from
   * outside: either structurally, because a participant came from memory rather
   * than from what is in front of it — one cannot act on what one only
   * remembers — or because the simulation reports back what actually happened,
   * "you wanted to eat the stone -> your mouth hurts".
   *
   * Letting the creature assert its own wanting would give it a label to learn
   * instead of a connection: nothing would ground the claim, and a proposal
   * marked WANT would look identical whether or not anything followed from it.
   */
  want: 0x0011,
} as const;

export const BINARY_LAYOUT_PLAN_VERSION = 3;

export const BRAIN_LIMITS = {
  recordWidth: 8,

  /**
   * Two logical frames, one physical buffer.
   *
   * Perception is rewritten every tick and sized by how rich the scene is;
   * memory is resident and grows with the life of the creature. Sharing one
   * budget made every new percept compete with everything the creature knew,
   * so the two are now defined apart and merely CONCATENATED when packed. The
   * device sees one flat slot array and is unaffected by where the boundary
   * sits — which is what makes moving it cheap.
   */
  perceptRecordSlots: 304,
  memoryRecordSlots: 128,
  frameRecordSlots: 304 + 128,
  frameTokens: (304 + 128) * 8,
  perceptBands: 7,
  memoryBands: 1,
  frameBands: 8,
  fixedRecordBindings: 32,

  maxRecordSchemas: 0x100,
  maxRecordFields: 0x800,
  maxReferencesPerRecord: 8,

  maxActionIntents: 0x100,
  /**
   * Roles a reified relation can bind, and the closed list is the point: a
   * fixed arity is maskable on the device with one comparison per slot, where
   * an open role vocabulary would need a per-record search. See `RelationRole`.
   */
  relationArity: 6,
  maxIntentProposals: 8,
  maxActiveIntents: 16,

  maxQueries: 8,
  maxMemorySlots: 128,

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
 * Perceptual frame — 304 slots, rewritten every tick.
 *
 * There are no named senses here. A creature may live in a world with
 * echolocation and no eyes, so a band called `vision` would be this engine
 * asserting which senses exist — a fact about one world, frozen into the ABI of
 * all of them. Instead there is ONE perception band, and which sense a record
 * came from is a token carried by the record, drawn from that world's own
 * vocabulary. Per-channel quotas are the simulation's business.
 */
export const PERCEPT_FRAME_BANDS = [
  { kind: "system", recordOffset: 0, recordCapacity: 8, tokenOffset: 0, tokenCapacity: 64, placement: "fixed", overflow: "error" },
  { kind: "homeostasis", recordOffset: 8, recordCapacity: 16, tokenOffset: 64, tokenCapacity: 128, placement: "fixed", overflow: "error" },
  { kind: "body", recordOffset: 24, recordCapacity: 24, tokenOffset: 192, tokenCapacity: 192, placement: "fixed", overflow: "error" },
  { kind: "perception", recordOffset: 48, recordCapacity: 192, tokenOffset: 384, tokenCapacity: 1536, placement: "shuffled_records", overflow: "truncate_low_salience" },
  { kind: "focus", recordOffset: 240, recordCapacity: 16, tokenOffset: 1920, tokenCapacity: 128, placement: "stable_resident", overflow: "drop_oldest" },
  { kind: "query", recordOffset: 256, recordCapacity: 8, tokenOffset: 2048, tokenCapacity: 64, placement: "stable_resident", overflow: "drop_oldest" },
  { kind: "catalog", recordOffset: 264, recordCapacity: 40, tokenOffset: 2112, tokenCapacity: 320, placement: "fixed", overflow: "error" },
] as const;

/**
 * Memory frame — resident, its own revision counter and eviction policy.
 *
 * Holds perceptual facts, 'before -> then' transition traces, causal rules and
 * standing goals. Its offsets continue the perceptual frame's because the two
 * are packed back to back into one `BrainFrameGpu`.
 */
export const MEMORY_FRAME_BANDS = [
  { kind: "memory", recordOffset: 304, recordCapacity: 128, tokenOffset: 2432, tokenCapacity: 1024, placement: "stable_resident", overflow: "evict_low_priority" },
] as const;

/** The packed frame: perception then memory, in slot order. */
export const BRAIN_FRAME_BANDS = [...PERCEPT_FRAME_BANDS, ...MEMORY_FRAME_BANDS] as const;

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

  perceptualFocus: 240,
  thoughtFocus: 241,
  speechTopic: 242,

  primaryQuery: 256,
  catalogBase: 264,
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
  negated: 1 << 8, // Operator NOT applied to token
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
  // Proposition Modality Bits
  modalityImperative: 1 << 11,    // '!' Action command to effector
  modalityInterrogative: 1 << 12,  // '?' Epistemic query
  modalityImplicative: 1 << 13,    // '->' before -> then / causal rule
  /**
   * This record IS a reified relation, not an entity.
   *
   * Set so a role mask can tell the two apart without inspecting tokens. A
   * participant slot takes things, not events: nothing in the engine binds a
   * relation as a participant, so admitting one would only let the creature
   * form "eat the eating". That is a grammatical restriction, not a claim about
   * what is edible — the creature is still perfectly free to try the stone.
   */
  relation: 1 << 14,
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
  negated: 1 << 12, // NOT relation
  /**
   * WANTED rather than done — see `KRYSTAL_SENTINEL_TOKENS.want`.
   *
   * Set by the engine when a proposal reaches for something only remembered,
   * and by the simulation when it reports an attempt back. Both write the same
   * bit on purpose: the creature has to be able to learn that its own reaching
   * and the outcome it hears about afterwards are the same event.
   */
  volitive: 1 << 13,
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

/**
 * The closed role list of a reified relation.
 *
 * Order IS the slot index: a relation's roles are a fixed-size array indexed by
 * this list, so a device-side mask is one comparison per slot rather than a
 * search. Closed on purpose — an open role vocabulary would put a grammatical
 * decision inside the attention kernel.
 *
 * `agent` and `patient` cover what used to be subject/object. The remaining
 * four are what Krystal's own grammar marks with prepositions: `z` (instrument),
 * `v`/`na` (location), `podćas` (time) and the `kio` correlative (reason).
 */
export const RELATION_ROLES = ["agent", "patient", "instrument", "location", "time", "reason"] as const;

export type RelationRoleName = (typeof RELATION_ROLES)[number];

export const RELATION_ROLE_INDEX: Readonly<Record<RelationRoleName, number>> = Object.fromEntries(
  RELATION_ROLES.map((role, index) => [role, index]),
) as Readonly<Record<RelationRoleName, number>>;

export const RELATION_ROLE_FLAGS = {
  acceptsAny: 1 << 0,
  required: 1 << 1,
  /** This slot is declared by the relation at all. An unused slot is inert. */
  present: 1 << 2,
} as const;

export const INTENT_PROPOSAL_FLAGS = {
  /** Patient mirrors agent — what a unary relation means once roles are named. */
  reflexive: 1 << 0,
  patientUnknown: 1 << 1,
  patientExistential: 1 << 2,
  /**
   * A reach toward something remembered rather than perceived.
   *
   * Not executable as it stands — the creature cannot act on what is not in
   * front of it — and that is exactly what makes it a wanting. Derived from
   * where the participants came from, never chosen by a head.
   */
  volitive: 1 << 3,
} as const;

export const MEMORY_FLAGS = {
  pinned: 1 << 0,
  autobiographical: 1 << 1,
  goal: 1 << 2,
  intent: 1 << 3,
  transitionTrace: 1 << 4, // Represents a 'before -> then' state transition
  causalRule: 1 << 5,      // Represents an 'if -> then' world model rule
  eligibleForEviction: 1 << 6,
} as const;

// Geometry validation
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
// The split must hold on its own: perception is packed first, memory second,
// and each has to add up to the limit it declares or the concatenation lands
// records in the wrong frame.
const perceptSlots = PERCEPT_FRAME_BANDS.reduce((total, band) => total + band.recordCapacity, 0);
const memorySlots = MEMORY_FRAME_BANDS.reduce((total, band) => total + band.recordCapacity, 0);
if (
  perceptSlots !== BRAIN_LIMITS.perceptRecordSlots ||
  memorySlots !== BRAIN_LIMITS.memoryRecordSlots ||
  memorySlots !== BRAIN_LIMITS.maxMemorySlots ||
  PERCEPT_FRAME_BANDS.length !== BRAIN_LIMITS.perceptBands ||
  MEMORY_FRAME_BANDS.length !== BRAIN_LIMITS.memoryBands
) {
  throw new Error("Invalid perceptual/memory frame split");
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

  /**
   * Bands are structural, never sensory. `perception` is one band for every
   * sense the world happens to have; which sense a record came from lives in
   * `BrainRecordHeader.channelToken`, as a symbol from that world's vocabulary.
   */
  BrainBandKind:
    "'system' | 'homeostasis' | 'body' | 'perception' | 'memory' | 'focus' | 'query' | 'catalog'",

  /** Closed role list of a reified relation; see `RELATION_ROLES`. */
  RelationRole: "'agent' | 'patient' | 'instrument' | 'location' | 'time' | 'reason'",

  /**
   * Modality of a proposition in the Language of Thought:
   *   declarative    (.) - Observable fact or current state
   *   imperative     (!) - Motor or cognitive command executed by an effector
   *   interrogative  (?) - Epistemic query matching patterns across bands
   *   implicative    (->) - Temporal transition (before -> then) or causal rule (cause -> effect)
   */
  PropositionModality: "'declarative' | 'imperative' | 'interrogative' | 'implicative'",

  BandPlacementPolicy: "'fixed' | 'shuffled_records' | 'stable_resident'",
  BandOverflowPolicy: "'error' | 'truncate_low_salience' | 'evict_low_priority' | 'drop_oldest'",

  RecordSource:
    "'runtime' | 'sensor' | 'body' | 'homeostasis' | 'memory' | 'focus' | 'query' | 'creator' | 'intent_feedback'",

  RuntimeRefKind:
    "'none' | 'entity' | 'value' | 'memory' | 'event' | 'goal' | 'intent' | 'transition' | 'snapshot' | 'controller' | 'topic'",

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

  /**
   * One participant of a record.
   *
   * On a reified relation the `role` is the whole point: the record IS the
   * relation, and its references are the things standing in each role. On an
   * ordinary entity record the role is `agent` and means nothing in particular.
   */
  BrainReferenceBinding: {
    localTokenIndex: `LocalTokenIndex = ${INVALID_U32}`,
    fieldId: "FieldId = 0",
    role: "RelationRole = 'agent'",
    flags: "u32 = 0",
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
    /**
     * Which sense produced this, as a vocabulary symbol. Metadata rather than
     * one of the eight content tokens: those are already scarce, and a channel
     * is not something the record is ABOUT.
     */
    channelToken: "KrystalTokenId = 0",
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
    /**
     * The question's participants, indexed by `RELATION_ROLES`. A query is a
     * relation like any other — it just carries the interrogative modality —
     * so it binds the same six roles rather than a privileged subject/object
     * pair. An unbound role is what is being ASKED about.
     */
    roles: `ConceptRef[] == ${BRAIN_LIMITS.relationArity}`,
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

  MemoryTraceKind: "'none' | 'entity' | 'event' | 'goal' | 'intent' | 'transition' | 'rule' | 'topic' | 'observation'",
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
   * Working memory trace. Capable of storing:
   *   - A recognized world entity (kind: 'entity', modality: '.')
   *   - An observed state transition (kind: 'transition', modality: '->', e.g. before -> then)
   *   - A learned causal rule (kind: 'rule', modality: '->', e.g. action -> outcome)
   *   - An active sub-goal / intention (kind: 'goal', modality: '!')
   */
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

  /**
   * One role slot of a relation.
   *
   * Structural constraints only. There is no accepted-token set and no accepted
   * channel set, deliberately: what may stand in a role is not a fact any world
   * gets to declare, it is what the creature learns from watching. What remains
   * is what the FRAME makes true — which band a candidate may come from, and
   * whether it must carry a live reference to be acted upon at all.
   *
   * `role` is redundant with the slot's position in `ActionIntentDescriptor.roles`
   * and is kept anyway: one u32 buys a check that the catalog and the role list
   * have not drifted apart, which is otherwise a silent corruption.
   */
  RelationRoleDescriptor: {
    role: "RelationRole = 'agent'",
    roleToken: "KrystalTokenId = 0",
    valueKind: "BrainValueKind",
    candidateBandMask: "BandMask = 0",
    flags: "u32 = 0",
    reserved0: "u32 = 0",
  },

  ActionIntentDescriptor: {
    intentId: "IntentId",
    actionToken: "KrystalTokenId",
    semanticIntentToken: "KrystalTokenId",
    domain: "ActionIntentDomain",

    flags: "u32 = 0",
    effectClassToken: "KrystalTokenId = 0",
    capabilityClassToken: "KrystalTokenId = 0",

    preconditionClassToken: "KrystalTokenId = 0",
    preferredControllerRole: "KrystalTokenId = 0",
    reserved0: "u32 = 0",
    reserved1: "u32 = 0",

    /**
     * Every role slot, indexed by `RELATION_ROLES`. There is no privileged
     * subject any more: `Self` is simply a bank candidate the agent role
     * admits, selected the same way every other participant is.
     */
    roles: `RelationRoleDescriptor[] == ${BRAIN_LIMITS.relationArity}`,
  },

  RelationRoleAuthoringSpec: {
    role: "RelationRole",
    name: "string",
    roleToken: "number",
    valueKind: "BrainValueKind",
    "candidateBands?": "BrainBandKind[]",
    "doc?": "string",
  },

  ActionIntentAuthoringSpec: {
    name: "string",
    actionToken: "number",
    semanticIntentToken: "number",
    domain: "ActionIntentDomain",
    roles: "RelationRoleAuthoringSpec[]",
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

    /** One selection per role slot, indexed by `RELATION_ROLES`. */
    roles: `SelectedConceptRef[] == ${BRAIN_LIMITS.relationArity}`,
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