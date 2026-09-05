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

export const INVALID_U32 = 0xffff_ffff;

export const schema = scope({
  ...wgsl.import(),

  /**
   * Bands are structural, never sensory. `perception` is one band for every
   * sense the world happens to have; which sense a record came from lives in
   * `BrainRecordHeader.channelToken`, as a symbol from that world's vocabulary.
   */
  BrainBandKind:
    "'system' | 'homeostasis' | 'body' | 'perception' | 'memory' | 'focus' | 'query' | 'catalog'",

  BrainValueKind:
    "'none' | 'token' | 'context_ref' | 'record_ref' | 'boolean_class' | 'scalar_band' | 'quantity_projection' | 'opaque_payload'",




  /**
   * The frame as the model reads it: six additive lookups per token and one
   * active list, in columns.
   *
   * Three buffers left when nothing turned out to read them. `attentionMask`
   * was written every frame and never looked at — padding is found by the PAD
   * sentinel. `runtimeRefs` and `recordFlags` served role filters that belonged
   * to a world contract the host now owns; a reference to a world entity is the
   * host's to resolve, and the model never learned anything from one.
   */
  BrainFrameGpu: {
    tokenIds: `u32[] == ${BRAIN_LIMITS.frameTokens}`,
    fieldRoles: `u32[] == ${BRAIN_LIMITS.frameTokens}`,
    schemaIds: `u32[] == ${BRAIN_LIMITS.frameRecordSlots}`,
    bandIds: `u32[] == ${BRAIN_LIMITS.frameRecordSlots}`,
    activeRecordIndices: `u32[] == ${BRAIN_LIMITS.frameRecordSlots}`,
  },
});