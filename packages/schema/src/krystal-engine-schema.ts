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
 *   - ActionIntent catalog as binary relations, soft-gather results and the
 *     concurrent IntentSet,
 *   - generic tutorial orchestration,
 *   - engine/runtime state and telemetry.
 *
 * Explicitly out of scope:
 *   - world/entity/component schemas,
 *   - sensory simulation and feature extraction (including how the temporal
 *     band's motion relations are actually derived),
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
 *    The model sees `frameRecordSlots` fixed record slots, eight logical token
 *    positions per record. Band and selected structural record positions are
 *    stable. Dynamic records may be shuffled only inside their band's fixed
 *    slot range. Record-local token positions are interpreted by schema
 *    metadata rather than by a universal flat grammar.
 *
 * 2. BinaryLayoutPlan (intentionally not frozen yet)
 *
 *    The GPU implementation will probably lower the logical AoS records into
 *    SoA buffers resembling:
 *
 *      tokenIds[recordSlot][localToken]
 *      attentionMask[recordSlot][localToken]
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
  /**
   * Semantic/token ABI. v1 widens tokens from 12 to 16 bits and splits the
   * space into an embedded semantic half and a bound reference half.
   */
  tokenAbiVersion: 1,
  /** Neural/frame architecture described by KRYSTAL_BRAIN_ARCHITECTURE_V2. */
  architectureVersion: 2,
  /** Frame geometry in this file; independent from both above. v2 adds the
   *  temporal band and binary-relation intents. */
  frameLayoutVersion: 2,

  /**
   * Token IDs are 16 bits. 16 is chosen over 13/14 because it is the machine
   * boundary the eventual two-u16-per-word packing wants anyway, and it costs
   * nothing while IDs are still stored as u32.
   *
   * The space has two halves that are NOT interchangeable:
   *
   *   semantic  [semanticStart, semanticEnd]
   *     Concept vocabulary. Every ID owns a learned embedding row, so this
   *     half is the one that costs parameters and must stay bounded.
   *
   *   reference [refSpaceStart, refSpaceEnd]
   *     Dynamic runtime symbols (the old 0xExx context class). These are
   *     POINTERS: their meaning comes entirely from the matching
   *     `BrainReferenceBinding` and from the record they denote, never from a
   *     per-ID learned row. Reference #37 has no stable semantics across
   *     frames, so a dense embedding table over this half would be both waste
   *     and noise. A reference token's input vector is composed from a small
   *     shared pool of `refEmbeddingRows` role rows plus the content of its
   *     referent. That is what makes 32k references affordable.
   */
  tokenBits: 16,
  tokenSpaceSize: 0x10000,

  semanticStart: 0x0000,
  semanticEnd: 0x7fff,
  /** Embedded rows. This is the number that sizes the embedding matrix. */
  semanticVocabSize: 0x8000,

  refSpaceStart: 0x8000,
  /**
   * 0xffff is deliberately NOT allocatable. It is the reserved empty-state
   * sentinel of the sibling GPU guide schema (GPU_SCHEMA_SENTINELS), which
   * relies on no token id ever reaching it so fixed-size tables can mark an
   * empty slot without a parallel validity bitmap. Under the old 12-bit space
   * that was free; at 16 bits the two spaces meet, so one reference out of
   * 32768 is given up to keep the invariant.
   */
  refSpaceEnd: 0xfffe,
  refSpaceSize: 0x7fff,
  reservedEmptyToken: 0xffff,
  /** Shared role rows backing every reference token; NOT one row per ref. */
  refEmbeddingRows: 0x100,

  /**
   * Capacity of the learned semantic embedding table, in rows.
   *
   * A token's row is its MANIFEST INDEX, not its token id. The id space is a
   * sparse class grid (objects at 0x1800, properties at 0x2000, ...), so a
   * table indexed by id would have to span all 0x8000 ids — 4.2M parameters at
   * h=128 to carry a few hundred live symbols. Indexing by manifest position
   * makes the table proportional to the vocabulary that actually exists.
   *
   * This is a fixed CAPACITY, deliberately not `activeTokenCount`. Sizing the
   * table to the current vocabulary would make every content addition a change
   * of tensor shape. 4096 rows is ~16x the pirapitinga grammar's present 248
   * symbols and costs 524K parameters at h=128 — cheap enough that growing the
   * world never becomes a model-architecture decision.
   *
   * Two invariants the compiler must enforce, because violating either
   * silently destroys trained weights rather than failing:
   *   1. activeTokenCount <= semanticEmbeddingRows.
   *   2. Manifest indices are APPEND-ONLY. A row is a learned vector; if index
   *      assignment is re-derived (say, by sorting symbol names) then adding
   *      one symbol renumbers every later row and every embedding it trained.
   */
  semanticEmbeddingRows: 0x1000,

  /**
   * The semantic half is a uniform grid: 16 class slots of 0x800 each,
   * exactly filling 0x8000. Class index of a semantic token is `id >> 11`.
   * `domain` deliberately spans four slots because game-authored vocabulary is
   * the part that actually grows.
   */
  tokenClassBits: 4,
  tokenClassSize: 0x800,
} as const;

/**
 * Reserved system tokens with runtime mechanics attached. These are not merely
 * conventional IDs: the compiler and runtime branch on them, so they are frozen
 * here rather than left to a vocabulary fixture.
 *
 * The three "nothing" tokens are deliberately distinct, because collapsing them
 * destroys information a creature needs:
 *
 *   pad          Structural absence. Nothing was sampled into this position.
 *                Carries zero information and MUST be hard-masked out of
 *                attention rather than learned-around; a PAD the network still
 *                attends to is white noise that costs capacity.
 *
 *   void         Sensed emptiness. The sense looked and there is genuinely
 *                nothing there — empty space, silence. This is a PERCEPT and
 *                must reach the model as an ordinary token.
 *
 *   unavailable  The sense itself is not reporting: eyes closed, darkness, ear
 *                blocked. Also a percept, and a different one — "silence"
 *                should settle a creature while "I cannot hear" should not.
 *
 * Separate from all three is the epistemic pair:
 *
 *   unknown      A referent exists but its identity is not known. Licenses a
 *                query; a calibrated model is right to emit it.
 *   something    Existential quantifier / bound variable, for goals and query
 *                objects. Does not license a query — it IS the query's target.
 *
 * And separate again from every token above is `RuntimeRefStatus.invalid`:
 * "the handle broke" is a runtime fault, not a belief. Never project a fault
 * into a sentinel token, or a broken binding becomes indistinguishable from
 * honest ignorance and the model learns to report ignorance when the engine
 * is at fault.
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
export const BINARY_LAYOUT_PLAN_VERSION = 2;

export const BRAIN_LIMITS = {
  recordWidth: 8,
  frameRecordSlots: 304,
  frameTokens: 304 * 8,
  frameBands: 13,
  fixedRecordBindings: 32,

  maxRecordSchemas: 0x100,
  maxRecordFields: 0x800,
  /**
   * How many token ids one relation role may admit.
   *
   * Small on purpose. A role that needs a long list is a role whose world has
   * no word for what it means — the fix there is a category symbol the records
   * carry, not more slots here, and a hard ceiling is what forces that
   * question to be asked instead of quietly answered with enumeration.
   */
  maxRoleAcceptedTokens: 16,
  // A record can legally contain a reference in every logical token position.
  maxReferencesPerRecord: 8,

  maxActionIntents: 0x100,
  /**
   * Every relation is binary. Not a cap that happens to be two — a structural
   * commitment. Unary predicates take object = subject (`LAUGH` becomes "I
   * rejoice myself", the construction Polish spells with the reflexive
   * "cieszę SIĘ"), so there is no void-argument case at all: the object
   * selection head always predicts something and training never sees a masked
   * argument slot. Higher-arity verbs reify instead — an emitted relation owns
   * a `RuntimeRefHandle`, so `GIVE(self, apple)` plus `RECIPIENT(that, mother)`
   * expresses the ditransitive as two binary facts.
   */
  relationArity: 2,
  maxIntentProposals: 8,
  maxActiveIntents: 16,

  maxQueries: 8,
  maxMemorySlots: 32,

  maxTutorialBeats: 0x100,
  maxTutorialProbes: 0x40,
  maxTutorialTokens: 0x1000,
} as const;

/**
 * Class grid over the semantic half, plus `context` which is the whole
 * reference half. Every semantic class is one 0x800 slot except `domain`,
 * which spans four.
 */
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
  /** Motion, change, rate and duration concepts fed by the temporal sense. */
  temporal: [0x5000, 0x57ff],
  domain: [0x5800, 0x77ff],
  experimental: [0x7800, 0x7fff],
  /**
   * The reference half: dynamic runtime symbols, bound and not embedded.
   * Stops at 0xfffe — see KRYSTAL_ABI.reservedEmptyToken.
   */
  context: [0x8000, 0xfffe],
} as const;

/** Token classes in range order; the index is the class id. */
export const KRYSTAL_TOKEN_CLASS_ORDER = Object.keys(
  KRYSTAL_TOKEN_RANGES,
) as readonly (keyof typeof KRYSTAL_TOKEN_RANGES)[];

/**
 * Which class a token id falls in, -1 if none does.
 *
 * The classes are contiguous and disjoint by construction, so this is the
 * inverse of the grid rather than a second table that could disagree with it.
 */
export function tokenClassIndex(tokenId: number): number {
  for (let index = 0; index < KRYSTAL_TOKEN_CLASS_ORDER.length; index++) {
    const [lo, hi] = KRYSTAL_TOKEN_RANGES[KRYSTAL_TOKEN_CLASS_ORDER[index]!];
    if (tokenId >= lo && tokenId <= hi) return index;
  }
  return -1;
}

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
  { kind: "system", recordOffset: 0, recordCapacity: 8, tokenOffset: 0, tokenCapacity: 64, placement: "fixed", overflow: "error" },
  { kind: "homeostasis", recordOffset: 8, recordCapacity: 16, tokenOffset: 64, tokenCapacity: 128, placement: "fixed", overflow: "error" },
  { kind: "body", recordOffset: 24, recordCapacity: 24, tokenOffset: 192, tokenCapacity: 192, placement: "fixed", overflow: "error" },
  { kind: "vision", recordOffset: 48, recordCapacity: 64, tokenOffset: 384, tokenCapacity: 512, placement: "shuffled_records", overflow: "truncate_low_salience" },
  { kind: "audio", recordOffset: 112, recordCapacity: 24, tokenOffset: 896, tokenCapacity: 192, placement: "shuffled_records", overflow: "truncate_low_salience" },
  { kind: "olfaction", recordOffset: 136, recordCapacity: 12, tokenOffset: 1088, tokenCapacity: 96, placement: "shuffled_records", overflow: "truncate_low_salience" },
  { kind: "taste", recordOffset: 148, recordCapacity: 8, tokenOffset: 1184, tokenCapacity: 64, placement: "shuffled_records", overflow: "truncate_low_salience" },
  { kind: "touch", recordOffset: 156, recordCapacity: 24, tokenOffset: 1248, tokenCapacity: 192, placement: "shuffled_records", overflow: "truncate_low_salience" },
  { kind: "memory", recordOffset: 180, recordCapacity: 52, tokenOffset: 1440, tokenCapacity: 416, placement: "stable_resident", overflow: "evict_low_priority" },
  { kind: "focus", recordOffset: 232, recordCapacity: 12, tokenOffset: 1856, tokenCapacity: 96, placement: "stable_resident", overflow: "drop_oldest" },
  { kind: "query", recordOffset: 244, recordCapacity: 12, tokenOffset: 1952, tokenCapacity: 96, placement: "stable_resident", overflow: "drop_oldest" },
  // The ActionIntent catalog is creator-authored, not perceived: the intent
  // selector scores these records, so their count is the ceiling on how many
  // actions a game may declare. It owns a band rather than squatting `focus`
  // slots, which capped the catalog at six without saying so.
  //
  // Appended LAST on purpose. A band's array index is its embedding row and its
  // `candidateBandMask` bit, so inserting one anywhere else renumbers the bands
  // after it and silently changes the meaning of every stored mask.
  { kind: "catalog", recordOffset: 256, recordCapacity: 32, tokenOffset: 2048, tokenCapacity: 256, placement: "fixed", overflow: "error" },
  // Temporal sense. A BrainFrame is a snapshot, so without this band the model
  // has no access to derivatives at all — only `observedAt` and `freshness`,
  // from which it would have to infer motion. The band is its own sense rather
  // than extra fields on `vision` because motion is cross-modal (something can
  // be heard approaching) and because self-motion hangs off no exterior sense.
  //
  // Its records are ordinary binary relations, which is why this falls out of
  // the relation rework for free: APPROACHING(ball, self) carries speed in
  // `intensity`, and because the relation is reified it can itself become the
  // object of an intent — AVOID(self, <that approaching>).
  //
  // Appended after `catalog` for the same reason `catalog` was appended last:
  // a band's array index is its embedding row and its `candidateBandMask` bit.
  { kind: "temporal", recordOffset: 288, recordCapacity: 16, tokenOffset: 2304, tokenCapacity: 128, placement: "shuffled_records", overflow: "truncate_low_salience" },
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

  perceptualFocus: 232,
  thoughtFocus: 233,
  speechTopic: 234,

  primaryQuery: 244,

  /** First slot of the ActionIntent catalog band; entry i lives at +i. */
  catalogBase: 256,
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
  /**
   * The relation is authored as unary: the compiler fills `object` from
   * `subject`. Purely diagnostic — the emitted proposal is indistinguishable
   * from a genuinely reflexive one by its argument pair alone, and this bit is
   * what lets the runtime tell `HURT(self, self)` meaning "I am in pain" from
   * the same pair meaning "I hurt myself".
   */
  canonicallyReflexive: 1 << 8,
} as const;

/**
 * Algebraic properties of a relation token, stored in the HIGH bits of
 * `VocabManifestEntry.flags` (TOKEN_FLAGS owns bits 0..7).
 *
 * These are properties of the relation TYPE, not of an instance, which is why
 * they live in the vocabulary manifest. They buy the runtime cheap deduction
 * that the network then never has to learn:
 *
 *   transitive     INSIDE(apple, basket) + INSIDE(basket, room)
 *                  yields INSIDE(apple, room) as a materialized record.
 *                  Closure MUST skip self-loops R(x, x), which are now the
 *                  normal form of every unary predicate and would otherwise
 *                  flood working memory with derived garbage.
 *   symmetric      NEXT_TO, SIMILAR_TO. Pairs with `inverseToken`.
 *   antisymmetric  Contradiction detection for orderings.
 *   functional     Single-valued: HAS_COLOR, IS_AT. A new instance invalidates
 *                  the previous one without the model learning to overwrite.
 *
 * There is deliberately no `reflexive` flag. Under object-defaults-to-subject
 * every unary predicate is reflexive, so the bit would be true almost
 * everywhere and carry no signal.
 */
export const RELATION_FLAGS = {
  transitive: 1 << 8,
  symmetric: 1 << 9,
  antisymmetric: 1 << 10,
  functional: 1 << 11,
} as const;

/**
 * Monotonicity of a quantifier, in the HIGH bits of `VocabManifestEntry.flags`
 * above RELATION_FLAGS. Entailment direction differs per argument position, so
 * both are recorded: `all` is downward-entailing on its restrictor (all dogs
 * bark => all brown dogs bark) and upward on its scope; `some` is upward on
 * both. This is the quantifier counterpart of the relation algebra — inference
 * the runtime can perform without the network learning it.
 */
export const QUANTIFIER_FLAGS = {
  restrictorUpward: 1 << 12,
  restrictorDownward: 1 << 13,
  scopeUpward: 1 << 14,
  scopeDownward: 1 << 15,
} as const;

/**
 * Discretization thresholds. These live here, in the engine, and not in the
 * simulation that produces the numbers.
 *
 * A band is a token; a token owns an embedding row; the row is trained. If a
 * simulation owned the threshold, moving "near" from 4 to 5 units would keep
 * every symbol and every row identical while silently changing what the trained
 * NEAR vector denotes — training would continue and loss would fall, against a
 * shifted meaning. It is the same failure class as renumbering the manifest.
 * Thresholds belong next to the tokens they define, under the same hash.
 *
 * The second reason is cross-modal consistency: if sight and hearing each
 * banded distance their own way, DIST_NEAR would name two different things and
 * stop being learnable.
 *
 * Note `proportion` has no entry. Quantifier boundaries are logic, not
 * perception — `none` is exactly 0, `all` exactly 1, `most` exactly above a
 * half — so there is nothing to calibrate and approximating them would destroy
 * the operators' inferential force.
 */
export const QUANTITY_BANDS = {
  /** |v| at or below this reads as the zero CATEGORY, not as a small value. */
  signedDeadzone: 0.05,
  /** Magnitude cuts for signed values; matches the comfort encoding. */
  signedMagnitude: [0.25, 0.5, 0.75],
  /** Monotone cuts for 0..1 values. */
  unipolar: [0.25, 0.5, 0.75],
  /**
   * Count cuts. Bottom-heavy on purpose: 1 against 2 against 3 is a large
   * perceptual difference and 47 against 52 is none. The upper cut sits at the
   * subitizing limit, past which a glance yields "many" rather than a number.
   */
  count: [1, 3],
} as const;

/**
 * Flags on `RelationRoleDescriptor.flags`.
 *
 * `acceptsAny` exists because an empty acceptance bitset is ambiguous on its
 * own: it reads identically as "nothing may fill this" and "nothing was
 * narrowed". A world that declines to restrict a role has not thereby
 * forbidden it, and the flag is what carries that difference.
 */
export const RELATION_ROLE_FLAGS = {
  acceptsAny: 1 << 0,
  required: 1 << 1,
} as const;

/** Per-proposal flags on `IntentProposal.flags`. */
export const INTENT_PROPOSAL_FLAGS = {
  /** `object` was filled from `subject` by the unary rule, not selected. */
  objectFromSubject: 1 << 0,
  /** The object head resolved to the `unknown` sentinel; emit a query. */
  objectUnknown: 1 << 1,
  /** The object head resolved to the `something` sentinel (open goal). */
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
    "'system' | 'structure' | 'operation' | 'object' | 'property' | 'quantity' | 'action' | 'reference' | 'relation' | 'logic' | 'temporal' | 'domain' | 'experimental' | 'context'",

  BrainBandKind:
    "'system' | 'homeostasis' | 'body' | 'vision' | 'audio' | 'olfaction' | 'taste' | 'touch' | 'memory' | 'focus' | 'query' | 'catalog' | 'temporal'",

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
    /** Id space of the embedded half. Not the table size — see below. */
    vocabSize: `u32 = ${KRYSTAL_ABI.semanticVocabSize}`,
    /**
     * Live symbols in this manifest. Each owns one embedding row at its
     * manifest index, so this must not exceed `embeddingRows`, and indices
     * must be append-only across manifest versions.
     */
    activeTokenCount: "u32 = 0",
    /** Row capacity of the semantic embedding table. */
    embeddingRows: `u32 = ${KRYSTAL_ABI.semanticEmbeddingRows}`,
    manifestHashLo: "u32 = 0",
    manifestHashHi: "u32 = 0",
    reserved0: "u32 = 0",
    reserved1: "u32 = 0",
  },

  /**
   * Device/compiler representation; human symbol strings stay host-only.
   *
   * `flags` carries TOKEN_FLAGS in bits 0..7 and RELATION_FLAGS above them.
   * `inverseToken` is the argument-swapped relation and is only meaningful now
   * that every relation is binary. `arity` survives as an authoring assertion
   * (0 or 1 means "unary, expect object == subject"); it is no longer a
   * variable shape the runtime has to honour.
   */
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

  /**
   * How a numeric field becomes tokens. The simulation sends exact numbers; the
   * engine discretizes them (see QUANTITY_BANDS for why that direction).
   *
   *   signed      -1..1. Structure sits around zero, and zero is a CATEGORY:
   *               "not moving" is a different percept from "barely approaching"
   *               and "barely receding". Emits TWO tokens, sign then magnitude,
   *               so such a field declares tokenWidth 2. Comfort already uses
   *               this shape: FEEL_BAD/FEEL_GOOD plus MILD/MODERATE/SEVERE.
   *   unipolar    0..1. No zero crossing; zero is an extreme, not a neutral.
   *               Distance normalized by a sense's own range lands here, which
   *               is also the perceptually right framing — what matters is "far
   *               for my eyes", not an absolute length.
   *   count       0..inf, discrete. Subitizing bands.
   *   proportion  0..1, but a fraction OF a reference set, so it is inherently
   *               relational — `MOST(white, sheep)` names both the subset and
   *               the set. Its boundaries are logical rather than perceptual
   *               and its endpoints are exact tests: 0.99 is not `all`.
   *
   * There is deliberately no absolute-scale kind. A unit is a concept, not a
   * value kind: "four kilometres" is a count plus the KILOMETRE symbol, and a
   * creature without that symbol cannot perceive kilometres. Comparative
   * magnitude ("the apple's distance is four times the tree's") is likewise a
   * relation between two quantities, not a scalar. Both are symbolic operations
   * built on percepts rather than percepts themselves — which is also why an
   * exact count is not a sensory value: a glance at a flock yields "many", and
   * producing "47" is counting, not seeing.
   */
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
    /** 2 for a `signed` quantity (sign + magnitude), 1 otherwise. */
    tokenWidth: "u32 = 1",
    roleToken: "KrystalTokenId",
    valueKind: "BrainValueKind",
    /** How to discretize, when valueKind is a scalar/quantity projection. */
    quantityKind: "QuantityKind = 'unipolar'",
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

    // Cheap first derivative, available to every band without a dedicated
    // sense: when this record was previously observed, and how much it changed
    // since. "This thing moved" costs two words here; "what moved, relative to
    // what, how fast" is the temporal band's job.
    previousObservedAt: `u32 = ${INVALID_U32}`,
    changeMagnitude: "f32 = 0",
    reserved0: "u32 = 0",
    reserved1: "u32 = 0",
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
    // Elapsed time since the previous frame, stated rather than inferred.
    // Every rate the temporal sense reports is divided by this, and without it
    // the model would have to recover the timebase from `observedAt` deltas.
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
   * activeRecordCount.
   *
   * `attentionMask` is 1 for a token the model may attend to and 0 otherwise,
   * and consuming it is mandatory, not advisory. A PAD token that still enters
   * the attention softmax is not neutral — it is white noise the network has
   * to spend capacity learning to ignore, and sensory bands are mostly empty
   * in a typical frame (a vision band of 64 slots holding six visible objects
   * is 58 slots of nothing). Masking at the kernel is free; learning around it
   * is not. It is a separate buffer rather than a flags bit so a kernel cannot
   * silently forget to apply it.
   *
   * Note this masks only structural absence. Sensed emptiness and an
   * unavailable sense are real percepts carrying the `void` and `unavailable`
   * sentinel tokens, and their positions stay unmasked.
   */
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
    // Same binary shape as an intent: a query is the relation whose object the
    // model does not yet know, so its object usually carries the `unknown` or
    // `something` sentinel.
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
    relationArity: `u32 = ${BRAIN_LIMITS.relationArity}`,
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
  /**
   * Type constraint on one side of a relation. Inlined into the intent
   * descriptor rather than kept in a side table: with arity frozen at two
   * there is nothing left for `argumentOffset`/`argumentCount` to vary, and
   * the indirection only bought a chance to read the wrong row.
   */
  RelationRoleDescriptor: {
    roleToken: "KrystalTokenId = 0",
    valueKind: "BrainValueKind",
    /**
     * Token ids this role admits, unused entries zero (PAD is never accepted).
     *
     * A SET rather than a single id, because acceptance is genuinely plural:
     * EAT admits anything edible, not one exemplar. Carrying the set here is
     * also what makes the catalog self-contained — the previous single id had
     * to be widened at mask time by looking a capability up by NAME in a
     * host-side table, which tied the forward pass to one particular
     * vocabulary and made any other world's catalog unusable.
     *
     * TOKENS, not record schema ids, and the difference is the whole point.
     * A record has exactly one schema and many tokens, so a schema-keyed
     * acceptance can only ever name individuals: `Apple`, `Berry`, `Bread`,
     * one bit each. What a role usually means is a CLASS — "anything edible" —
     * and a class lives in the record's tokens beside its identity. Matching
     * on tokens is therefore what makes the generalization work at all: a
     * berry the creature has never seen is edible because it carries the
     * category, without anyone extending the catalog. It also removes a
     * silent failure: schema ids are 8 bits, so projecting a 16-bit token
     * into one collided (in a 163-symbol grammar, 30 times), and a collision
     * does not merely fail to match — it matches the WRONG record, emits the
     * intent, and trains on it.
     */
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

    /** Type expected of the subject; was `actorSchemaId`. */
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

  /**
   * `object` omitted means the relation is unary: the compiler sets
   * ACTION_INTENT_FLAGS.canonicallyReflexive and mirrors the subject role.
   */
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

  /**
   * The universal argument: one concept, addressed at both levels at once.
   *
   * `token` is what the model can reason about symbolically; `handle` is the
   * exact runtime identity the simulation acts on. Queries, relations and
   * intents all take their participants as this single type, so there is one
   * notion of "a thing that can stand in a relation" instead of a different
   * argument shape per subsystem.
   */
  ConceptRef: {
    kind: "BrainValueKind",
    token: "KrystalTokenId = 0",
    flags: "u32 = 0",
    reserved0: "u32 = 0",
    handle: "RuntimeRefHandle",
  },

  /**
   * A ConceptRef plus the diagnostics of how the selection heads arrived at
   * it. Kept separate from `ConceptRef` because identity and provenance have
   * different lifetimes and very different sizes: a query stores participants
   * it was handed, and has no gather distribution to record.
   */
  SelectedConceptRef: {
    concept: "ConceptRef",
    selector: "SoftGatherResult",
  },

  IntentLifecycle: "'empty' | 'start' | 'maintain' | 'stop' | 'resume'",
  IntentExecutionStatus:
    "'empty' | 'proposed' | 'accepted' | 'active' | 'succeeded' | 'partial' | 'failed' | 'cancelled' | 'forgotten'",

  /**
   * One desired effect, expressed as a binary relation between two concepts.
   * Not a guaranteed exclusive controller command: multiple proposals may
   * overlap in body/controller preferences, and the motor layer combines them
   * into soft goals that physics resolves.
   *
   * `subject` and `object` are always both populated. For a relation authored
   * as unary the compiler copies `subject` into `object` and sets
   * `INTENT_PROPOSAL_FLAGS.objectFromSubject`, so the object head never faces
   * an absent-argument case and training sees one uniform shape.
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
    // How firmly the network chose, from the selection heads: intent top-1
    // probability × distribution peakedness × argument support. 0.5 is the ABI
    // default for empty slots (no selection exists). This used to be called
    // `intensity`, which was a misnomer — it never measured magnitude.
    commitment: "f32 = 0.5",
    // How much of the relation: how hard to push, how fast to move, how
    // strongly it holds. A learned magnitude head, and the same field the
    // temporal band uses to carry speed on APPROACHING/RECEDING. Deliberately
    // NOT merged with `commitment`: one float cannot carry two gradients
    // without the two objectives fighting each other.
    intensity: "f32 = 0",
    persistence: "f32 = 0",
    confidence: "f32 = 0",

    subject: "SelectedConceptRef",
    object: "SelectedConceptRef",
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
