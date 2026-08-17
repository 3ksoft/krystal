/**
 * S2-S10 grounded-policy curriculum bridge (docs/S2_S10_CURRICULUM_TASK.md).
 *
 * Ownership matches the Pira/Krystal bridge (docs/krystal-sensory-bridge.md):
 * Pira owns complete raw sensory truth; Krystal owns representation. Pira
 * does not yet export S2-S10 scenario streams, so this module builds
 * deterministic synthetic raw-snapshot fixtures in Krystal first, with a
 * narrow adapter surface (RawResource -> record) so a future pira stream can
 * be consumed unchanged.
 *
 *   RawEpisode (stage, seed, frames of RawResource)
 *     -> lowerPolicyFrame(...) -> BrainFrame
 *     -> packBrainFrame -> BrainFrameGpu (shuffled slots, exact 0xExx refs)
 *
 * The lowerer deliberately reuses the Step-1 machinery: the homeostasis
 * signal records, the kaleidoscope noise (same mix32 + coordinate addressing
 * as comfort.ts, so noise is a pure function of the per-band seed) and the
 * query probe. On top it places real sensory records (Apple/Mother/distractors
 * in shuffled vision/memory slots, each carrying an exact runtime-ref
 * sidecar) and the full 6-intent ActionIntent catalog (CRY/LAUGH/EAT/
 * MOVE_TOWARDS/LOOK/WAIT) so the intent selector must pick among a structural
 * legal set, not a curated pair.
 *
 * The gold target for each frame is the exact ResourceRef (never a fixed
 * vocabulary position): the curriculum evaluator compares the emitted
 * proposal's resolved handle to this ref.
 */
import {
  BRAIN_FIXED_RECORDS,
  BRAIN_FRAME_BANDS,
  BRAIN_LIMITS,
  INVALID_U32,
  RECORD_FLAGS,
  TOKEN_FLAGS,
} from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import { PAD_TOKEN_ID } from "../frame/packer.ts";
import { fixtureTokenId } from "../fixtures/vocabulary.ts";
import { ACTION_INTENT_SCHEMA_ID } from "../fixtures/frame.ts";
import { mulberry32 } from "../forward/model.ts";
import {
  COMFORT_NOISE_ALPHABET,
  comfortBandIndex,
  comfortNoiseToken,
  comfortSignToken,
  comfortMagnitudeToken,
  mix32,
} from "./comfort.ts";

// ---------------------------------------------------------------------------
// Raw-snapshot episode model (the adapter surface for future pira streams)
// ---------------------------------------------------------------------------

export type PolicyStage =
  | "S1" // comfort replay (CRY/LAUGH)
  | "S2" // comfort + vision
  | "S3" // multi-frame mother delivery
  | "S4" // reference choice with distractors
  | "S5" // spatial availability / reachability
  | "S6" // active perception (LOOK reveal)
  | "S7" // capability (TARGET_OF(EAT), not Apple identity)
  | "S8" // consequence (poisoned/feces worsen comfort)
  | "S9" // working memory (vision leaves, ref stays)
  | "S10"; // full randomized policy

export type PolicyAction = "CRY" | "LAUGH" | "EAT" | "MOVE_TOWARDS" | "LOOK" | "WAIT";

export type ResourceKind =
  | "apple"
  | "berry"
  | "bread"
  | "mother"
  | "stone"
  | "feces"
  | "unknown";

/** One physical resource in a raw snapshot (Pira-side truth, minimal form). */
export interface RawResource {
  readonly kind: ResourceKind;
  /** Exact runtime reference token (0xE00..0xEFF), unique per episode. */
  readonly refToken: number;
  readonly generation: number;
  readonly band: "vision" | "memory";
  /** Revealed property token symbols (color/size/distance/poison trait). */
  readonly properties: readonly string[];
}

/** One frame of a raw episode: sensory truth + the gold decision. */
export interface PolicyRawFrame {
  readonly tick: number;
  readonly comfort: number;
  readonly resources: readonly RawResource[];
  readonly gold: {
    readonly action: PolicyAction;
    /** Exact gold ref; undefined for arity-0 actions. */
    readonly refToken?: number;
  };
}

/** A deterministic curriculum episode: an ordered sequence of raw frames. */
export interface PolicyEpisode {
  readonly stage: PolicyStage;
  readonly seed: number;
  readonly frames: readonly PolicyRawFrame[];
}

// ---------------------------------------------------------------------------
// Deterministic per-stage generators
// ---------------------------------------------------------------------------

/**
 * Deterministic ref-token stream in the dynamic-context band (0xE00..0xEFF).
 * Train and eval use disjoint sub-bands by construction, so a held-out eval
 * episode can never reuse a train resource id (FOLLOW_UP.md §1): train refs
 * live in 0xE00..0xE9F and eval refs in 0xEA0..0xEDF.
 */
export type RefBand = "train" | "eval";
export function policyRefToken(seed: number, offset: number, band: RefBand = "train"): number {
  const base = band === "eval" ? 0xea0 : 0xe00;
  const span = band === "eval" ? 0x40 : 0xa0;
  return base + ((mix32((seed >>> 0) + offset * 0x9e3779b1) % span) >>> 0);
}

function resource(kind: ResourceKind, refToken: number, band: "vision" | "memory", properties: readonly string[] = []): RawResource {
  return { kind, refToken, generation: 1, band, properties };
}

const NEAR: readonly string[] = ["NEAR"];
const FAR: readonly string[] = ["FAR"];
const APPLE_PROPS: readonly string[] = ["RED", "SMALL"];

/**
 * Build one episode for a stage from a seed. All randomness is a pure
 * function of (stage, seed), so episodes replay exactly and train/eval seeds
 * give disjoint resource ids and record orders.
 */
export function generatePolicyEpisode(stage: PolicyStage, seed: number, band: RefBand = "train"): PolicyEpisode {
  switch (stage) {
    case "S1": return s1Episode(seed, band);
    case "S2": return s2Episode(seed, band);
    case "S3": return s3Episode(seed, band);
    case "S4": return s4Episode(seed, band);
    case "S5": return s5Episode(seed, band);
    case "S6": return s6Episode(seed, band);
    case "S7": return s7Episode(seed, band);
    case "S8": return s8Episode(seed, band);
    case "S9": return s9Episode(seed, band);
    case "S10": return s10Episode(seed, band);
  }
}

/** S1 replay: comfort-only CRY/LAUGH (the regression proof, policy frame form). */
function s1Episode(seed: number, _band: RefBand = "train"): PolicyEpisode {
  const bad = seed % 2 === 0;
  return {
    stage: "S1",
    seed,
    frames: [{
      tick: 10,
      comfort: bad ? -1 : 1,
      resources: [],
      gold: { action: bad ? "CRY" : "LAUGH" },
    }],
  };
}

/** S2: bad + visible Apple -> EAT(Apple); bad + no Apple -> CRY; good -> LAUGH. */
function s2Episode(seed: number, band: RefBand = "train"): PolicyEpisode {
  // The variant is decoupled from the curriculum stage pick (which uses
  // `seed % stages.length`): an independently salted hash, so every S2
  // variant can occur regardless of which stage bucket the seed lands in
  // (FOLLOW_UP.md §1 — the old `seed % 3` made the mixture always draw
  // variant 0, so "bad + no Apple -> CRY" and "good -> LAUGH" never trained).
  // Keep the stream independent from the stage bucket while also avoiding a
  // severe small-split skew in the canonical M-A train/eval ranges. The
  // audited canonical ranges contain 3/3/3 train variants and 4/4/3 held-out
  // variants instead of the former 1/3/5 vs 7/3/1 skew.
  const variant = (mix32((seed >>> 0) ^ 0x54) % 3) >>> 0;
  const appleRef = policyRefToken(seed, 0, band);
  if (variant === 0) {
    return {
      stage: "S2", seed,
      frames: [{ tick: 10, comfort: -1, resources: [resource("apple", appleRef, "vision", APPLE_PROPS)], gold: { action: "EAT", refToken: appleRef } }],
    };
  }
  if (variant === 1) {
    return {
      stage: "S2", seed,
      frames: [{ tick: 10, comfort: -1, resources: [], gold: { action: "CRY" } }],
    };
  }
  return {
    stage: "S2", seed,
    frames: [{ tick: 10, comfort: 1, resources: [resource("apple", appleRef, "vision", APPLE_PROPS)], gold: { action: "LAUGH" } }],
  };
}

/** S3: two frames — CRY with no Apple, then Mother delivers -> EAT(delivered ref). */
function s3Episode(seed: number, band: RefBand = "train"): PolicyEpisode {
  const motherRef = policyRefToken(seed, 0, band);
  const deliveredRef = policyRefToken(seed, 1, band);
  return {
    stage: "S3", seed,
    frames: [
      { tick: 10, comfort: -1, resources: [resource("mother", motherRef, "vision")], gold: { action: "CRY" } },
      { tick: 11, comfort: -1, resources: [resource("mother", motherRef, "vision"), resource("apple", deliveredRef, "vision", APPLE_PROPS)], gold: { action: "EAT", refToken: deliveredRef } },
    ],
  };
}

/** S4: Apple + Mother + distractors, permuted order/ids -> EAT(AppleRef). */
function s4Episode(seed: number, band: RefBand = "train"): PolicyEpisode {
  const appleRef = policyRefToken(seed, 0, band);
  const motherRef = policyRefToken(seed, 1, band);
  const stoneRef = policyRefToken(seed, 2, band);
  const fecesRef = policyRefToken(seed, 3, band);
  // Deterministic slot order permutation: the apple is never positionally fixed.
  const rng = mulberry32(seed * 7 + 1);
  const pool: RawResource[] = [
    resource("apple", appleRef, "vision", APPLE_PROPS),
    resource("mother", motherRef, "vision"),
    resource("stone", stoneRef, "vision"),
    resource("feces", fecesRef, "vision"),
  ];
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [pool[i], pool[j]] = [pool[j]!, pool[i]!];
  }
  return {
    stage: "S4", seed,
    frames: [{ tick: 10, comfort: -1, resources: pool, gold: { action: "EAT", refToken: appleRef } }],
  };
}

/** S5: far Apple -> MOVE_TOWARDS(ref); after transition (reachable) -> EAT(same ref). */
function s5Episode(seed: number, band: RefBand = "train"): PolicyEpisode {
  const appleRef = policyRefToken(seed, 0, band);
  return {
    stage: "S5", seed,
    frames: [
      { tick: 10, comfort: -1, resources: [resource("apple", appleRef, "vision", [...APPLE_PROPS, ...FAR])], gold: { action: "MOVE_TOWARDS", refToken: appleRef } },
      { tick: 11, comfort: -1, resources: [resource("apple", appleRef, "vision", [...APPLE_PROPS, ...NEAR])], gold: { action: "EAT", refToken: appleRef } },
    ],
  };
}

/** S6: unknown record -> LOOK(ref); after a deterministic reveal -> EAT(ref) or reject. */
function s6Episode(seed: number, band: RefBand = "train"): PolicyEpisode {
  const unknownRef = policyRefToken(seed, 0, band);
  // Stage selection for a two-stage milestone currently uses seed parity.
  // Salt and hash the reveal outcome so choosing S6 cannot collapse the
  // curriculum to only one branch (the old even/odd rule made every sampled
  // S6 episode take the reject path in M-B).
  const revealed = (mix32((seed >>> 0) ^ 0x53) & 1) === 0;
  return {
    stage: "S6", seed,
    frames: [
      { tick: 10, comfort: -1, resources: [resource("unknown", unknownRef, "vision")], gold: { action: "LOOK", refToken: unknownRef } },
      {
        tick: 11, comfort: -1,
        resources: revealed
          ? [resource("apple", unknownRef, "vision", APPLE_PROPS)]
          // A rejected zoom must still reveal new evidence. Keeping the
          // record byte-for-byte "unknown" made frame 0 (LOOK) and frame 1
          // (CRY) observationally identical and therefore unlearnable.
          : [resource("stone", unknownRef, "vision", ["SHINY"])],
        gold: revealed ? { action: "EAT", refToken: unknownRef } : { action: "CRY" },
      },
    ],
  };
}

/** S7: any edible (Apple/Berry/Bread) via TARGET_OF(EAT); Stone/Feces negative. */
function s7Episode(seed: number, band: RefBand = "train"): PolicyEpisode {
  const edible: ResourceKind[] = ["apple", "berry", "bread"];
  const negative: ResourceKind[] = ["stone", "feces"];
  const rng = mulberry32(seed * 13 + 1);
  const pick = edible[Math.floor(rng() * edible.length)]!;
  const neg = negative[Math.floor(rng() * negative.length)]!;
  const edibleRef = policyRefToken(seed, 0, band);
  const negRef = policyRefToken(seed, 1, band);
  return {
    stage: "S7", seed,
    frames: [{
      tick: 10, comfort: -1,
      resources: [resource(pick, edibleRef, "vision", APPLE_PROPS), resource(neg, negRef, "vision")],
      gold: { action: "EAT", refToken: edibleRef },
    }],
  };
}

/** S8: consequence — ordinary food improves comfort; poisoned/feces worsen it. */
function s8Episode(seed: number, band: RefBand = "train"): PolicyEpisode {
  // Keep the consequence independent from the curriculum's stage bucket.
  // With S7+S8 selected by seed parity, the old `seed % 2` outcome collapsed
  // every sampled S8 episode to the same branch.
  const bad = (mix32((seed >>> 0) ^ 0x6d2b79f5) & 1) === 0;
  const foods: readonly ResourceKind[] = ["apple", "berry", "bread"];
  const food = foods[mix32((seed >>> 0) ^ 0x8f1bbcdc) % foods.length]!;
  const foodRef = policyRefToken(seed, 0, band);
  const secondComfort = bad ? -1 : 1;
  return {
    stage: "S8", seed,
    frames: [
      // Before consumption no negative consequence is known, so exploration
      // is consistently labelled EAT. This never conflicts with the known-
      // POISONED adversarial frame, whose correct action is CRY.
      { tick: 10, comfort: -1, resources: [resource(food, foodRef, "vision", APPLE_PROPS)], gold: { action: "EAT", refToken: foodRef } },
      {
        tick: 11,
        comfort: secondComfort,
        resources: [resource(food, foodRef, "vision", bad ? [...APPLE_PROPS, "POISONED"] : APPLE_PROPS)],
        gold: bad ? { action: "CRY" } : { action: "LAUGH" },
      },
    ],
  };
}

/** S9: an observed target leaves vision while its exact ref stays in memory. */
function s9Episode(seed: number, band: RefBand = "train"): PolicyEpisode {
  const appleRef = policyRefToken(seed, 0, band);
  const far = seed % 2 === 0;
  return {
    stage: "S9", seed,
    frames: [
      {
        tick: 10, comfort: -1,
        resources: [resource("apple", appleRef, "vision", [...APPLE_PROPS, ...(far ? FAR : NEAR)])],
        gold: far ? { action: "MOVE_TOWARDS", refToken: appleRef } : { action: "EAT", refToken: appleRef },
      },
      {
        tick: 11, comfort: -1,
        // Vision drops the apple; memory keeps the exact ref.
        resources: [resource("apple", appleRef, "memory", [...APPLE_PROPS, ...(far ? FAR : NEAR)])],
        gold: far ? { action: "MOVE_TOWARDS", refToken: appleRef } : { action: "EAT", refToken: appleRef },
      },
    ],
  };
}

/** S10: randomized full policy — layouts, distance, distractors, noise. */
function s10Episode(seed: number, band: RefBand = "train"): PolicyEpisode {
  const rng = mulberry32(seed * 17 + 3);
  const bad = rng() < 0.6;
  const appleRef = policyRefToken(seed, 0, band);
  const distractorRef = policyRefToken(seed, 1, band);
  const far = rng() < 0.5;
  const withDistractor = rng() < 0.5;
  const resources: RawResource[] = [];
  if (bad) resources.push(resource("apple", appleRef, "vision", [...APPLE_PROPS, ...(far ? FAR : NEAR)]));
  if (withDistractor) resources.push(resource("mother", distractorRef, "vision"));
  const action: PolicyAction = !bad ? "LAUGH" : far ? "MOVE_TOWARDS" : "EAT";
  return {
    stage: "S10", seed,
    frames: [{ tick: 10, comfort: bad ? -1 : 1, resources, gold: { action, refToken: action === "EAT" || action === "MOVE_TOWARDS" ? appleRef : undefined } }],
  };
}

// ---------------------------------------------------------------------------
// Record lowering
// ---------------------------------------------------------------------------

/** Map a raw resource kind to the fixture record-schema id. */
export function resourceSchemaId(kind: ResourceKind): number {
  switch (kind) {
    case "apple": return 2; // Apple
    case "berry": return 5; // Berry
    case "bread": return 6; // Bread
    case "mother": return 7; // Mother
    case "stone": return 8; // Stone
    case "feces": return 9; // Feces
    case "unknown": return 10; // UnknownObject
  }
}

function tokenMeta(roleToken: number, flags: number = 0): v1_0_0.BrainTokenMeta {
  return { fieldId: 0, roleToken, flags, referenceBinding: INVALID_U32 };
}

function refBinding(localTokenIndex: number, tokenId: number, generation: number): v1_0_0.BrainReferenceBinding {
  return {
    localTokenIndex,
    fieldId: 0,
    flags: 0,
    reserved0: 0,
    handle: { tokenId, generation, kind: "entity", status: "live" },
  };
}

interface PolicyRecordSpec {
  readonly slot: number;
  readonly band: v1_0_0.BrainBandKind;
  readonly schemaId: number;
  readonly tokens: readonly number[];
  readonly roleTokens: readonly number[];
  readonly refs?: readonly v1_0_0.BrainReferenceBinding[];
  readonly source: v1_0_0.RecordSource;
  readonly flags?: number;
}

export interface PolicyLowerOptions {
  /** Noise per band (0 = none). Default: every leftover slot. */
  readonly noisePerBand?: number;
  /** Skip the kaleidoscope noise entirely. */
  readonly ablateNoise?: boolean;
  /** Deterministic slot-order permutation seed (default: the episode seed). */
  readonly shuffleSeed?: number;
}

/** Band capacity of the vision/memory bands (where real records live). */
const VISION_BAND = BRAIN_FRAME_BANDS.find((band) => band.kind === "vision")!;
const MEMORY_BAND = BRAIN_FRAME_BANDS.find((band) => band.kind === "memory")!;

/** Deterministically permute `count` slots inside [start, start+capacity). */
function shuffledSlots(start: number, capacity: number, count: number, seed: number): number[] {
  const rng = mulberry32(seed);
  const slots: number[] = [];
  const pool = Array.from({ length: capacity }, (_, i) => start + i);
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [pool[i], pool[j]] = [pool[j]!, pool[i]!];
  }
  for (let i = 0; i < count && i < pool.length; i++) slots.push(pool[i]!);
  return slots.sort((a, b) => a - b);
}

/**
 * Build the record specs for one policy frame: signal records (homeostasis
 * summary + the 6-intent catalog + query) plus real resource records in
 * shuffled vision/memory slots plus kaleidoscope noise in every leftover
 * sensory slot.
 */
export function buildPolicyRecords(frame: PolicyRawFrame, episode: PolicyEpisode, options: PolicyLowerOptions): PolicyRecordSpec[] {
  const specs: PolicyRecordSpec[] = [];
  const recordWidth = BRAIN_LIMITS.recordWidth;
  const noisePerBand = options.noisePerBand ?? Number.POSITIVE_INFINITY;
  const shuffleSeed = options.shuffleSeed ?? episode.seed;

  // Homeostasis signal (identical machinery to the Step-1 lowerer).
  const signalTokens: readonly number[] = [
    fixtureTokenId("COMFORT"),
    comfortSignToken(frame.comfort),
    comfortMagnitudeToken(frame.comfort),
  ];
  specs.push({
    slot: BRAIN_FIXED_RECORDS.homeostasisSummary,
    band: "homeostasis",
    schemaId: 3, // HomeostasisQuery
    tokens: signalTokens,
    roleTokens: signalTokens.map((token) => token),
    source: "homeostasis",
    flags: RECORD_FLAGS.occupied | RECORD_FLAGS.fixed,
  });

  // The full 6-intent ActionIntent catalog (focus band, fixed roles).
  const catalogActions: readonly PolicyAction[] = ["LOOK", "EAT", "MOVE_TOWARDS", "WAIT", "CRY", "LAUGH"];
  for (let i = 0; i < catalogActions.length; i++) {
    const token = fixtureTokenId(catalogActions[i]!);
    specs.push({
      slot: BRAIN_FIXED_RECORDS.perceptualFocus + i,
      band: "focus",
      schemaId: ACTION_INTENT_SCHEMA_ID,
      tokens: [token, ...new Array<number>(recordWidth - 1).fill(PAD_TOKEN_ID)],
      roleTokens: [token, ...new Array<number>(recordWidth - 1).fill(0)],
      source: "creator",
      flags: RECORD_FLAGS.occupied | RECORD_FLAGS.fixed | RECORD_FLAGS.creatorAuthored,
    });
  }

  // Real sensory records in shuffled slots (deterministic per episode seed).
  const visionResources = frame.resources.filter((r) => r.band === "vision");
  const memoryResources = frame.resources.filter((r) => r.band === "memory");
  const visionSlots = shuffledSlots(VISION_BAND.recordOffset, VISION_BAND.recordCapacity, visionResources.length, shuffleSeed);
  const memorySlots = shuffledSlots(MEMORY_BAND.recordOffset, MEMORY_BAND.recordCapacity, memoryResources.length, shuffleSeed);
  for (let i = 0; i < visionResources.length; i++) {
    specs.push(resourceRecord(visionResources[i]!, visionSlots[i]!, "vision"));
  }
  for (let i = 0; i < memoryResources.length; i++) {
    specs.push(resourceRecord(memoryResources[i]!, memorySlots[i]!, "memory"));
  }

  // Kaleidoscope noise in every leftover sensory band slot (incl. leftover
  // vision/memory slots after the real records), deterministic per band seed.
  if (!options.ablateNoise) {
    const occupied = new Set(specs.map((spec) => spec.slot));
    for (const band of BRAIN_FRAME_BANDS) {
      if (band.kind === "system" || band.kind === "focus" || band.kind === "query" || band.kind === "homeostasis") continue;
      const bandIndexValue = comfortBandIndex(band.kind);
      const bandSeed = mix32((episode.seed >>> 0) ^ (bandIndexValue + 1) * 0x9e3779b1) >>> 0;
      let emitted = 0;
      for (let slot = band.recordOffset; slot < band.recordOffset + band.recordCapacity; slot++) {
        if (occupied.has(slot)) continue;
        if (emitted >= noisePerBand) break;
        // Noise is observable background, never edible: VisionObject for the
        // sensory bands and MemoryObject for the memory band, so an EAT
        // argument mask can never resolve to a noise record.
        const schemaId = band.kind === "memory" ? 4 : 1;
        const tokens = Array.from({ length: recordWidth }, (_, local) =>
          comfortNoiseToken(COMFORT_NOISE_ALPHABET, bandSeed, bandIndexValue, slot, local),
        );
        specs.push({
          slot,
          band: band.kind,
          schemaId,
          tokens,
          roleTokens: new Array<number>(recordWidth).fill(0),
          source: "sensor",
        });
        emitted++;
      }
    }
  }

  // The active query probe (decision-head query row).
  specs.push({
    slot: BRAIN_FIXED_RECORDS.primaryQuery,
    band: "query",
    schemaId: 3,
    tokens: signalTokens,
    roleTokens: signalTokens.map((token) => token),
    source: "query",
    flags: RECORD_FLAGS.occupied | RECORD_FLAGS.fixed | RECORD_FLAGS.query,
  });

  return specs;
}

function resourceRecord(r: RawResource, slot: number, band: "vision" | "memory"): PolicyRecordSpec {
  const recordWidth = BRAIN_LIMITS.recordWidth;
  const familyToken = fixtureTokenId(resourceFamilySymbol(r.kind));
  const tokens = [familyToken, r.refToken, ...r.properties.map((p) => fixtureTokenId(p))];
  const refs = [refBinding(1, r.refToken, r.generation)];
  const padded = [...tokens, ...new Array<number>(Math.max(0, recordWidth - tokens.length)).fill(PAD_TOKEN_ID)];
  return {
    slot,
    band,
    schemaId: resourceSchemaId(r.kind),
    tokens: padded,
    roleTokens: padded.map((token) => (token === PAD_TOKEN_ID ? 0 : token)),
    refs,
    source: "sensor",
    flags: band === "memory" ? RECORD_FLAGS.occupied | RECORD_FLAGS.remembered : RECORD_FLAGS.occupied,
  };
}

function resourceFamilySymbol(kind: ResourceKind): string {
  switch (kind) {
    case "apple": return "APPLE";
    case "berry": return "BERRY";
    case "bread": return "BREAD";
    case "mother": return "MOTHER";
    case "stone": return "STONE";
    case "feces": return "FECES";
    case "unknown": return "UNKNOWN";
  }
}

/** Lower one raw frame to a canonical AoS BrainFrame (with noise + shuffle). */
export function lowerPolicyFrame(
  frame: PolicyRawFrame,
  episode: PolicyEpisode,
  options: PolicyLowerOptions = {},
): v1_0_0.BrainFrame {
  const recordWidth = BRAIN_LIMITS.recordWidth;
  const maxReferencesPerRecord = BRAIN_LIMITS.maxReferencesPerRecord;
  const records: v1_0_0.BrainRecordSlot[] = [];
  for (let slot = 0; slot < BRAIN_LIMITS.frameRecordSlots; slot++) {
    records.push({
      header: {
        schemaId: 0, band: "system", source: "runtime", flags: 0,
        tokenCount: 0, referenceCount: 0, observedAt: 0, revision: 0,
        primaryReference: INVALID_U32, continuationRecord: INVALID_U32,
        salience: 0, freshness: 0,
      },
      tokens: new Array<number>(recordWidth).fill(PAD_TOKEN_ID),
      tokenMeta: new Array<v1_0_0.BrainTokenMeta>(recordWidth).fill(tokenMeta(0, TOKEN_FLAGS.padding)),
      references: new Array<v1_0_0.BrainReferenceBinding>(maxReferencesPerRecord).fill(refBinding(INVALID_U32, 0, 0)),
    });
  }

  let activeRecordCount = 0;
  let activeTokenCount = 0;
  const specs = buildPolicyRecords(frame, episode, options);
  for (const spec of specs) {
    const record = records[spec.slot]!;
    record.header = {
      schemaId: spec.schemaId,
      band: spec.band,
      source: spec.source,
      flags: spec.flags ?? RECORD_FLAGS.occupied,
      tokenCount: spec.tokens.filter((token) => token !== PAD_TOKEN_ID).length,
      referenceCount: spec.refs?.length ?? 0,
      observedAt: frame.tick,
      revision: 1,
      primaryReference: spec.refs?.[0]?.handle.tokenId ?? INVALID_U32,
      continuationRecord: INVALID_U32,
      salience: 0.5,
      freshness: 1.0,
    };
    record.tokens = spec.tokens.slice();
    record.tokenMeta = spec.tokens.map((token, localToken) =>
      token === PAD_TOKEN_ID
        ? tokenMeta(0, TOKEN_FLAGS.padding)
        : tokenMeta(spec.roleTokens[localToken] ?? 0, TOKEN_FLAGS.structural),
    );
    record.references = spec.refs?.slice() ?? record.references;
    activeRecordCount++;
    activeTokenCount += record.header.tokenCount;
  }

  const bands = BRAIN_FRAME_BANDS.map((band) => {
    const active = specs.filter((spec) => spec.band === band.kind);
    return {
      kind: band.kind,
      activeRecords: active.length,
      activeTokens: active.reduce((sum, spec) => sum + spec.tokens.filter((t) => t !== PAD_TOKEN_ID).length, 0),
      overflowRecords: 0,
      truncatedRecords: 0,
      revision: 0,
      flags: 0,
      reserved0: 0,
    } satisfies v1_0_0.BrainBandState;
  });

  return {
    header: {
      tokenAbiVersion: 0,
      architectureVersion: 2,
      layoutVersion: 1,
      tick: frame.tick,
      snapshot: 1,
      activeRecordCount,
      activeTokenCount,
      activeQueryRecord: BRAIN_FIXED_RECORDS.primaryQuery,
      actorRecord: BRAIN_FIXED_RECORDS.actor,
      frameRevision: 1,
      memoryRevision: 0,
      intentRevision: 0,
      flags: 0,
    },
    bands,
    records,
  };
}
