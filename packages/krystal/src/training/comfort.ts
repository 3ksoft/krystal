/**
 * Pira → Krystal sensory compiler for TRAINING.md Step 1 (Comfort).
 *
 * Consumes the `comfort-episodes@1` JSON artifact exported by the pira CLI
 * (`pira episodes`): semantic CRY/LAUGH counterfactual pairs per noise seed.
 * The artifact is deliberately pre-frame: per the Pira/Krystal bridge
 * (docs/krystal-sensory-bridge.md ownership split) Pira owns the scenario and
 * Krystal owns representation. This module is the Krystal-side lowerer:
 *
 *   episode -> homeostasis signal record (COMFORT channel + signed valence +
 *              magnitude projection) + CRY/LAUGH ActionIntent catalog records
 *              + the homeostasis query record + kaleidoscope noise records in
 *              every other sensory band slot.
 *
 * Noise is a pure function of (per-band seed, band, record offset, local
 * token offset), so the two members of a counterfactual pair (identical
 * noiseSeed) receive byte-identical noise and only the homeostasis signal
 * differs. `mix32` and the coordinate addressing mirror pira's
 * `kaleidoscopeToken` exactly so both sides agree on the algorithm; the legal
 * alphabet is Krystal's choice (context/experimental ranges, decision: small
 * step-1 noise alphabet rather than PAD).
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
import type { ActiveFrame } from "../forward/masks.ts";

// ---------------------------------------------------------------------------
// Wire format: pira `comfort-episodes@1` export
// ---------------------------------------------------------------------------

export type ComfortStage = "1A-extremes" | "1B-scale";
export type ComfortAction = "CRY" | "LAUGH";

export interface ComfortNoiseField {
  readonly band: string;
  readonly seed: number;
}

export interface ComfortEpisode {
  readonly curriculumVersion: "comfort-v1";
  readonly stage: ComfortStage;
  readonly noiseSeed: number;
  readonly comfort: number;
  readonly activeBand: "homeostasis";
  readonly inactiveNoise: readonly ComfortNoiseField[];
  readonly creator: {
    readonly source: "creator";
    readonly trusted: true;
    readonly directive: "RESPOND_TO_HOMEOSTASIS";
  };
  readonly target: { readonly action: ComfortAction; readonly arguments: readonly [] };
}

export interface ComfortEpisodesArtifact {
  readonly contract: "comfort-episodes@1";
  readonly curriculumVersion: "comfort-v1";
  readonly stages: readonly ComfortStage[];
  readonly seedCount: number;
  readonly episodes: readonly ComfortEpisode[];
}

// ---------------------------------------------------------------------------
// Kaleidoscope noise (mirrors pira's mix32 + coordinate addressing)
// ---------------------------------------------------------------------------

/** Step-1 legal noise alphabet: context/experimental tokens (never PAD). */
export const COMFORT_NOISE_ALPHABET: readonly number[] = [
  0x8001, 0x8002, 0x8003, 0x8004, 0x8005, 0x8006, 0x8007, 0x8008,
  0xf01, 0xf02, 0xf03, 0xf04, 0xf05, 0xf06, 0xf07, 0xf08,
];

/** Counter-based 32-bit mixer (identical to pira's). */
export function mix32(value: number): number {
  let x = value >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}

/**
 * One deterministic noise token for one eventual frame coordinate. A pure
 * function of (bandSeed, band, recordOffset, localTokenOffset), so identical
 * seeds yield identical noise regardless of the episode's comfort/target.
 */
export function comfortNoiseToken(
  alphabet: readonly number[],
  bandSeed: number,
  bandIndexValue: number,
  recordOffset: number,
  localTokenOffset: number,
): number {
  const coordinate =
    (Math.imul(bandIndexValue + 1, 0x9e3779b1) ^
      Math.imul(recordOffset + 1, 0x85ebca6b) ^
      Math.imul(localTokenOffset + 1, 0xc2b2ae35)) >>>
    0;
  return alphabet[mix32((bandSeed >>> 0) ^ coordinate) % alphabet.length]!;
}

// ---------------------------------------------------------------------------
// Signal lowering
// ---------------------------------------------------------------------------

/** Route-kind classes for the decision head (config.routeKindCount = 2). */
export const COMFORT_ROUTE_CRY = 0;
export const COMFORT_ROUTE_LAUGH = 1;

export function comfortRouteKind(episode: ComfortEpisode): number {
  return episode.target.action === "CRY" ? COMFORT_ROUTE_CRY : COMFORT_ROUTE_LAUGH;
}

export function comfortSignToken(comfort: number): number {
  return comfort < 0 ? fixtureTokenId("FEEL_BAD") : fixtureTokenId("FEEL_GOOD");
}

/**
 * QuantityProjectionProfile for comfort magnitude (4 bands, bridge doc):
 * neutral (|c| <= 0.25, not emitted in Step 1), MILD, MODERATE, SEVERE.
 */
export function comfortMagnitudeToken(comfort: number): number {
  const magnitude = Math.abs(comfort);
  if (!(magnitude > 0.25 && magnitude <= 1)) {
    throw new Error(
      `Comfort magnitude ${magnitude} outside Step 1 training range (0.25, 1]; the neutral region is deferred`,
    );
  }
  if (magnitude <= 0.5) return fixtureTokenId("MILD");
  if (magnitude <= 0.75) return fixtureTokenId("MODERATE");
  return fixtureTokenId("SEVERE");
}

/** Per-band kaleidoscope seeds from the episode (band name -> seed). */
export function comfortBandSeeds(episode: ComfortEpisode): ReadonlyMap<string, number> {
  return new Map(episode.inactiveNoise.map((field) => [field.band, field.seed]));
}

/** Band index of one band name in the frozen frame layout. */
export function comfortBandIndex(band: string): number {
  const entry = BRAIN_FRAME_BANDS.find((candidate) => candidate.kind === band);
  if (!entry) throw new Error(`Unknown frame band: ${band}`);
  return BRAIN_FRAME_BANDS.indexOf(entry);
}

// ---------------------------------------------------------------------------
// Frame lowering
// ---------------------------------------------------------------------------

export interface ComfortLowerOptions {
  /**
   * Max noise records per band (0 = no noise). Default: half the band's
   * capacity, so enlarging a band adds room for real records rather than more
   * distractors (see the same budget in training/policy.ts).
   */
  readonly noisePerBand?: number;
  /**
   * Comfort ablation: drop the homeostasisSummary record and strip the query
   * record's valence/magnitude tokens (channel only). Noise is untouched, so
   * an accurate model must drop to chance.
   */
  readonly ablateComfort?: boolean;
  /** Noise ablation: emit no noise records at all. Signal untouched. */
  readonly ablateNoise?: boolean;
}

/** Record schema ids used for noise records per sensory band. */
export const COMFORT_NOISE_SCHEMA_BY_BAND: Readonly<Record<string, number>> = {
  body: 0, // Self
  vision: 1, // VisionObject
  audio: 1, // VisionObject
  olfaction: 1, // VisionObject
  taste: 2, // Apple (food-adjacent taste channel)
  touch: 1, // VisionObject
  memory: 4, // MemoryObject
};

function tokenMeta(roleToken: number, flags: number = 0): v1_0_0.BrainTokenMeta {
  return { fieldId: 0, roleToken, flags, referenceBinding: INVALID_U32 };
}

function refBinding(localTokenIndex: number, tokenId: number): v1_0_0.BrainReferenceBinding {
  return {
    localTokenIndex,
    fieldId: 0,
    flags: 0,
    reserved0: 0,
    handle: { tokenId, generation: 0, kind: "entity", status: "live" },
  };
}

interface ComfortRecordSpec {
  readonly slot: number;
  readonly band: v1_0_0.BrainBandKind;
  readonly schemaId: number;
  readonly tokens: readonly number[];
  readonly roleTokens: readonly number[];
  readonly source: v1_0_0.RecordSource;
  readonly flags?: number;
}

/**
 * Build the record specs for one comfort episode: signal records (fixed
 * homeostasisSummary + the two catalog records + the query record) plus
 * kaleidoscope noise records in every other sensory slot.
 */
function buildComfortRecords(episode: ComfortEpisode, options: ComfortLowerOptions): ComfortRecordSpec[] {
  const specs: ComfortRecordSpec[] = [];
  const recordWidth = BRAIN_LIMITS.recordWidth;
  const noiseBudget = (capacity: number): number =>
    options.noisePerBand ?? Math.floor(capacity / 2);
  const bandSeeds = comfortBandSeeds(episode);

  const signalTokens = (ablate: boolean): readonly number[] =>
    ablate
      ? [fixtureTokenId("COMFORT"), PAD_TOKEN_ID, PAD_TOKEN_ID]
      : [
          fixtureTokenId("COMFORT"),
          comfortSignToken(episode.comfort),
          comfortMagnitudeToken(episode.comfort),
        ];
  const signalRoles = (ablate: boolean): readonly number[] =>
    ablate ? [fixtureTokenId("COMFORT"), 0, 0] : signalTokens(false).map((token) => token);

  if (!options.ablateComfort) {
    specs.push({
      slot: BRAIN_FIXED_RECORDS.homeostasisSummary,
      band: "homeostasis",
      schemaId: 3, // HomeostasisQuery
      tokens: signalTokens(false),
      roleTokens: signalRoles(false),
      source: "homeostasis",
      flags: RECORD_FLAGS.occupied | RECORD_FLAGS.fixed,
    });
  }

  // CRY / LAUGH ActionIntent catalog records (catalog band fixed roles). The
  // intent selector must choose between exactly these two.
  const catalog: readonly ComfortAction[] = ["CRY", "LAUGH"];
  for (let i = 0; i < catalog.length; i++) {
    const action = catalog[i]!;
    const token = fixtureTokenId(action);
    specs.push({
      slot: BRAIN_FIXED_RECORDS.catalogBase + i,
      band: "catalog",
      schemaId: ACTION_INTENT_SCHEMA_ID,
      tokens: [token, ...new Array<number>(recordWidth - 1).fill(PAD_TOKEN_ID)],
      roleTokens: [token, ...new Array<number>(recordWidth - 1).fill(0)],
      source: "creator",
      flags: RECORD_FLAGS.occupied | RECORD_FLAGS.fixed | RECORD_FLAGS.creatorAuthored,
    });
  }

  // The active homeostasis query probe (decision-head query row).
  specs.push({
    slot: BRAIN_FIXED_RECORDS.primaryQuery,
    band: "query",
    schemaId: 3, // HomeostasisQuery
    tokens: signalTokens(options.ablateComfort ?? false),
    roleTokens: signalRoles(options.ablateComfort ?? false),
    source: "query",
    flags: RECORD_FLAGS.occupied | RECORD_FLAGS.fixed | RECORD_FLAGS.query,
  });

  // Kaleidoscope noise in every other sensory band slot.
  if (!options.ablateNoise) {
    for (const [band, schemaId] of Object.entries(COMFORT_NOISE_SCHEMA_BY_BAND)) {
      const bandDef = BRAIN_FRAME_BANDS.find((candidate) => candidate.kind === band)!;
      const bandIndexValue = comfortBandIndex(band);
      const bandSeed = bandSeeds.get(band) ?? 0;
      const budget = noiseBudget(bandDef.recordCapacity);
      let emitted = 0;
      for (let slot = bandDef.recordOffset; slot < bandDef.recordOffset + bandDef.recordCapacity; slot++) {
        if (emitted >= budget) break;
        const tokens = Array.from({ length: recordWidth }, (_, local) =>
          comfortNoiseToken(COMFORT_NOISE_ALPHABET, bandSeed, bandIndexValue, slot, local),
        );
        specs.push({
          slot,
          band: band as v1_0_0.BrainBandKind,
          schemaId,
          tokens,
          roleTokens: new Array<number>(recordWidth).fill(0),
          source: "sensor",
        });
        emitted++;
      }
    }
  }

  return specs;
}

/**
 * Lower one comfort episode to a canonical AoS BrainFrame. The noise records
 * make the frame look like full sensory background: occupied records with
 * legal (context/experimental) tokens, identical across the counterfactual
 * pair by construction.
 */
export function lowerComfortEpisode(
  episode: ComfortEpisode,
  options: ComfortLowerOptions = {},
): v1_0_0.BrainFrame {
  const recordWidth = BRAIN_LIMITS.recordWidth;
  const maxReferencesPerRecord = BRAIN_LIMITS.maxReferencesPerRecord;
  const records: v1_0_0.BrainRecordSlot[] = [];
  for (let slot = 0; slot < BRAIN_LIMITS.frameRecordSlots; slot++) {
    records.push({
      header: {
        schemaId: 0,
        band: "system",
        source: "runtime",
        flags: 0,
        tokenCount: 0,
        referenceCount: 0,
        observedAt: 0,
        revision: 0,
        primaryReference: INVALID_U32,
        continuationRecord: INVALID_U32,
        salience: 0,
        freshness: 0,
        previousObservedAt: INVALID_U32,
        changeMagnitude: 0,
        reserved0: 0,
        reserved1: 0,
      },
      tokens: new Array<number>(recordWidth).fill(PAD_TOKEN_ID),
      tokenMeta: new Array<v1_0_0.BrainTokenMeta>(recordWidth).fill(
        tokenMeta(0, TOKEN_FLAGS.padding),
      ),
      references: new Array<v1_0_0.BrainReferenceBinding>(maxReferencesPerRecord).fill(
        refBinding(INVALID_U32, 0),
      ),
    });
  }

  let activeRecordCount = 0;
  let activeTokenCount = 0;
  for (const spec of buildComfortRecords(episode, options)) {
    const record = records[spec.slot]!;
    record.header = {
      schemaId: spec.schemaId,
      band: spec.band,
      source: spec.source,
      flags: spec.flags ?? RECORD_FLAGS.occupied,
      tokenCount: spec.tokens.filter((token) => token !== PAD_TOKEN_ID).length,
      referenceCount: 0,
      observedAt: 10,
      revision: 1,
      primaryReference: INVALID_U32,
      continuationRecord: INVALID_U32,
      salience: 0.5,
      freshness: 1.0,
      previousObservedAt: INVALID_U32,
      changeMagnitude: 0,
      reserved0: 0,
      reserved1: 0,
    };
    record.tokens = spec.tokens.slice();
    record.tokenMeta = spec.tokens.map((token, localToken) =>
      token === PAD_TOKEN_ID
        ? tokenMeta(0, TOKEN_FLAGS.padding)
        : tokenMeta(spec.roleTokens[localToken] ?? 0, TOKEN_FLAGS.structural),
    );
    activeRecordCount++;
    activeTokenCount += record.header.tokenCount;
  }

  const bands = BRAIN_FRAME_BANDS.map((band) => {
    const active = buildComfortRecords(episode, options).filter((spec) => spec.band === band.kind);
    return {
      kind: band.kind,
      activeRecords: active.length,
      activeTokens: active.reduce(
        (sum, spec) => sum + spec.tokens.filter((token) => token !== PAD_TOKEN_ID).length,
        0,
      ),
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
      tick: 10,
      snapshot: 1,
      deltaMillis: 0,
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

/**
 * Bank index of the catalog record whose family token is `actionToken`
 * (CRY/LAUGH). The intent pointer-loss gold is a bank index (matching the
 * selector's argmax semantics).
 */
export function catalogBankIndex(active: ActiveFrame, frame: v1_0_0.BrainFrameGpu, actionToken: number): number {
  for (let j = 0; j < active.bankRecords.length; j++) {
    const slot = active.bankRecords[j]!;
    if (frame.schemaIds[slot] !== ACTION_INTENT_SCHEMA_ID) continue;
    if (frame.tokenIds[slot * BRAIN_LIMITS.recordWidth] === actionToken) return j;
  }
  throw new Error(`Catalog record 0x${actionToken.toString(16)} not found in the frame bank`);
}
