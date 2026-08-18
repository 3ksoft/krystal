/**
 * Canonical fixture BrainFrame (AoS form) exercising all record kinds used by
 * the first milestones: Self (body), Apple (first vision slot),
 * HomeostasisQuery (homeostasis), MemoryObject (first memory slot), the
 * LOOK/EAT/WAIT ActionIntent catalog records (catalog band, which the intent
 * selector scores against) and the active query (query band, which the M2b
 * mixer cross-attends to the record bank). Slots come from the band table
 * rather than literals, so a layout change moves them together. Token
 * patterns follow the schema's illustrative lowerings; reference bindings
 * carry exact 0xExx handles with generations.
 */

/**
 * Schema id of the ActionIntent catalog records. Not part of the record-schema
 * manifest: the ActionIntent catalog (fixtures/action-intents.ts) is a separate
 * device manifest, and the selector masks use this id to mark catalog records
 * as intent candidates (first-forward fixture convention). The id is derived
 * from the record-manifest length so it can never collide with a record
 * schema (e.g. the S2-S10 `Mother` schema occupies index 5).
 */
export const ACTION_INTENT_SCHEMA_ID = FIXTURE_RECORD_SCHEMAS.length;
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
import { FIXTURE_RECORD_SCHEMAS } from "./record-schemas.ts";
import { fixtureTokenId } from "./vocabulary.ts";

function tokenMeta(roleToken: number, flags: number = 0): v1_0_0.BrainTokenMeta {
  return { fieldId: 0, roleToken, flags, referenceBinding: INVALID_U32 };
}

function refBinding(
  localTokenIndex: number,
  tokenId: number,
  generation: number,
): v1_0_0.BrainReferenceBinding {
  return {
    localTokenIndex,
    fieldId: 0,
    flags: 0,
    reserved0: 0,
    handle: { tokenId, generation, kind: "entity", status: "live" },
  };
}

export interface FixtureRecordSpec {
  readonly slot: number;
  readonly band: v1_0_0.BrainBandKind;
  readonly schemaId: number;
  readonly tokens: readonly number[];
  readonly roleTokens: readonly number[];
  readonly refs?: readonly v1_0_0.BrainReferenceBinding[];
  readonly source: v1_0_0.RecordSource;
  readonly flags?: number;
}

export const FIXTURE_APPLE_REF = 0xe11;
export const FIXTURE_MEMORY_REF = 0xe43;

const VISION_BAND_OFFSET = BRAIN_FRAME_BANDS.find((band) => band.kind === "vision")!.recordOffset;
const MEMORY_BAND_OFFSET = BRAIN_FRAME_BANDS.find((band) => band.kind === "memory")!.recordOffset;

export function buildFixtureRecords(): readonly FixtureRecordSpec[] {
  const APPLE = fixtureTokenId("APPLE");
  const SELF = fixtureTokenId("SELF");
  const FEEL_BAD = fixtureTokenId("FEEL_BAD");
  const NEED = fixtureTokenId("NEED");
  const SATIATED = fixtureTokenId("SATIATED");
  const RED = fixtureTokenId("RED");
  const SMALL = fixtureTokenId("SMALL");
  const REMEMBER = fixtureTokenId("REMEMBER");
  const STICK = fixtureTokenId("STICK");
  const LAST_ACTION = fixtureTokenId("LAST_ACTION");
  const HOLD = fixtureTokenId("HOLD");
  const LOOK = fixtureTokenId("LOOK");
  const EAT = fixtureTokenId("EAT");
  const WAIT = fixtureTokenId("WAIT");

  return [
    {
      slot: BRAIN_FIXED_RECORDS.homeostasisSummary,
      band: "homeostasis",
      schemaId: 3, // HomeostasisQuery
      tokens: [FEEL_BAD, NEED, SATIATED, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
      roleTokens: [FEEL_BAD, NEED, SATIATED, 0, 0, 0, 0, 0],
      source: "homeostasis",
      flags: RECORD_FLAGS.occupied | RECORD_FLAGS.fixed,
    },
    {
      slot: BRAIN_FIXED_RECORDS.self,
      band: "body",
      schemaId: 0, // Self
      tokens: [SELF, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
      roleTokens: [SELF, 0, 0, 0, 0, 0, 0, 0],
      source: "body",
      flags: RECORD_FLAGS.occupied | RECORD_FLAGS.fixed,
    },
    {
      slot: VISION_BAND_OFFSET, // first vision slot
      band: "vision",
      schemaId: 2, // Apple
      tokens: [APPLE, FIXTURE_APPLE_REF, RED, SMALL, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
      roleTokens: [APPLE, 0, RED, SMALL, 0, 0, 0, 0],
      refs: [refBinding(1, FIXTURE_APPLE_REF, 3)],
      source: "sensor",
      flags: RECORD_FLAGS.occupied,
    },
    {
      slot: MEMORY_BAND_OFFSET, // first memory slot
      band: "memory",
      schemaId: 4, // MemoryObject
      tokens: [REMEMBER, FIXTURE_MEMORY_REF, STICK, LAST_ACTION, HOLD, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
      roleTokens: [REMEMBER, 0, STICK, LAST_ACTION, HOLD, 0, 0, 0],
      refs: [refBinding(1, FIXTURE_MEMORY_REF, 1)],
      source: "memory",
      flags: RECORD_FLAGS.occupied | RECORD_FLAGS.remembered,
    },
    {
      slot: BRAIN_FIXED_RECORDS.primaryQuery,
      band: "query",
      schemaId: 3, // HomeostasisQuery doubles as the first query/need record
      tokens: [FEEL_BAD, NEED, SATIATED, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
      roleTokens: [FEEL_BAD, NEED, SATIATED, 0, 0, 0, 0, 0],
      source: "query",
      flags: RECORD_FLAGS.occupied | RECORD_FLAGS.fixed | RECORD_FLAGS.query,
    },
    // ActionIntent catalog records (catalog band): the intent selector scores
    // the mixed query state against their pooled keys. The action token is the
    // embedded family/discriminator (LOOK vs EAT vs WAIT).
    {
      slot: BRAIN_FIXED_RECORDS.catalogBase,
      band: "catalog",
      schemaId: ACTION_INTENT_SCHEMA_ID,
      tokens: [LOOK, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
      roleTokens: [LOOK, 0, 0, 0, 0, 0, 0, 0],
      source: "creator",
      flags: RECORD_FLAGS.occupied | RECORD_FLAGS.fixed | RECORD_FLAGS.creatorAuthored,
    },
    {
      slot: BRAIN_FIXED_RECORDS.catalogBase + 1,
      band: "catalog",
      schemaId: ACTION_INTENT_SCHEMA_ID,
      tokens: [EAT, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
      roleTokens: [EAT, 0, 0, 0, 0, 0, 0, 0],
      source: "creator",
      flags: RECORD_FLAGS.occupied | RECORD_FLAGS.fixed | RECORD_FLAGS.creatorAuthored,
    },
    {
      slot: BRAIN_FIXED_RECORDS.catalogBase + 2,
      band: "catalog",
      schemaId: ACTION_INTENT_SCHEMA_ID,
      tokens: [WAIT, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
      roleTokens: [WAIT, 0, 0, 0, 0, 0, 0, 0],
      source: "creator",
      flags: RECORD_FLAGS.occupied | RECORD_FLAGS.fixed | RECORD_FLAGS.creatorAuthored,
    },
  ];
}

/** Build the canonical fixture BrainFrame (AoS). */
export function buildFixtureFrame(): v1_0_0.BrainFrame {
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
      },
      tokens: new Array<number>(BRAIN_LIMITS.recordWidth).fill(PAD_TOKEN_ID),
      tokenMeta: new Array<v1_0_0.BrainTokenMeta>(BRAIN_LIMITS.recordWidth).fill(
        tokenMeta(0, TOKEN_FLAGS.padding),
      ),
      references: new Array<v1_0_0.BrainReferenceBinding>(BRAIN_LIMITS.maxReferencesPerRecord).fill(
        refBinding(INVALID_U32, 0, 0),
      ),
    });
  }

  let activeRecordCount = 0;
  let activeTokenCount = 0;
  for (const spec of buildFixtureRecords()) {
    const record = records[spec.slot]!;
    record.header = {
      schemaId: spec.schemaId,
      band: spec.band,
      source: spec.source,
      flags: spec.flags ?? RECORD_FLAGS.occupied,
      tokenCount: spec.tokens.filter((token) => token !== PAD_TOKEN_ID).length,
      referenceCount: spec.refs?.length ?? 0,
      observedAt: 10,
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
    const active = buildFixtureRecords().filter((spec) => spec.band === band.kind);
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
