/**
 * Canonical fixture BrainFrame (AoS form) exercising all four record kinds
 * used by the first milestones: Self (body, fixed slot 12), Apple (vision,
 * slot 24), HomeostasisQuery (homeostasis, fixed slot 4) and MemoryObject
 * (memory, slot 90). Token patterns follow the schema's illustrative
 * lowerings; reference bindings carry exact 0xExx handles with generations.
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
      slot: 24, // first vision slot
      band: "vision",
      schemaId: 2, // Apple
      tokens: [APPLE, FIXTURE_APPLE_REF, RED, SMALL, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
      roleTokens: [APPLE, 0, RED, SMALL, 0, 0, 0, 0],
      refs: [refBinding(1, FIXTURE_APPLE_REF, 3)],
      source: "sensor",
      flags: RECORD_FLAGS.occupied,
    },
    {
      slot: 90, // first memory slot
      band: "memory",
      schemaId: 4, // MemoryObject
      tokens: [REMEMBER, FIXTURE_MEMORY_REF, STICK, LAST_ACTION, HOLD, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
      roleTokens: [REMEMBER, 0, STICK, LAST_ACTION, HOLD, 0, 0, 0],
      refs: [refBinding(1, FIXTURE_MEMORY_REF, 1)],
      source: "memory",
      flags: RECORD_FLAGS.occupied | RECORD_FLAGS.remembered,
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
