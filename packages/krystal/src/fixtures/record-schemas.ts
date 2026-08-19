/**
 * Provisional fixture record-schema catalog (concerns answer 20).
 *
 * Hand-authored for the first forward/training milestones: `Self`,
 * `VisionObject`, `Apple`, `HomeostasisQuery` and `MemoryObject`. This is a
 * test fixture catalog, not the final world/domain catalog. The compiled
 * manifest mirrors the schema's `RecordSchemaEntry`/`RecordFieldEntry` device
 * forms; the host authoring forms stay build-time data.
 */
import { BRAIN_LIMITS } from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import { bandIndex } from "../binary-layout-plan.ts";

type BrainBandKind = v1_0_0.BrainBandKind;
type BrainValueKind = v1_0_0.BrainValueKind;
type RecordFieldEntry = v1_0_0.RecordFieldEntry;
type RecordSchemaAuthoringSpec = v1_0_0.RecordSchemaAuthoringSpec;
type RecordSchemaEntry = v1_0_0.RecordSchemaEntry;
type RecordSchemaManifestHeader = v1_0_0.RecordSchemaManifestHeader;
import { hashU32s } from "../hash.ts";
import { FIXTURE_TOKENS, fixtureTokenId } from "./vocabulary.ts";

export const FIXTURE_RECORD_MANIFEST_VERSION = 0;

/**
 * Field flags are not frozen in the schema; this provisional fixture
 * convention uses the low bits of `RecordFieldEntry.flags`.
 */
export const FIXTURE_FIELD_FLAGS = {
  required: 1 << 0,
  exactRuntime: 1 << 1,
} as const;

/** Canonical BrainValueKind order for manifest hashing. */
export const BRAIN_VALUE_KIND_ORDER: readonly BrainValueKind[] = [
  "none",
  "token",
  "context_ref",
  "record_ref",
  "boolean_class",
  "scalar_band",
  "quantity_projection",
  "opaque_payload",
];

export const FIXTURE_RECORD_SCHEMAS: readonly RecordSchemaAuthoringSpec[] = [
  {
    name: "Self",
    familyToken: fixtureTokenId("SELF"),
    defaultBand: "body",
    fields: [
      {
        name: "identity",
        localTokenIndex: 0,
        roleToken: fixtureTokenId("SELF"),
        valueKind: "token",
        required: true,
        doc: "the actor identity anchor record",
      },
    ],
    doc: "body-band identity record; fixed slot 12",
  },
  {
    name: "VisionObject",
    familyToken: fixtureTokenId("APPLE"),
    defaultBand: "vision",
    fields: [
      { name: "family", localTokenIndex: 0, roleToken: fixtureTokenId("APPLE"), valueKind: "token", required: true },
      { name: "referent", localTokenIndex: 1, roleToken: 0, valueKind: "context_ref", required: true, exactRuntime: true },
      { name: "color", localTokenIndex: 2, roleToken: fixtureTokenId("RED"), valueKind: "token" },
      { name: "shape", localTokenIndex: 3, roleToken: fixtureTokenId("ROUND"), valueKind: "token" },
      { name: "surface", localTokenIndex: 4, roleToken: fixtureTokenId("SHINY"), valueKind: "token" },
      { name: "size", localTokenIndex: 5, roleToken: fixtureTokenId("SMALL"), valueKind: "token" },
    ],
    doc: "[APPLE, #ref, RED, ROUND, SHINY, SMALL, PAD, PAD]",
  },
  {
    name: "Apple",
    familyToken: fixtureTokenId("APPLE"),
    defaultBand: "vision",
    fields: [
      { name: "family", localTokenIndex: 0, roleToken: fixtureTokenId("APPLE"), valueKind: "token", required: true },
      { name: "referent", localTokenIndex: 1, roleToken: 0, valueKind: "context_ref", required: true, exactRuntime: true },
      { name: "color", localTokenIndex: 2, roleToken: fixtureTokenId("RED"), valueKind: "token" },
      { name: "size", localTokenIndex: 3, roleToken: fixtureTokenId("SMALL"), valueKind: "token" },
    ],
    doc: "concrete apple instance schema (vision band)",
  },
  {
    name: "HomeostasisQuery",
    familyToken: fixtureTokenId("FEEL_BAD"),
    defaultBand: "homeostasis",
    fields: [
      { name: "family", localTokenIndex: 0, roleToken: fixtureTokenId("FEEL_BAD"), valueKind: "token", required: true },
      { name: "need", localTokenIndex: 1, roleToken: fixtureTokenId("NEED"), valueKind: "token" },
      { name: "desired", localTokenIndex: 2, roleToken: fixtureTokenId("SATIATED"), valueKind: "token" },
    ],
    doc: "[FEEL_BAD, NEED, SATIATED, PAD, ...]",
  },
  {
    name: "MemoryObject",
    familyToken: fixtureTokenId("REMEMBER"),
    defaultBand: "memory",
    fields: [
      { name: "family", localTokenIndex: 0, roleToken: fixtureTokenId("REMEMBER"), valueKind: "token", required: true },
      { name: "referent", localTokenIndex: 1, roleToken: 0, valueKind: "context_ref", required: true, exactRuntime: true },
      { name: "relation", localTokenIndex: 2, roleToken: fixtureTokenId("STICK"), valueKind: "token" },
      { name: "target", localTokenIndex: 3, roleToken: fixtureTokenId("LAST_ACTION"), valueKind: "token" },
      { name: "action", localTokenIndex: 4, roleToken: fixtureTokenId("HOLD"), valueKind: "token" },
    ],
    doc: "[REMEMBER, #ref, STICK, LAST_ACTION, HOLD, PAD, ...]",
  },
  {
    name: "Berry",
    familyToken: fixtureTokenId("BERRY"),
    defaultBand: "vision",
    fields: [
      { name: "family", localTokenIndex: 0, roleToken: fixtureTokenId("BERRY"), valueKind: "token", required: true },
      { name: "referent", localTokenIndex: 1, roleToken: 0, valueKind: "context_ref", required: true, exactRuntime: true },
      { name: "color", localTokenIndex: 2, roleToken: fixtureTokenId("RED"), valueKind: "token" },
      { name: "size", localTokenIndex: 3, roleToken: fixtureTokenId("SMALL"), valueKind: "token" },
    ],
    doc: "concrete berry instance schema (vision band); S7 edible candidate",
  },
  {
    name: "Bread",
    familyToken: fixtureTokenId("BREAD"),
    defaultBand: "vision",
    fields: [
      { name: "family", localTokenIndex: 0, roleToken: fixtureTokenId("BREAD"), valueKind: "token", required: true },
      { name: "referent", localTokenIndex: 1, roleToken: 0, valueKind: "context_ref", required: true, exactRuntime: true },
      { name: "color", localTokenIndex: 2, roleToken: fixtureTokenId("RED"), valueKind: "token" },
      { name: "size", localTokenIndex: 3, roleToken: fixtureTokenId("SMALL"), valueKind: "token" },
    ],
    doc: "concrete bread instance schema (vision band); S7 edible candidate",
  },
  {
    name: "Mother",
    familyToken: fixtureTokenId("MOTHER"),
    defaultBand: "vision",
    fields: [
      { name: "family", localTokenIndex: 0, roleToken: fixtureTokenId("MOTHER"), valueKind: "token", required: true },
      { name: "referent", localTokenIndex: 1, roleToken: 0, valueKind: "context_ref", required: true, exactRuntime: true },
      { name: "role", localTokenIndex: 2, roleToken: fixtureTokenId("MOTHER"), valueKind: "token" },
    ],
    doc: "agent record; never a valid EAT argument (S4 distractor, S3 deliverer)",
  },
  {
    name: "Stone",
    familyToken: fixtureTokenId("STONE"),
    defaultBand: "vision",
    fields: [
      { name: "family", localTokenIndex: 0, roleToken: fixtureTokenId("STONE"), valueKind: "token", required: true },
      { name: "referent", localTokenIndex: 1, roleToken: 0, valueKind: "context_ref", required: true, exactRuntime: true },
      { name: "shape", localTokenIndex: 2, roleToken: fixtureTokenId("ROUND"), valueKind: "token" },
    ],
    doc: "negative EAT candidate (S7); LOOK-observable",
  },
  {
    name: "Feces",
    familyToken: fixtureTokenId("FECES"),
    defaultBand: "vision",
    fields: [
      { name: "family", localTokenIndex: 0, roleToken: fixtureTokenId("FECES"), valueKind: "token", required: true },
      { name: "referent", localTokenIndex: 1, roleToken: 0, valueKind: "context_ref", required: true, exactRuntime: true },
      { name: "shape", localTokenIndex: 2, roleToken: fixtureTokenId("ROUND"), valueKind: "token" },
    ],
    doc: "negative EAT candidate (S7/S8); consequence-worsening consumable",
  },
  {
    name: "UnknownObject",
    familyToken: fixtureTokenId("UNKNOWN"),
    defaultBand: "vision",
    fields: [
      { name: "family", localTokenIndex: 0, roleToken: fixtureTokenId("UNKNOWN"), valueKind: "token", required: true },
      { name: "referent", localTokenIndex: 1, roleToken: 0, valueKind: "context_ref", required: true, exactRuntime: true },
      { name: "surface", localTokenIndex: 2, roleToken: fixtureTokenId("SHINY"), valueKind: "token" },
    ],
    doc: "partially/not-yet observed record; LOOK argument candidate (S6)",
  },
  {
    name: "Dog",
    familyToken: fixtureTokenId("DOG"),
    defaultBand: "vision",
    fields: [
      { name: "family", localTokenIndex: 0, roleToken: fixtureTokenId("DOG"), valueKind: "token", required: true },
      { name: "referent", localTokenIndex: 1, roleToken: 0, valueKind: "context_ref", required: true, exactRuntime: true },
      { name: "case", localTokenIndex: 2, roleToken: fixtureTokenId("ACCUSATIVE"), valueKind: "token", doc: "grammatical case marker: NOMINATIVE or ACCUSATIVE" },
    ],
    doc: "animate vision record; W2 case-binding assay (docs/word_attention_bias.md)",
  },
  {
    name: "Cat",
    familyToken: fixtureTokenId("CAT"),
    defaultBand: "vision",
    fields: [
      { name: "family", localTokenIndex: 0, roleToken: fixtureTokenId("CAT"), valueKind: "token", required: true },
      { name: "referent", localTokenIndex: 1, roleToken: 0, valueKind: "context_ref", required: true, exactRuntime: true },
      { name: "case", localTokenIndex: 2, roleToken: fixtureTokenId("ACCUSATIVE"), valueKind: "token", doc: "grammatical case marker: NOMINATIVE or ACCUSATIVE" },
    ],
    doc: "animate vision record; W2 case-binding assay (docs/word_attention_bias.md)",
  },
];

export class FixtureRecordCatalogError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FixtureRecordCatalogError";
  }
}

function valueKindIndex(valueKind: BrainValueKind): number {
  return BRAIN_VALUE_KIND_ORDER.indexOf(valueKind);
}

function bandKindIndex(band: BrainBandKind): number {
  return bandIndex(band);
}

function validateFixtureRecordSchemas(): void {
  const knownTokens = new Set(FIXTURE_TOKENS.map((spec) => spec.id));
  const schemaNames = new Set(FIXTURE_RECORD_SCHEMAS.map((schema) => schema.name));
  const schemaIds = new Set<number>();
  for (const schema of FIXTURE_RECORD_SCHEMAS) {
    if (!knownTokens.has(schema.familyToken)) {
      throw new FixtureRecordCatalogError(
        `${schema.name}: familyToken 0x${schema.familyToken.toString(16)} not in fixture vocabulary`,
      );
    }
    bandKindIndex(schema.defaultBand); // throws on unknown band
    const positions = new Set<number>();
    for (const field of schema.fields) {
      if (field.localTokenIndex >= BRAIN_LIMITS.recordWidth) {
        throw new FixtureRecordCatalogError(
          `${schema.name}.${field.name}: localTokenIndex ${field.localTokenIndex} >= recordWidth ${BRAIN_LIMITS.recordWidth}`,
        );
      }
      if (positions.has(field.localTokenIndex)) {
        throw new FixtureRecordCatalogError(
          `${schema.name}.${field.name}: duplicate localTokenIndex ${field.localTokenIndex}`,
        );
      }
      positions.add(field.localTokenIndex);
      if (field.roleToken !== 0 && !knownTokens.has(field.roleToken)) {
        throw new FixtureRecordCatalogError(
          `${schema.name}.${field.name}: roleToken 0x${field.roleToken.toString(16)} not in fixture vocabulary`,
        );
      }
      if (field.acceptedSchema && !schemaNames.has(field.acceptedSchema)) {
        throw new FixtureRecordCatalogError(
          `${schema.name}.${field.name}: acceptedSchema "${field.acceptedSchema}" does not resolve`,
        );
      }
      if (field.valueKind === "context_ref" && !field.required) {
        throw new FixtureRecordCatalogError(
          `${schema.name}.${field.name}: context_ref fields must be required in fixtures`,
        );
      }
    }
    // The same family token may legitimately appear in several schemas
    // (VisionObject and Apple both use APPLE), so no uniqueness check here.
    schemaIds.add(schema.familyToken);
  }
}

export interface CompiledRecordManifest {
  header: RecordSchemaManifestHeader;
  entries: RecordSchemaEntry[];
  fields: RecordFieldEntry[];
}

/** Compile the fixture record-schema catalog into device manifest forms. */
export function buildFixtureRecordManifest(): CompiledRecordManifest {
  validateFixtureRecordSchemas();
  const entries: RecordSchemaEntry[] = [];
  const fields: RecordFieldEntry[] = [];
  let fieldOffset = 0;
  for (let schemaId = 0; schemaId < FIXTURE_RECORD_SCHEMAS.length; schemaId++) {
    const schema = FIXTURE_RECORD_SCHEMAS[schemaId]!;
    let tokenCount = 0;
    for (const field of schema.fields) tokenCount = Math.max(tokenCount, field.localTokenIndex + 1);
    entries.push({
      schemaId,
      familyToken: schema.familyToken,
      defaultBand: schema.defaultBand,
      tokenCount,
      fieldOffset,
      fieldCount: schema.fields.length,
      flags: 0,
      reserved0: 0,
    });
    for (let fieldId = 0; fieldId < schema.fields.length; fieldId++) {
      const field = schema.fields[fieldId]!;
      const flags =
        (field.required ? FIXTURE_FIELD_FLAGS.required : 0) |
        (field.exactRuntime ? FIXTURE_FIELD_FLAGS.exactRuntime : 0);
      fields.push({
        schemaId,
        fieldId,
        localTokenIndex: field.localTokenIndex,
        // Fixture default: no fixture field is a scalar projection yet, so the
        // discretization law is unused. A real grammar declares it per field.
        quantityKind: "unipolar",
        tokenWidth: 1,
        roleToken: field.roleToken,
        valueKind: field.valueKind,
        acceptedSchemaId: field.acceptedSchema
          ? FIXTURE_RECORD_SCHEMAS.findIndex((candidate) => candidate.name === field.acceptedSchema)
          : 0,
        allowedBandMask: 0,
        flags,
        reserved0: 0,
      });
    }
    fieldOffset += schema.fields.length;
  }

  const words: number[] = [];
  for (const entry of entries) {
    words.push(
      entry.schemaId,
      entry.familyToken,
      bandKindIndex(entry.defaultBand),
      entry.tokenCount,
      entry.fieldOffset,
      entry.fieldCount,
      entry.flags,
    );
  }
  for (const field of fields) {
    words.push(
      field.schemaId,
      field.fieldId,
      field.localTokenIndex,
      field.tokenWidth,
      field.roleToken,
      valueKindIndex(field.valueKind),
      field.acceptedSchemaId,
      field.allowedBandMask,
      field.flags,
    );
  }
  const hash = hashU32s(words);
  const header: RecordSchemaManifestHeader = {
    version: FIXTURE_RECORD_MANIFEST_VERSION,
    schemaCount: entries.length,
    fieldCount: fields.length,
    maxRecordTokens: BRAIN_LIMITS.recordWidth,
    schemaHashLo: hash.lo,
    schemaHashHi: hash.hi,
    reserved0: 0,
    reserved1: 0,
  };
  return { header, entries, fields };
}

/** Lookup helper: schema name -> compiled entry, throws on unknown name. */
export function fixtureSchema(name: string): RecordSchemaEntry {
  const manifest = buildFixtureRecordManifest();
  const entry = manifest.entries.find((candidate) => {
    return FIXTURE_RECORD_SCHEMAS[candidate.schemaId]!.name === name;
  });
  if (!entry) throw new FixtureRecordCatalogError(`Unknown fixture record schema: ${name}`);
  return entry;
}
