// M2a fixture validity tests (concerns answers 19/20): every fixture token
// stays inside a legal ABI range, the record-schema and ActionIntent catalogs
// resolve against the vocabulary, and all manifest hashes are deterministic.
import { expect, test } from "bun:test";
import {
  KRYSTAL_ABI,
  KRYSTAL_TOKEN_RANGES,
} from "../packages/schema/src/krystal-engine-schema.ts";
import {
  FIXTURE_TOKENS,
  FIXTURE_PAD_TOKEN,
  buildFixtureVocabManifest,
  fixtureTokenId,
} from "../packages/krystal/src/fixtures/vocabulary.ts";
import {
  FIXTURE_RECORD_SCHEMAS,
  buildFixtureRecordManifest,
} from "../packages/krystal/src/fixtures/record-schemas.ts";
import {
  FIXTURE_ACTION_INTENTS,
  buildFixtureActionCatalog,
  fixtureIntent,
} from "../packages/krystal/src/fixtures/action-intents.ts";

test("vocabulary: all tokens are in legal ABI ranges with matching class", () => {
  const manifest = buildFixtureVocabManifest();
  expect(manifest.header.tokenAbiVersion).toBe(KRYSTAL_ABI.tokenAbiVersion);
  expect(manifest.header.vocabSize).toBe(KRYSTAL_ABI.vocabSize);
  expect(manifest.header.activeTokenCount).toBe(FIXTURE_TOKENS.length);
  expect(manifest.header.manifestVersion).toBe(0);

  const entries = new Map(manifest.entries.map((entry) => [entry.tokenId, entry]));
  expect(entries.size).toBe(FIXTURE_TOKENS.length);
  for (const spec of FIXTURE_TOKENS) {
    const entry = entries.get(spec.id)!;
    expect(entry.tokenClass).toBe(spec.tokenClass);
    // classOf(token) = token >> 8 must match the declared class range.
    const range = Object.entries(KRYSTAL_TOKEN_RANGES).find(
      ([, [start, end]]) => spec.id >= start && spec.id <= end,
    )!;
    const [rangeName] = range as [string, readonly [number, number]];
    expect(rangeName).toBe(spec.tokenClass);
  }
});

test("vocabulary: PAD and dynamic reference space are respected", () => {
  expect(FIXTURE_PAD_TOKEN).toBe(0x000);
  expect(fixtureTokenId("PAD")).toBe(FIXTURE_PAD_TOKEN);
  // No static manifest entry may live in the dynamic context range 0xE00..0xEFF.
  for (const spec of FIXTURE_TOKENS) {
    expect(spec.id < 0xe00 || spec.id > 0xeff).toBe(true);
  }
});

test("vocabulary: manifest hash is deterministic", () => {
  const a = buildFixtureVocabManifest();
  const b = buildFixtureVocabManifest();
  expect(a.header.manifestHashLo).toBe(b.header.manifestHashLo);
  expect(a.header.manifestHashHi).toBe(b.header.manifestHashHi);
});

test("record schemas: catalog compiles and resolves against the vocabulary", () => {
  const manifest = buildFixtureRecordManifest();
  expect(manifest.header.schemaCount).toBe(FIXTURE_RECORD_SCHEMAS.length);
  expect(manifest.header.maxRecordTokens).toBe(8);
  expect(manifest.entries).toHaveLength(FIXTURE_RECORD_SCHEMAS.length);

  const tokenIds = new Set(FIXTURE_TOKENS.map((spec) => spec.id));
  for (const entry of manifest.entries) {
    expect(tokenIds.has(entry.familyToken)).toBe(true);
    expect(entry.tokenCount).toBeGreaterThan(0);
    expect(entry.tokenCount).toBeLessThanOrEqual(8);
  }

  // fieldOffset accumulates across schemas in catalog order.
  let cumulative = 0;
  for (const entry of manifest.entries) {
    expect(entry.fieldOffset).toBe(cumulative);
    cumulative += entry.fieldCount;
  }
  expect(manifest.header.fieldCount).toBe(cumulative);
});

test("record schemas: every field references legal role tokens and valid positions", () => {
  const manifest = buildFixtureRecordManifest();
  const tokenIds = new Set(FIXTURE_TOKENS.map((spec) => spec.id));
  for (const field of manifest.fields) {
    expect(field.localTokenIndex).toBeLessThan(8);
    expect(field.tokenWidth).toBe(1);
    if (field.roleToken !== 0) expect(tokenIds.has(field.roleToken)).toBe(true);
  }
});

test("record schemas: manifest hash is deterministic", () => {
  const a = buildFixtureRecordManifest();
  const b = buildFixtureRecordManifest();
  expect(a.header.schemaHashLo).toBe(b.header.schemaHashLo);
  expect(a.header.schemaHashHi).toBe(b.header.schemaHashHi);
});

test("action catalog: LOOK/EAT/MOVE_TOWARDS/WAIT/CRY/LAUGH compile with resolved arguments", () => {
  const catalog = buildFixtureActionCatalog();
  expect(catalog.header.intentCount).toBe(FIXTURE_ACTION_INTENTS.length);
  expect(catalog.descriptors).toHaveLength(6);

  const look = fixtureIntent("LOOK");
  expect(look.argumentCount).toBe(1);
  expect(look.flags & 0b100).toBe(0b100); // perceptual flag
  expect(catalog.arguments[look.argumentOffset]!.valueKind).toBe("context_ref");

  const eat = fixtureIntent("EAT");
  expect(eat.argumentCount).toBe(1);
  expect(eat.flags & 1).toBe(1); // durative
  // EAT's argument identity is the Apple schema (schemaId 2 in catalog order);
  // the capability layer (S7) widens acceptance to every edible schema.
  expect(catalog.arguments[eat.argumentOffset]!.acceptedSchemaId).toBe(2);

  const move = fixtureIntent("MOVE_TOWARDS");
  expect(move.argumentCount).toBe(1);
  expect(move.flags & 1).toBe(1); // durative
  expect(catalog.arguments[move.argumentOffset]!.valueKind).toBe("context_ref");

  const wait = fixtureIntent("WAIT");
  expect(wait.argumentCount).toBe(0);
  expect(wait.flags & 1).toBe(1); // durative

  // Step-1 communicative intents: zero-arity, non-durative, communicative flag.
  const cry = fixtureIntent("CRY");
  expect(cry.argumentCount).toBe(0);
  expect(cry.flags & 1).toBe(0); // not durative
  const laugh = fixtureIntent("LAUGH");
  expect(laugh.argumentCount).toBe(0);
  expect(laugh.flags & 1).toBe(0);
});

test("action catalog: argument band masks and hash are deterministic", () => {
  const a = buildFixtureActionCatalog();
  const b = buildFixtureActionCatalog();
  expect(a.header.catalogHashLo).toBe(b.header.catalogHashLo);
  expect(a.header.catalogHashHi).toBe(b.header.catalogHashHi);

  // LOOK accepts vision + memory bands (band ids 3 and 8).
  const look = fixtureIntent("LOOK");
  expect(a.arguments[look.argumentOffset]!.candidateBandMask).toBe((1 << 3) | (1 << 8));
});

test("action catalog: the header counts and the hash cover argument descriptors", () => {
  const catalog = buildFixtureActionCatalog();

  // The header must describe the compiled arguments, not zero of them: it is
  // the device-side length of the argument-descriptor section.
  expect(catalog.arguments.length).toBeGreaterThan(0);
  expect(catalog.header.argumentCount).toBe(catalog.arguments.length);
  expect(catalog.header.argumentCount).toBe(
    catalog.descriptors.reduce((sum, descriptor) => sum + descriptor.argumentCount, 0),
  );

  // The hash is the catalog's integrity mechanism, so it must move when an
  // argument descriptor changes even though every intent stays identical.
  // Narrowing EAT's candidate bands to vision alone is exactly the kind of
  // edit a widened catalog will make.
  const narrowed = FIXTURE_ACTION_INTENTS.map((intent) =>
    intent.name !== "EAT"
      ? intent
      : { ...intent, arguments: intent.arguments.map((arg) => ({ ...arg, candidateBands: ["vision"] as const })) },
  );
  const variant = buildFixtureActionCatalog(narrowed);
  expect(variant.header.intentCount).toBe(catalog.header.intentCount);
  expect(variant.header.argumentCount).toBe(catalog.header.argumentCount);
  expect(
    variant.header.catalogHashLo !== catalog.header.catalogHashLo ||
      variant.header.catalogHashHi !== catalog.header.catalogHashHi,
  ).toBe(true);
});
