/**
 * Provisional fixture vocabulary (concerns answer 19).
 *
 * This is a small test manifest for the first forward/training milestones. It
 * is explicitly NOT the production vocabulary: it uses legal ABI token ranges
 * (KRYSTAL_ABI_V0.md section 4) and labels itself provisional in the manifest
 * header. Dynamic context symbols (0xE00..0xEFF) are runtime entities and are
 * deliberately absent from the static manifest.
 */
import {
  KRYSTAL_ABI,
  KRYSTAL_TOKEN_RANGES,
  TOKEN_FLAGS,
} from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import { hashU32s } from "../hash.ts";

type KrystalTokenClass = v1_0_0.KrystalTokenClass;
type VocabManifestEntry = v1_0_0.VocabManifestEntry;
type VocabManifestHeader = v1_0_0.VocabManifestHeader;

export const FIXTURE_VOCAB_VERSION = 0;

/** Canonical order of token classes for manifest hashing. */
export const KRYSTAL_TOKEN_CLASS_ORDER: readonly KrystalTokenClass[] = [
  "system",
  "structure",
  "operation",
  "object",
  "property",
  "quantity",
  "action",
  "reference",
  "relation",
  "logic",
  "domain",
  "context",
  "experimental",
];

export interface FixtureTokenSpec {
  readonly id: number;
  readonly symbol: string;
  readonly tokenClass: KrystalTokenClass;
  readonly flags?: number;
  readonly arity?: number;
  readonly semanticTypeToken?: number;
  readonly inverseToken?: number;
  readonly doc?: string;
}

export const FIXTURE_TOKENS: readonly FixtureTokenSpec[] = [
  // System / control (0x0xx)
  { id: 0x000, symbol: "PAD", tokenClass: "system", flags: TOKEN_FLAGS.padding, doc: "unused token position; masked" },
  { id: 0x001, symbol: "BOS", tokenClass: "system" },
  { id: 0x002, symbol: "EOS", tokenClass: "system" },
  { id: 0x003, symbol: "TRUE", tokenClass: "system" },
  { id: 0x004, symbol: "FALSE", tokenClass: "system" },
  { id: 0x005, symbol: "UNKNOWN", tokenClass: "system" },
  { id: 0x006, symbol: "BEGIN", tokenClass: "system", flags: TOKEN_FLAGS.structural },
  { id: 0x007, symbol: "END", tokenClass: "system", flags: TOKEN_FLAGS.structural },

  // Basic object concepts (0x3xx)
  { id: 0x300, symbol: "APPLE", tokenClass: "object" },
  { id: 0x301, symbol: "SELF", tokenClass: "object" },

  // Properties / qualities (0x4xx)
  { id: 0x400, symbol: "RED", tokenClass: "property" },
  { id: 0x401, symbol: "ROUND", tokenClass: "property" },
  { id: 0x402, symbol: "SHINY", tokenClass: "property" },
  { id: 0x403, symbol: "SMALL", tokenClass: "property" },
  { id: 0x404, symbol: "SATIATED", tokenClass: "property" },
  { id: 0x405, symbol: "FEEL_BAD", tokenClass: "property" },
  { id: 0x406, symbol: "NEED", tokenClass: "property" },

  // Actions / state changes (0x6xx)
  { id: 0x600, symbol: "LOOK", tokenClass: "action", arity: 1, doc: "perceptual LOOK(ref)" },
  { id: 0x601, symbol: "EAT", tokenClass: "action", arity: 1, doc: "EAT(ref)" },
  { id: 0x602, symbol: "WAIT", tokenClass: "action", arity: 0 },
  { id: 0x603, symbol: "HOLD", tokenClass: "action", arity: 1 },
  { id: 0x604, symbol: "REMEMBER", tokenClass: "action", arity: 2 },

  // Relations (0x8xx)
  { id: 0x800, symbol: "STICK", tokenClass: "relation", doc: "persistent attachment relation" },

  // Logic / intent (0x9xx)
  { id: 0x900, symbol: "LAST_ACTION", tokenClass: "logic" },
];

export const FIXTURE_PAD_TOKEN = 0x000;
export const FIXTURE_BOS_TOKEN = 0x001;
export const FIXTURE_EOS_TOKEN = 0x002;

export class FixtureVocabularyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FixtureVocabularyError";
  }
}

export function classIndexOf(tokenClass: KrystalTokenClass): number {
  return KRYSTAL_TOKEN_CLASS_ORDER.indexOf(tokenClass);
}

function validateFixtureVocabulary(): void {
  const seen = new Set<number>();
  for (const spec of FIXTURE_TOKENS) {
    if (spec.id < 0 || spec.id > 0xfff) {
      throw new FixtureVocabularyError(`${spec.symbol}: token id 0x${spec.id.toString(16)} out of 12-bit range`);
    }
    if (seen.has(spec.id)) {
      throw new FixtureVocabularyError(`${spec.symbol}: duplicate token id 0x${spec.id.toString(16)}`);
    }
    seen.add(spec.id);
    const range = Object.entries(KRYSTAL_TOKEN_RANGES).find(
      ([, [start, end]]) => spec.id >= start && spec.id <= end,
    );
    if (!range) {
      throw new FixtureVocabularyError(`${spec.symbol}: token id outside legal ABI ranges`);
    }
    const [rangeName] = range as [KrystalTokenClass, readonly [number, number]];
    if (rangeName !== spec.tokenClass) {
      throw new FixtureVocabularyError(
        `${spec.symbol}: token class ${spec.tokenClass} does not match its range (${rangeName})`,
      );
    }
    if (spec.id >= 0xe00 && spec.id <= 0xeff) {
      throw new FixtureVocabularyError(
        `${spec.symbol}: dynamic context symbols (0xExx) must not be in the static manifest`,
      );
    }
  }
}

/** Compile the fixture vocabulary into a manifest header + entries. */
export function buildFixtureVocabManifest(): {
  header: VocabManifestHeader;
  entries: VocabManifestEntry[];
} {
  validateFixtureVocabulary();
  const entries: VocabManifestEntry[] = FIXTURE_TOKENS.map((spec, index) => ({
    tokenId: spec.id,
    tokenClass: spec.tokenClass,
    flags: spec.flags ?? 0,
    arity: spec.arity ?? 0,
    semanticTypeToken: spec.semanticTypeToken ?? 0,
    inverseToken: spec.inverseToken ?? 0,
    reserved0: index, // manifest order, stable per version
    reserved1: 0,
  }));
  const words: number[] = [];
  for (const entry of entries) {
    words.push(
      entry.tokenId,
      classIndexOf(entry.tokenClass),
      entry.flags,
      entry.arity,
      entry.semanticTypeToken,
      entry.inverseToken,
    );
  }
  const hash = hashU32s(words);
  const header: VocabManifestHeader = {
    tokenAbiVersion: KRYSTAL_ABI.tokenAbiVersion,
    manifestVersion: FIXTURE_VOCAB_VERSION,
    vocabSize: KRYSTAL_ABI.vocabSize,
    activeTokenCount: entries.length,
    manifestHashLo: hash.lo,
    manifestHashHi: hash.hi,
    reserved0: 0,
    reserved1: 0,
  };
  return { header, entries };
}

/** Lookup helper: symbol -> token id, throws on unknown symbol. */
export function fixtureTokenId(symbol: string): number {
  const spec = FIXTURE_TOKENS.find((candidate) => candidate.symbol === symbol);
  if (!spec) throw new FixtureVocabularyError(`Unknown fixture token symbol: ${symbol}`);
  return spec.id;
}
