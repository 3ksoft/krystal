/**
 * Provisional fixture vocabulary (concerns answer 19).
 *
 * This is a small test manifest for the first forward/training milestones. It
 * is explicitly NOT the production vocabulary: it uses legal ABI token ranges
 * and labels itself provisional in the manifest header. Reference-half symbols
 * (0x8000..0xFFFF) are runtime entities with no embedding row of their own and
 * are deliberately absent from the static manifest.
 */
import {
  KRYSTAL_ABI,
  KRYSTAL_TOKEN_RANGES,
  QUANTIFIER_FLAGS,
  RELATION_FLAGS,
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
  "temporal",
  "domain",
  "experimental",
  "context",
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
  { id: 0x0000, symbol: "PAD", tokenClass: "system", flags: TOKEN_FLAGS.padding, doc: "unused token position; masked" },
  { id: 0x0001, symbol: "BOS", tokenClass: "system" },
  { id: 0x0002, symbol: "EOS", tokenClass: "system" },
  { id: 0x0003, symbol: "TRUE", tokenClass: "system" },
  { id: 0x0004, symbol: "FALSE", tokenClass: "system" },
  { id: 0x0005, symbol: "UNKNOWN", tokenClass: "system" },
  { id: 0x0006, symbol: "BEGIN", tokenClass: "system", flags: TOKEN_FLAGS.structural },
  { id: 0x0007, symbol: "END", tokenClass: "system", flags: TOKEN_FLAGS.structural },
  // The three "nothing"s are distinct on purpose (KRYSTAL_SENTINEL_TOKENS).
  // PAD above is structural absence and is masked out of attention entirely;
  // these two are percepts and reach the model like any other token.
  { id: 0x0008, symbol: "VOID", tokenClass: "system", doc: "sensed emptiness: the sense looked and there is nothing there" },
  { id: 0x0009, symbol: "UNAVAILABLE", tokenClass: "system", doc: "the sense is not reporting: eyes shut, darkness, blocked ear" },
  // Epistemic pair. UNKNOWN (0x0005) licenses a query; SOMETHING is the bound
  // variable a query is asking about and licenses none.
  { id: 0x000a, symbol: "SOMETHING", tokenClass: "system", doc: "existential: some referent, identity open" },

  // Grammatical structure markers (0x1xx). Case is carried by an explicit
  // marker token, never by token order: the packed record has no word order to
  // read a role off (W2 case binding, docs/word_attention_bias.md).
  { id: 0x0800, symbol: "NOMINATIVE", tokenClass: "structure", doc: "grammatical case marker: agent/subject of the predicate" },
  { id: 0x0801, symbol: "ACCUSATIVE", tokenClass: "structure", doc: "grammatical case marker: patient/direct object of the predicate" },

  // Basic object concepts (0x3xx)
  { id: 0x1800, symbol: "APPLE", tokenClass: "object" },
  { id: 0x1801, symbol: "SELF", tokenClass: "object" },
  { id: 0x1802, symbol: "BERRY", tokenClass: "object" },
  { id: 0x1803, symbol: "BREAD", tokenClass: "object" },
  { id: 0x1804, symbol: "MOTHER", tokenClass: "object" },
  { id: 0x1805, symbol: "STONE", tokenClass: "object" },
  { id: 0x1806, symbol: "FECES", tokenClass: "object" },
  { id: 0x1807, symbol: "DOG", tokenClass: "object", doc: "animate vision object; W2 case-binding assay" },
  { id: 0x1808, symbol: "CAT", tokenClass: "object", doc: "animate vision object; W2 case-binding assay" },

  // Properties / qualities (0x4xx)
  { id: 0x2000, symbol: "RED", tokenClass: "property" },
  { id: 0x2001, symbol: "ROUND", tokenClass: "property" },
  { id: 0x2002, symbol: "SHINY", tokenClass: "property" },
  { id: 0x2003, symbol: "SMALL", tokenClass: "property" },
  { id: 0x2004, symbol: "SATIATED", tokenClass: "property" },
  { id: 0x2005, symbol: "FEEL_BAD", tokenClass: "property" },
  { id: 0x2006, symbol: "NEED", tokenClass: "property" },
  { id: 0x2007, symbol: "FEEL_GOOD", tokenClass: "property", doc: "positive homeostasis valence (comfort > 0)" },
  { id: 0x2008, symbol: "NEAR", tokenClass: "property", doc: "object is within reach (S5 spatial availability)" },
  { id: 0x2009, symbol: "FAR", tokenClass: "property", doc: "object is out of reach (S5 spatial availability)" },
  { id: 0x200a, symbol: "POISONED", tokenClass: "property", doc: "consumable carries a negative consequence (S8)" },
  { id: 0x200b, symbol: "YELLOW", tokenClass: "property", doc: "word-binding assay colour (docs/word_attention_bias.md)" },

  // Quantities / projected channels (0x5xx)
  { id: 0x2800, symbol: "COMFORT", tokenClass: "quantity", doc: "homeostasis comfort channel token" },
  { id: 0x2801, symbol: "MILD", tokenClass: "quantity", doc: "comfort magnitude band (0.25 < |c| <= 0.5)" },
  { id: 0x2802, symbol: "MODERATE", tokenClass: "quantity", doc: "comfort magnitude band (0.5 < |c| <= 0.75)" },
  { id: 0x2803, symbol: "SEVERE", tokenClass: "quantity", doc: "comfort magnitude band (0.75 < |c| <= 1)" },
  // Count bands (subitizing, QUANTITY_BANDS.count). These answer "how many",
  // and are NOT the quantifiers below: "some sheep" is three of them, while
  // "some of the sheep" is a fraction of a flock. Sharing one symbol between
  // the two would give two incompatible concepts a single embedding row.
  { id: 0x2804, symbol: "SOME", tokenClass: "quantity", doc: "count band: 2..3 (word-binding assay, docs/word_attention_bias.md)" },
  { id: 0x2805, symbol: "MUCH", tokenClass: "quantity", doc: "count band: above the subitizing limit (word-binding assay)" },
  { id: 0x2806, symbol: "FEW", tokenClass: "quantity", doc: "count band: 0..1" },

  // Actions / state changes (0x6xx)
  { id: 0x3000, symbol: "LOOK", tokenClass: "action", arity: 1, doc: "perceptual LOOK(ref)" },
  { id: 0x3001, symbol: "EAT", tokenClass: "action", arity: 1, doc: "EAT(ref)" },
  { id: 0x3002, symbol: "WAIT", tokenClass: "action", arity: 0 },
  { id: 0x3003, symbol: "HOLD", tokenClass: "action", arity: 1 },
  { id: 0x3004, symbol: "REMEMBER", tokenClass: "action", arity: 2 },
  { id: 0x3005, symbol: "CRY", tokenClass: "action", arity: 0, doc: "communicative CRY() — negative homeostasis valence" },
  { id: 0x3006, symbol: "LAUGH", tokenClass: "action", arity: 0, doc: "communicative LAUGH() — positive homeostasis valence" },
  { id: 0x3007, symbol: "MOVE_TOWARDS", tokenClass: "action", arity: 1, doc: "MOVE_TOWARDS(ref) — approach a spatially distant target (S5)" },
  { id: 0x3008, symbol: "CHASE", tokenClass: "action", arity: 1, doc: "CHASE(ref) — pursue the accusative-marked participant (W2 case-binding assay)" },

  // Relations (0x40xx). Algebraic properties live in the high flag bits and let
  // the runtime close/invalidate without the network learning to.
  { id: 0x4000, symbol: "STICK", tokenClass: "relation", doc: "persistent attachment relation" },
  { id: 0x4001, symbol: "INSIDE", tokenClass: "relation", arity: 2, flags: RELATION_FLAGS.transitive, doc: "containment; transitively closed by the runtime" },
  { id: 0x4002, symbol: "NEXT_TO", tokenClass: "relation", arity: 2, flags: RELATION_FLAGS.symmetric, inverseToken: 0x4002, doc: "proximity; its own inverse" },
  { id: 0x4003, symbol: "IS_AT", tokenClass: "relation", arity: 2, flags: RELATION_FLAGS.functional, doc: "location; single-valued, a new instance invalidates the old" },

  // Temporal sense (0x50xx). Motion is expressed as an ordinary binary
  // relation whose `intensity` carries the rate, which is why the temporal
  // band needs no machinery of its own.
  { id: 0x5000, symbol: "CHANGING", tokenClass: "temporal", arity: 1, doc: "unary: subject is changing; object == subject" },
  { id: 0x5001, symbol: "APPROACHING", tokenClass: "temporal", arity: 2, inverseToken: 0x5002, doc: "subject closes on object; intensity = closing speed" },
  { id: 0x5002, symbol: "RECEDING", tokenClass: "temporal", arity: 2, inverseToken: 0x5001, doc: "subject moves away from object; intensity = speed" },
  { id: 0x5003, symbol: "STILL", tokenClass: "temporal", arity: 1, doc: "unary: subject is not moving" },

  // Logic (0x48xx).
  { id: 0x4800, symbol: "LAST_ACTION", tokenClass: "logic" },

  // Quantifiers over a reference set. They live here rather than beside the
  // magnitude bands because they are operators, not sizes: their boundaries are
  // logic rather than perception (none is exactly 0, all exactly 1, most above
  // a half), so unlike NEAR/FAR there is no threshold to calibrate, and
  // approximating the endpoints would destroy the inference they license —
  // ALL_OF says something about any member, MOST_OF says nothing about one.
  //
  // Each is a binary relation between a subset and its reference set, which is
  // also what makes aggregate records expressive: a distant flock collapsing
  // into one record can report MOST_OF(white, sheep) rather than only a count.
  //
  // Monotonicity flags let the runtime entail for free, as the relation algebra
  // does for transitivity.
  { id: 0x4801, symbol: "NONE_OF", tokenClass: "logic", arity: 2, flags: QUANTIFIER_FLAGS.restrictorDownward | QUANTIFIER_FLAGS.scopeDownward, doc: "exactly 0 of the reference set" },
  { id: 0x4802, symbol: "SOME_OF", tokenClass: "logic", arity: 2, flags: QUANTIFIER_FLAGS.restrictorUpward | QUANTIFIER_FLAGS.scopeUpward, doc: "above 0; the set-level counterpart of the SOMETHING sentinel" },
  { id: 0x4803, symbol: "MOST_OF", tokenClass: "logic", arity: 2, flags: QUANTIFIER_FLAGS.scopeUpward, doc: "above one half" },
  { id: 0x4804, symbol: "ALL_OF", tokenClass: "logic", arity: 2, flags: QUANTIFIER_FLAGS.restrictorDownward | QUANTIFIER_FLAGS.scopeUpward, doc: "exactly 1 of the reference set" },
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
  if (FIXTURE_TOKENS.length > KRYSTAL_ABI.semanticEmbeddingRows) {
    throw new FixtureVocabularyError(
      `${FIXTURE_TOKENS.length} symbols exceed the ${KRYSTAL_ABI.semanticEmbeddingRows}-row embedding table`,
    );
  }
  const seen = new Set<number>();
  for (const spec of FIXTURE_TOKENS) {
    if (spec.id < 0 || spec.id > KRYSTAL_ABI.semanticEnd) {
      throw new FixtureVocabularyError(
        `${spec.symbol}: token id 0x${spec.id.toString(16)} outside the embedded semantic half`,
      );
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
    if (spec.id >= KRYSTAL_ABI.refSpaceStart) {
      throw new FixtureVocabularyError(
        `${spec.symbol}: reference-half symbols are bound at runtime and carry no embedding row, so they must not appear in the static manifest`,
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
    // Manifest index == embedding row. Append-only: reordering this list
    // renumbers rows and invalidates every embedding trained against them.
    reserved0: index,
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
    vocabSize: KRYSTAL_ABI.semanticVocabSize,
    activeTokenCount: entries.length,
    embeddingRows: KRYSTAL_ABI.semanticEmbeddingRows,
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

/**
 * Token id -> embedding row.
 *
 * The two halves of the token space are projected differently, which is the
 * whole reason the space was split:
 *
 *   semantic   Row is the token's MANIFEST INDEX. Ids are a sparse class grid
 *              (objects at 0x1800, properties at 0x2000...), so a table indexed
 *              by id would span 0x8000 rows to carry a few hundred symbols.
 *   reference  Rows are SHARED. A reference token is a pointer whose meaning
 *              comes from its binding and its referent, and reference #37 means
 *              something different every frame — a private learned row per
 *              reference would be memorising noise. They fold into a small pool
 *              of `refEmbeddingRows` positional rows instead, which is what
 *              makes a 32k reference space affordable.
 *
 * Unknown semantic ids map to PAD's row rather than throwing: a frame carrying
 * a symbol outside the compiled manifest is a compiler bug, but it must not be
 * able to read past the end of the embedding table.
 */
export function buildTokenRowTable(
  tokens: readonly FixtureTokenSpec[] = FIXTURE_TOKENS,
): Uint32Array {
  const table = new Uint32Array(KRYSTAL_ABI.tokenSpaceSize);
  for (let index = 0; index < tokens.length; index++) {
    table[tokens[index]!.id] = index;
  }
  const refBase = KRYSTAL_ABI.semanticEmbeddingRows;
  for (let id = KRYSTAL_ABI.refSpaceStart; id <= KRYSTAL_ABI.refSpaceEnd; id++) {
    table[id] = refBase + ((id - KRYSTAL_ABI.refSpaceStart) % KRYSTAL_ABI.refEmbeddingRows);
  }
  return table;
}

/** Row table for the fixture manifest. */
export const FIXTURE_TOKEN_ROWS: Uint32Array = buildTokenRowTable();

/** Project a whole token id buffer into embedding rows. */
export function toEmbeddingRows(
  tokenIds: ArrayLike<number>,
  table: Uint32Array = FIXTURE_TOKEN_ROWS,
): Uint32Array {
  const rows = new Uint32Array(tokenIds.length);
  for (let i = 0; i < tokenIds.length; i++) rows[i] = table[tokenIds[i]!]!;
  return rows;
}
