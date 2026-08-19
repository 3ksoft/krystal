/**
 * Agent construction: where a simulation's grammar becomes a brain.
 *
 * The vocabulary is NOT owned by Krystal. A simulation declares what exists in
 * its world and hands that over here; Krystal contributes only the symbols its
 * own runtime branches on. That split is not a convention to remember, it is
 * the token-class grid: `system` and `structure` are reserved, everything from
 * `object` upward belongs to the simulation. A supplied symbol landing below
 * `RESERVED_TOKEN_END` is rejected, so the rule needs no list of forbidden
 * names.
 *
 * The second job of this module is less obvious and more important. An
 * embedding row is identified by manifest index, so a grammar and a set of
 * trained weights are only meaningful together: feed a checkpoint a grammar
 * whose symbol order has shifted and every row it learned now means something
 * else. Nothing about that failure is visible at runtime — training simply
 * proceeds against scrambled meanings. Binding the manifest hash into the
 * agent turns that into an error at construction, which is why `createAgent`
 * is the right place for it rather than an incidental detail of loading.
 */
import {
  ACTION_INTENT_FLAGS,
  BRAIN_FRAME_BANDS,
  BRAIN_LIMITS,
  KRYSTAL_ABI,
  RELATION_ROLE_FLAGS,
  KRYSTAL_SENTINEL_TOKENS,
  KRYSTAL_TOKEN_RANGES,
  TOKEN_FLAGS,
  tokenClassIndex,
} from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import { hashU32s } from "../hash.ts";
import { BAND_SYMBOLS, BAND_TOKEN_IDS, type Polarity } from "./quantize.ts";

type KrystalTokenClass = v1_0_0.KrystalTokenClass;

/**
 * End of the Krystal-reserved block. Everything at or below this is structural
 * and fixed by the engine; the simulation's own vocabulary starts above it.
 */
export const RESERVED_TOKEN_END = KRYSTAL_TOKEN_RANGES.structure[1];

/** Classes a simulation may declare symbols in. */
export const SIM_TOKEN_CLASSES: readonly KrystalTokenClass[] = [
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
];

/**
 * Symbols the engine contributes to every agent, in fixed row order.
 *
 * These occupy embedding rows 0..N-1 permanently. Simulation symbols start at
 * row N, so a world that grows never disturbs a structural row. The runtime
 * branches on each of these — PAD is masked out of attention entirely, UNKNOWN
 * licenses a query while SOMETHING does not, VOID and UNAVAILABLE are percepts
 * that must reach the model — so they cannot be supplied as data.
 */
export const RESERVED_SYMBOLS: readonly {
  readonly symbol: string;
  readonly id: number;
  readonly tokenClass: KrystalTokenClass;
  readonly flags?: number;
  readonly doc?: string;
}[] = [
  { symbol: "PAD", id: KRYSTAL_SENTINEL_TOKENS.pad, tokenClass: "system", flags: TOKEN_FLAGS.padding, doc: "structural absence; hard-masked, never attended" },
  { symbol: "BOS", id: KRYSTAL_SENTINEL_TOKENS.bos, tokenClass: "system" },
  { symbol: "EOS", id: KRYSTAL_SENTINEL_TOKENS.eos, tokenClass: "system" },
  { symbol: "TRUE", id: KRYSTAL_SENTINEL_TOKENS.boolTrue, tokenClass: "system" },
  { symbol: "FALSE", id: KRYSTAL_SENTINEL_TOKENS.boolFalse, tokenClass: "system" },
  { symbol: "UNKNOWN", id: KRYSTAL_SENTINEL_TOKENS.unknown, tokenClass: "system", doc: "a referent exists, identity not known; licenses a query" },
  { symbol: "BEGIN", id: KRYSTAL_SENTINEL_TOKENS.begin, tokenClass: "system", flags: TOKEN_FLAGS.structural },
  { symbol: "END", id: KRYSTAL_SENTINEL_TOKENS.end, tokenClass: "system", flags: TOKEN_FLAGS.structural },
  { symbol: "VOID", id: KRYSTAL_SENTINEL_TOKENS.void, tokenClass: "system", doc: "sensed emptiness; a percept, not a gap" },
  { symbol: "UNAVAILABLE", id: KRYSTAL_SENTINEL_TOKENS.unavailable, tokenClass: "system", doc: "the sense is not reporting; also a percept" },
  { symbol: "SOMETHING", id: KRYSTAL_SENTINEL_TOKENS.something, tokenClass: "system", doc: "existential; the target of a query, not a trigger for one" },
  /**
   * Something previously perceived is absent now.
   *
   * Reserved, and the only member of its family that has to be: an appearance
   * needs no token because the new record carries `previousObservedAt` invalid,
   * and a change needs none because its magnitude is a quantity that bands like
   * any other. A disappearance is different in kind — the record is GONE, so
   * there is nothing to hang a flag on, and an absence cannot be observed in the
   * frame it is absent from. The engine therefore synthesizes a temporal record
   * for it, which is also what gives working memory something to hold.
   */
  { symbol: "VANISHED", id: KRYSTAL_TOKEN_RANGES.structure[0] + 2, tokenClass: "structure", doc: "perceived before, absent now" },
  { symbol: "NOMINATIVE", id: KRYSTAL_TOKEN_RANGES.structure[0], tokenClass: "structure", doc: "case marker: subject of the relation" },
  { symbol: "ACCUSATIVE", id: KRYSTAL_TOKEN_RANGES.structure[0] + 1, tokenClass: "structure", doc: "case marker: object of the relation" },
  // Band symbols. Reserved because each is a pure consequence of a threshold in
  // QUANTITY_BANDS: owning the cut without owning the token it produces would
  // leave the two free to drift apart.
  ...Object.values(BAND_SYMBOLS).map((symbol) => ({
    symbol,
    id: BAND_TOKEN_IDS[symbol]!,
    tokenClass: "structure" as KrystalTokenClass,
  })),
];

export class AgentGrammarError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AgentGrammarError";
  }
}

/** One symbol a simulation declares. */
export interface SimGrammarSymbol {
  readonly symbol: string;
  readonly tokenId: number;
  readonly tokenClass: KrystalTokenClass;
  readonly flags?: number;
  readonly arity?: number;
  readonly semanticTypeToken?: number;
  readonly inverseToken?: number;
}

/**
 * A numeric field the simulation reports, declared once rather than per record.
 *
 * The kind is a property of the field, not of an observation, so it belongs
 * here: a field that was `count` last frame and `unipolar` this one would be
 * incoherent, and repeating it on every record invites exactly that. Declaring
 * it once also keeps the wire small — a snapshot then carries only names and
 * numbers.
 *
 * `polarity` names what each direction of a signed field MEANS, which is domain
 * knowledge the engine does not have: negative comfort is FEEL_BAD, negative
 * radial motion is RECEDING. The engine decides only how far from zero a value
 * must be before it has a direction at all.
 */
export interface SimQuantityField {
  readonly field: string;
  readonly kind: v1_0_0.QuantityKind;
  /** Required for `signed`. Both symbols must exist in the grammar. */
  readonly polarity?: Polarity;
}

/**
 * An action the creature can perform: one binary relation, with what may fill
 * each side.
 *
 * The relation names a grammar symbol, and crucially the SAME symbol an
 * observed event uses. That sharing is what grounds the catalog: an entry is
 * otherwise an opaque option the creature has no way to interpret, whereas a
 * token it has also seen someone else act out carries a meaning learned by
 * watching. Doing and seeing done end up in the same representation.
 *
 * Omitting `object` declares a unary action, whose object mirrors its subject —
 * the reflexive reading rather than a missing argument.
 */
export interface SimActionV2 {
  readonly relation: string;
  /** What may act. Defaults to the actor itself. */
  readonly subject?: RelationRoleV2;
  /** What may be acted upon. Omitted means unary. */
  readonly object?: RelationRoleV2;
}

export interface RelationRoleV2 {
  /** Grammar symbols this side accepts; empty accepts anything perceived. */
  readonly accepts?: readonly string[];
  /** Bands the filler may be drawn from. */
  readonly candidateBands?: readonly string[];
}

/**
 * Schema id stamped on catalog records, so the intent mask can find them.
 *
 * Engine-owned and fixed: which records are "things I could do" is not a fact
 * about any world.
 */
export const CATALOG_SCHEMA_ID = 0xfe;

/**
 * Record schema id for a grammar symbol: the token class of its symbol.
 *
 * This was `token & 0xff`, and that was not a projection but a collision. A
 * 163-symbol grammar spread across the class ranges produced 30 pairs sharing
 * a low byte, so `resource:Apple` and `field:Satiation` were the same schema —
 * which, while acceptance was schema-keyed, meant a role admitting apples also
 * admitted satiation readings, silently, and trained on whichever the head
 * pointed at.
 *
 * The class is the right granularity for what a schema id actually still does.
 * Identity lives in the record's tokens (that is where acceptance now reads
 * it), so the schema id is left with one job: a coarse "what kind of thing is
 * this record" feature for the embedding, plus keeping catalog records apart
 * from everything else. A class answers exactly that, is stable no matter how
 * large a world's vocabulary grows, and can never reach CATALOG_SCHEMA_ID.
 */
export function schemaIdOf(grammar: CompiledGrammar, symbol: string): number {
  const token = grammar.tokenBySymbol.get(symbol);
  if (token === undefined) throw new AgentGrammarError(`symbol '${symbol}' is not in the grammar`);
  const classId = tokenClassIndex(token);
  if (classId < 0) {
    throw new AgentGrammarError(
      `symbol '${symbol}' has token id 0x${token.toString(16)}, which is in no token class`,
    );
  }
  return classId;
}

/** Pack accepted token ids into a role's fixed-width acceptance list. */
function acceptedTokenList(symbol: string, tokens: readonly number[]): number[] {
  if (tokens.length > BRAIN_LIMITS.maxRoleAcceptedTokens) {
    throw new AgentGrammarError(
      `action '${symbol}' names ${tokens.length} accepted symbols for one role, over the ${BRAIN_LIMITS.maxRoleAcceptedTokens} a role may carry. ` +
        "A list that long is usually a missing word: declare a category symbol the records carry and accept that instead.",
    );
  }
  const list = new Array<number>(BRAIN_LIMITS.maxRoleAcceptedTokens).fill(0);
  for (let i = 0; i < tokens.length; i++) list[i] = tokens[i]!;
  return list;
}

/**
 * The compiled catalog the selection machinery consumes.
 *
 * Structurally identical to what the fixture catalog produces, and made of
 * nothing but schema types on purpose: the forward pass can then mask against a
 * catalog compiled from any grammar, which it could not while acceptance was a
 * capability name resolved in one particular vocabulary's tables.
 */
export interface CompiledCatalog {
  readonly header: v1_0_0.ActionIntentCatalogHeader;
  readonly descriptors: v1_0_0.ActionIntentDescriptor[];
}

/**
 * Compile declared actions into catalog descriptors.
 *
 * A role with no `accepts` admits everything the frame offers, because a world
 * that has not narrowed an action has not thereby forbidden it. Narrowing is
 * how a world says "only edible things"; silence is not a prohibition.
 */
export function compileActionCatalog(grammar: CompiledGrammar): CompiledCatalog {
  const role = (
    spec: RelationRoleV2 | undefined,
    relation: string,
  ): v1_0_0.RelationRoleDescriptor => ({
    roleToken: 0,
    valueKind: "context_ref",
    // Acceptance travels as TOKENS, which is what lets a role name a class:
    // a record carries its identity and its categories side by side, so
    // `category:Edible` admits every edible thing including ones this world
    // has not shown the creature yet.
    acceptedTokens: acceptedTokenList(
      relation,
      (spec?.accepts ?? []).map((symbol) => {
        const token = grammar.tokenBySymbol.get(symbol);
        if (token === undefined) {
          throw new AgentGrammarError(`action '${relation}' accepts '${symbol}', which is not in the grammar`);
        }
        return token;
      }),
    ),
    candidateBandMask: bandMaskOf(spec?.candidateBands),
    // An empty acceptance set means "anything", and the mask alone cannot say
    // that apart from "nothing"; the flag carries the difference.
    flags: (spec?.accepts ?? []).length === 0 ? RELATION_ROLE_FLAGS.acceptsAny : 0,
    reserved0: 0,
  });

  // An undeclared subject is the actor, not "anything". A world that does not
  // say who acts has not thereby said that anyone may: the creature's own body
  // is the default, and for a unary action the object mirrors it, which is what
  // makes CRY resolve to the crier rather than to whatever happens to be in
  // view.
  const ACTOR: RelationRoleV2 = { candidateBands: ["body"] };

  const descriptors = grammar.actions.map((action, intentId) => {
    const unary = action.object === undefined;
    return {
      intentId,
      actionToken: grammar.tokenBySymbol.get(action.relation)!,
      semanticIntentToken: grammar.tokenBySymbol.get(action.relation)!,
      domain: "external" as const,
      subjectSchemaId: 0,
      flags: unary ? ACTION_INTENT_FLAGS.canonicallyReflexive : 0,
      effectClassToken: 0,
      capabilityClassToken: 0,
      preconditionClassToken: 0,
      preferredControllerRole: 0,
      reserved0: 0,
      reserved1: 0,
      subjectRole: role(action.subject ?? ACTOR, action.relation),
      objectRole: role(action.object ?? action.subject ?? ACTOR, action.relation),
    };
  });

  const words: number[] = [];
  for (const d of descriptors) {
    words.push(d.intentId, d.actionToken, d.flags);
    for (const r of [d.subjectRole, d.objectRole]) {
      words.push(...r.acceptedTokens, r.candidateBandMask, r.flags);
    }
  }
  const hash = hashU32s(words);

  return {
    header: {
      version: 0,
      intentCount: descriptors.length,
      relationArity: BRAIN_LIMITS.relationArity,
      flags: 0,
      catalogHashLo: hash.lo,
      catalogHashHi: hash.hi,
      reserved0: 0,
      reserved1: 0,
    },
    descriptors,
  };
}

function bandMaskOf(bands: readonly string[] | undefined): number {
  if (!bands) return 0;
  let mask = 0;
  for (const band of bands) {
    const index = BRAIN_FRAME_BANDS.findIndex((entry) => entry.kind === band);
    if (index < 0) throw new AgentGrammarError(`unknown candidate band '${band}'`);
    mask |= 1 << index;
  }
  return mask >>> 0;
}

/** A simulation's declared vocabulary. */
export interface SimGrammar {
  readonly contract: "pira-grammar@2";
  readonly symbols: readonly SimGrammarSymbol[];
  /** Numeric fields and how to read them. */
  readonly quantities?: readonly SimQuantityField[];
  /**
   * Declared only by a world that has space. Motion is not engine structure —
   * a world need not have anywhere to move — so its polarity is named here
   * rather than reserved, and a snapshot carrying motion without this
   * declaration is refused instead of guessed at.
   */
  readonly motion?: {
    readonly radial: Polarity;
    readonly angular?: Polarity;
  };
  /**
   * What the creature can do. These become records in the frame's catalog band,
   * which is how the intent selector has anything to score at all: an agent
   * that cannot see its own options never proposes, never acts, and so never
   * generates the experience it would learn from.
   */
  readonly actions?: readonly SimActionV2[];
}

/** A resolved vocabulary: reserved block first, simulation symbols after. */
export interface CompiledGrammar {
  readonly header: v1_0_0.VocabManifestHeader;
  readonly entries: readonly v1_0_0.VocabManifestEntry[];
  /** Token id -> embedding row, for the forward and backward passes. */
  readonly tokenRows: Uint32Array;
  /** Symbol -> token id, for lowering a snapshot without a hardcoded map. */
  readonly tokenBySymbol: ReadonlyMap<string, number>;
  /** Token id -> symbol, for telemetry and readable failures. */
  readonly symbolByToken: ReadonlyMap<number, string>;
  /** Declared numeric fields, by name. */
  readonly quantities: ReadonlyMap<string, SimQuantityField>;
  /** Motion polarity, when this world has space. */
  readonly motion: SimGrammar["motion"];
  /** Declared actions, in catalog order; the index is the intent id. */
  readonly actions: readonly SimActionV2[];
  readonly reservedCount: number;
}

/**
 * Compile the reserved block plus a simulation grammar into one manifest.
 *
 * Rejects rather than drops. A symbol the engine cannot place is a compile
 * error here, not a `null` at lowering time — the previous bridge silently
 * discarded every resource outside a hardcoded set of six, which meant a
 * simulation could show the brain a tree and the brain would report seeing
 * nothing at all, with no diagnostic anywhere.
 */
export function compileGrammar(grammar: SimGrammar): CompiledGrammar {
  if (grammar.contract !== "pira-grammar@2") {
    throw new AgentGrammarError(`Unsupported grammar contract '${grammar.contract}'`);
  }

  const entries: v1_0_0.VocabManifestEntry[] = [];
  const tokenBySymbol = new Map<string, number>();
  const symbolByToken = new Map<number, string>();
  const seenIds = new Set<number>();

  const push = (
    symbol: string,
    tokenId: number,
    tokenClass: KrystalTokenClass,
    extra: { flags?: number; arity?: number; semanticTypeToken?: number; inverseToken?: number },
  ): void => {
    if (seenIds.has(tokenId)) {
      throw new AgentGrammarError(
        `${symbol}: token id 0x${tokenId.toString(16)} already taken by ${symbolByToken.get(tokenId)}`,
      );
    }
    if (tokenBySymbol.has(symbol)) throw new AgentGrammarError(`duplicate symbol '${symbol}'`);
    seenIds.add(tokenId);
    tokenBySymbol.set(symbol, tokenId);
    symbolByToken.set(tokenId, symbol);
    entries.push({
      tokenId,
      tokenClass,
      flags: extra.flags ?? 0,
      arity: extra.arity ?? 0,
      semanticTypeToken: extra.semanticTypeToken ?? 0,
      inverseToken: extra.inverseToken ?? 0,
      // Manifest index == embedding row.
      reserved0: entries.length,
      reserved1: 0,
    });
  };

  for (const reserved of RESERVED_SYMBOLS) {
    push(reserved.symbol, reserved.id, reserved.tokenClass, { flags: reserved.flags });
  }
  const reservedCount = entries.length;

  for (const symbol of grammar.symbols) {
    if (symbol.tokenId <= RESERVED_TOKEN_END) {
      throw new AgentGrammarError(
        `${symbol.symbol}: token id 0x${symbol.tokenId.toString(16)} is inside the engine-reserved block (<= 0x${RESERVED_TOKEN_END.toString(16)})`,
      );
    }
    if (symbol.tokenId > KRYSTAL_ABI.semanticEnd) {
      throw new AgentGrammarError(
        `${symbol.symbol}: token id 0x${symbol.tokenId.toString(16)} is outside the embedded semantic half; reference-half symbols are bound at runtime and carry no row`,
      );
    }
    if (!SIM_TOKEN_CLASSES.includes(symbol.tokenClass)) {
      throw new AgentGrammarError(
        `${symbol.symbol}: class '${symbol.tokenClass}' is engine-reserved`,
      );
    }
    const range = KRYSTAL_TOKEN_RANGES[symbol.tokenClass as keyof typeof KRYSTAL_TOKEN_RANGES];
    if (symbol.tokenId < range[0] || symbol.tokenId > range[1]) {
      throw new AgentGrammarError(
        `${symbol.symbol}: token id 0x${symbol.tokenId.toString(16)} is outside its class range 0x${range[0].toString(16)}..0x${range[1].toString(16)}`,
      );
    }
    push(symbol.symbol, symbol.tokenId, symbol.tokenClass, symbol);
  }

  if (entries.length > KRYSTAL_ABI.semanticEmbeddingRows) {
    throw new AgentGrammarError(
      `${entries.length} symbols exceed the ${KRYSTAL_ABI.semanticEmbeddingRows}-row embedding table`,
    );
  }

  const quantities = new Map<string, SimQuantityField>();
  for (const field of grammar.quantities ?? []) {
    if (quantities.has(field.field)) throw new AgentGrammarError(`duplicate quantity field '${field.field}'`);
    if (field.kind === "signed") {
      if (!field.polarity) {
        throw new AgentGrammarError(
          `${field.field}: a signed field must name what each direction means; the engine knows the threshold, not the meaning`,
        );
      }
      for (const symbol of [field.polarity.negative, field.polarity.positive]) {
        if (!tokenBySymbol.has(symbol)) {
          throw new AgentGrammarError(`${field.field}: polarity symbol '${symbol}' is not in the grammar`);
        }
      }
    } else if (field.polarity) {
      throw new AgentGrammarError(`${field.field}: only a signed field has polarity`);
    }
    quantities.set(field.field, field);
  }

  for (const polarity of [grammar.motion?.radial, grammar.motion?.angular]) {
    if (!polarity) continue;
    for (const symbol of [polarity.negative, polarity.positive]) {
      if (!tokenBySymbol.has(symbol)) {
        throw new AgentGrammarError(`motion polarity symbol '${symbol}' is not in the grammar`);
      }
    }
  }

  const actions = grammar.actions ?? [];
  if (actions.length === 0) {
    throw new AgentGrammarError(
      "the grammar declares no actions: the intent selector scores catalog records, so a creature with none " +
        "never proposes, never acts, and never generates the experience it would learn from — which from " +
        "outside is indistinguishable from a timid policy. Declare actions, or say explicitly that this agent " +
        "is meant to be inert.",
    );
  }
  const catalogCapacity = BRAIN_FRAME_BANDS.find((band) => band.kind === "catalog")!.recordCapacity;
  if (actions.length > catalogCapacity) {
    throw new AgentGrammarError(
      `${actions.length} actions exceed the catalog band's ${catalogCapacity} records; the band is the ceiling on how many a world may declare`,
    );
  }
  for (const action of actions) {
    if (!tokenBySymbol.has(action.relation)) {
      throw new AgentGrammarError(`action '${action.relation}' is not in the grammar`);
    }
    for (const role of [action.subject, action.object]) {
      for (const accepted of role?.accepts ?? []) {
        if (!tokenBySymbol.has(accepted)) {
          throw new AgentGrammarError(`action '${action.relation}' accepts '${accepted}', which is not in the grammar`);
        }
      }
    }
  }

  const words: number[] = [];
  for (const entry of entries) {
    words.push(
      entry.tokenId,
      entry.flags,
      entry.arity,
      entry.semanticTypeToken,
      entry.inverseToken,
      entry.reserved0,
    );
  }
  const hash = hashU32s(words);

  // Token id -> embedding row. Semantic ids take their manifest index; the
  // whole reference half folds into a shared pool, because a reference is a
  // pointer whose meaning changes every frame and a private learned row per
  // reference would be memorising noise.
  const tokenRows = new Uint32Array(KRYSTAL_ABI.tokenSpaceSize);
  for (const entry of entries) tokenRows[entry.tokenId] = entry.reserved0;
  const refBase = KRYSTAL_ABI.semanticEmbeddingRows;
  for (let id = KRYSTAL_ABI.refSpaceStart; id <= KRYSTAL_ABI.refSpaceEnd; id++) {
    tokenRows[id] = refBase + ((id - KRYSTAL_ABI.refSpaceStart) % KRYSTAL_ABI.refEmbeddingRows);
  }

  return {
    header: {
      tokenAbiVersion: KRYSTAL_ABI.tokenAbiVersion,
      manifestVersion: 0,
      vocabSize: KRYSTAL_ABI.semanticVocabSize,
      activeTokenCount: entries.length,
      embeddingRows: KRYSTAL_ABI.semanticEmbeddingRows,
      manifestHashLo: hash.lo,
      manifestHashHi: hash.hi,
      reserved0: 0,
      reserved1: 0,
    },
    entries,
    tokenRows,
    tokenBySymbol,
    symbolByToken,
    quantities,
    motion: grammar.motion,
    actions,
    reservedCount,
  };
}

/** Weights plus the grammar they were trained against. */
export interface AgentCheckpoint {
  readonly manifestHashLo: number;
  readonly manifestHashHi: number;
  readonly weights: Float32Array;
}

export interface CreateAgentInput {
  readonly grammar: SimGrammar;
  readonly checkpoint?: AgentCheckpoint | undefined;
}

export interface Agent {
  readonly grammar: CompiledGrammar;
  readonly weights: Float32Array | undefined;
}

export class AgentCheckpointMismatchError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AgentCheckpointMismatchError";
  }
}

/**
 * Build an agent from a simulation grammar, optionally resuming a checkpoint.
 *
 * A checkpoint carries the hash of the manifest it was trained against, and a
 * mismatch is refused. The check exists because the alternative is silent: row
 * assignment follows manifest order, so inserting one symbol shifts every later
 * row and each trained vector quietly starts denoting a different concept.
 * Training would continue and loss would even fall — against scrambled
 * meanings. Failing here also means a simulation is free to reorder or rename
 * its vocabulary; it only costs a retrain, and it says so.
 */
export function createAgent(input: CreateAgentInput): Agent {
  const grammar = compileGrammar(input.grammar);
  const checkpoint = input.checkpoint;
  if (!checkpoint) return { grammar, weights: undefined };

  if (
    checkpoint.manifestHashLo !== grammar.header.manifestHashLo ||
    checkpoint.manifestHashHi !== grammar.header.manifestHashHi
  ) {
    throw new AgentCheckpointMismatchError(
      "checkpoint was trained against a different grammar: embedding rows follow manifest order, " +
        "so these weights no longer denote the symbols this grammar declares. Retrain, or restore the grammar it was trained with.",
    );
  }
  return { grammar, weights: checkpoint.weights };
}
