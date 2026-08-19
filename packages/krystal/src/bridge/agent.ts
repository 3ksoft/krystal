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

export const RESERVED_TOKEN_END = KRYSTAL_TOKEN_RANGES.structure[1];

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
  { symbol: "NOT", id: KRYSTAL_SENTINEL_TOKENS.not, tokenClass: "system" },
  { symbol: "ISA", id: KRYSTAL_SENTINEL_TOKENS.isA, tokenClass: "system" },
  { symbol: "PARTOF", id: KRYSTAL_SENTINEL_TOKENS.partOf, tokenClass: "system" },
  { symbol: "NOT", id: KRYSTAL_SENTINEL_TOKENS.not, tokenClass: "system" },
  { symbol: "UNKNOWN", id: KRYSTAL_SENTINEL_TOKENS.unknown, tokenClass: "system", doc: "a referent exists, identity not known; licenses a query" },
  { symbol: "BEGIN", id: KRYSTAL_SENTINEL_TOKENS.begin, tokenClass: "system", flags: TOKEN_FLAGS.structural },
  { symbol: "END", id: KRYSTAL_SENTINEL_TOKENS.end, tokenClass: "system", flags: TOKEN_FLAGS.structural },
  { symbol: "VOID", id: KRYSTAL_SENTINEL_TOKENS.void, tokenClass: "system", doc: "sensed emptiness; a percept, not a gap" },
  { symbol: "UNAVAILABLE", id: KRYSTAL_SENTINEL_TOKENS.unavailable, tokenClass: "system", doc: "the sense is not reporting; also a percept" },
  { symbol: "SOMETHING", id: KRYSTAL_SENTINEL_TOKENS.something, tokenClass: "system", doc: "existential; the target of a query, not a trigger for one" },
  { symbol: "VANISHED", id: KRYSTAL_TOKEN_RANGES.structure[0] + 2, tokenClass: "structure", doc: "perceived before, absent now" },
  { symbol: "NOMINATIVE", id: KRYSTAL_TOKEN_RANGES.structure[0], tokenClass: "structure", doc: "case marker: subject of the relation" },
  { symbol: "ACCUSATIVE", id: KRYSTAL_TOKEN_RANGES.structure[0] + 1, tokenClass: "structure", doc: "case marker: object of the relation" },
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

export interface SimGrammarSymbol {
  readonly symbol: string;
  readonly tokenId: number;
  readonly tokenClass: KrystalTokenClass;
  readonly flags?: number;
  readonly arity?: number;
  readonly semanticTypeToken?: number;
  readonly inverseToken?: number;
}

export interface SimQuantityField {
  readonly field: string;
  readonly kind: v1_0_0.QuantityKind;
  readonly polarity?: Polarity;
}

export interface SimActionV2 {
  readonly relation: string;
  readonly subject?: RelationRoleV2;
  readonly object?: RelationRoleV2;
}

export interface RelationRoleV2 {
  readonly accepts?: readonly string[];
  readonly candidateBands?: readonly string[];
}

export const CATALOG_SCHEMA_ID = 0xfe;

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

export interface CompiledCatalog {
  readonly header: v1_0_0.ActionIntentCatalogHeader;
  readonly descriptors: v1_0_0.ActionIntentDescriptor[];
}

export function compileActionCatalog(grammar: CompiledGrammar): CompiledCatalog {
  const role = (
    spec: RelationRoleV2 | undefined,
    relation: string,
  ): v1_0_0.RelationRoleDescriptor => ({
    roleToken: 0,
    valueKind: "context_ref",
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
            flags: (spec?.accepts ?? []).length === 0 ? RELATION_ROLE_FLAGS.acceptsAny : 0,
    reserved0: 0,
  });

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

export interface SimGrammar {
  readonly contract: "pira-grammar@2";
  readonly symbols: readonly SimGrammarSymbol[];
  readonly quantities?: readonly SimQuantityField[];
  readonly motion?: {
    readonly radial: Polarity;
    readonly angular?: Polarity;
  };
  readonly actions?: readonly SimActionV2[];
}

export interface CompiledGrammar {
  readonly header: v1_0_0.VocabManifestHeader;
  readonly entries: readonly v1_0_0.VocabManifestEntry[];
  readonly tokenRows: Uint32Array;
  readonly tokenBySymbol: ReadonlyMap<string, number>;
  readonly symbolByToken: ReadonlyMap<number, string>;
  readonly quantities: ReadonlyMap<string, SimQuantityField>;
  readonly motion: SimGrammar["motion"];
  readonly actions: readonly SimActionV2[];
  readonly reservedCount: number;
}

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
