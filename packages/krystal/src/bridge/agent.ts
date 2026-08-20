import {
  ACTION_INTENT_FLAGS,
  BRAIN_FRAME_BANDS,
  BRAIN_LIMITS,
  KRYSTAL_ABI,
  RELATION_ROLES,
  RELATION_ROLE_FLAGS,
  RELATION_ROLE_INDEX,
  KRYSTAL_SENTINEL_TOKENS,
  KRYSTAL_TOKEN_RANGES,
  TOKEN_FLAGS,
  tokenClassIndex,
  type RelationRoleName,
} from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import type { v1_0_0 as world } from "../../../schema/generated/world.types.ts";
import { hashU32s } from "../hash.ts";
import { BAND_SYMBOLS, BAND_TOKEN_IDS } from "./quantize.ts";

type KrystalTokenClass = v1_0_0.KrystalTokenClass;

export const RESERVED_TOKEN_END = KRYSTAL_TOKEN_RANGES.structure[1];

/**
 * Token classes a simulation may assign to.
 *
 * `system` and `structure` belong to the engine: a world symbol landing there
 * would redefine a sentinel the runtime branches on.
 */
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
  { symbol: "BEFORE", id: KRYSTAL_SENTINEL_TOKENS.before, tokenClass: "system", doc: "temporal prime: the earlier side of a transition" },
  { symbol: "NOW", id: KRYSTAL_SENTINEL_TOKENS.now, tokenClass: "system" },
  { symbol: "THEN", id: KRYSTAL_SENTINEL_TOKENS.then, tokenClass: "system", doc: "temporal prime: the later side of a transition" },
  { symbol: "WANT", id: KRYSTAL_SENTINEL_TOKENS.want, tokenClass: "system", doc: "operator: reached for rather than done; never self-asserted" },
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

export class VocabularyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "VocabularyError";
  }
}

export const CATALOG_SCHEMA_ID = 0xfe;

/**
 * A world's vocabulary, compiled into what the engine indexes by.
 *
 * `tokenRows` is the whole reason order matters: an embedding row is a learned
 * vector indexed by MANIFEST POSITION, so inserting a symbol in the middle
 * redefines every later row and training continues against scrambled meanings.
 */
export interface CompiledVocabulary {
  readonly header: v1_0_0.VocabManifestHeader;
  readonly entries: readonly v1_0_0.VocabManifestEntry[];
  readonly tokenRows: Uint32Array;
  readonly tokenBySymbol: ReadonlyMap<string, number>;
  readonly symbolByToken: ReadonlyMap<number, string>;
  readonly quantities: ReadonlyMap<string, world.WorldQuantity>;
  /** Sensory channels this world declares, by symbol. */
  readonly channels: ReadonlyMap<string, world.WorldChannel>;
  readonly relations: readonly world.WorldRelation[];
  readonly reservedCount: number;
}

export function schemaIdOf(vocabulary: CompiledVocabulary, symbol: string): number {
  const token = vocabulary.tokenBySymbol.get(symbol);
  if (token === undefined) throw new VocabularyError(`symbol '${symbol}' is not in the vocabulary`);
  const classId = tokenClassIndex(token);
  if (classId < 0) {
    throw new VocabularyError(
      `symbol '${symbol}' has token id 0x${token.toString(16)}, which is in no token class`,
    );
  }
  return classId;
}

export function compileVocabulary(vocabulary: world.WorldVocabulary): CompiledVocabulary {
  if (vocabulary.contract !== "krystal-world@3") {
    throw new VocabularyError(`Unsupported world contract '${vocabulary.contract}'`);
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
      throw new VocabularyError(
        `${symbol}: token id 0x${tokenId.toString(16)} already taken by ${symbolByToken.get(tokenId)}`,
      );
    }
    if (tokenBySymbol.has(symbol)) throw new VocabularyError(`duplicate symbol '${symbol}'`);
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

  for (const symbol of vocabulary.symbols) {
    if (symbol.tokenId <= RESERVED_TOKEN_END) {
      throw new VocabularyError(
        `${symbol.symbol}: token id 0x${symbol.tokenId.toString(16)} is inside the engine-reserved block (<= 0x${RESERVED_TOKEN_END.toString(16)})`,
      );
    }
    if (symbol.tokenId > KRYSTAL_ABI.semanticEnd) {
      throw new VocabularyError(
        `${symbol.symbol}: token id 0x${symbol.tokenId.toString(16)} is outside the embedded semantic half; reference-half symbols are bound at runtime and carry no row`,
      );
    }
    if (!SIM_TOKEN_CLASSES.includes(symbol.tokenClass as KrystalTokenClass)) {
      throw new VocabularyError(`${symbol.symbol}: class '${symbol.tokenClass}' is engine-reserved`);
    }
    const range = KRYSTAL_TOKEN_RANGES[symbol.tokenClass as keyof typeof KRYSTAL_TOKEN_RANGES];
    if (symbol.tokenId < range[0] || symbol.tokenId > range[1]) {
      throw new VocabularyError(
        `${symbol.symbol}: token id 0x${symbol.tokenId.toString(16)} is outside its class range 0x${range[0].toString(16)}..0x${range[1].toString(16)}`,
      );
    }
    push(symbol.symbol, symbol.tokenId, symbol.tokenClass as KrystalTokenClass, symbol);
  }

  if (entries.length > KRYSTAL_ABI.semanticEmbeddingRows) {
    throw new VocabularyError(
      `${entries.length} symbols exceed the ${KRYSTAL_ABI.semanticEmbeddingRows}-row embedding table`,
    );
  }

  // Channels are ordinary symbols: the engine has no sense inventory of its
  // own, so a channel must be declared like anything else the world names.
  const channels = new Map<string, world.WorldChannel>();
  for (const channel of vocabulary.channels) {
    if (channels.has(channel.symbol)) throw new VocabularyError(`duplicate channel '${channel.symbol}'`);
    if (!tokenBySymbol.has(channel.symbol)) {
      throw new VocabularyError(
        `channel '${channel.symbol}' is not in the vocabulary; a sense is a symbol like any other and needs its own token`,
      );
    }
    channels.set(channel.symbol, channel);
  }
  if (channels.size === 0) {
    throw new VocabularyError(
      "the world declares no sensory channels: every percept names the sense that produced it, so a creature with none can perceive nothing",
    );
  }

  const quantities = new Map<string, world.WorldQuantity>();
  for (const field of vocabulary.quantities) {
    if (quantities.has(field.field)) throw new VocabularyError(`duplicate quantity field '${field.field}'`);
    if (field.kind === "signed") {
      if (!field.polarity) {
        throw new VocabularyError(
          `${field.field}: a signed field must name what each direction means; the engine knows the threshold, not the meaning`,
        );
      }
      for (const symbol of [field.polarity.negative, field.polarity.positive]) {
        if (!tokenBySymbol.has(symbol)) {
          throw new VocabularyError(`${field.field}: polarity symbol '${symbol}' is not in the vocabulary`);
        }
      }
    } else if (field.polarity) {
      throw new VocabularyError(`${field.field}: only a signed field has polarity`);
    }
    quantities.set(field.field, field);
  }

  const relations = vocabulary.relations;
  if (relations.length === 0) {
    throw new VocabularyError(
      "the world declares no relations: the selector scores catalog records, so a creature with none " +
        "never proposes, never acts, and never generates the experience it would learn from — which from " +
        "outside is indistinguishable from a timid policy. Declare relations, or say explicitly that this " +
        "agent is meant to be inert.",
    );
  }
  const catalogCapacity = BRAIN_FRAME_BANDS.find((band) => band.kind === "catalog")!.recordCapacity;
  if (relations.length > catalogCapacity) {
    throw new VocabularyError(
      `${relations.length} relations exceed the catalog band's ${catalogCapacity} records; the band is the ceiling on how many a world may declare`,
    );
  }
  const seenRelations = new Set<string>();
  for (const relation of relations) {
    if (seenRelations.has(relation.relation)) {
      throw new VocabularyError(`duplicate relation '${relation.relation}'`);
    }
    seenRelations.add(relation.relation);
    if (!tokenBySymbol.has(relation.relation)) {
      throw new VocabularyError(`relation '${relation.relation}' is not in the vocabulary`);
    }
    const seenRoles = new Set<string>();
    for (const role of relation.roles) {
      if (seenRoles.has(role.role)) {
        throw new VocabularyError(`relation '${relation.relation}' declares role '${role.role}' twice`);
      }
      seenRoles.add(role.role);
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
    channels,
    relations,
    reservedCount,
  };
}

export interface CompiledCatalog {
  readonly header: v1_0_0.ActionIntentCatalogHeader;
  readonly descriptors: v1_0_0.ActionIntentDescriptor[];
}

/**
 * Bands a role will never draw a participant from.
 *
 * The catalog holds the creature's OPTIONS, not things in the world, so a
 * relation whose agent could be another catalog entry would be proposing to act
 * on its own menu. The world does not declare this — it is a fact about the
 * frame, not about any world — which is why it is applied here rather than
 * being something a simulation could get wrong.
 */
const CATALOG_BAND_INDEX = BRAIN_FRAME_BANDS.findIndex((band) => band.kind === "catalog");
const PARTICIPANT_BAND_MASK = ~(1 << CATALOG_BAND_INDEX) >>> 0;

/**
 * Compile the declared relations into the catalog the selector scores.
 *
 * Every relation carries all six role slots; the ones it did not declare are
 * inert. There is no privileged subject: an agent role is scored against the
 * bank like any other, and `Self` is simply a candidate it admits.
 *
 * A declared role carries no acceptance set. What may fill it is not something
 * this world gets to assert — the creature learns that by watching — so every
 * role admits whatever the frame holds, and the only constraints are structural.
 */
export function compileRelationCatalog(vocabulary: CompiledVocabulary): CompiledCatalog {
  const tokenOf = (symbol: string, context: string): number => {
    const token = vocabulary.tokenBySymbol.get(symbol);
    if (token === undefined) throw new VocabularyError(`${context}: '${symbol}' is not in the vocabulary`);
    return token;
  };

  const emptyRole = (role: RelationRoleName): v1_0_0.RelationRoleDescriptor => ({
    role,
    roleToken: 0,
    valueKind: "context_ref",
    candidateBandMask: 0,
    flags: 0,
    reserved0: 0,
  });

  const descriptors: v1_0_0.ActionIntentDescriptor[] = vocabulary.relations.map((relation, intentId) => {
    const roles = RELATION_ROLES.map(emptyRole);

    for (const declared of relation.roles) {
      roles[RELATION_ROLE_INDEX[declared.role]] = {
        role: declared.role,
        roleToken: 0,
        valueKind: "context_ref",
        candidateBandMask: PARTICIPANT_BAND_MASK,
        flags: RELATION_ROLE_FLAGS.present | RELATION_ROLE_FLAGS.acceptsAny,
        reserved0: 0,
      };
    }

    // A relation that binds only an agent is reflexive: the patient mirrors it.
    // That is what a unary relation means once roles are named.
    const declaresPatient = relation.roles.some((role) => role.role === "patient");
    const actionToken = tokenOf(relation.relation, `relation '${relation.relation}'`);

    return {
      intentId,
      actionToken,
      semanticIntentToken: actionToken,
      domain: relation.domain ?? "external",
      flags: declaresPatient ? 0 : ACTION_INTENT_FLAGS.canonicallyReflexive,
      effectClassToken: 0,
      capabilityClassToken: 0,
      preconditionClassToken: 0,
      preferredControllerRole: 0,
      reserved0: 0,
      reserved1: 0,
      roles,
    };
  });

  const words: number[] = [];
  for (const descriptor of descriptors) {
    words.push(descriptor.intentId, descriptor.actionToken, descriptor.flags);
    for (const role of descriptor.roles) {
      words.push(role.candidateBandMask, role.flags);
    }
  }
  const hash = hashU32s(words);

  return {
    header: {
      version: 3,
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

/** Weights plus the vocabulary they were trained against. */
export interface AgentCheckpoint {
  readonly manifestHashLo: number;
  readonly manifestHashHi: number;
  readonly weights: Float32Array;
}

export interface CreateAgentInput {
  readonly vocabulary: world.WorldVocabulary;
  readonly checkpoint?: AgentCheckpoint | undefined;
}

export interface Agent {
  readonly vocabulary: CompiledVocabulary;
  readonly weights: Float32Array | undefined;
}

export class AgentCheckpointMismatchError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AgentCheckpointMismatchError";
  }
}

export function createAgent(input: CreateAgentInput): Agent {
  const vocabulary = compileVocabulary(input.vocabulary);
  const checkpoint = input.checkpoint;
  if (!checkpoint) return { vocabulary, weights: undefined };

  if (
    checkpoint.manifestHashLo !== vocabulary.header.manifestHashLo ||
    checkpoint.manifestHashHi !== vocabulary.header.manifestHashHi
  ) {
    throw new AgentCheckpointMismatchError(
      "checkpoint was trained against a different vocabulary: embedding rows follow manifest order, " +
        "so these weights no longer denote the symbols this world declares. Retrain, or restore the vocabulary it was trained with.",
    );
  }
  return { vocabulary, weights: checkpoint.weights };
}
