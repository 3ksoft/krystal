import {
  BRAIN_FRAME_BANDS,
  BRAIN_LIMITS,
  INVALID_U32,
  KRYSTAL_ABI,
  KRYSTAL_SENTINEL_TOKENS,
  RECORD_FLAGS,
  REFERENCE_FLAGS,
  TOKEN_FLAGS,
  type RelationRoleName,
} from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import type { v1_0_0 as world } from "../../../schema/generated/world.types.ts";
import { CATALOG_SCHEMA_ID, schemaIdOf, type CompiledVocabulary } from "./agent.ts";
import { BAND_SYMBOLS, quantize } from "./quantize.ts";

/**
 * Percept -> BrainFrame.
 *
 * This fills the PERCEPTUAL frame only. Memory is resident and has its own
 * lifetime — it is not rewritten from a tick, which is the whole reason the two
 * frames were split. A relation observed now is a percept; whether it also
 * becomes a memory trace is a decision made later, against activation and
 * eviction, not here.
 */

/**
 * The instance id standing for one emitted proposal.
 *
 * Proposals are not records in the frame, so they have no identity of their
 * own; this manufactures a stable one from the `intentRef` the proposal was
 * emitted with, and the reference table then treats it like any other instance.
 */
export function intentInstanceId(intentRef: number): string {
  return `intent:${intentRef}`;
}

/** A relation the engine itself performed, fed back as its own percept. */
export interface PerformedRelation {
  readonly relation: string;
  readonly roles: readonly world.PerceptRoleBinding[];
  readonly quantities?: readonly world.PerceptQuantity[];
}

interface RoleBinding {
  readonly role: RelationRoleName;
  readonly token: number;
}

interface LoweredRecord {
  readonly band: v1_0_0.BrainBandKind;
  readonly modality: v1_0_0.PropositionModality;
  readonly schema: string;
  /** Sensory channel symbol; absent for engine-authored records. */
  readonly channel?: string | undefined;
  readonly instanceId?: string | undefined;
  /** Bound participants — what makes this record a reified relation. */
  readonly roles?: readonly RoleBinding[] | undefined;
  readonly tokens: readonly string[];
  readonly quantities?: readonly world.PerceptQuantity[] | undefined;
  readonly count?: number | undefined;
  readonly salience?: number | undefined;
  readonly observedAt: number;
  readonly emptiness?: "void" | "unavailable" | undefined;
  readonly catalogIndex?: number | undefined;
}

export class LoweringError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LoweringError";
  }
}

export interface BandOverflow {
  readonly band: v1_0_0.BrainBandKind;
  readonly offered: number;
  readonly admitted: number;
}

export interface LoweredFrame {
  readonly frame: v1_0_0.BrainFrame;
  readonly valenceDelta: number | undefined;
  readonly instanceByToken: ReadonlyMap<number, string>;
  readonly overflow: readonly BandOverflow[];
  readonly truncatedRecords: number;
}

export class ReferenceTable {
  private readonly tokenByInstance = new Map<string, number>();
  private readonly instanceByToken = new Map<number, string>();
  private readonly generations = new Map<number, number>();
  private next = KRYSTAL_ABI.refSpaceStart;

  tokenFor(instanceId: string): number {
    const existing = this.tokenByInstance.get(instanceId);
    if (existing !== undefined) return existing;
    if (this.next > KRYSTAL_ABI.refSpaceEnd) {
      throw new LoweringError("reference space exhausted");
    }
    const token = this.next++;
    this.tokenByInstance.set(instanceId, token);
    this.instanceByToken.set(token, instanceId);
    this.generations.set(token, 1);
    return token;
  }

  generationOf(token: number): number {
    return this.generations.get(token) ?? 1;
  }

  instanceFor(token: number): string | undefined {
    return this.instanceByToken.get(token);
  }

  snapshotInstances(): ReadonlyMap<number, string> {
    return new Map(this.instanceByToken);
  }
}

function emptyRecord(): v1_0_0.BrainRecordSlot {
  return {
    header: {
      schemaId: 0,
      band: "system",
      source: "runtime",
      modality: "declarative",
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
      channelToken: 0,
      reserved1: 0,
    },
    tokens: Array.from({ length: BRAIN_LIMITS.recordWidth }, () => KRYSTAL_SENTINEL_TOKENS.pad),
    tokenMeta: Array.from({ length: BRAIN_LIMITS.recordWidth }, () => ({
      fieldId: 0,
      roleToken: 0,
      flags: TOKEN_FLAGS.padding,
      referenceBinding: INVALID_U32,
    })),
    references: Array.from({ length: BRAIN_LIMITS.maxReferencesPerRecord }, () => ({
      localTokenIndex: INVALID_U32,
      fieldId: 0,
      role: "agent" as const,
      flags: 0,
      handle: { tokenId: 0, generation: 0, kind: "none" as const, status: "invalid" as const },
    })),
  };
}

/**
 * The record's own tokens: what it is, who is in it, then what is true of it.
 *
 * A relation's participants are tokens, not merely sidecar entries. The
 * reference table carries the exact handle and the role, but the TOKEN is what
 * attention sees: a participant kept only in the sidecar would be invisible to
 * every head that reads the frame, and the packer refuses such a binding for
 * exactly that reason. So a role costs a token slot, which is the real ceiling
 * on how many roles one relation can usefully bind.
 */
function recordTokens(
  record: LoweredRecord,
  vocabulary: CompiledVocabulary,
  refToken: number | undefined,
): { tokens: number[]; refIndex: number; roleTokenIndices: number[] } {
  const tokens: number[] = [];
  let refIndex = -1;
  const roleTokenIndices: number[] = [];

  const push = (symbol: string): void => {
    const id = vocabulary.tokenBySymbol.get(symbol);
    if (id === undefined) throw new LoweringError(`symbol '${symbol}' is not in the vocabulary`);
    tokens.push(id);
  };

  push(record.schema);
  for (const binding of record.roles ?? []) {
    roleTokenIndices.push(tokens.length);
    tokens.push(binding.token);
  }
  if (refToken !== undefined) {
    if (refToken >= KRYSTAL_ABI.refSpaceStart) refIndex = tokens.length;
    tokens.push(refToken);
  }
  if (record.emptiness) {
    tokens.push(
      record.emptiness === "void"
        ? KRYSTAL_SENTINEL_TOKENS.void
        : KRYSTAL_SENTINEL_TOKENS.unavailable,
    );
  }
  if (record.count !== undefined) {
    for (const symbol of quantize(record.count, "count").tokens) push(symbol);
  }
  for (const symbol of record.tokens) push(symbol);
  for (const quantity of record.quantities ?? []) {
    const declared = vocabulary.quantities.get(quantity.field);
    if (!declared) throw new LoweringError(`quantity field '${quantity.field}' is not declared`);
    for (const symbol of quantize(
      quantity.value,
      declared.kind as v1_0_0.QuantityKind,
      declared.polarity,
    ).tokens) {
      push(symbol);
    }
  }

  return { tokens, refIndex, roleTokenIndices };
}

export function lowerPercept(
  percept: world.Percept,
  vocabulary: CompiledVocabulary,
  references: ReferenceTable,
  previous?: ReadonlyMap<string, { observedAt: number; tokens: readonly number[] }>,
  selfRelations?: readonly PerformedRelation[],
  previousValence?: number,
): LoweredFrame {
  const records: v1_0_0.BrainRecordSlot[] = Array.from(
    { length: BRAIN_LIMITS.frameRecordSlots },
    () => emptyRecord(),
  );

  const byBand = new Map<v1_0_0.BrainBandKind, LoweredRecord[]>();
  const perception: LoweredRecord[] = [];

  for (const record of percept.records) {
    perception.push({
      band: "perception",
      modality: "declarative",
      schema: record.schema,
      channel: record.channel,
      instanceId: record.instanceId,
      tokens: record.tokens,
      quantities: record.quantities,
      count: record.count,
      salience: record.salience,
      observedAt: record.observedAt,
      emptiness: record.emptiness,
    });
  }

  perception.push(
    ...relationRecords(percept, vocabulary, references, selfRelations),
    ...vanishedRecords(percept, references, previous),
  );
  if (perception.length > 0) byBand.set("perception", perception);

  const valenceDelta =
    previousValence === undefined ? undefined : percept.valence - previousValence;
  if (valenceDelta !== undefined) {
    byBand.set("homeostasis", [valenceRecord(percept, valenceDelta)]);
  }

  byBand.set("query", [queryRecord(percept, valenceDelta)]);

  byBand.set(
    "catalog",
    vocabulary.relations.map((relation, index) => ({
      band: "catalog" as const,
      modality: "imperative" as const,
      schema: relation.relation,
      tokens: [],
      salience: 0,
      observedAt: percept.tick,
      catalogIndex: index,
    })),
  );

  const overflow: BandOverflow[] = [];
  let truncatedRecords = 0;
  let activeRecordCount = 0;
  let activeTokenCount = 0;

  for (const [band, offered] of byBand) {
    const layout = BRAIN_FRAME_BANDS.find((candidate) => candidate.kind === band);
    if (!layout) throw new LoweringError(`no layout for band '${band}'`);

    const ranked = [...offered].sort((a, b) => (b.salience ?? 0) - (a.salience ?? 0));
    const admitted = Math.min(ranked.length, layout.recordCapacity);
    if (admitted < ranked.length) {
      overflow.push({ band, offered: ranked.length, admitted });
    }

    for (let i = 0; i < admitted; i++) {
      const raw = ranked[i]!;
      const slot = layout.recordOffset + i;
      const target = records[slot]!;

      // A relation's participants live in its references; an entity record's
      // single identity is folded into its tokens as before.
      const isRelation = raw.roles !== undefined && raw.roles.length > 0;
      const refToken =
        isRelation || raw.instanceId === undefined ? undefined : references.tokenFor(raw.instanceId);
      const { tokens, refIndex, roleTokenIndices } = recordTokens(raw, vocabulary, refToken);

      const width = Math.min(tokens.length, BRAIN_LIMITS.recordWidth);
      if (tokens.length > BRAIN_LIMITS.recordWidth) truncatedRecords++;

      for (let t = 0; t < width; t++) {
        target.tokens[t] = tokens[t]!;
        target.tokenMeta[t] = {
          fieldId: t,
          roleToken: 0,
          flags: t === refIndex ? TOKEN_FLAGS.reference : 0,
          referenceBinding: t === refIndex ? 0 : INVALID_U32,
        };
      }

      let referenceCount = 0;
      if (isRelation) {
        raw.roles!.forEach((binding, index) => {
          const localTokenIndex = roleTokenIndices[index];
          // A participant whose token was truncated away has no binding: the
          // reference must point at a token that survived, or the packer is
          // describing a slot that is not there.
          if (localTokenIndex === undefined || localTokenIndex >= width) return;
          if (referenceCount >= BRAIN_LIMITS.maxReferencesPerRecord) return;
          // A sentinel participant (UNKNOWN, SOMETHING) names no referent, so
          // there is no handle to bind — it stays a plain token. Its role is
          // then carried by position in the relation rather than by a binding,
          // which is the honest encoding: there is nothing to point at.
          if (binding.token < KRYSTAL_ABI.refSpaceStart) return;
          target.references[referenceCount] = {
            localTokenIndex,
            fieldId: localTokenIndex,
            role: binding.role,
            flags: (referenceCount === 0 ? REFERENCE_FLAGS.primary : 0) | REFERENCE_FLAGS.live,
            handle: {
              tokenId: binding.token,
              generation: references.generationOf(binding.token),
              kind: "entity",
              status: "live",
            },
          };
          target.tokenMeta[localTokenIndex] = {
            fieldId: localTokenIndex,
            roleToken: 0,
            flags: TOKEN_FLAGS.reference,
            referenceBinding: referenceCount,
          };
          referenceCount++;
        });
      } else if (refIndex >= 0 && refToken !== undefined) {
        target.references[0] = {
          localTokenIndex: refIndex,
          fieldId: refIndex,
          role: "agent",
          flags: REFERENCE_FLAGS.primary | REFERENCE_FLAGS.live,
          handle: {
            tokenId: refToken,
            generation: references.generationOf(refToken),
            kind: "entity",
            status: "live",
          },
        };
        referenceCount = 1;
      }

      const prior = raw.instanceId === undefined ? undefined : previous?.get(raw.instanceId);
      target.header = {
        ...target.header,
        schemaId:
          raw.catalogIndex === undefined ? schemaIdOf(vocabulary, raw.schema) : CATALOG_SCHEMA_ID,
        band,
        source: "sensor",
        modality: raw.modality,
        flags:
          RECORD_FLAGS.occupied |
          (isRelation ? RECORD_FLAGS.relation : 0) |
          (tokens.length > BRAIN_LIMITS.recordWidth ? RECORD_FLAGS.truncated : 0) |
          (raw.emptiness === "unavailable" ? RECORD_FLAGS.unavailable : 0),
        tokenCount: width,
        referenceCount,
        observedAt: raw.observedAt,
        primaryReference: referenceCount > 0 ? 0 : INVALID_U32,
        salience: raw.salience ?? 0,
        freshness: 1,
        previousObservedAt: prior?.observedAt ?? INVALID_U32,
        changeMagnitude: prior ? changeBetween(prior.tokens, tokens.slice(0, width)) : 0,
        channelToken:
          raw.channel === undefined ? 0 : (vocabulary.tokenBySymbol.get(raw.channel) ?? 0),
      };

      activeRecordCount++;
      activeTokenCount += width;
    }
  }

  const bands: v1_0_0.BrainBandState[] = BRAIN_FRAME_BANDS.map((layout) => {
    const spill = overflow.find((entry) => entry.band === layout.kind);
    const offered = byBand.get(layout.kind)?.length ?? 0;
    return {
      kind: layout.kind,
      activeRecords: Math.min(offered, layout.recordCapacity),
      activeTokens: 0,
      overflowRecords: spill ? spill.offered - spill.admitted : 0,
      truncatedRecords: 0,
      revision: 0,
      flags: 0,
      reserved0: 0,
    };
  });

  const frame: v1_0_0.BrainFrame = {
    header: {
      tokenAbiVersion: KRYSTAL_ABI.tokenAbiVersion,
      architectureVersion: KRYSTAL_ABI.architectureVersion,
      layoutVersion: KRYSTAL_ABI.frameLayoutVersion,
      tick: percept.tick,
      snapshot: percept.tick,
      deltaMillis: percept.deltaMillis,
      activeRecordCount,
      activeTokenCount,
      activeQueryRecord: INVALID_U32,
      actorRecord: 2,
      frameRevision: percept.tick,
      memoryRevision: 0,
      intentRevision: 0,
      flags: 0,
    },
    bands,
    records,
  };

  return {
    frame,
    valenceDelta,
    instanceByToken: references.snapshotInstances(),
    overflow,
    truncatedRecords,
  };
}

/**
 * The standing query: the row the selectors score candidates against.
 *
 * Engine-authored, because asking is not something the world reports — it is
 * the shape of the creature's own turn.
 */
function queryRecord(
  percept: world.Percept,
  valenceDelta: number | undefined,
): LoweredRecord {
  const tokens =
    valenceDelta === undefined
      ? []
      : quantize(Math.max(-1, Math.min(1, valenceDelta)), "signed", {
          negative: BAND_SYMBOLS.worse,
          positive: BAND_SYMBOLS.better,
        }).tokens;
  return {
    band: "query",
    modality: "interrogative",
    schema: BAND_SYMBOLS.neither,
    tokens,
    salience: 1,
    observedAt: percept.tick,
  };
}

function valenceRecord(percept: world.Percept, delta: number): LoweredRecord {
  const banded = quantize(Math.max(-1, Math.min(1, delta)), "signed", {
    negative: BAND_SYMBOLS.worse,
    positive: BAND_SYMBOLS.better,
  });
  return {
    band: "homeostasis",
    modality: "declarative",
    schema: banded.tokens[0]!,
    tokens: banded.tokens.slice(1),
    salience: Math.abs(delta),
    observedAt: percept.tick,
  };
}

/**
 * Perceived relations, plus the creature's own acts fed back to it.
 *
 * Its own acts are not sent by the world — the engine emitted them and already
 * knows — but they are perceived all the same, and sharing one shape with
 * everyone else's is what makes the agency distinction free: the same relation
 * differs only in who stands in the agent role.
 */
function relationRecords(
  percept: world.Percept,
  vocabulary: CompiledVocabulary,
  references: ReferenceTable,
  selfRelations?: readonly PerformedRelation[],
): LoweredRecord[] {
  const operandToken = (operand: world.PerceptOperand): number => {
    switch (operand.kind) {
      case "instance":
        return references.tokenFor(operand.instanceId);
      case "symbol": {
        const token = vocabulary.tokenBySymbol.get(operand.symbol);
        if (token === undefined) {
          throw new LoweringError(`symbol '${operand.symbol}' is not in the vocabulary`);
        }
        return token;
      }
      case "unknown":
        return KRYSTAL_SENTINEL_TOKENS.unknown;
      case "something":
        return KRYSTAL_SENTINEL_TOKENS.something;
      case "intent":
        // The creature's own earlier reach, given the same kind of identity as
        // any other participant. It is a thing that happened and can be pointed
        // at, so it earns a reference like an entity does.
        return references.tokenFor(intentInstanceId(operand.intentRef));
    }
  };

  const bind = (roles: readonly world.PerceptRoleBinding[]): RoleBinding[] =>
    roles.map((binding) => ({
      role: binding.role as RelationRoleName,
      token: operandToken(binding.operand),
    }));

  const records: LoweredRecord[] = [];

  for (const relation of percept.relations ?? []) {
    records.push({
      band: "perception",
      // A standing relation is a fact; a punctual one is a transition, and the
      // implicative modality is what marks a 'before -> then' edge.
      modality: relation.aspect === "state" ? "declarative" : "implicative",
      schema: relation.relation,
      channel: relation.channel,
      roles: bind(relation.roles),
      tokens: [],
      quantities: relation.quantities,
      salience: relation.salience ?? 0.5,
      observedAt: relation.observedAt,
    });
  }

  for (const performed of selfRelations ?? []) {
    records.push({
      band: "perception",
      modality: "implicative",
      schema: performed.relation,
      roles: bind(performed.roles),
      tokens: [],
      quantities: performed.quantities,
      salience: 1,
      observedAt: percept.tick,
    });
  }

  return records;
}

/**
 * What is gone.
 *
 * A vanished thing is precisely what is NOT perceived, so no world can report
 * it. The engine derives it from a stable instanceId missing where it was.
 */
function vanishedRecords(
  percept: world.Percept,
  references: ReferenceTable,
  previous?: ReadonlyMap<string, { observedAt: number; tokens: readonly number[] }>,
): LoweredRecord[] {
  if (!previous) return [];
  const presentNow = new Set<string>();
  for (const record of percept.records) {
    if (record.instanceId) presentNow.add(record.instanceId);
  }

  const records: LoweredRecord[] = [];
  for (const instanceId of previous.keys()) {
    if (presentNow.has(instanceId)) continue;
    references.tokenFor(instanceId);
    records.push({
      band: "perception",
      modality: "declarative",
      schema: "VANISHED",
      instanceId,
      tokens: [],
      salience: 1,
      observedAt: percept.tick,
    });
  }
  return records;
}

/** Fraction of a record's tokens that differ from its previous observation. */
function changeBetween(before: readonly number[], after: readonly number[]): number {
  const seen = new Set(before);
  let changed = 0;
  for (const token of after) if (!seen.has(token)) changed++;
  return after.length === 0 ? 0 : changed / after.length;
}
