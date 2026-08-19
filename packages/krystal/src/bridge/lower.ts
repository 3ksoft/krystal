import {
  BRAIN_FRAME_BANDS,
  BRAIN_LIMITS,
  INVALID_U32,
  KRYSTAL_ABI,
  KRYSTAL_SENTINEL_TOKENS,
  RECORD_FLAGS,
  REFERENCE_FLAGS,
  TOKEN_FLAGS,
} from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import { CATALOG_SCHEMA_ID, schemaIdOf, type CompiledGrammar } from "./agent.ts";
import type { ConceptOperandV2, RawRecordV2, RawSnapshotV2, SensoryBand } from "./contract.ts";
import { BAND_SYMBOLS, quantize } from "./quantize.ts";

export interface PerformedAction {
  readonly relation: string;
  readonly object?: string;
  readonly intensity?: number;
}

type LoweredRecord = Omit<RawRecordV2, "band"> & {
  band: v1_0_0.BrainBandKind;
  catalogIndex?: number;
  subjectToken?: number;
  objectToken?: number;
};

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
      reserved0: 0,
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
      flags: 0,
      reserved0: 0,
      handle: { tokenId: 0, generation: 0, kind: "none" as const, status: "invalid" as const },
    })),
  };
}

function recordTokens(
  record: LoweredRecord,
  grammar: CompiledGrammar,
  refToken: number | undefined,
  objectRefToken: number | undefined,
): { tokens: number[]; refIndex: number } {
  const tokens: number[] = [];
  let refIndex = -1;

  const push = (symbol: string): void => {
    const id = grammar.tokenBySymbol.get(symbol);
    if (id === undefined) throw new LoweringError(`symbol '${symbol}' is not in the grammar`);
    tokens.push(id);
  };

  push(record.schema);
  if (refToken !== undefined) {
    if (refToken >= KRYSTAL_ABI.refSpaceStart) refIndex = tokens.length;
    tokens.push(refToken);
  }
  if (objectRefToken !== undefined) tokens.push(objectRefToken);
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
    const declared = grammar.quantities.get(quantity.field);
    if (!declared) throw new LoweringError(`quantity field '${quantity.field}' is not declared`);
    for (const symbol of quantize(quantity.value, declared.kind, declared.polarity).tokens) {
      push(symbol);
    }
  }

  return { tokens, refIndex };
}

export function lowerSnapshot(
  snapshot: RawSnapshotV2,
  grammar: CompiledGrammar,
  references: ReferenceTable,
  previous?: ReadonlyMap<string, { observedAt: number; tokens: readonly number[] }>,
  selfActions?: readonly PerformedAction[],
  previousValence?: number,
): LoweredFrame {
  const records: v1_0_0.BrainRecordSlot[] = Array.from(
    { length: BRAIN_LIMITS.frameRecordSlots },
    () => emptyRecord(),
  );

  const byBand = new Map<v1_0_0.BrainBandKind, LoweredRecord[]>();
  for (const record of snapshot.records) {
    const list = byBand.get(record.band) ?? [];
    list.push(record);
    byBand.set(record.band, list);
  }

  const valenceDelta =
    previousValence === undefined ? undefined : snapshot.valence - previousValence;
  if (valenceDelta !== undefined) {
    byBand.set("homeostasis", [
      ...(byBand.get("homeostasis") ?? []),
      valenceRecord(snapshot, valenceDelta),
    ]);
  }

  byBand.set("query", [queryRecord(snapshot, valenceDelta)]);

  if (grammar.actions.length > 0) {
    byBand.set("catalog", grammar.actions.map((action, index) => ({
      band: "catalog" as const,
      modality: "catalog",
      schema: action.relation,
      tokens: [],
      salience: 0,
      observedAt: snapshot.tick,
      catalogIndex: index,
    })));
  }

  const temporal = temporalRecords(snapshot, grammar, references, previous, selfActions);
  if (temporal.length > 0) byBand.set("temporal", temporal);

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
      const refToken =
        raw.subjectToken ??
        (raw.instanceId === undefined ? undefined : references.tokenFor(raw.instanceId));
      const objectRefToken =
        raw.objectToken ??
        (raw.objectInstanceId === undefined ? undefined : references.tokenFor(raw.objectInstanceId));
      const { tokens, refIndex } = recordTokens(raw, grammar, refToken, objectRefToken);

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

      const prior = raw.instanceId === undefined ? undefined : previous?.get(raw.instanceId);
      target.header = {
        ...target.header,
        schemaId:
          raw.catalogIndex === undefined
            ? schemaIdOf(grammar, raw.schema)
            : CATALOG_SCHEMA_ID,
        band,
        source: "sensor",
        flags:
          RECORD_FLAGS.occupied |
          (tokens.length > BRAIN_LIMITS.recordWidth ? RECORD_FLAGS.truncated : 0) |
          (raw.emptiness === "unavailable" ? RECORD_FLAGS.unavailable : 0),
        tokenCount: width,
        referenceCount: refIndex >= 0 ? 1 : 0,
        observedAt: raw.observedAt,
        primaryReference: refIndex >= 0 ? 0 : INVALID_U32,
        salience: raw.salience ?? 0,
        freshness: 1,
        previousObservedAt: prior?.observedAt ?? INVALID_U32,
        changeMagnitude: prior ? changeBetween(prior.tokens, tokens.slice(0, width)) : 0,
      };

      if (refIndex >= 0 && refToken !== undefined) {
        target.references[0] = {
          localTokenIndex: refIndex,
          fieldId: refIndex,
          flags: REFERENCE_FLAGS.primary | REFERENCE_FLAGS.live,
          reserved0: 0,
          handle: {
            tokenId: refToken,
            generation: references.generationOf(refToken),
            kind: "entity",
            status: "live",
          },
        };
      }

      activeRecordCount++;
      activeTokenCount += width;
    }
  }

  const bands: v1_0_0.BrainBandState[] = BRAIN_FRAME_BANDS.map((layout) => {
    const spill = overflow.find((entry) => entry.band === layout.kind);
    const offered = byBand.get(layout.kind as SensoryBand)?.length ?? 0;
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
      tick: snapshot.tick,
      snapshot: snapshot.tick,
      deltaMillis: snapshot.deltaMillis,
      activeRecordCount,
      activeTokenCount,
      activeQueryRecord: INVALID_U32,
      actorRecord: 2,
      frameRevision: snapshot.tick,
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
 * the shape of the creature's own turn. Its content is deliberately thin for
 * now; what a richer query should carry is a question about attention and
 * goals, not about the sensory format.
 */
function queryRecord(snapshot: RawSnapshotV2, valenceDelta: number | undefined): LoweredRecord {
  const tokens =
    valenceDelta === undefined
      ? []
      : quantize(Math.max(-1, Math.min(1, valenceDelta)), "signed", {
        negative: BAND_SYMBOLS.worse,
        positive: BAND_SYMBOLS.better,
      }).tokens;
  return {
    band: "query",
    modality: "query",
    schema: BAND_SYMBOLS.neither,
    tokens,
    salience: 1,
    observedAt: snapshot.tick,
  };
}

function valenceRecord(snapshot: RawSnapshotV2, delta: number): LoweredRecord {
  const banded = quantize(Math.max(-1, Math.min(1, delta)), "signed", {
    negative: BAND_SYMBOLS.worse,
    positive: BAND_SYMBOLS.better,
  });
  return {
    band: "homeostasis",
    modality: "derived",
    schema: banded.tokens[0]!,
    tokens: banded.tokens.slice(1),
    salience: Math.abs(delta),
    observedAt: snapshot.tick,
  };
}

function temporalRecords(
  snapshot: RawSnapshotV2,
  grammar: CompiledGrammar,
  references: ReferenceTable,
  previous?: ReadonlyMap<string, { observedAt: number; tokens: readonly number[] }>,
  selfActions?: readonly PerformedAction[],
): LoweredRecord[] {
  const operandToken = (operand: ConceptOperandV2): number => {
    switch (operand.kind) {
      case "instance":
        return references.tokenFor(operand.instanceId);
      case "symbol": {
        const token = grammar.tokenBySymbol.get(operand.symbol);
        if (token === undefined) throw new LoweringError(`symbol '${operand.symbol}' is not in the grammar`);
        return token;
      }
      case "unknown":
        return KRYSTAL_SENTINEL_TOKENS.unknown;
      case "something":
        return KRYSTAL_SENTINEL_TOKENS.something;
    }
  };

  const eventRecord = (
    relation: string,
    subject: ConceptOperandV2,
    object: ConceptOperandV2 | undefined,
    intensity: number | undefined,
    meta: { modality: string; salience: number; observedAt: number },
  ): LoweredRecord => ({
    band: "temporal",
    modality: meta.modality,
    schema: relation,
    subjectToken: operandToken(subject),
    objectToken: operandToken(object ?? subject),
    tokens: intensity === undefined ? [] : quantize(intensity, "unipolar").tokens,
    salience: meta.salience,
    observedAt: meta.observedAt,
  });

  const records: LoweredRecord[] = [];
  const presentNow = new Set<string>();
  for (const record of snapshot.records) {
    if (record.instanceId) presentNow.add(record.instanceId);
  }

  for (const motion of snapshot.motion ?? []) {
    if (!grammar.motion) {
      throw new LoweringError(
        "snapshot carries motion but the grammar declares none; a world with space must name what each direction of movement means",
      );
    }
    const polarity = grammar.motion.radial;
    const banded = quantize(motion.radial, "signed", polarity);
    if (banded.tokens[0] === BAND_SYMBOLS.neither) continue;
    records.push({
      band: "temporal",
      modality: "motion",
      schema: banded.tokens[0]!,
      instanceId: motion.instanceId,
      tokens: banded.tokens.slice(1),
      salience: Math.abs(motion.radial),
      observedAt: snapshot.tick,
    });
  }

  for (const event of snapshot.events ?? []) {
    records.push(
      eventRecord(event.relation, event.subject, event.object, event.intensity, {
        modality: "event",
        salience: event.salience ?? 0.5,
        observedAt: event.observedAt,
      }),
    );
  }

  for (const action of selfActions ?? []) {
    records.push(
      eventRecord(
        action.relation,
        { kind: "instance", instanceId: snapshot.actorId },
        action.object === undefined ? undefined : { kind: "instance", instanceId: action.object },
        action.intensity,
        {
          modality: "self",
          salience: 1,
          observedAt: snapshot.tick,
        },
      ),
    );
  }

  if (previous) {
    for (const instanceId of previous.keys()) {
      if (presentNow.has(instanceId)) continue;
      references.tokenFor(instanceId);
      records.push({
        band: "temporal",
        modality: "derived",
        schema: "VANISHED",
        instanceId,
        tokens: [],
        salience: 1,
        observedAt: snapshot.tick,
      });
    }
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
