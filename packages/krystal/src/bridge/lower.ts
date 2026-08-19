/**
 * `pira-raw-sensory@2` -> `BrainFrame`.
 *
 * This is where perception becomes the model's input, and the whole job is
 * fitting an unbounded scene into a fixed geometry: a record is eight token
 * slots, a band is a fixed number of records, and a snapshot may describe far
 * more than either. Everything here is therefore about what to keep.
 *
 * Three rules govern that, and each exists because the obvious alternative
 * fails quietly rather than loudly:
 *
 *   Overflow is reported, never silently truncated. A band that drops its tail
 *   without saying so looks identical to a scene that was simply emptier than
 *   expected.
 *
 *   Unoccupied slots are masked out, not padded and attended. A PAD the network
 *   still attends to is white noise it must spend capacity learning to ignore,
 *   and sensory bands are mostly empty in a typical frame.
 *
 *   Identity comes from the sidecar, never from a token. A reference token is a
 *   local symbol whose meaning is the binding beside it; the model reads that
 *   two things are the same reference, and the runtime resolves what it points
 *   at.
 */
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

/**
 * An action the engine emitted last tick, fed back so the creature can see what
 * it did. Not an outcome: it says an action was taken, never what came of it.
 */
export interface PerformedAction {
  readonly relation: string;
  readonly object?: string;
  readonly intensity?: number;
}

/**
 * A record on its way into the frame.
 *
 * Widens `RawRecordV2`'s band, because the two have different rights: a
 * simulation may only write the sensory bands, while the engine also authors
 * the query row and its own derivatives. Keeping the contract narrow is what
 * stops a world from writing into the brain's own workings; keeping this one
 * wide is what lets the engine do so.
 */
type LoweredRecord = Omit<RawRecordV2, "band"> & {
  band: v1_0_0.BrainBandKind;
  /** Set on catalog records; its position is the intent id. */
  catalogIndex?: number;
  /**
   * Pre-resolved participants for a relation record. Tokens rather than
   * instance ids, because a participant may be a sentinel — an unidentified
   * thing has no instance to name.
   */
  subjectToken?: number;
  objectToken?: number;
};

export class LoweringError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LoweringError";
  }
}

/** Per-band accounting, so a frame that lost information says which band. */
export interface BandOverflow {
  readonly band: v1_0_0.BrainBandKind;
  readonly offered: number;
  readonly admitted: number;
}

export interface LoweredFrame {
  readonly frame: v1_0_0.BrainFrame;
  /**
   * Change in valence since the previous frame, signed, and the engine's own
   * learning signal. Undefined on the first frame, where there is nothing to
   * difference against — which is honest: a creature that has just started
   * cannot yet know whether anything got better.
   */
  readonly valenceDelta: number | undefined;
  /** Runtime reference token -> the instance it denotes, for reading intents back. */
  readonly instanceByToken: ReadonlyMap<number, string>;
  readonly overflow: readonly BandOverflow[];
  /** Records whose token bag did not fit in eight slots. */
  readonly truncatedRecords: number;
}

/**
 * Mints and remembers a reference token per world instance.
 *
 * The mapping must outlive a frame: a handle identifies an entity ACROSS ticks,
 * so re-minting would make the same object read as a new one every frame and
 * leave the memory band nothing to hold on to.
 */
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

/**
 * Token bag for one observed record, in priority order.
 *
 * Order is the truncation policy, because eight slots will not always be
 * enough. Identity first, then the reference that lets the record be pointed
 * at, then declared measurements, then loose category tokens — so what survives
 * a crowded record is what makes it addressable and distinguishable, and what
 * is lost is descriptive detail.
 */
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
    // Only a reference-half token names a runtime binding. A sentinel occupies
    // the slot as an ordinary symbol, so recording it as a reference would
    // promise the runtime an identity that does not exist.
    if (refToken >= KRYSTAL_ABI.refSpaceStart) refIndex = tokens.length;
    tokens.push(refToken);
  }
  // A relation record carries both participants: subject, then object. The
  // object slot is always filled — a reflexive event repeats the subject — so
  // the record has the same shape whether the relation was unary or not.
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
  // Categories before quantity bands, because a record holds eight tokens and
  // the tail is what truncation takes. A lost band costs detail — the thing is
  // still there, merely less finely described. A lost category costs the record
  // its class, and with it every role that admits the class rather than the
  // individual: the apple stops being edible. Order by what cannot be spared.
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

/**
 * Lower a validated snapshot into a BrainFrame.
 *
 * `previous` supplies the derivatives the engine computes for itself: how long
 * since a record was last seen and how much it changed. Those need only a
 * stable instance id and the last frame, which is why they are not asked of the
 * simulation — unlike radial motion, which cannot be recovered from banded
 * distances and must be sent.
 */
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

  // The temporal band is not read off the snapshot's records like the senses
  // are: its contents are relations, some sent and some derived. Build them
  // first so they compete for the band's slots on the same terms.
  // Valence arrives as a level and is trained on as a change, so the engine
  // differences it here rather than asking the simulation for both. Same
  // pattern as every other cross-frame derivative: what needs only memory is
  // the engine's to compute.
  const valenceDelta =
    previousValence === undefined ? undefined : snapshot.valence - previousValence;
  if (valenceDelta !== undefined) {
    byBand.set("homeostasis", [
      ...(byBand.get("homeostasis") ?? []),
      valenceRecord(snapshot, valenceDelta),
    ]);
  }

  // Every frame needs a query row. It is the position the selectors score
  // against — "given all this, what now" — so a frame without one produces no
  // proposal and no gradient at all. Nothing errors in that case, the model
  // simply has nothing to be asked, which is a failure that looks exactly like
  // a quiet tick.
  //
  // The valence delta rides here when there is one: the standing question a
  // creature carries is about its own condition, and the direction that
  // condition just moved is the most relevant thing to ask from.
  byBand.set("query", [queryRecord(snapshot, valenceDelta)]);

  // The catalog: what the creature could do, in the frame where it can see it.
  //
  // Not an optimisation — the intent selector scores exactly these records, so
  // a frame without them yields no proposal, hence no action, hence no
  // experience to learn from. An agent that cannot perceive its own options is
  // inert in a way nothing reports.
  //
  // Every option is present every tick, including ones that will fail. The
  // schema is explicit that capability and precondition are descriptive rather
  // than exclusive, and that is what makes failure learnable: mask away what
  // cannot work and the creature never discovers why it cannot.
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

    // Salience decides what survives a full band. Ties break on observation
    // order so the same scene lowers identically twice.
    const ranked = [...offered].sort((a, b) => (b.salience ?? 0) - (a.salience ?? 0));
    const admitted = Math.min(ranked.length, layout.recordCapacity);
    if (admitted < ranked.length) {
      overflow.push({ band, offered: ranked.length, admitted });
    }

    for (let i = 0; i < admitted; i++) {
      const raw = ranked[i]!;
      const slot = layout.recordOffset + i;
      const target = records[slot]!;
      // A relation record arrives with its participants already resolved,
      // because one of them may be a sentinel rather than a thing: an
      // unidentified participant has no instance to mint a reference for.
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
        // Catalog records carry a fixed schema id so the intent mask can find
        // them; everything else carries its symbol's token class, which is the
        // coarse family the embedding wants. Identity is not here — it is the
        // record's first token, which is also where acceptance reads it.
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
        // Engine-derived: how much of this record's content is new. Cheap, and
        // available to every band without a sense dedicated to change.
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

/**
 * The engine-derived valence change, as an ordinary homeostasis record.
 *
 * It is a percept, not merely a training target: a creature should be able to
 * notice that things have just got worse, independently of learning from it.
 * The level itself is already perceived through whatever channels the world
 * declares; this is the derivative the world does not send.
 */
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
    // A change for the worse should compete well for a slot.
    salience: Math.abs(delta),
    observedAt: snapshot.tick,
  };
}

/**
 * Relations for the temporal band: motion the simulation sent, and
 * disappearance the engine derives.
 *
 * The asymmetry between the two is the point. Motion cannot be recovered here —
 * distance bands are coarse enough that an animal can cross the whole of `near`
 * without changing one — so it must be sent. Disappearance is the opposite: it
 * needs only a stable instance id and the previous frame, and it CANNOT be sent,
 * because the simulation reports what is perceived and a vanished thing is
 * precisely what is not.
 *
 * Disappearance is also the one member of the change family that needs its own
 * record. An appearance is already visible in a record whose `previousObservedAt`
 * is invalid, and a change in content is a magnitude that bands like any other
 * quantity; only an absence has nothing left to carry it.
 */
function temporalRecords(
  snapshot: RawSnapshotV2,
  grammar: CompiledGrammar,
  references: ReferenceTable,
  previous?: ReadonlyMap<string, { observedAt: number; tokens: readonly number[] }>,
  selfActions?: readonly PerformedAction[],
): LoweredRecord[] {
  /**
   * An event is a relation between two participants, so it does not fit the
   * entity-shaped record the senses produce: `instanceId` carries the subject
   * and `objectInstanceId` the object. A reflexive event repeats the subject,
   * which is the same default that makes unary relations work everywhere else.
   */
  /**
   * Token for one side of an observed event.
   *
   * An operand rather than an instance id, because a participant may be seen
   * without being identified — the point of the `unknown` sentinel. A relation
   * with an unidentifiable side is still a relation worth recording; refusing
   * it would leave the creature able to learn only from what happens close
   * enough to make out.
   */
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
    // Inside the deadzone nothing moved, and the temporal band carries change.
    // A record saying "this is still where it was" would spend a slot to report
    // the absence of news, and the band is small on the assumption it never has
    // to. This is also the deadzone's better justification: it is the test for
    // whether movement happened at all, not merely where the sign gets unclear.
    if (banded.tokens[0] === BAND_SYMBOLS.neither) continue;
    records.push({
      band: "temporal",
      modality: "motion",
      // The relation itself is the record's identity: APPROACHING(x, self).
      schema: banded.tokens[0]!,
      instanceId: motion.instanceId,
      tokens: banded.tokens.slice(1),
      // Something closing on the actor outranks something drifting away.
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

  // The actor's own last actions, written by the engine rather than reported by
  // the simulation: the engine emitted them and so already knows. They take the
  // same shape as an observed action, with `self` as the subject, which is what
  // lets a creature relate what it did to what it has seen others do.
  for (const action of selfActions ?? []) {
    records.push(
      eventRecord(
        action.relation,
        { kind: "instance", instanceId: snapshot.actorId },
        action.object === undefined ? undefined : { kind: "instance", instanceId: action.object },
        action.intensity,
        {
          modality: "self",
          // One's own action is worth attending to: without it in the frame the
          // creature cannot condition on what it did, and its own agency becomes
          // invisible — the world would merely seem to vary.
          salience: 1,
          observedAt: snapshot.tick,
        },
      ),
    );
  }

  if (previous) {
    for (const instanceId of previous.keys()) {
      if (presentNow.has(instanceId)) continue;
      // The reference survives the thing leaving perception; that is what lets
      // working memory keep pointing at it.
      references.tokenFor(instanceId);
      records.push({
        band: "temporal",
        modality: "derived",
        schema: "VANISHED",
        instanceId,
        tokens: [],
        // A disappearance is worth attending to: it is the last chance to
        // notice, and there will be no record of it next frame either.
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
