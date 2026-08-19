/**
 * `pira-raw-sensory@2` — the simulation-facing contract.
 *
 * What changed from @1, and why:
 *
 *   Vocabulary moved out.  @1 carried a hardcoded map of six resource names and
 *   returned `null` for anything else, so a simulation with 77 resources had 71
 *   of them vanish at the boundary with no error and no counter. Symbols are
 *   now looked up in the compiled grammar (see `agent.ts`), and an unknown one
 *   is reported, never dropped.
 *
 *   Bands are derived.  @1 restated the sensory band list as a hand-written
 *   union, which meant a sense added to the ABI could not cross the boundary
 *   until someone remembered to edit the union. `temporal` was exactly that
 *   case. The set is now filtered from BRAIN_FRAME_BANDS.
 *
 *   Intents are binary relations.  @1 had `kind` / `subject` / `instrument` /
 *   `target`, a four-slot shape with optional members. Every relation now has
 *   exactly a subject and an object; anything further is expressed by relating
 *   to the intent itself, which is possible because an emitted intent owns a
 *   runtime reference.
 */
import { BRAIN_FRAME_BANDS } from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import type { SimQuantityField } from "./agent.ts";
import { quantize } from "./quantize.ts";

/**
 * Bands a simulation may report into: the senses, plus the two bands that carry
 * the body and its internal state. `memory`, `focus`, `query` and `catalog` are
 * the brain's own workings and are deliberately not writable from outside.
 */
export const SENSORY_BANDS = BRAIN_FRAME_BANDS.map((band) => band.kind).filter(
  (kind) =>
    kind !== "system" &&
    kind !== "memory" &&
    kind !== "focus" &&
    kind !== "query" &&
    kind !== "catalog",
);

export type SensoryBand = (typeof SENSORY_BANDS)[number];

export function isSensoryBand(value: string): value is SensoryBand {
  return (SENSORY_BANDS as readonly string[]).includes(value);
}

/**
 * A numeric observation, sent exact and discretized by the engine.
 *
 * The simulation must NOT pre-band. `Size.Medium` and `Speed.Fast` are already
 * decisions, and they are the engine's to make: a band is a token, a token owns
 * a trained embedding row, and a threshold that moves upstream would silently
 * redefine what that row denotes. Send `{ field: "size", value: 0.6 }`.
 *
 * How to read the number is declared once in the grammar, not repeated here: a
 * field's kind and its polarity are properties of the field, and restating them
 * per observation would allow them to disagree between frames.
 */
export interface RawQuantityV2 {
  /** Must name a quantity field declared in the grammar. */
  readonly field: string;
  readonly value: number;
}

/**
 * One observed thing.
 *
 * `schema` names a symbol in the compiled grammar; `instanceId` identifies a
 * persistent world entity and is what earns a runtime reference token. A record
 * without one is an observation of a kind rather than of a thing and cannot be
 * pointed at — which also means the same entity seen and heard must carry the
 * SAME instanceId in both records, or the brain perceives two things.
 *
 * Only what the sense can actually pick up belongs in `tokens` and
 * `quantities`. A simulation's own bookkeeping — a patrol's phase and period, a
 * spawner id, a child list — is not perceivable and must not travel this
 * channel. The cost of leaking it is not wasted slots: the model learns to use
 * it, is rewarded for predicting movement it could never have seen, and arrives
 * at a scenario without patrols having never learned to watch.
 */
export interface RawRecordV2 {
  readonly band: SensoryBand;
  /** Which sense produced this. The same entity may appear under several. */
  readonly modality: string;
  /** Grammar symbol for what was perceived. */
  readonly schema: string;
  /** Stable identity of a world entity; omitted for a kind-level observation. */
  readonly instanceId?: string;
  /**
   * Second participant, for a temporal record standing for a relation rather
   * than an entity: `instanceId` is then the subject and this the object. A
   * reflexive event repeats the subject, the same default that makes unary
   * relations work elsewhere.
   */
  readonly objectInstanceId?: string;
  /** Perceived category tokens, as grammar symbols. */
  readonly tokens: readonly string[];
  /** Exact numeric observations; the engine bands them. */
  readonly quantities?: readonly RawQuantityV2[];
  /**
   * Aggregate records stand for several things at once, which is how a distant
   * flock stays inside a bounded band instead of consuming a slot per sheep.
   * `count` feeds a subitizing band, and quantifiers over the group ride in
   * `quantities` as `proportion` entries.
   */
  readonly count?: number;
  readonly salience?: number;
  readonly observedAt: number;
  /**
   * Set when the sense looked and found nothing (VOID) or could not report at
   * all (UNAVAILABLE). Both are percepts and reach the model as tokens; leaving
   * the record out entirely is a third and different thing, meaning the sense
   * was never sampled, and is masked out of attention.
   */
  readonly emptiness?: "void" | "unavailable";
}

/**
 * Motion, as a relation rather than a property.
 *
 * This is the distinction the sensory format has to preserve: `speed` is a
 * property of the dog, `approaching` is a relation between the dog and me. A
 * fast dog running away and a fast dog running at me share a speed and share
 * nothing that matters, so a `Speed.Fast` token cannot stand in for this.
 *
 * The simulation must supply it because the engine cannot recover it. Distance
 * bands are coarse by design, so an animal crossing the whole of `near` toward
 * the actor changes no band for many ticks; and exact positions are not sent,
 * because coordinates are not perceivable — closing distance is (looming is a
 * primitive percept, a coordinate is not).
 *
 * What the engine does derive on its own is everything that needs only memory:
 * content change, appearance and disappearance follow from a stable instanceId
 * and the previous frame, which is what `previousObservedAt` and
 * `changeMagnitude` on the record header are for.
 */
export interface RawMotionV2 {
  /** Entity in motion; must match a record's instanceId in this snapshot. */
  readonly instanceId: string;
  /**
   * Signed radial rate toward the actor, normalized to -1..1: positive
   * approaches, negative recedes, and a magnitude inside the signed deadzone
   * reads as the distinct category "not moving" rather than as a small value.
   */
  readonly radial: number;
  /** Signed angular rate, normalized to -1..1: how tracking gets learned. */
  readonly angular?: number;
}

/**
 * Something that happened: an action performed, by anyone.
 *
 * Deliberately NOT an outcome. An outcome would have to say where its effects
 * end, and there is no principled answer — drop a ball on a trampoline and the
 * fall, the bounce and the second bounce are all equally "the result". Any cut
 * is arbitrary, so this format refuses to cut: it records that an action
 * occurred, and whatever follows is simply more events. Whether one caused
 * another is something a creature may learn, not something the contract asserts.
 *
 * Participants are operands rather than bare instance ids, and that matters
 * most at a distance. Watching someone handle a thing you cannot identify is a
 * real percept — you see that the mother picked something up, and you do not
 * know what — so an unidentified participant degrades to `unknown` instead of
 * costing the whole event.
 *
 * `unknown` rather than `something`: a specific thing was handled and the
 * observer failed to identify it, which is the epistemic gap that licenses a
 * query. `something` is the existential, and it asks nothing.
 *
 * Dropping such events would bias the imitation channel toward whatever happens
 * within arm's reach — precisely backwards, since the demonstrations most worth
 * learning from are the distant ones a creature cannot yet walk over to.
 *
 * The same shape carries the actor's own actions, with `self` in the subject
 * slot. That is what makes the agency distinction free — `PICKUP(self, ball)`
 * against `PICKUP(mother, ball)` differ only in who is the subject — and it is
 * also what grounds the action catalog. A catalog entry is otherwise an opaque
 * symbol the creature has no way to interpret; sharing its token with an
 * observed event means watching someone act teaches what one's own option does.
 * Seeing an action and performing one end up with the same representation,
 * which is imitation for free.
 */
export interface RawEventV2 {
  /** Grammar symbol; the same one the action catalog uses for this relation. */
  readonly relation: string;
  /** Who acted. */
  readonly subject: ConceptOperandV2;
  /** What was acted upon. Omitted means reflexive: the subject itself. */
  readonly object?: ConceptOperandV2;
  /** How hard, how fast. Not confidence. */
  readonly intensity?: number;
  readonly salience?: number;
  readonly observedAt: number;
}

/** The actor's own movement. Whole-field flow is its own percept. */
export interface RawSelfMotionV2 {
  readonly speed: number;
  readonly turning?: number;
}

export interface RawSnapshotV2 {
  readonly contract: "pira-raw-sensory@2";
  readonly tick: number;
  /** Milliseconds since the previous snapshot. Rates are derived from this. */
  readonly deltaMillis: number;
  /**
   * How the actor is doing, 0..1, where 0 is dead.
   *
   * Named rather than left as one declared quantity among others, because the
   * engine trains against it and so has to know which number it is. How a world
   * arrives at it — from satiation, warmth, health, whatever it has — is the
   * simulation's business entirely.
   *
   * Unipolar and not signed, because death is a floor and an absorbing state:
   * there is nothing as good as death is bad, so a scale symmetric about a
   * neutral point would assert a symmetry the world does not have. It also
   * spares the simulation from having to decide where "neutral" sits, which is
   * a calibration that quietly determines whether any signal exists at all.
   *
   * The learning signal is not this level but its CHANGE, and that difference
   * is signed whatever the level is measured in. The engine derives it, the way
   * it derives every other cross-frame quantity.
   */
  readonly valence: number;
  readonly actorId: string;
  readonly records: readonly RawRecordV2[];
  /**
   * Feeds the temporal band, which carries CHANGE and nothing else: only things
   * that actually moved or actually did something appear there. A world that
   * held still contributes no temporal records at all, and the band is small
   * precisely because it is not a state channel.
   */
  readonly motion?: readonly RawMotionV2[];
  readonly events?: readonly RawEventV2[];
  readonly selfMotion?: RawSelfMotionV2 | undefined;
}

/**
 * One side of a relation as the simulation names it: either a world entity by
 * instance id, a concept by symbol, or one of the epistemic sentinels.
 */
export type ConceptOperandV2 =
  | { readonly kind: "instance"; readonly instanceId: string }
  | { readonly kind: "symbol"; readonly symbol: string }
  | { readonly kind: "unknown" }
  | { readonly kind: "something" };

/**
 * An intent the brain produced, as a binary relation.
 *
 * `object` is always present. For a relation the catalog declares unary it
 * equals `subject` — the reflexive reading, not a missing argument.
 * Higher-arity meanings attach to `intentRef`: GIVE(self, apple) plus
 * RECIPIENT(<that intent>, mother).
 */
export interface AgentIntentV2 {
  readonly relation: string;
  readonly subject: ConceptOperandV2;
  readonly object: ConceptOperandV2;
  /** Magnitude: how hard, how fast, how strongly. Not confidence. */
  readonly intensity: number;
  /** How firmly the network chose. Not magnitude. */
  readonly commitment: number;
  /** Reference to this intent, so further relations can take it as an operand. */
  readonly intentRef?: number;
  readonly source: "learned";
}

/**
 * There is deliberately no outcome type here.
 *
 * An outcome would have to say where its own effects stop, and no cut is
 * defensible: drop a ball on a trampoline and the fall, the bounce and the
 * second bounce are all equally "what resulted". Any boundary the format drew
 * would be arbitrary, and a creature trained on it would learn that arbitrary
 * boundary as if it were a fact about the world.
 *
 * So the contract records only that an action happened, as a `RawEventV2`, and
 * whatever follows arrives as further events. Whether one caused another is
 * left to be learned rather than asserted — which is also why the actor's own
 * actions travel the same channel as everybody else's.
 */

export class SensoryContractError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SensoryContractError";
  }
}

/** Counted rejections, so a lowering failure is legible instead of silent. */
export interface LoweringDiagnostics {
  unknownSymbols: Map<string, number>;
  unknownBands: Map<string, number>;
  droppedRecords: number;
}

export function emptyDiagnostics(): LoweringDiagnostics {
  return { unknownSymbols: new Map(), unknownBands: new Map(), droppedRecords: 0 };
}

/**
 * Validate a snapshot against the contract and the compiled grammar.
 *
 * Strict by default. `onUnknown: "count"` exists for a simulation mid-migration
 * whose grammar has not caught up with its content, but it still records what
 * was lost — the failure mode being designed against is not rejection, it is a
 * boundary that quietly forgets.
 */
export function validateSnapshot(
  snapshot: RawSnapshotV2,
  tokenBySymbol: ReadonlyMap<string, number>,
  options: {
    onUnknown?: "throw" | "count";
    /** Declared quantity fields; when given, values are checked against them. */
    quantities?: ReadonlyMap<string, SimQuantityField>;
  } = {},
): LoweringDiagnostics {
  const quantities = options.quantities;
  if (snapshot.contract !== "pira-raw-sensory@2") {
    throw new SensoryContractError(`Unsupported sensory contract '${snapshot.contract}'`);
  }
  if (!Number.isFinite(snapshot.deltaMillis) || snapshot.deltaMillis < 0) {
    throw new SensoryContractError(`snapshot.deltaMillis must be a non-negative number`);
  }
  if (!Number.isFinite(snapshot.valence) || snapshot.valence < 0 || snapshot.valence > 1) {
    throw new SensoryContractError(
      `snapshot.valence must be within 0..1, where 0 is dead; got ${snapshot.valence}`,
    );
  }

  const strict = (options.onUnknown ?? "throw") === "throw";
  const diagnostics = emptyDiagnostics();

  const instances = new Set<string>();
  for (const record of snapshot.records) {
    if (record.instanceId) instances.add(record.instanceId);
    for (const quantity of record.quantities ?? []) {
      if (!Number.isFinite(quantity.value)) {
        throw new SensoryContractError(
          `${record.schema}.${quantity.field}: quantity must be a finite number, not a pre-banded label — the engine owns the thresholds`,
        );
      }
      const declared = quantities?.get(quantity.field);
      if (quantities && !declared) {
        throw new SensoryContractError(
          `${record.schema}.${quantity.field}: no such quantity field in the grammar; declare its kind before sending values`,
        );
      }
      if (declared) {
        try {
          quantize(quantity.value, declared.kind, declared.polarity);
        } catch (error) {
          throw new SensoryContractError(`${record.schema}.${quantity.field}: ${(error as Error).message}`);
        }
      }
    }
    if (!isSensoryBand(record.band)) {
      if (strict) throw new SensoryContractError(`record band '${record.band}' is not writable by the simulation`);
      diagnostics.unknownBands.set(record.band, (diagnostics.unknownBands.get(record.band) ?? 0) + 1);
      diagnostics.droppedRecords++;
      continue;
    }
    if (!tokenBySymbol.has(record.schema)) {
      if (strict) {
        throw new SensoryContractError(
          `record schema '${record.schema}' is not in the compiled grammar; declare it or the brain cannot perceive it`,
        );
      }
      diagnostics.unknownSymbols.set(record.schema, (diagnostics.unknownSymbols.get(record.schema) ?? 0) + 1);
      diagnostics.droppedRecords++;
    }
  }

  for (const event of snapshot.events ?? []) {
    if (!tokenBySymbol.has(event.relation)) {
      throw new SensoryContractError(
        `event relation '${event.relation}' is not in the compiled grammar`,
      );
    }
    for (const participant of [event.subject, event.object]) {
      if (participant === undefined) continue;
      if (participant.kind === "instance") {
        if (!instances.has(participant.instanceId) && participant.instanceId !== snapshot.actorId) {
          throw new SensoryContractError(
            `event '${event.relation}' names instance '${participant.instanceId}', which is not perceived in this snapshot; ` +
              `report an unidentified participant as { kind: "unknown" } rather than dropping the event`,
          );
        }
      } else if (participant.kind === "symbol" && !tokenBySymbol.has(participant.symbol)) {
        throw new SensoryContractError(
          `event '${event.relation}' names symbol '${participant.symbol}', which is not in the compiled grammar`,
        );
      }
    }
  }

  // Motion names an entity, so it can only describe something also perceived.
  // A rate attached to nothing visible would be knowledge of the world rather
  // than of the scene.
  for (const motion of snapshot.motion ?? []) {
    if (!instances.has(motion.instanceId)) {
      throw new SensoryContractError(
        `motion references instance '${motion.instanceId}', which is not perceived in this snapshot`,
      );
    }
    if (!Number.isFinite(motion.radial) || Math.abs(motion.radial) > 1) {
      throw new SensoryContractError(`motion '${motion.instanceId}': radial rate must be within -1..1`);
    }
  }
  return diagnostics;
}

/**
 * Map an emitted `IntentSet` back into contract intents.
 *
 * References become instance ids, which is the boundary's whole job here: the
 * engine reasons in reference tokens because their identity is exact and
 * stable, and the simulation knows nothing of them.
 *
 * A proposal whose reference does not resolve is dropped rather than reported
 * with a fabricated participant. Every intent is binary, so a missing side is
 * not a partial answer — it is not an answer.
 */
export function toAgentIntents(
  intentSet: v1_0_0.IntentSet,
  relationOf: (intentId: number) => string | undefined,
  instanceOf: (refToken: number) => string | undefined,
): AgentIntentV2[] {
  const intents: AgentIntentV2[] = [];
  for (const proposal of intentSet.proposals) {
    if (proposal.lifecycle === "empty") continue;
    const relation = relationOf(proposal.intentId);
    if (relation === undefined) continue;

    const operand = (side: v1_0_0.SelectedConceptRef): ConceptOperandV2 | undefined => {
      const token = side.concept.handle.tokenId;
      const instanceId = token === 0 ? undefined : instanceOf(token);
      return instanceId === undefined ? undefined : { kind: "instance", instanceId };
    };

    const subject = operand(proposal.subject);
    const object = operand(proposal.object);
    if (!subject || !object) continue;

    intents.push({
      relation,
      subject,
      object,
      intensity: proposal.intensity,
      commitment: proposal.commitment,
      intentRef:
        proposal.intentRef.tokenId === 0 ? undefined : proposal.intentRef.tokenId,
      source: "learned",
    });
  }
  return intents;
}
