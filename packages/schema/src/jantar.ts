import {
  BRAIN_LIMITS,
  KRYSTAL_SENTINEL_TOKENS,
  RELATION_ROLES,
  type RelationRoleName,
} from "./krystal-engine-schema";
import type { v1_0_0 } from "../generated/krystal.types";

/**
 * Jantar — the grammar of thought.
 *
 * This layer answers exactly one question: which records are WELL FORMED. It
 * constrains what the creature can compose, simple concepts and complex ones
 * alike, and it stops there. It does not know whether an apple is within reach,
 * whether a hand is free, or whether eating will nourish anything — those are
 * the simulation's verdicts, reached after the act. `JA CIESZĘ SIĘ` is a
 * perfectly grammatical thought whether or not any joy follows.
 *
 * The boundary matters in both directions. A rule here must never encode a fact
 * about one world (that is the vocabulary's job), and never a physical
 * precondition (that is the simulation's). What is left is the shape of a
 * thought: what may stand where, what a modality demands, and how a complex
 * concept nests inside a simple one.
 *
 * Nothing here is about letters. The written forms in docs/jantar.md — `-o` for
 * a noun, `-n` for the accusative, `za-` for the perfective — are a way of
 * WRITING a record down for a human. The model never sees them; it sees slots
 * and tokens, and these are the rules those obey.
 */

export class JantarError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "JantarError";
  }
}

/**
 * Sentence intent, as jantar's terminators and connectives.
 *
 * Four of the five map onto a proposition modality directly. The fifth, `~`,
 * is a hypothetical rather than a real condition — "had I the keys" against "if
 * it rains" — and the difference is real: a real condition licenses acting on
 * it, a hypothetical one explicitly does not. It shares the implicative
 * modality and is distinguished by `RELATION_FEATURE_FLAGS.hypothetical`,
 * because both are one antecedent binding one consequent and only their
 * standing toward the world differs.
 */
export const JANTAR_INTENT = {
  ".": { modality: "declarative", hypothetical: false },
  "!": { modality: "imperative", hypothetical: false },
  "?": { modality: "interrogative", hypothetical: false },
  "->": { modality: "implicative", hypothetical: false },
  "~": { modality: "implicative", hypothetical: true },
} as const satisfies Record<string, { modality: v1_0_0.PropositionModality; hypothetical: boolean }>;

export type JantarIntentMark = keyof typeof JANTAR_INTENT;

/**
 * Features a relation carries beyond its identity.
 *
 * These are properties of the RELATION, not extra participants, which is why
 * they are flags rather than slots: they would otherwise compete for the eight
 * token positions with the participants themselves.
 */
export const RELATION_FEATURE_FLAGS = {
  /** Completed or one-off, written `za-`. Absent means ongoing or repeated. */
  perfective: 1 << 0,
  past: 1 << 1,
  future: 1 << 2,
  /** `nie` before the relation. There are no double negatives. */
  negated: 1 << 3,
  /** A `~` antecedent: entertained, never acted on. */
  hypothetical: 1 << 4,
  /**
   * Takes another relation as its patient — `muśi`, `mógi`, `hći`.
   *
   * This is where complex concepts come from. Only the outer relation carries
   * tense; the inner one stays in its base form, which is jantar's verb
   * sequence rule and the reason there is no separate infinitive.
   */
  modal: 1 << 5,
} as const;

/** Past and future are exclusive; a relation is in one tense or none. */
export function validateFeatures(flags: number): void {
  if ((flags & RELATION_FEATURE_FLAGS.past) !== 0 && (flags & RELATION_FEATURE_FLAGS.future) !== 0) {
    throw new JantarError("a relation cannot be both past and future");
  }
}

/** What one slot of a record under construction may hold. */
export type SlotKind =
  /** Slot 0: what the record is ABOUT — the relation or the quality. */
  | "predicate"
  /** A participant standing in one named role. */
  | "participant"
  /** A quality or quantity said of the record. */
  | "modifier";

export interface SlotPlan {
  readonly index: number;
  readonly kind: SlotKind;
  /** Set when `kind` is "participant". */
  readonly role?: RelationRoleName;
  /**
   * Whether the record is ill-formed without this slot filled.
   *
   * The agent is always required — jantar has no impersonal constructions, so
   * "trzeba" and "należy" must be rephrased until somebody is doing something.
   * A record with nobody in it is not a thought about the world.
   */
  readonly required: boolean;
  /**
   * Whether an unbound slot is what the record ASKS about.
   *
   * Only in the interrogative: a question is a proposition with a hole, and the
   * hole is the question. Everywhere else an unbound required slot is an error
   * rather than an enquiry.
   */
  readonly interrogable: boolean;
}

export interface RecordPlan {
  readonly modality: v1_0_0.PropositionModality;
  readonly slots: readonly SlotPlan[];
}

/**
 * The slot order of a record, given its modality and the roles its relation
 * declares.
 *
 * Order carries focus. Jantar has no passive voice: `koto uovi myśon` and
 * `myśon uovi koto` state the same fact and differ only in what the thought is
 * ABOUT, and that difference survives here as the order participants are
 * planned in. So a caller that wants the patient in focus passes it first —
 * the record is the same relation either way, and nothing about its truth
 * changes.
 */
export function planRecord(
  modality: v1_0_0.PropositionModality,
  declaredRoles: readonly RelationRoleName[],
): RecordPlan {
  const seen = new Set<RelationRoleName>();
  for (const role of declaredRoles) {
    if (seen.has(role)) throw new JantarError(`role '${role}' declared twice`);
    seen.add(role);
  }
  if (!seen.has("agent")) {
    throw new JantarError(
      "every record needs an explicit agent: jantar has no impersonal constructions, " +
        "so a thought with nobody doing anything must be rephrased until somebody is",
    );
  }

  const slots: SlotPlan[] = [{ index: 0, kind: "predicate", required: true, interrogable: modality === "interrogative" }];

  // Participants follow the predicate, in the order given. Roles the relation
  // did not declare are simply absent — there is no slot to leave empty.
  const ordered = RELATION_ROLES.filter((role) => seen.has(role));
  const focusFirst = declaredRoles.filter((role) => seen.has(role));
  const participants = focusFirst.length === ordered.length ? focusFirst : ordered;

  for (const role of participants) {
    slots.push({
      index: slots.length,
      kind: "participant",
      role,
      // The agent is required outright. A patient is required of a declarative
      // or an imperative — one does not command an act upon nothing — but a
      // question may leave exactly that hole.
      required: role === "agent" || (role === "patient" && modality !== "interrogative"),
      interrogable: modality === "interrogative",
    });
  }

  // Whatever token width is left takes qualities and quantities. This is why
  // arity has a real ceiling: every participant costs a slot that a modifier
  // could have had.
  for (let index = slots.length; index < BRAIN_LIMITS.recordWidth; index++) {
    slots.push({ index, kind: "modifier", required: false, interrogable: false });
  }

  return { modality, slots };
}

/**
 * Token classes admissible in the predicate slot, by modality.
 *
 * An imperative must name something the creature can DO, so its predicate is
 * narrowed further — to the relations the world declared — by the caller, which
 * is the only party that knows them.
 */
export const PREDICATE_CLASSES: Readonly<
  Record<v1_0_0.PropositionModality, readonly v1_0_0.KrystalTokenClass[]>
> = {
  // A fact may be a relation holding or a quality obtaining.
  declarative: ["action", "relation", "property", "object"],
  // A command is an act.
  imperative: ["action", "relation"],
  // A question may ask after either.
  interrogative: ["action", "relation", "property", "object"],
  // An implication binds two propositions; its predicate is the connective.
  implicative: ["logic", "relation", "action"],
};

/**
 * The sentinel that stands in an unbound slot.
 *
 * `UNKNOWN` and `SOMETHING` are not interchangeable, and the difference decides
 * what the creature does next. A question's hole is `UNKNOWN`: a referent
 * exists and which one is precisely what is being asked. `SOMETHING` is the
 * existential — it is what an answer may quantify over, never what prompts the
 * asking.
 */
export function unboundToken(slot: SlotPlan): number {
  if (slot.interrogable && slot.kind === "participant") return KRYSTAL_SENTINEL_TOKENS.unknown;
  return KRYSTAL_SENTINEL_TOKENS.pad;
}

/**
 * Check a finished record against its plan.
 *
 * Well-formedness only. Every diagnostic here is about the SHAPE of the
 * thought — a missing agent, a role the relation never declared, a question
 * with no hole in it. Whether the thought is true, or achievable, or wise, is
 * not asked and could not be answered here.
 */
export function validatePlan(
  plan: RecordPlan,
  boundRoles: ReadonlySet<RelationRoleName>,
): string[] {
  const problems: string[] = [];

  for (const slot of plan.slots) {
    if (slot.kind !== "participant" || !slot.role) continue;
    if (slot.required && !boundRoles.has(slot.role)) {
      problems.push(`role '${slot.role}' is required by a ${plan.modality} record and is unbound`);
    }
  }

  if (plan.modality === "interrogative") {
    const holes = plan.slots.filter(
      (slot) => slot.kind === "participant" && slot.role && !boundRoles.has(slot.role),
    );
    if (holes.length === 0) {
      problems.push("an interrogative record with every role bound asks nothing");
    }
  }

  const participants = plan.slots.filter((slot) => slot.kind === "participant").length;
  if (participants > BRAIN_LIMITS.relationArity) {
    problems.push(
      `${participants} participants exceed the ${BRAIN_LIMITS.relationArity} roles a relation may bind`,
    );
  }

  return problems;
}
