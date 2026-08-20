import {} from "./env";
import { scope } from "arktype";
import { BRAIN_LIMITS, RELATION_ROLES } from "./krystal-engine-schema";

/**
 * What a simulation tells Krystal — the whole boundary, in one place.
 *
 * Two documents, versioned apart because they change for different reasons:
 *
 *   krystal-world@3     what exists in this world.   Sent once, at agent creation.
 *   krystal-percept@3   what the actor perceives.    Sent every tick.
 *
 * and one shape comes back: what the creature wants to do.
 *
 * This is a VOCABULARY, not a grammar. It says which concepts a world contains
 * and which relations its creatures may attempt. What may be composed FROM
 * those concepts is grammar, and lives elsewhere; whether an attempt has any
 * physical effect is the simulation's own business and is never declared here.
 *
 * Validated rather than compiled: these cross the wire as JSON, so the scope is
 * deliberately outside `build.ts` — there is no std430 layout to derive.
 */

/**
 * The role list, spelled out because arktype needs a literal to infer from.
 * Asserted against the engine's list below, so the two cannot drift apart in
 * silence — a wire role the engine has no slot for would otherwise be accepted
 * here and dropped there.
 */
const ROLE_ALTERNATIVES = "'agent' | 'patient' | 'instrument' | 'location' | 'time' | 'reason'";

{
  const spelled = ROLE_ALTERNATIVES.split("|").map((part) => part.trim().slice(1, -1));
  if (spelled.length !== RELATION_ROLES.length || spelled.some((role, i) => role !== RELATION_ROLES[i])) {
    throw new Error(`world contract roles ${spelled.join(",")} disagree with RELATION_ROLES ${RELATION_ROLES.join(",")}`);
  }
}

/**
 * Every type here carries `"+": "reject"`.
 *
 * This is a wire boundary: a field the contract does not know is either a
 * simulation internal that must not cross, or a misspelling of one that
 * matters. Both look identical to a validator that shrugs, and the second costs
 * a day of "why can it not see the apple".
 */
export const world = scope({
  /** Closed role list of a reified relation; mirrors the engine's own. */
  RelationRole: ROLE_ALTERNATIVES,

  /**
   * A sense this world has.
   *
   * Krystal has no opinion about which senses exist — a creature may echolocate
   * and have no eyes — so a channel is declared here and carried by percepts as
   * an ordinary symbol. `quota` is the simulation's own cap on how much of the
   * perceptual band this channel may fill in one tick; Krystal does not enforce
   * it, because a world that floods one sense is describing itself accurately.
   */
  WorldChannel: {
    "+": "reject",
    symbol: "string",
    "quota?": "number > 0",
  },

  /**
   * A numeric field the world reports, declared once for the whole world.
   *
   * Kind is a property of the FIELD, never of an observation: a value that was
   * `count` last frame and `unipolar` this one would be incoherent, and
   * repeating the kind per record is exactly how that happens.
   */
  WorldQuantity: {
    "+": "reject",
    field: "string",
    kind: "'signed' | 'unipolar' | 'count' | 'proportion'",
    /** Required for `signed`: what each direction MEANS, as symbols. */
    "polarity?": { "+": "reject", negative: "string", positive: "string" },
    /** `proportion` only: the reference set the fraction is of. */
    "of?": "string",
  },

  WorldSymbol: {
    "+": "reject",
    symbol: "string",
    tokenId: "number >= 0",
    tokenClass: "string",
    "flags?": "number",
    "arity?": "number",
    "semanticTypeToken?": "number",
    "inverseToken?": "number",
  },

  /**
   * One role of a relation — its name, and nothing else.
   *
   * A world says that EAT has an agent and a patient. It does NOT say what may
   * stand in them, and the omission is the whole point: "apples are edible" is
   * something the creature has to learn by watching someone eat one and seeing
   * it vanish, not something handed to it as a label. A role that declared
   * `accepts: ["category:Edible"]` would be the simulation thinking on the
   * creature's behalf, and a creature that cannot form the thought "eat the
   * stone" can never find out why that does nothing.
   *
   * Categories still exist — in the simulation, which needs them to decide what
   * eating actually does. They just do not cross this boundary.
   */
  WorldRelationRole: {
    "+": "reject",
    role: "RelationRole",
  },

  /**
   * A relation the creature may attempt.
   *
   * Declare relations that will often fail. Capability and precondition are
   * descriptive, never exclusive: a creature that cannot attempt something can
   * never learn why it does not work — and whether an attempt succeeds is
   * decided by the simulation, after the fact, not by this declaration.
   *
   * `relation` names the same symbol an OBSERVED relation uses, and that
   * sharing is what makes the catalog mean anything: a token the creature has
   * watched someone else act out carries a meaning learned by watching.
   */
  WorldRelation: {
    "+": "reject",
    relation: "string",
    "domain?": "'external' | 'perceptual' | 'internal' | 'communicative' | 'postural'",
    roles: `WorldRelationRole[] <= ${BRAIN_LIMITS.relationArity}`,
  },

  WorldVocabulary: {
    "+": "reject",
    contract: "'krystal-world@3'",
    symbols: "WorldSymbol[]",
    channels: "WorldChannel[]",
    quantities: "WorldQuantity[]",
    relations: "WorldRelation[]",
  },

  // -------------------------------------------------------------------------
  // Per tick
  // -------------------------------------------------------------------------

  PerceptQuantity: {
    "+": "reject",
    field: "string",
    value: "number",
  },

  /**
   * One perceived thing.
   *
   * `channel` replaces the old sensory band: which sense reported this, by
   * symbol, so the engine's geometry stays free of any world's sense inventory.
   */
  PerceptRecord: {
    "+": "reject",
    channel: "string",
    schema: "string",
    /**
     * Stable world identity. Omitted for a kind-level observation, which cannot
     * be pointed at; present, it must be the SAME id in every record of this
     * entity this tick and across ticks, or the brain perceives several things
     * where there is one and working memory has nothing to hold.
     */
    "instanceId?": "string",
    tokens: `string[] <= ${BRAIN_LIMITS.recordWidth}`,
    "quantities?": "PerceptQuantity[]",
    /** Several things at once — how a distant flock stays inside one record. */
    "count?": "number >= 0",
    /** Retention hint only. Every overflow decision belongs to Krystal. */
    "salience?": "0 <= number <= 1",
    observedAt: "number >= 0",
    /**
     * Looked and found nothing (`void`), or the sense is not reporting
     * (`unavailable`). Both are percepts; omitting the record entirely is a
     * third and different thing — the sense was never sampled.
     */
    "emptiness?": "'void' | 'unavailable'",
  },

  /**
   * One side of a relation, as the simulation names it.
   *
   * `unknown` and `something` are not interchangeable. `unknown` means a
   * referent EXISTS and its identity is not known — it licenses a query, and
   * curiosity is the right response. `something` is the existential quantifier
   * and licenses nothing; it is the target of a question, never a trigger.
   */
  PerceptInstanceRef: { "+": "reject", kind: "'instance'", instanceId: "string" },
  PerceptSymbolRef: { "+": "reject", kind: "'symbol'", symbol: "string" },
  PerceptUnknownRef: { "+": "reject", kind: "'unknown'" },
  PerceptSomethingRef: { "+": "reject", kind: "'something'" },
  /**
   * The creature's own earlier proposal, by the `intentRef` it was emitted with.
   *
   * This is how an attempt gets an outcome. The simulation knows which reach it
   * is answering, and saying so lets it report "what you wanted -> what came of
   * it" as one implication rather than two unrelated facts the creature would
   * have to pair up by coincidence.
   */
  PerceptIntentRef: { "+": "reject", kind: "'intent'", intentRef: "number >= 0" },
  PerceptOperand:
    "PerceptInstanceRef | PerceptSymbolRef | PerceptUnknownRef | PerceptSomethingRef | PerceptIntentRef",

  PerceptRoleBinding: {
    "+": "reject",
    role: "RelationRole",
    operand: "PerceptOperand",
  },

  /**
   * A relation, perceived — reified, and the same shape whether it is a state
   * that holds or an act that occurred.
   *
   * Deliberately NOT an outcome. An outcome would have to say where its effects
   * end, and there is no principled answer: drop a ball on a trampoline and the
   * fall, the bounce and the second bounce are equally "the result". This
   * records that a relation obtains; whatever follows arrives as further ones.
   *
   * The actor's own acts are not sent back — the engine emitted them and knows.
   */
  PerceptRelation: {
    "+": "reject",
    channel: "string",
    relation: "string",
    roles: `PerceptRoleBinding[] <= ${BRAIN_LIMITS.relationArity}`,
    /** Punctual (it happened) versus standing (it holds). */
    aspect: "'event' | 'state'",
    /**
     * How much of it, as declared quantities — how hard the blow, how fast the
     * closing. Exact numbers, never bands: a band is a token, a token owns a
     * trained vector, and a threshold moved upstream would redefine that vector
     * without changing a symbol.
     *
     * Motion is nothing but this. Approaching is a relation between the dog and
     * me, not a property of the dog, so it arrives as one — with a `signed`
     * quantity whose polarity the vocabulary declares, and the engine picks
     * RECEDING or APPROACHING from the sign. A fast dog running away and a fast
     * dog running at me share a speed and share nothing that matters.
     */
    "quantities?": "PerceptQuantity[]",
    "salience?": "0 <= number <= 1",
    observedAt: "number >= 0",
  },

  Percept: {
    "+": "reject",
    contract: "'krystal-percept@3'",
    tick: "number >= 0",
    deltaMillis: "number >= 0",
    /**
     * How the actor is doing, 0..1, where 0 is dead. Unipolar rather than
     * signed because death is a floor and an absorbing state: nothing is as
     * good as death is bad. Send the LEVEL, never its change — the engine
     * derives the learning signal itself, so a world cannot forget to report a
     * reward, and cannot misreport one either.
     */
    valence: "0 <= number <= 1",
    actorId: "string",
    records: "PerceptRecord[]",
    /**
     * Everything relational, including movement and the actor's own motion —
     * the latter is simply a relation whose only bound role is the agent.
     *
     * Send only what changed or what holds NOW: a world standing still
     * contributes nothing here, which is why this stays small.
     */
    "relations?": "PerceptRelation[]",
  },

  /**
   * A lesson: this scene, and what the creature was supposed to do in it.
   *
   * Deliberately NOT a field on `Percept`. What a teacher knows is not
   * something the creature perceives — folding it into the tick would let the
   * answer arrive through the same channel as the world, and a creature that
   * can see the answer is not learning the task. So a lesson is its own
   * document, sent to its own endpoint, and the percept inside it is exactly
   * the percept the creature would have received anyway.
   *
   * `expect` names the relation and the participants a teacher wants chosen.
   * Participants are given as ordinary operands, so a lesson refers to things
   * by the same instance ids the scene does.
   */
  Lesson: {
    "+": "reject",
    contract: "'krystal-lesson@3'",
    percept: "Percept",
    expect: {
      "+": "reject",
      relation: "string",
      roles: `PerceptRoleBinding[] <= ${BRAIN_LIMITS.relationArity}`,
    },
    /** Human-facing label, so a curriculum stays readable in a GUI. */
    "label?": "string",
  },

  // -------------------------------------------------------------------------
  // Back out
  // -------------------------------------------------------------------------

  /**
   * What the creature wants to do.
   *
   * `commitment` is how firmly the network chose; `intensity` is how much of
   * the thing. Different quantities, deliberately not merged.
   */
  AgentIntent: {
    "+": "reject",
    relation: "string",
    roles: `PerceptRoleBinding[] <= ${BRAIN_LIMITS.relationArity}`,
    /** How much of the thing — the brain's own output, not a perceived value. */
    intensity: "number",
    commitment: "0 <= number <= 1",
    /**
     * Handle for this reach, to be quoted back when reporting what came of it.
     * See `PerceptIntentRef`.
     */
    intentRef: "number >= 0",
    /**
     * The creature reached for something it only remembers, so this is a
     * wanting rather than an act — it is not executable as it stands. The
     * simulation is free to answer it with what WOULD have happened, which is
     * how a want acquires consequences.
     */
    volitive: "boolean",
  },
});

export const $world = world;

// ---------------------------------------------------------------------------
// Runtime validators
// ---------------------------------------------------------------------------
//
// The TYPES are generated from this scope into `generated/world.types.ts` —
// runtime-free interfaces a host can use without arktype. What lives here is
// the other half: the validators themselves, for the boundary that has to
// refuse a bad document rather than merely describe a good one.

export const {
  WorldChannel,
  WorldQuantity,
  WorldSymbol,
  WorldRelationRole,
  WorldRelation,
  WorldVocabulary,
  PerceptQuantity,
  PerceptRecord,
  PerceptOperand,
  PerceptIntentRef,
  PerceptRoleBinding,
  PerceptRelation,
  Percept,
  Lesson,
  AgentIntent,
} = world.export();
