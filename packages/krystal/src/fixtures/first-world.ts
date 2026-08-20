import type { v1_0_0 as world } from "../../../schema/generated/world.types.ts";

/**
 * The smallest world that can teach anything.
 *
 * A child, its mother, an apple and a stone. The apple and the stone are both
 * round, and nothing in what the child receives says which one is food: the
 * apple is `round red sweetsmell`, the stone is `round grey hard`. No
 * `category:Edible` anywhere, because that is the conclusion, not the input.
 *
 * What the child does get is a mother who picks the apple up and eats it while
 * the child watches. That observation and the consequences of its own attempts
 * are the only evidence available, which is the point of the fixture: a world
 * where the answer is learnable but never given.
 *
 * The vocabulary is append-only. An embedding row is a learned vector indexed
 * by manifest POSITION, so inserting a symbol in the middle would redefine
 * every later row and training would continue against scrambled meanings.
 */

const T = {
  senseSight: 0x5800,
  senseSmell: 0x5801,
  senseTouch: 0x5802,

  person: 0x1800,
  thing: 0x1801,

  round: 0x2000,
  red: 0x2001,
  grey: 0x2002,
  hard: 0x2003,
  soft: 0x2004,
  sweetsmell: 0x2005,
  hurt: 0x2006,

  hunger: 0x2800,

  EAT: 0x3000,
  PICKUP: 0x3001,
  LOOK: 0x3002,
  CRY: 0x3003,
} as const;

export const FIRST_WORLD: world.WorldVocabulary = {
  contract: "krystal-world@3",
  symbols: [
    { symbol: "sense:Sight", tokenId: T.senseSight, tokenClass: "domain" },
    { symbol: "sense:Smell", tokenId: T.senseSmell, tokenClass: "domain" },
    { symbol: "sense:Touch", tokenId: T.senseTouch, tokenClass: "domain" },

    { symbol: "person", tokenId: T.person, tokenClass: "object" },
    { symbol: "thing", tokenId: T.thing, tokenClass: "object" },

    { symbol: "round", tokenId: T.round, tokenClass: "property" },
    { symbol: "red", tokenId: T.red, tokenClass: "property" },
    { symbol: "grey", tokenId: T.grey, tokenClass: "property" },
    { symbol: "hard", tokenId: T.hard, tokenClass: "property" },
    { symbol: "soft", tokenId: T.soft, tokenClass: "property" },
    { symbol: "sweetsmell", tokenId: T.sweetsmell, tokenClass: "property" },
    { symbol: "hurt", tokenId: T.hurt, tokenClass: "property" },

    { symbol: "hunger", tokenId: T.hunger, tokenClass: "quantity" },

    { symbol: "EAT", tokenId: T.EAT, tokenClass: "action" },
    { symbol: "PICKUP", tokenId: T.PICKUP, tokenClass: "action" },
    { symbol: "LOOK", tokenId: T.LOOK, tokenClass: "action" },
    { symbol: "CRY", tokenId: T.CRY, tokenClass: "action" },
  ],
  channels: [
    { symbol: "sense:Sight" },
    { symbol: "sense:Smell" },
    { symbol: "sense:Touch" },
  ],
  quantities: [{ field: "hunger", kind: "unipolar" }],
  // A role declares its name and nothing more. What may fill it is what the
  // creature learns; declaring it here would be this world thinking for it.
  relations: [
    { relation: "EAT", roles: [{ role: "agent" }, { role: "patient" }] },
    { relation: "PICKUP", roles: [{ role: "agent" }, { role: "patient" }] },
    { relation: "LOOK", roles: [{ role: "agent" }, { role: "patient" }] },
    { relation: "CRY", roles: [{ role: "agent" }] },
  ],
};

export interface FirstWorldEntity {
  readonly instanceId: string;
  readonly schema: "person" | "thing";
  readonly tokens: readonly string[];
  /** Only the simulation knows this. It never crosses to the creature. */
  readonly nourishing: boolean;
  present: boolean;
}

export interface FirstWorldState {
  tick: number;
  /** 0 = starving, 1 = sated. Valence is derived from it. */
  satiation: number;
  hurt: boolean;
  /** Ticks since the apple was eaten; it regrows after a few. */
  appleGoneSince: number;
  entities: FirstWorldEntity[];
}

export function firstWorldState(): FirstWorldState {
  return {
    tick: 0,
    satiation: 0.6,
    hurt: false,
    appleGoneSince: 0,
    entities: [
      { instanceId: "child", schema: "person", tokens: [], nourishing: false, present: true },
      { instanceId: "mother", schema: "person", tokens: [], nourishing: false, present: true },
      { instanceId: "apple", schema: "thing", tokens: ["round", "red", "sweetsmell", "soft"], nourishing: true, present: true },
      { instanceId: "stone", schema: "thing", tokens: ["round", "grey", "hard"], nourishing: false, present: true },
    ],
  };
}

/**
 * One tick of perception.
 *
 * `demonstrate` puts the mother through picking the apple up and eating it —
 * the observation the creature has to generalize from. It happens early and
 * then stops, so what follows tests whether anything was learned rather than
 * continuing to hand over the answer.
 */
export function firstWorldPercept(
  state: FirstWorldState,
  options: { readonly demonstrate?: boolean } = {},
): world.Percept {
  const records: world.PerceptRecord[] = [];

  for (const entity of state.entities) {
    if (!entity.present) continue;
    const tokens = [...entity.tokens];
    if (entity.instanceId === "child" && state.hurt) tokens.push("hurt");
    records.push({
      channel: entity.tokens.includes("sweetsmell") ? "sense:Smell" : "sense:Sight",
      schema: entity.schema,
      instanceId: entity.instanceId,
      tokens,
      observedAt: state.tick,
      salience: entity.instanceId === "child" ? 1 : 0.8,
    });
  }

  // The creature's own hunger is an ordinary percept on a homeostasis-bound
  // record, distinct from valence: "I am hungry" and "I am doing badly" are
  // different things and the creature should be able to tell them apart.
  records.push({
    channel: "sense:Touch",
    schema: "person",
    instanceId: "child",
    tokens: [],
    quantities: [{ field: "hunger", value: 1 - state.satiation }],
    observedAt: state.tick,
    salience: 0.9,
  });

  const relations: world.PerceptRelation[] = [];
  const apple = state.entities.find((entity) => entity.instanceId === "apple");
  if (options.demonstrate && apple?.present) {
    for (const relation of ["PICKUP", "EAT"] as const) {
      relations.push({
        channel: "sense:Sight",
        relation,
        aspect: "event",
        observedAt: state.tick,
        roles: [
          { role: "agent", operand: { kind: "instance", instanceId: "mother" } },
          { role: "patient", operand: { kind: "instance", instanceId: "apple" } },
        ],
      });
    }
  }

  return {
    contract: "krystal-percept@3",
    tick: state.tick,
    deltaMillis: 16,
    // Unipolar: 0 is dead and an absorbing state, so nothing is as good as
    // death is bad. How a world arrives at the number is its own business.
    valence: Math.max(0, Math.min(1, state.satiation - (state.hurt ? 0.2 : 0))),
    actorId: "child",
    records,
    ...(relations.length === 0 ? {} : { relations }),
  };
}

/**
 * What an attempt actually does — the simulation's verdict, and nowhere else's.
 *
 * The creature is free to propose eating the stone. This is where that turns
 * out badly, and the only place that knows the difference between an apple and
 * a stone in the first place.
 */
export function firstWorldApply(
  state: FirstWorldState,
  intent: { readonly relation: string; readonly patient?: string; readonly volitive: boolean },
): string {
  state.hurt = false;

  // A reach toward something only remembered is not executed. It is a wanting,
  // and the world has nothing to do about it.
  if (intent.volitive) return "wanted, not done";

  const target = state.entities.find((entity) => entity.instanceId === intent.patient);

  if (intent.relation === "EAT" && target) {
    if (!target.present) return `${target.instanceId} is gone`;
    if (target.nourishing) {
      state.satiation = Math.min(1, state.satiation + 0.35);
      target.present = false;
      return `ate ${target.instanceId}, sated`;
    }
    state.hurt = true;
    return `bit ${target.instanceId}, hurt`;
  }

  return `${intent.relation} did nothing`;
}

/**
 * Hunger creeps up on its own; that is what makes anything matter.
 *
 * An eaten apple comes back after a while. Not realism — recurrence: a creature
 * learning from consequences needs the same opportunity to arise more than
 * once, or its one good outcome is an anecdote it can never test.
 */
export const APPLE_REGROWS_AFTER = 6;

export function firstWorldStep(state: FirstWorldState): void {
  state.tick++;
  state.satiation = Math.max(0, state.satiation - 0.02);

  const apple = state.entities.find((entity) => entity.instanceId === "apple")!;
  if (!apple.present) {
    state.appleGoneSince++;
    if (state.appleGoneSince >= APPLE_REGROWS_AFTER) {
      apple.present = true;
      state.appleGoneSince = 0;
    }
  }
}
