/**
 * Provisional fixture ActionIntent catalog (concerns answer 20).
 *
 * A test fixture catalog, not the final capability catalog. Descriptors carry
 * capability/precondition classes as descriptive conditioning only; they never
 * act as exclusive resource locks.
 *
 * Every intent is a binary relation. The fixture actor is always Self, so each
 * spec declares the same subject role and differs only in its object. An intent
 * that was previously nullary (WAIT, CRY, LAUGH) simply omits `object`: the
 * compiler mirrors the subject into it, which reads as the reflexive
 * construction a natural language would use anyway — LAUGH() is "I rejoice
 * myself", the shape Polish spells "cieszę SIĘ".
 */
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import {
  ACTION_INTENT_FLAGS,
  BRAIN_LIMITS,
} from "../../../schema/src/krystal-engine-schema.ts";

type RelationRoleDescriptor = v1_0_0.RelationRoleDescriptor;
type RelationRoleAuthoringSpec = v1_0_0.RelationRoleAuthoringSpec;
type ActionIntentAuthoringSpec = v1_0_0.ActionIntentAuthoringSpec;
type ActionIntentCatalogHeader = v1_0_0.ActionIntentCatalogHeader;
type ActionIntentDescriptor = v1_0_0.ActionIntentDescriptor;
import { bandMask } from "../binary-layout-plan.ts";
import { hashU32s } from "../hash.ts";
import { FIXTURE_TOKENS, fixtureTokenId } from "./vocabulary.ts";
import { argumentRequiredCapability, schemaFamilyToken, schemaTokensWithCapability } from "./capabilities.ts";
import { BRAIN_VALUE_KIND_ORDER, FIXTURE_RECORD_SCHEMAS } from "./record-schemas.ts";

export const FIXTURE_ACTION_CATALOG_VERSION = 0;

/** Pack accepted token ids into the descriptor's fixed-width acceptance list. */
export function acceptedTokenList(tokens: readonly number[]): number[] {
  if (tokens.length > BRAIN_LIMITS.maxRoleAcceptedTokens) {
    throw new Error(
      `a relation role may accept at most ${BRAIN_LIMITS.maxRoleAcceptedTokens} tokens, got ${tokens.length}`,
    );
  }
  const list = new Array<number>(BRAIN_LIMITS.maxRoleAcceptedTokens).fill(0);
  for (let i = 0; i < tokens.length; i++) list[i] = tokens[i]!;
  return list;
}

/**
 * The fixture actor. Shared by every intent because in this catalog Krystal is
 * always the subject; a game-derived catalog would vary it.
 */
const SELF_SUBJECT: RelationRoleAuthoringSpec = {
  name: "actor",
  roleToken: 0,
  valueKind: "record_ref",
  acceptedSchema: "Self",
  candidateBands: ["body"],
  doc: "the acting subject",
};

export const FIXTURE_ACTION_INTENTS: readonly ActionIntentAuthoringSpec[] = [
  {
    name: "LOOK",
    actionToken: fixtureTokenId("LOOK"),
    semanticIntentToken: fixtureTokenId("LOOK"),
    domain: "perceptual",
    subject: SELF_SUBJECT,
    object: {
      name: "target",
      roleToken: 0,
      valueKind: "context_ref",
      acceptedSchema: "VisionObject",
      candidateBands: ["vision", "memory"],
      doc: "the object to look at",
    },
    doc: "LOOK(self, ref)",
  },
  {
    name: "EAT",
    actionToken: fixtureTokenId("EAT"),
    semanticIntentToken: fixtureTokenId("EAT"),
    domain: "external",
    durative: true,
    subject: SELF_SUBJECT,
    object: {
      name: "target",
      roleToken: 0,
      valueKind: "context_ref",
      acceptedSchema: "Apple",
      candidateBands: ["vision", "memory"],
      doc: "the food to eat (capability: edible)",
    },
    doc: "EAT(self, ref)",
  },
  {
    name: "MOVE_TOWARDS",
    actionToken: fixtureTokenId("MOVE_TOWARDS"),
    semanticIntentToken: fixtureTokenId("MOVE_TOWARDS"),
    domain: "postural",
    durative: true,
    subject: SELF_SUBJECT,
    object: {
      name: "target",
      roleToken: 0,
      valueKind: "context_ref",
      acceptedSchema: "VisionObject",
      candidateBands: ["vision", "memory"],
      doc: "the spatially distant target to approach (S5)",
    },
    doc: "MOVE_TOWARDS(self, ref) — approach a distant target",
  },
  // The three unary intents below omit `object` and are compiled with
  // ACTION_INTENT_FLAGS.canonicallyReflexive. Nothing about the emitted
  // proposal is special-cased: the object head still predicts, and the runtime
  // still receives a fully populated pair.
  {
    name: "WAIT",
    actionToken: fixtureTokenId("WAIT"),
    semanticIntentToken: fixtureTokenId("WAIT"),
    domain: "internal",
    durative: true,
    subject: SELF_SUBJECT,
    doc: "do nothing for the current beat",
  },
  {
    name: "CRY",
    actionToken: fixtureTokenId("CRY"),
    semanticIntentToken: fixtureTokenId("CRY"),
    domain: "communicative",
    durative: false,
    subject: SELF_SUBJECT,
    doc: "communicative CRY() — negative homeostasis valence",
  },
  {
    name: "LAUGH",
    actionToken: fixtureTokenId("LAUGH"),
    semanticIntentToken: fixtureTokenId("LAUGH"),
    domain: "communicative",
    durative: false,
    subject: SELF_SUBJECT,
    doc: "communicative LAUGH() — positive homeostasis valence",
  },
  {
    // W2 case-binding assay (docs/word_attention_bias.md). CHASE is the first
    // fixture intent whose object is identified by a *grammatical* role rather
    // than by feasibility: every animate record is an equally legal filler, so
    // the candidate mask admits all of them and only the ACCUSATIVE marking
    // distinguishes the patient from the agent. That is deliberate — an object
    // the mask could pre-select would make the assay measure the mask, not the
    // case binding.
    name: "CHASE",
    actionToken: fixtureTokenId("CHASE"),
    semanticIntentToken: fixtureTokenId("CHASE"),
    domain: "postural",
    durative: true,
    subject: SELF_SUBJECT,
    object: {
      name: "target",
      // The object's grammatical role, declared rather than implied by
      // position: the filler is the participant marked ACCUSATIVE.
      roleToken: fixtureTokenId("ACCUSATIVE"),
      valueKind: "context_ref",
      // Identity exemplar (as EAT names Apple); the "animate" capability is
      // what actually widens acceptance to Dog/Cat/Mother.
      acceptedSchema: "Dog",
      candidateBands: ["vision", "memory"],
      doc: "the pursued participant — the accusative-marked entity (capability: animate)",
    },
    doc: "CHASE(self, ref) — pursue the accusative-marked animate participant",
  },
];

export class FixtureActionCatalogError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FixtureActionCatalogError";
  }
}

/** Domain -> descriptor flag projection (provisional fixture convention). */
function domainFlag(domain: ActionIntentAuthoringSpec["domain"]): number {
  switch (domain) {
    case "perceptual":
      return ACTION_INTENT_FLAGS.perceptual;
    case "external":
    case "postural":
      return ACTION_INTENT_FLAGS.motor;
    case "internal":
      return ACTION_INTENT_FLAGS.internal;
    case "communicative":
      return ACTION_INTENT_FLAGS.communicative;
    default: {
      const exhaustive: never = domain;
      throw new FixtureActionCatalogError(`Unknown intent domain: ${exhaustive}`);
    }
  }
}

function validateFixtureActionIntents(intents: readonly ActionIntentAuthoringSpec[]): void {
  const knownTokens = new Set(FIXTURE_TOKENS.map((spec) => spec.id));
  const recordNames = new Set(FIXTURE_RECORD_SCHEMAS.map((schema) => schema.name));
  const intentNames = new Set<string>();
  for (const intent of intents) {
    if (intentNames.has(intent.name)) {
      throw new FixtureActionCatalogError(`duplicate intent name: ${intent.name}`);
    }
    intentNames.add(intent.name);
    if (!knownTokens.has(intent.actionToken)) {
      throw new FixtureActionCatalogError(
        `${intent.name}: actionToken 0x${intent.actionToken.toString(16)} not in fixture vocabulary`,
      );
    }
    if (intent.semanticIntentToken !== 0 && !knownTokens.has(intent.semanticIntentToken)) {
      throw new FixtureActionCatalogError(
        `${intent.name}: semanticIntentToken not in fixture vocabulary`,
      );
    }
    // `object` may be absent (unary), `subject` never may.
    for (const role of [intent.subject, intent.object]) {
      if (!role) continue;
      if (role.roleToken !== 0 && !knownTokens.has(role.roleToken)) {
        throw new FixtureActionCatalogError(
          `${intent.name}.${role.name}: roleToken not in fixture vocabulary`,
        );
      }
      if (role.acceptedSchema && !recordNames.has(role.acceptedSchema)) {
        throw new FixtureActionCatalogError(
          `${intent.name}.${role.name}: acceptedSchema "${role.acceptedSchema}" does not resolve`,
        );
      }
    }
  }
}

export interface CompiledActionCatalog {
  header: ActionIntentCatalogHeader;
  descriptors: ActionIntentDescriptor[];
}

/**
 * Compile an ActionIntent catalog into device manifest forms. `intents`
 * defaults to the fixture catalog; it is a parameter so a caller (and the
 * catalog-integrity test) can compile a variant without mutating the module
 * constant. A later game-derived catalog compiles through this same path.
 */
export function buildFixtureActionCatalog(
  intents: readonly ActionIntentAuthoringSpec[] = FIXTURE_ACTION_INTENTS,
): CompiledActionCatalog {
  validateFixtureActionIntents(intents);
  const schemaIdOf = (name: string | undefined): number =>
    name ? FIXTURE_RECORD_SCHEMAS.findIndex((schema) => schema.name === name) : 0;

  // Capability widening happens HERE, at catalog compile time, and its result
  // is baked into the descriptor. It used to happen at mask time by looking the
  // capability up by name, which is why the forward pass could only ever mask
  // against this one vocabulary.
  const role = (
    spec: RelationRoleAuthoringSpec,
    intentName: string,
  ): RelationRoleDescriptor => {
    const capability = argumentRequiredCapability(intentName, spec.name);
    const acceptedToken = spec.acceptedSchema ? schemaFamilyToken(spec.acceptedSchema) : undefined;
    const accepted = capability
      ? schemaTokensWithCapability(capability)
      : acceptedToken === undefined
        ? []
        : [acceptedToken];
    return {
      roleToken: spec.roleToken,
      valueKind: spec.valueKind,
      acceptedTokens: acceptedTokenList(accepted),
      candidateBandMask: spec.candidateBands ? bandMask(spec.candidateBands) : 0,
      flags: 0,
      reserved0: 0,
    };
  };

  const descriptors: ActionIntentDescriptor[] = [];
  for (let intentId = 0; intentId < intents.length; intentId++) {
    const intent = intents[intentId]!;
    // No `object` means unary. The object role mirrors the subject so the
    // candidate mask and type check stay well-defined, and the flag records
    // that the pair was authored reflexive rather than chosen that way.
    const unary = intent.object === undefined;
    const flags =
      (intent.durative ? ACTION_INTENT_FLAGS.durative : 0) |
      (unary ? ACTION_INTENT_FLAGS.canonicallyReflexive : 0) |
      domainFlag(intent.domain);
    descriptors.push({
      intentId,
      actionToken: intent.actionToken,
      semanticIntentToken: intent.semanticIntentToken,
      domain: intent.domain,
      subjectSchemaId: schemaIdOf(intent.subject.acceptedSchema),
      flags,
      effectClassToken: intent.effectClassToken ?? 0,
      capabilityClassToken: intent.capabilityClassToken ?? 0,
      preconditionClassToken: intent.preconditionClassToken ?? 0,
      preferredControllerRole: intent.preferredControllerRole ?? 0,
      reserved0: 0,
      reserved1: 0,
      subjectRole: role(intent.subject, intent.name),
      objectRole: role(intent.object ?? intent.subject, intent.name),
    });
  }

  const words: number[] = [];
  for (const descriptor of descriptors) {
    words.push(
      descriptor.intentId,
      descriptor.actionToken,
      descriptor.semanticIntentToken,
      descriptor.subjectSchemaId,
      descriptor.flags,
      descriptor.effectClassToken,
      descriptor.capabilityClassToken,
      descriptor.preconditionClassToken,
      descriptor.preferredControllerRole,
    );
    for (const r of [descriptor.subjectRole, descriptor.objectRole]) {
      words.push(
        r.roleToken,
        BRAIN_VALUE_KIND_ORDER.indexOf(r.valueKind),
        ...r.acceptedTokens,
        r.candidateBandMask,
        r.flags,
      );
    }
  }
  const hash = hashU32s(words);
  const header: ActionIntentCatalogHeader = {
    version: FIXTURE_ACTION_CATALOG_VERSION,
    intentCount: descriptors.length,
    relationArity: BRAIN_LIMITS.relationArity,
    flags: 0,
    catalogHashLo: hash.lo,
    catalogHashHi: hash.hi,
    reserved0: 0,
    reserved1: 0,
  };
  return { header, descriptors };
}

/** Lookup helper: intent name -> descriptor, throws on unknown name. */
export function fixtureIntent(name: string): ActionIntentDescriptor {
  const catalog = buildFixtureActionCatalog();
  const descriptor = catalog.descriptors.find((candidate) => {
    return FIXTURE_ACTION_INTENTS[candidate.intentId]!.name === name;
  });
  if (!descriptor) throw new FixtureActionCatalogError(`Unknown fixture action intent: ${name}`);
  return descriptor;
}
