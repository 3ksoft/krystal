/**
 * Provisional fixture ActionIntent catalog (concerns answer 20).
 *
 * `LOOK(ref)`, `EAT(ref)` and `WAIT` — the minimal set the first forward
 * milestone needs. This is a test fixture catalog, not the final capability
 * catalog. Descriptors carry capability/precondition classes as descriptive
 * conditioning only; they never act as exclusive resource locks.
 */
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import { ACTION_INTENT_FLAGS } from "../../../schema/src/krystal-engine-schema.ts";

type ActionArgumentDescriptor = v1_0_0.ActionArgumentDescriptor;
type ActionArgumentAuthoringSpec = v1_0_0.ActionArgumentAuthoringSpec;
type ActionIntentAuthoringSpec = v1_0_0.ActionIntentAuthoringSpec;
type ActionIntentCatalogHeader = v1_0_0.ActionIntentCatalogHeader;
type ActionIntentDescriptor = v1_0_0.ActionIntentDescriptor;
import { bandMask } from "../binary-layout-plan.ts";
import { hashU32s } from "../hash.ts";
import { FIXTURE_TOKENS, fixtureTokenId } from "./vocabulary.ts";
import { BRAIN_VALUE_KIND_ORDER, FIXTURE_RECORD_SCHEMAS } from "./record-schemas.ts";

export const FIXTURE_ACTION_CATALOG_VERSION = 0;

export const FIXTURE_ACTION_INTENTS: readonly ActionIntentAuthoringSpec[] = [
  {
    name: "LOOK",
    actionToken: fixtureTokenId("LOOK"),
    semanticIntentToken: fixtureTokenId("LOOK"),
    domain: "perceptual",
    arguments: [
      {
        name: "target",
        roleToken: 0,
        valueKind: "context_ref",
        acceptedSchema: "VisionObject",
        candidateBands: ["vision", "memory"],
        required: true,
        doc: "the object to look at",
      },
    ],
    doc: "perceptual LOOK(ref)",
  },
  {
    name: "EAT",
    actionToken: fixtureTokenId("EAT"),
    semanticIntentToken: fixtureTokenId("EAT"),
    domain: "external",
    durative: true,
    arguments: [
      {
        name: "target",
        roleToken: 0,
        valueKind: "context_ref",
        acceptedSchema: "Apple",
        candidateBands: ["vision", "memory"],
        required: true,
        doc: "the food to eat (capability: edible)",
      },
    ],
    doc: "EAT(ref)",
  },
  {
    name: "MOVE_TOWARDS",
    actionToken: fixtureTokenId("MOVE_TOWARDS"),
    semanticIntentToken: fixtureTokenId("MOVE_TOWARDS"),
    domain: "postural",
    durative: true,
    arguments: [
      {
        name: "target",
        roleToken: 0,
        valueKind: "context_ref",
        acceptedSchema: "VisionObject",
        candidateBands: ["vision", "memory"],
        required: true,
        doc: "the spatially distant target to approach (S5)",
      },
    ],
    doc: "MOVE_TOWARDS(ref) — approach a distant target",
  },
  {
    name: "WAIT",
    actionToken: fixtureTokenId("WAIT"),
    semanticIntentToken: fixtureTokenId("WAIT"),
    domain: "internal",
    durative: true,
    arguments: [],
    doc: "do nothing for the current beat",
  },
  {
    name: "CRY",
    actionToken: fixtureTokenId("CRY"),
    semanticIntentToken: fixtureTokenId("CRY"),
    domain: "communicative",
    durative: false,
    arguments: [],
    doc: "communicative CRY() — negative homeostasis valence",
  },
  {
    name: "LAUGH",
    actionToken: fixtureTokenId("LAUGH"),
    semanticIntentToken: fixtureTokenId("LAUGH"),
    domain: "communicative",
    durative: false,
    arguments: [],
    doc: "communicative LAUGH() — positive homeostasis valence",
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

function validateFixtureActionIntents(): void {
  const knownTokens = new Set(FIXTURE_TOKENS.map((spec) => spec.id));
  const recordNames = new Set(FIXTURE_RECORD_SCHEMAS.map((schema) => schema.name));
  const intentNames = new Set<string>();
  for (const intent of FIXTURE_ACTION_INTENTS) {
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
    if (intent.arguments.length > 4) {
      throw new FixtureActionCatalogError(`${intent.name}: exceeds maxActionArguments (4)`);
    }
    for (const argument of intent.arguments) {
      if (argument.roleToken !== 0 && !knownTokens.has(argument.roleToken)) {
        throw new FixtureActionCatalogError(
          `${intent.name}.${argument.name}: roleToken not in fixture vocabulary`,
        );
      }
      if (argument.acceptedSchema && !recordNames.has(argument.acceptedSchema)) {
        throw new FixtureActionCatalogError(
          `${intent.name}.${argument.name}: acceptedSchema "${argument.acceptedSchema}" does not resolve`,
        );
      }
      if (argument.valueKind === "context_ref" && !argument.required) {
        throw new FixtureActionCatalogError(
          `${intent.name}.${argument.name}: context_ref arguments must be required in fixtures`,
        );
      }
    }
  }
}

export interface CompiledActionCatalog {
  header: ActionIntentCatalogHeader;
  descriptors: ActionIntentDescriptor[];
  arguments: ActionArgumentDescriptor[];
}

/** Compile the fixture ActionIntent catalog into device manifest forms. */
export function buildFixtureActionCatalog(): CompiledActionCatalog {
  validateFixtureActionIntents();
  const schemaIdOf = (name: string | undefined): number =>
    name ? FIXTURE_RECORD_SCHEMAS.findIndex((schema) => schema.name === name) : 0;

  const descriptors: ActionIntentDescriptor[] = [];
  const argDescriptors: ActionArgumentDescriptor[] = [];
  let argumentOffset = 0;
  for (let intentId = 0; intentId < FIXTURE_ACTION_INTENTS.length; intentId++) {
    const intent = FIXTURE_ACTION_INTENTS[intentId]!;
    const flags =
      (intent.durative ? ACTION_INTENT_FLAGS.durative : 0) | domainFlag(intent.domain);
    descriptors.push({
      intentId,
      actionToken: intent.actionToken,
      semanticIntentToken: intent.semanticIntentToken,
      domain: intent.domain,
      actorSchemaId: schemaIdOf("Self"),
      argumentOffset,
      argumentCount: intent.arguments.length,
      flags,
      effectClassToken: intent.effectClassToken ?? 0,
      capabilityClassToken: intent.capabilityClassToken ?? 0,
      preconditionClassToken: intent.preconditionClassToken ?? 0,
      preferredControllerRole: intent.preferredControllerRole ?? 0,
    });
    for (let argumentIndex = 0; argumentIndex < intent.arguments.length; argumentIndex++) {
      const argument: ActionArgumentAuthoringSpec = intent.arguments[argumentIndex]!;
      argDescriptors.push({
        intentId,
        argumentIndex,
        roleToken: argument.roleToken,
        valueKind: argument.valueKind,
        acceptedSchemaId: schemaIdOf(argument.acceptedSchema),
        candidateBandMask: argument.candidateBands ? bandMask(argument.candidateBands) : 0,
        flags: argument.required ? 1 : 0, // provisional fixture convention
        reserved0: 0,
      });
    }
    argumentOffset += intent.arguments.length;
  }

  const words: number[] = [];
  for (const descriptor of descriptors) {
    words.push(
      descriptor.intentId,
      descriptor.actionToken,
      descriptor.semanticIntentToken,
      descriptor.actorSchemaId,
      descriptor.argumentOffset,
      descriptor.argumentCount,
      descriptor.flags,
      descriptor.effectClassToken,
      descriptor.capabilityClassToken,
      descriptor.preconditionClassToken,
      descriptor.preferredControllerRole,
    );
  }
  for (const argument of arguments) {
    words.push(
      argument.intentId,
      argument.argumentIndex,
      argument.roleToken,
      BRAIN_VALUE_KIND_ORDER.indexOf(argument.valueKind),
      argument.acceptedSchemaId,
      argument.candidateBandMask,
      argument.flags,
    );
  }
  const hash = hashU32s(words);
  const header: ActionIntentCatalogHeader = {
    version: FIXTURE_ACTION_CATALOG_VERSION,
    intentCount: descriptors.length,
    argumentCount: arguments.length,
    flags: 0,
    catalogHashLo: hash.lo,
    catalogHashHi: hash.hi,
    reserved0: 0,
    reserved1: 0,
  };
  return { header, descriptors, arguments: argDescriptors };
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
