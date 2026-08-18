/**
 * Host-side fixture capability traits (S2-S10 task, capability work).
 *
 * Capabilities are policy predicates, deliberately separate from resource
 * class/schema identity: `TARGET_OF(EAT)` is a capability ("edible"), not an
 * `Apple` schema check. The fixture record-schema catalog (record-schemas.ts)
 * stays identity-only; this module maps schema names to traits and intent
 * arguments to the traits they require. The arg-mask lowering (masks.ts)
 * consults these tables so S7+ candidates (Apple, Berry, Bread through
 * "edible"; Stone/Feces negative) fall out of the capability, not a hardcoded
 * schema list.
 */
import { FIXTURE_RECORD_SCHEMAS } from "./record-schemas.ts";
import { FIXTURE_ACTION_INTENTS } from "./action-intents.ts";

/** Capability traits attached to fixture record schemas (by schema name). */
export const FIXTURE_SCHEMA_CAPABILITIES: Readonly<Record<string, readonly string[]>> = {
  Self: [],
  VisionObject: ["observable"],
  Apple: ["observable", "edible"],
  Berry: ["observable", "edible"],
  Bread: ["observable", "edible"],
  Mother: ["observable", "agent", "animate"],
  Stone: ["observable"],
  Feces: ["observable"],
  UnknownObject: ["observable"],
  Dog: ["observable", "animate"],
  Cat: ["observable", "animate"],
  HomeostasisQuery: [],
  MemoryObject: ["observable", "remembered"],
};

/**
 * Capability required by each intent argument, keyed `intentName.argName`.
 * `EAT.target` requires "edible" (S7: any edible record, not just Apple);
 * `LOOK.target` and `MOVE_TOWARDS.target` require "observable";
 * `CHASE.target` requires "animate" — a thing that can flee, so every animate
 * record is an equally legal filler and the mask cannot pre-select the patient
 * for the selector. Absent entries fall back to the argument's
 * `acceptedSchema` identity check.
 */
export const FIXTURE_ARGUMENT_CAPABILITIES: Readonly<Record<string, string>> = {
  "EAT.target": "edible",
  "LOOK.target": "observable",
  "MOVE_TOWARDS.target": "observable",
  "CHASE.target": "animate",
};

/** Capability traits of one schema, resolved by name (throws on unknown). */
export function schemaCapabilities(schemaName: string): readonly string[] {
  const traits = FIXTURE_SCHEMA_CAPABILITIES[schemaName];
  if (!traits) throw new Error(`Unknown fixture schema for capability lookup: ${schemaName}`);
  return traits;
}

/** Required capability of one intent argument, if declared. */
export function argumentRequiredCapability(intentName: string, argName: string): string | undefined {
  return FIXTURE_ARGUMENT_CAPABILITIES[`${intentName}.${argName}`];
}

/**
 * Schema ids (indexes into FIXTURE_RECORD_SCHEMAS) whose traits include
 * `capability`. Used by the arg-mask lowering to compile candidate lists from
 * the capability rather than a fixed identity list.
 */
export function schemaIdsWithCapability(capability: string): number[] {
  const ids: number[] = [];
  for (let schemaId = 0; schemaId < FIXTURE_RECORD_SCHEMAS.length; schemaId++) {
    const name = FIXTURE_RECORD_SCHEMAS[schemaId]!.name;
    if (FIXTURE_SCHEMA_CAPABILITIES[name]?.includes(capability)) ids.push(schemaId);
  }
  return ids;
}

/**
 * Accepted schema ids for one intent argument: the capability-derived set when
 * the argument declares a required capability, else the single acceptedSchema
 * identity. Mirrors how the arg mask treats the argument descriptor.
 */
export function argumentAcceptedSchemaIds(intentName: string, argName: string, acceptedSchemaName: string | undefined): number[] {
  const capability = argumentRequiredCapability(intentName, argName);
  if (capability) return schemaIdsWithCapability(capability);
  if (!acceptedSchemaName) return [];
  const schemaId = FIXTURE_RECORD_SCHEMAS.findIndex((schema) => schema.name === acceptedSchemaName);
  return schemaId >= 0 ? [schemaId] : [];
}

/**
 * Lookup helper: intent name -> authoring spec (host-side, for the mask and
 * emission lowering to resolve argument descriptors by name).
 */
export function fixtureIntentAuthoring(name: string) {
  const intent = FIXTURE_ACTION_INTENTS.find((candidate) => candidate.name === name);
  if (!intent) throw new Error(`Unknown fixture action intent: ${name}`);
  return intent;
}

/** Intent name by compiled intentId (catalog order == authoring order). */
export function intentNameById(intentId: number): string {
  const intent = FIXTURE_ACTION_INTENTS[intentId];
  if (!intent) throw new Error(`Unknown fixture intent id: ${intentId}`);
  return intent.name;
}

/** Argument name by (intentId, argumentIndex) in the authoring spec. */
export function argumentNameById(intentId: number, argumentIndex: number): string {
  const argument = FIXTURE_ACTION_INTENTS[intentId]?.arguments[argumentIndex];
  if (!argument) throw new Error(`Unknown fixture argument (intent ${intentId}, arg ${argumentIndex})`);
  return argument.name;
}
