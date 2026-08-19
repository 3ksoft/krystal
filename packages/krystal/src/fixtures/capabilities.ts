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
 * Family tokens of the schemas whose traits include `capability`.
 *
 * TOKENS rather than schema ids, because that is what acceptance now matches:
 * a record carries its family token in its first position, so a capability
 * compiles to "any record whose family is one of these" without the mask ever
 * needing to know what a schema id means in this particular vocabulary.
 * Duplicates collapse — several fixture schemas legitimately share a family.
 */
export function schemaTokensWithCapability(capability: string): number[] {
  const tokens = new Set<number>();
  for (const schema of FIXTURE_RECORD_SCHEMAS) {
    if (FIXTURE_SCHEMA_CAPABILITIES[schema.name]?.includes(capability)) {
      tokens.add(schema.familyToken);
    }
  }
  return [...tokens];
}

/** Family token of one fixture schema, by name. */
export function schemaFamilyToken(schemaName: string): number | undefined {
  return FIXTURE_RECORD_SCHEMAS.find((schema) => schema.name === schemaName)?.familyToken;
}

/**
 * Accepted tokens for one intent argument: the capability-derived set when the
 * argument declares a required capability, else the single acceptedSchema
 * identity. Mirrors how the arg mask treats the argument descriptor.
 */
export function argumentAcceptedTokens(intentName: string, argName: string, acceptedSchemaName: string | undefined): number[] {
  const capability = argumentRequiredCapability(intentName, argName);
  if (capability) return schemaTokensWithCapability(capability);
  if (!acceptedSchemaName) return [];
  const token = schemaFamilyToken(acceptedSchemaName);
  return token === undefined ? [] : [token];
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

/** Which side of a binary relation a lookup refers to. */
export type RelationRole = "subject" | "object";

/**
 * Role name by (intentId, role) in the authoring spec. A unary intent has no
 * authored object, so its object name is its subject name — the same mirroring
 * the catalog compiler applies to the descriptor.
 */
export function roleNameById(intentId: number, role: RelationRole): string {
  const intent = FIXTURE_ACTION_INTENTS[intentId];
  if (!intent) throw new Error(`Unknown fixture intent id: ${intentId}`);
  const spec = role === "subject" ? intent.subject : (intent.object ?? intent.subject);
  return spec.name;
}
