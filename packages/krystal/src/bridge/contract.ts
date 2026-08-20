import { type } from "arktype";
import { Percept as PerceptValidator } from "../../../schema/src/world.ts";
import type { v1_0_0 as world } from "../../../schema/generated/world.types.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import {
  BRAIN_LIMITS,
  INTENT_PROPOSAL_FLAGS,
  RELATION_ROLES,
  RELATION_ROLE_INDEX,
} from "../../../schema/src/krystal-engine-schema.ts";
import type { CompiledVocabulary } from "./agent.ts";
import { quantize } from "./quantize.ts";

/**
 * The tick boundary: what a simulation may send, checked before it becomes a
 * frame.
 *
 * Two passes, because they catch different things. `Percept` (the arktype
 * scope) settles SHAPE — the fields that exist, their types, their ranges, and
 * that nothing undeclared crossed. What is left is agreement with THIS world:
 * whether a symbol is in its vocabulary, whether a channel is one it declared,
 * whether a relation points at something actually perceived. No schema can know
 * that, because it is a fact about one agent rather than about the contract.
 *
 * Strict on purpose. An unknown symbol is refused, never dropped: a boundary
 * that quietly forgets is what makes "why can it not see the apple" cost a day.
 */

export class PerceptContractError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PerceptContractError";
  }
}

export interface PerceptDiagnostics {
  unknownSymbols: Map<string, number>;
  unknownChannels: Map<string, number>;
  droppedRecords: number;
}

export function emptyDiagnostics(): PerceptDiagnostics {
  return { unknownSymbols: new Map(), unknownChannels: new Map(), droppedRecords: 0 };
}

export interface ValidateOptions {
  readonly onUnknown?: "throw" | "count";
}

/**
 * Check one tick against the contract and against this world.
 *
 * Returns the percept narrowed to its validated type, so a caller cannot go on
 * using an unchecked value by accident.
 */
export function validatePercept(
  percept: unknown,
  vocabulary: CompiledVocabulary,
  options: ValidateOptions = {},
): { percept: world.Percept; diagnostics: PerceptDiagnostics } {
  const shaped = PerceptValidator(percept);
  if (shaped instanceof type.errors) {
    throw new PerceptContractError(`percept does not match krystal-percept@3: ${shaped.summary}`);
  }
  const checked = shaped as world.Percept;

  const strict = (options.onUnknown ?? "throw") === "throw";
  const diagnostics = emptyDiagnostics();
  const { tokenBySymbol, channels, quantities } = vocabulary;

  const requireSymbol = (symbol: string, context: string): boolean => {
    if (tokenBySymbol.has(symbol)) return true;
    if (strict) {
      throw new PerceptContractError(
        `${context}: '${symbol}' is not in this world's vocabulary; declare it or the brain cannot perceive it`,
      );
    }
    diagnostics.unknownSymbols.set(symbol, (diagnostics.unknownSymbols.get(symbol) ?? 0) + 1);
    return false;
  };

  const requireChannel = (channel: string, context: string): boolean => {
    if (channels.has(channel)) return true;
    if (strict) {
      throw new PerceptContractError(
        `${context}: channel '${channel}' is not one this world declared`,
      );
    }
    diagnostics.unknownChannels.set(channel, (diagnostics.unknownChannels.get(channel) ?? 0) + 1);
    return false;
  };

  const checkQuantities = (
    values: readonly world.PerceptQuantity[] | undefined,
    context: string,
  ): void => {
    for (const quantity of values ?? []) {
      const declared = quantities.get(quantity.field);
      if (!declared) {
        throw new PerceptContractError(
          `${context}.${quantity.field}: no such quantity field in this world; declare its kind before sending values`,
        );
      }
      try {
        // Banding is the engine's decision. Running it here is what refuses a
        // pre-banded or out-of-range value at the boundary rather than letting
        // it become a token that means something else.
        quantize(quantity.value, declared.kind as v1_0_0.QuantityKind, declared.polarity);
      } catch (error) {
        throw new PerceptContractError(`${context}.${quantity.field}: ${(error as Error).message}`);
      }
    }
  };

  const instances = new Set<string>();
  for (const record of checked.records) {
    if (record.instanceId) instances.add(record.instanceId);
  }

  for (const record of checked.records) {
    const context = `record '${record.schema}'`;
    let ok = requireChannel(record.channel, context);
    ok = requireSymbol(record.schema, context) && ok;
    for (const token of record.tokens) {
      ok = requireSymbol(token, `${context} token`) && ok;
    }
    checkQuantities(record.quantities, context);
    if (!ok) diagnostics.droppedRecords++;
  }

  for (const relation of checked.relations ?? []) {
    const context = `relation '${relation.relation}'`;
    requireChannel(relation.channel, context);
    requireSymbol(relation.relation, context);
    checkQuantities(relation.quantities, context);

    const seenRoles = new Set<string>();
    for (const binding of relation.roles) {
      if (seenRoles.has(binding.role)) {
        throw new PerceptContractError(`${context} binds role '${binding.role}' twice`);
      }
      seenRoles.add(binding.role);

      const operand = binding.operand;
      if (operand.kind === "instance") {
        if (!instances.has(operand.instanceId) && operand.instanceId !== checked.actorId) {
          throw new PerceptContractError(
            `${context} names instance '${operand.instanceId}' in role '${binding.role}', which is not perceived this tick; ` +
              `report an unidentified participant as { kind: "unknown" } rather than dropping the relation`,
          );
        }
      } else if (operand.kind === "symbol") {
        requireSymbol(operand.symbol, `${context} role '${binding.role}'`);
      }
      // An `intent` operand names a proposal this engine emitted. Its number
      // cannot be checked against anything here — the proposal is long gone —
      // so the shape check is all there is, and a wrong one simply resolves to
      // a reference the creature has no memory of.
    }
  }

  return { percept: checked, diagnostics };
}

/**
 * Turn what the brain chose into what the simulation reads.
 *
 * A proposal whose agent or patient resolved to nothing is dropped: the
 * simulation cannot act on a relation with no participants, and emitting one
 * would make an empty intent look like a decision.
 */
export function toAgentIntents(
  intentSet: v1_0_0.IntentSet,
  relationOf: (intentId: number) => string | undefined,
  instanceOf: (refToken: number) => string | undefined,
): world.AgentIntent[] {
  const intents: world.AgentIntent[] = [];

  for (const proposal of intentSet.proposals) {
    if (proposal.lifecycle === "empty") continue;
    const relation = relationOf(proposal.intentId);
    if (relation === undefined) continue;

    const roles: world.PerceptRoleBinding[] = [];
    for (const role of RELATION_ROLES) {
      const selected = proposal.roles[RELATION_ROLE_INDEX[role]];
      if (!selected) continue;
      const token = selected.concept.handle.tokenId;
      if (token === 0) continue;
      const instanceId = instanceOf(token);
      if (instanceId === undefined) continue;
      roles.push({ role, operand: { kind: "instance", instanceId } });
    }

    const bound = new Set(roles.map((binding) => binding.role));
    if (!bound.has("agent") || !bound.has("patient")) continue;

    intents.push({
      relation,
      roles,
      intensity: proposal.intensity,
      commitment: proposal.commitment,
      // Always emitted, and stable for this tick: it is the handle the
      // simulation quotes back when it reports what came of the attempt.
      intentRef: intentSet.tick * BRAIN_LIMITS.maxIntentProposals + proposal.proposalSlot,
      volitive: (proposal.flags & INTENT_PROPOSAL_FLAGS.volitive) !== 0,
    });
  }

  return intents;
}
