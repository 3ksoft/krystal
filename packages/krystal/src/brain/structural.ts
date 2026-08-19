import {
  BRAIN_LIMITS,
  KRYSTAL_SENTINEL_TOKENS,
  TOKEN_FLAGS,
} from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import type { CompiledGrammar } from "../agent.ts";
import type { SlotConstraint, ValidationDiagnostic } from "./types.ts";

export class StructuralALU {
  constructor(private readonly grammar: CompiledGrammar) {}

  public validateRecordStructure(record: v1_0_0.BrainRecordSlot): ValidationDiagnostic[] {
    const diagnostics: ValidationDiagnostic[] = [];
    const modality = record.header.modality;

    if (!["declarative", "imperative", "interrogative", "implicative"].includes(modality)) {
      diagnostics.push({
        code: "INVALID_MODALITY",
        message: `Unsupported record proposition modality: ${modality}`,
        slotIndex: 0,
      });
      return diagnostics;
    }

    const predicateToken = record.tokens[0];
    if (predicateToken === KRYSTAL_SENTINEL_TOKENS.pad) {
      return diagnostics;
    }

    if (modality === "imperative") {
      const isAction = this.grammar.actions.some(
        (a) => this.grammar.tokenBySymbol.get(a.relation) === predicateToken,
      );
      if (!isAction) {
        diagnostics.push({
          code: "UNKNOWN_ACTION",
          message: `Imperative proposition requires an action predicate; received token 0x${predicateToken.toString(16)}`,
          slotIndex: 0,
          receivedToken: predicateToken,
        });
      }
    }

    return diagnostics;
  }

  public getNextSlotConstraint(
    currentTokens: readonly number[],
    modality: v1_0_0.PropositionModality,
  ): SlotConstraint {
    const slotIndex = currentTokens.length;

    if (slotIndex === 0) {
      if (modality === "imperative") {
        const actionTokens = this.grammar.actions
          .map((a) => this.grammar.tokenBySymbol.get(a.relation))
          .filter((t): t is number => t !== undefined);
        return { allowedTokens: actionTokens };
      }
      return { allowedClasses: ["action", "relation", "system", "property"] };
    }

    if (slotIndex === 1) {
      return { candidateBands: ["body", "system"] };
    }

    if (slotIndex === 2) {
      return { candidateBands: ["vision", "touch", "audio", "memory", "body"] };
    }

    return { allowedClasses: ["property", "quantity", "reference"] };
  }

  public buildTokenLogitMask(
    constraint: SlotConstraint,
    vocabSize = BRAIN_LIMITS.frameTokens,
  ): Float32Array {
    const mask = new Float32Array(vocabSize).fill(-Infinity);

    if (constraint.allowedTokens) {
      for (const token of constraint.allowedTokens) {
        if (token < vocabSize) mask[token] = 0.0;
      }
      return mask;
    }

    return mask;
  }
}