import {
  KRYSTAL_ABI,
  KRYSTAL_SENTINEL_TOKENS,
  KRYSTAL_TOKEN_RANGES,
  RELATION_ROLES,
  RELATION_ROLE_FLAGS,
  RELATION_ROLE_INDEX,
  type RelationRoleName,
} from "../../../schema/src/krystal-engine-schema.ts";
import {
  PREDICATE_CLASSES,
  planRecord,
  validatePlan,
  type RecordPlan,
} from "../../../schema/src/jantar.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import type { CompiledCatalog, CompiledVocabulary } from "../bridge/agent.ts";
import type { SlotConstraint, ValidationDiagnostic } from "./types.ts";

/**
 * Generation, one slot at a time, under jantar.
 *
 * The creature does not pick a finished thought off a shelf; it builds one, and
 * at every step something has to say what may come next. That is this class:
 * jantar supplies the shape of a record, the vocabulary supplies what exists,
 * the catalog supplies what each role of the chosen relation will take, and the
 * product is a logit mask over the token space.
 *
 * What it never does is ask whether the thought could come true. A mask that
 * refused EAT because nothing edible is within reach would be teaching the
 * creature that unreachable food is unthinkable, and a creature that cannot
 * think it can never learn why reaching failed.
 */
export class StructuralALU {
  constructor(
    private readonly vocabulary: CompiledVocabulary,
    private readonly catalog: CompiledCatalog,
  ) {}

  /** Relation tokens this world declared — the imperative's whole predicate set. */
  private relationTokens(): number[] {
    return this.catalog.descriptors.map((descriptor) => descriptor.actionToken);
  }

  private descriptorFor(relationToken: number): v1_0_0.ActionIntentDescriptor | undefined {
    return this.catalog.descriptors.find((entry) => entry.actionToken === relationToken);
  }

  private declaredRoles(descriptor: v1_0_0.ActionIntentDescriptor): RelationRoleName[] {
    return RELATION_ROLES.filter((role) => {
      const slot = descriptor.roles[RELATION_ROLE_INDEX[role]];
      return slot !== undefined && (slot.flags & RELATION_ROLE_FLAGS.present) !== 0;
    });
  }

  /**
   * The record shape a relation implies under one modality.
   *
   * Throws when the relation declares no agent: jantar admits no impersonal
   * constructions, and a catalog entry with nobody to perform it is a
   * vocabulary bug rather than something to work around here.
   */
  public planFor(relationToken: number, modality: v1_0_0.PropositionModality): RecordPlan {
    const descriptor = this.descriptorFor(relationToken);
    if (!descriptor) {
      throw new Error(`token 0x${relationToken.toString(16)} names no declared relation`);
    }
    return planRecord(modality, this.declaredRoles(descriptor));
  }

  /**
   * What may fill the next empty slot.
   *
   * `currentTokens` is the record so far: its first entry, once present, is the
   * relation, and that is what conditions every slot after it. Before it there
   * is nothing to condition on, so the constraint is the modality's own.
   */
  public getNextSlotConstraint(
    currentTokens: readonly number[],
    modality: v1_0_0.PropositionModality,
  ): SlotConstraint {
    const slotIndex = currentTokens.length;

    if (slotIndex === 0) {
      // An imperative names something this world declared the creature can do,
      // so its predicate set is exact rather than a class range.
      if (modality === "imperative") return { allowedTokens: this.relationTokens() };
      return { allowedClasses: PREDICATE_CLASSES[modality] };
    }

    const relationToken = currentTokens[0]!;
    const descriptor = this.descriptorFor(relationToken);
    if (!descriptor) {
      // Not a declared relation: nothing downstream can be conditioned, so only
      // qualities remain admissible.
      return { allowedClasses: ["property", "quantity"] };
    }

    const plan = planRecord(modality, this.declaredRoles(descriptor));
    const slot = plan.slots[slotIndex];
    if (!slot || slot.kind === "modifier") {
      return { allowedClasses: ["property", "quantity", "reference"], complete: slot === undefined };
    }

    // A participant is not generated. It has to be something the creature
    // perceives or remembers, so the role selector picks it out of the bank and
    // the reference binds to that record. Generating a symbol here would let the
    // creature refer to a thing it has never encountered.
    return { role: slot.role, boundBySelection: true };
  }

  /**
   * The logit mask a sampler applies before drawing a token.
   *
   * Over the SEMANTIC half of the token space: the reference half is bound at
   * runtime to whatever the frame happens to hold, so it carries no learned row
   * and cannot be generated into.
   *
   * The old implementation returned an all-blocked mask whenever a constraint
   * named classes rather than exact tokens, which is every slot but the
   * imperative's first — so nothing could ever be generated at all.
   */
  public buildTokenLogitMask(
    constraint: SlotConstraint,
    vocabSize: number = KRYSTAL_ABI.semanticVocabSize,
  ): Float32Array {
    const mask = new Float32Array(vocabSize).fill(-Infinity);

    // Nothing to draw: this slot is filled by the role selector, from the bank.
    if (constraint.boundBySelection) return mask;

    if (constraint.allowedTokens && constraint.allowedTokens.length > 0) {
      for (const token of constraint.allowedTokens) {
        if (token < vocabSize) mask[token] = 0;
      }
      return mask;
    }

    for (const tokenClass of constraint.allowedClasses ?? []) {
      const range = KRYSTAL_TOKEN_RANGES[tokenClass as keyof typeof KRYSTAL_TOKEN_RANGES];
      if (!range) continue;
      const end = Math.min(range[1], vocabSize - 1);
      for (let token = range[0]; token <= end; token++) mask[token] = 0;
    }

    // Only tokens this world actually declared may be drawn. A class range is a
    // span of ADDRESSES, most of which name nothing: leaving them open would let
    // the creature emit a token with no meaning, no embedding row and no way for
    // anyone to say what it was trying to think.
    for (let token = 0; token < vocabSize; token++) {
      if (mask[token] === 0 && !this.vocabulary.symbolByToken.has(token)) mask[token] = -Infinity;
    }

    return mask;
  }

  /** Well-formedness of a finished record. Shape only — never feasibility. */
  public validateRecordStructure(record: v1_0_0.BrainRecordSlot): ValidationDiagnostic[] {
    const diagnostics: ValidationDiagnostic[] = [];
    const modality = record.header.modality;

    const predicateToken = record.tokens[0];
    if (predicateToken === undefined || predicateToken === KRYSTAL_SENTINEL_TOKENS.pad) {
      return diagnostics;
    }

    const descriptor = this.descriptorFor(predicateToken);
    if (modality === "imperative" && !descriptor) {
      diagnostics.push({
        code: "UNKNOWN_RELATION",
        message:
          `an imperative names an act this world declared; token 0x${predicateToken.toString(16)} is not one`,
        slotIndex: 0,
        receivedToken: predicateToken,
      });
      return diagnostics;
    }
    if (!descriptor) return diagnostics;

    const declared = new Set(this.declaredRoles(descriptor));
    const bound = new Set<RelationRoleName>();
    for (let i = 0; i < record.header.referenceCount; i++) {
      const binding = record.references[i];
      if (!binding || binding.handle.tokenId === 0) continue;
      const role = binding.role as RelationRoleName;
      if (bound.has(role)) {
        diagnostics.push({
          code: "DUPLICATE_ROLE",
          message: `role '${role}' is bound twice`,
          slotIndex: i,
        });
      }
      if (!declared.has(role)) {
        diagnostics.push({
          code: "ROLE_NOT_DECLARED",
          message: `this relation declares no '${role}' role`,
          slotIndex: i,
        });
      }
      bound.add(role);
    }

    const plan = planRecord(modality, this.declaredRoles(descriptor));
    for (const problem of validatePlan(plan, bound)) {
      diagnostics.push({
        code: problem.includes("agent")
          ? "MISSING_AGENT"
          : problem.includes("patient")
            ? "MISSING_PATIENT"
            : "ARITY_EXCEEDED",
        message: problem,
        slotIndex: 0,
      });
    }

    return diagnostics;
  }
}
