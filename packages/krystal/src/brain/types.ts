import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";

/**
 * What may stand in the next slot of a record under construction.
 *
 * Generation fills a record one token at a time, and this is the constraint the
 * logit mask is built from. It is a GRAMMATICAL limit, never a physical one:
 * whether the creature can reach the apple is the simulation's verdict, arrived
 * at after the fact. A record that says something impossible is still a
 * well-formed thought.
 */
export interface SlotConstraint {
  readonly allowedClasses?: readonly v1_0_0.KrystalTokenClass[];
  readonly allowedTokens?: readonly number[];
  readonly candidateBands?: readonly v1_0_0.BrainBandKind[];
  readonly requiredCategory?: number;
  /** Which role this slot binds, when it binds one. */
  readonly role?: v1_0_0.RelationRole;
  /**
   * This slot is filled by SELECTING from the frame, not by generating a token.
   *
   * A participant has to be something the creature currently perceives or
   * remembers — it cannot act on, or even refer to, a thing it has never seen.
   * So the generator does not draw a symbol here; the role selector picks a
   * record out of the bank and the reference binds to it. Generation proper is
   * for the predicate and the modifiers.
   */
  readonly boundBySelection?: boolean;
  /** Nothing further is admissible: the record is complete. */
  readonly complete?: boolean;
}

export interface ValidationDiagnostic {
  readonly code:
    | "INVALID_MODALITY"
    | "UNKNOWN_RELATION"
    | "MISSING_AGENT"
    | "MISSING_PATIENT"
    | "ROLE_NOT_DECLARED"
    | "DUPLICATE_ROLE"
    | "CATEGORY_MISMATCH"
    | "ARITY_EXCEEDED";
  readonly message: string;
  readonly slotIndex: number;
  readonly receivedToken?: number;
}
