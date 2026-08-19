import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";

export type PropositionModality = "declarative" | "imperative" | "interrogative" | "implicative";

export interface AffordanceRequirement {
  readonly maxDistance?: number;
  readonly requiresHolding?: boolean;
  readonly requiresEmptyHand?: boolean;
  readonly candidateBands?: readonly v1_0_0.BrainBandKind[];
  readonly acceptedCategories?: readonly number[];
  readonly preferredControllerRecord?: number;
}

export interface LegalActionCandidate {
  readonly intentId: number;
  readonly actionToken: number;
  readonly actionSymbol: string;
  readonly subjectToken: number;
  readonly objectToken: number;
  readonly controllerRecord: number;
  readonly targetRecordSlot: number;
  readonly priorityWeight: number;
}

export interface SlotConstraint {
  readonly allowedClasses?: readonly v1_0_0.KrystalTokenClass[];
  readonly allowedTokens?: readonly number[];
  readonly candidateBands?: readonly v1_0_0.BrainBandKind[];
  readonly requiredCategory?: number;
}

export interface ValidationDiagnostic {
  readonly code:
    | "INVALID_MODALITY"
    | "UNKNOWN_ACTION"
    | "MISSING_SUBJECT"
    | "MISSING_OBJECT"
    | "CATEGORY_MISMATCH"
    | "OUT_OF_REACH"
    | "CONTROLLER_BUSY"
    | "NOT_HOLDING_TARGET"
    | "HANDS_FULL";
  readonly message: string;
  readonly slotIndex: number;
  readonly receivedToken?: number;
}

export interface ActionMaskResult {
  readonly legalActions: readonly LegalActionCandidate[];
  readonly logitMask: Float32Array;
  readonly candidateCount: number;
}