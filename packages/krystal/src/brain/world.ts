
import {
  BRAIN_FRAME_BANDS,
  BRAIN_FIXED_RECORDS,
  BRAIN_LIMITS,
  KRYSTAL_ABI,
  KRYSTAL_SENTINEL_TOKENS,
} from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import type { CompiledGrammar, SimActionV2 } from "../agent.ts";
import type { OntologyGraph } from "./ontology.ts";
import type {
  ActionMaskResult,
  AffordanceRequirement,
  LegalActionCandidate,
  ValidationDiagnostic,
} from "./types.ts";

export interface EntityPercept {
  readonly instanceId: string;
  readonly refToken: number;
  readonly recordSlot: number;
  readonly band: v1_0_0.BrainBandKind;
  readonly distance: number;
  readonly categoryTokens: readonly number[];
  readonly isHolding: boolean;
}

export interface EvaluatedBodyState {
  readonly actorToken: number;
  readonly leftHandHoldingToken?: number;
  readonly rightHandHoldingToken?: number;
  readonly isMouthOccupied: boolean;
}

export class WorldALU {
  private readonly defaultAffordances = new Map<string, AffordanceRequirement>();

  constructor(
    private readonly grammar: CompiledGrammar,
    private readonly ontology: OntologyGraph,
  ) {
    this.initDefaultSimulationAffordances();
  }

  public registerAffordance(actionSymbol: string, affordance: AffordanceRequirement): void {
    this.defaultAffordances.set(actionSymbol, affordance);
  }

  public computeActionMask(
    frame: v1_0_0.BrainFrame,
    instanceByToken: ReadonlyMap<number, string>,
  ): ActionMaskResult {
    const legalActions: LegalActionCandidate[] = [];
    const totalActions = this.grammar.actions.length;
    const maxProposals = BRAIN_LIMITS.maxActionIntents;
    const logitMask = new Float32Array(maxProposals).fill(-Infinity);

    const body = this.evaluateBodyState(frame);
    const percepts = this.gatherSensoryEntities(frame, instanceByToken, body);

    for (let intentId = 0; intentId < totalActions; intentId++) {
      const action = this.grammar.actions[intentId]!;
      const actionToken = this.grammar.tokenBySymbol.get(action.relation);
      if (actionToken === undefined) continue;

      const affordance = this.defaultAffordances.get(action.relation) ?? {};
      const controllerRecord = affordance.preferredControllerRecord ?? BRAIN_FIXED_RECORDS.self;

      if (!action.object) {
        legalActions.push({
          intentId,
          actionToken,
          actionSymbol: action.relation,
          subjectToken: body.actorToken,
          objectToken: body.actorToken,
          controllerRecord,
          targetRecordSlot: BRAIN_FIXED_RECORDS.self,
          priorityWeight: 1.0,
        });
        if (intentId < maxProposals) logitMask[intentId] = 0.0;
        continue;
      }

      for (const percept of percepts) {
        const diagnostic = this.checkAffordance(action, affordance, percept, body);
        if (diagnostic === undefined) {
          legalActions.push({
            intentId,
            actionToken,
            actionSymbol: action.relation,
            subjectToken: body.actorToken,
            objectToken: percept.refToken,
            controllerRecord,
            targetRecordSlot: percept.recordSlot,
            priorityWeight: 1.0 / Math.max(1.0, percept.distance),
          });
          if (intentId < maxProposals) logitMask[intentId] = 0.0;
        }
      }
    }

    return {
      legalActions,
      logitMask,
      candidateCount: legalActions.length,
    };
  }

  public validateProposal(
    frame: v1_0_0.BrainFrame,
    proposal: v1_0_0.IntentProposal,
    instanceByToken: ReadonlyMap<number, string>,
  ): ValidationDiagnostic | undefined {
    const action = this.grammar.actions[proposal.intentId];
    if (!action) {
      return {
        code: "UNKNOWN_ACTION",
        message: `Intent ID ${proposal.intentId} is out of catalog bounds`,
        slotIndex: 0,
      };
    }

    const body = this.evaluateBodyState(frame);
    const objectToken = proposal.object.concept.handle.tokenId;

    if (!action.object) return undefined;

    const percepts = this.gatherSensoryEntities(frame, instanceByToken, body);
    const targetPercept = percepts.find((p) => p.refToken === objectToken);

    if (!targetPercept) {
      return {
        code: "OUT_OF_REACH",
        message: `Target entity 0x${objectToken.toString(16)} is not currently perceived`,
        slotIndex: 2,
        receivedToken: objectToken,
      };
    }

    const affordance = this.defaultAffordances.get(action.relation) ?? {};
    return this.checkAffordance(action, affordance, targetPercept, body);
  }

  private checkAffordance(
    action: SimActionV2,
    affordance: AffordanceRequirement,
    percept: EntityPercept,
    body: EvaluatedBodyState,
  ): ValidationDiagnostic | undefined {
    if (action.object?.accepts && action.object.accepts.length > 0) {
      const isAccepted = action.object.accepts.some((categorySymbol) => {
        const categoryToken = this.grammar.tokenBySymbol.get(categorySymbol);
        if (categoryToken === undefined) return false;
        return percept.categoryTokens.some((tok) => this.ontology.isA(tok, categoryToken));
      });

      if (!isAccepted) {
        return {
          code: "CATEGORY_MISMATCH",
          message: `Object does not satisfy category requirements for action '${action.relation}'`,
          slotIndex: 2,
          receivedToken: percept.refToken,
        };
      }
    }

    if (affordance.maxDistance !== undefined && percept.distance > affordance.maxDistance) {
      return {
        code: "OUT_OF_REACH",
        message: `Entity distance ${percept.distance.toFixed(2)} exceeds max affordance reach ${affordance.maxDistance}`,
        slotIndex: 2,
        receivedToken: percept.refToken,
      };
    }

    if (affordance.requiresHolding && !percept.isHolding) {
      return {
        code: "NOT_HOLDING_TARGET",
        message: `Action '${action.relation}' requires holding target object in hand`,
        slotIndex: 2,
        receivedToken: percept.refToken,
      };
    }

    if (affordance.requiresEmptyHand && body.leftHandHoldingToken && body.rightHandHoldingToken) {
      return {
        code: "HANDS_FULL",
        message: `Action '${action.relation}' requires at least one free hand`,
        slotIndex: 1,
        receivedToken: body.actorToken,
      };
    }

    return undefined;
  }

  private evaluateBodyState(frame: v1_0_0.BrainFrame): EvaluatedBodyState {
    const actorRecord = frame.records[BRAIN_FIXED_RECORDS.actor];
    const actorToken = actorRecord?.references[0]?.handle.tokenId ?? KRYSTAL_SENTINEL_TOKENS.pad;

    const leftHandRecord = frame.records[BRAIN_FIXED_RECORDS.leftHand];
    const rightHandRecord = frame.records[BRAIN_FIXED_RECORDS.rightHand];

    const leftRef = leftHandRecord?.references[0]?.handle.tokenId;
    const rightRef = rightHandRecord?.references[0]?.handle.tokenId;

    const leftHandHoldingToken =
      leftRef && leftRef >= KRYSTAL_ABI.refSpaceStart ? leftRef : undefined;
    const rightHandHoldingToken =
      rightRef && rightRef >= KRYSTAL_ABI.refSpaceStart ? rightRef : undefined;

    return {
      actorToken,
      leftHandHoldingToken,
      rightHandHoldingToken,
      isMouthOccupied: false,
    };
  }

  private gatherSensoryEntities(
    frame: v1_0_0.BrainFrame,
    instanceByToken: ReadonlyMap<number, string>,
    body: EvaluatedBodyState,
  ): EntityPercept[] {
    const percepts: EntityPercept[] = [];
    const sensoryBands: v1_0_0.BrainBandKind[] = ["vision", "touch", "body"];

    for (const bandKind of sensoryBands) {
      const layout = BRAIN_FRAME_BANDS.find((b) => b.kind === bandKind);
      if (!layout) continue;

      for (let i = 0; i < layout.recordCapacity; i++) {
        const slot = layout.recordOffset + i;
        const record = frame.records[slot]!;

        if ((record.header.flags & 1) === 0) continue;

        const refHandle = record.references[0]?.handle;
        if (!refHandle || refHandle.tokenId < KRYSTAL_ABI.refSpaceStart) continue;

        const refToken = refHandle.tokenId;
        const instanceId = instanceByToken.get(refToken) ?? `ref_${refToken.toString(16)}`;

        const isHolding =
          body.leftHandHoldingToken === refToken || body.rightHandHoldingToken === refToken;

        const distance =
          isHolding || bandKind === "touch" ? 0.0 : record.header.changeMagnitude || 1.0;

        const categoryTokens = record.tokens.filter(
          (t) => t !== KRYSTAL_SENTINEL_TOKENS.pad && t !== refToken,
        );

        percepts.push({
          instanceId,
          refToken,
          recordSlot: slot,
          band: bandKind,
          distance,
          categoryTokens,
          isHolding,
        });
      }
    }

    return percepts;
  }

  private initDefaultSimulationAffordances(): void {
    this.defaultAffordances.set("PICK_UP", {
      maxDistance: 1.2,
      requiresEmptyHand: true,
      preferredControllerRecord: BRAIN_FIXED_RECORDS.leftHand,
    });

    this.defaultAffordances.set("EAT", {
      maxDistance: 0.5,
      requiresHolding: true,
      preferredControllerRecord: BRAIN_FIXED_RECORDS.mouth,
    });

    this.defaultAffordances.set("DROP", {
      requiresHolding: true,
      preferredControllerRecord: BRAIN_FIXED_RECORDS.leftHand,
    });

    this.defaultAffordances.set("MOVE", {
      maxDistance: 50.0,
      preferredControllerRecord: BRAIN_FIXED_RECORDS.locomotion,
    });

    this.defaultAffordances.set("LOOK", {
      maxDistance: 100.0,
      preferredControllerRecord: BRAIN_FIXED_RECORDS.head,
    });
  }
}