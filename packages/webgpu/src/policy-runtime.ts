import { BRAIN_LIMITS, INVALID_U32 } from "../../schema/src/krystal-engine-schema.ts";
import { packBrainFrame, unpackRuntimeHandle } from "../../krystal/src/frame/packer.ts";
import { BRAIN_FORWARD_CONFIG, createBrainForwardWeights } from "../../krystal/src/forward/model.ts";
import { type SelectionMasks } from "./krystal-forward.ts";
import { KrystalForward } from "./krystal-forward.ts";
import type { CompiledActionCatalog } from "../../krystal/src/fixtures/action-intents.ts";
import { FIXTURE_ACTION_INTENTS } from "../../krystal/src/fixtures/action-intents.ts";
import { fixtureTokenId } from "../../krystal/src/fixtures/vocabulary.ts";
import { ACTION_INTENT_SCHEMA_ID } from "../../krystal/src/fixtures/frame.ts";
import { emitIntentSet } from "../../krystal/src/forward/intentset.ts";
import { argMaskFor, compileActiveFrame, compileIntentMask, type ActiveFrame, type WordBias } from "../../krystal/src/forward/masks.ts";
import type { PolicyAction, PolicyRawFrame } from "../../krystal/src/training/policy.ts";
import type { SelectionSlotResult } from "../../krystal/src/forward/oracle.ts";
import type { v1_0_0 } from "../../schema/generated/krystal.types.ts";

export { packBrainFrame, createBrainForwardWeights, BRAIN_FORWARD_CONFIG };
export type { v1_0_0 };

export const POLICY_CONFIG = {
  ...BRAIN_FORWARD_CONFIG,
  routeKindCount: 6,
};

export const ROUTE_KIND: Readonly<Record<PolicyAction, number>> = {
  CRY: 0,
  LAUGH: 1,
  EAT: 2,
  MOVE_TOWARDS: 3,
  LOOK: 4,
  WAIT: 5,
  // The typed decision head is a weight shape ([routeKindCount, 3H]), so a
  // seventh route would renumber every existing initialization. CHASE shares
  // the postural approach route with MOVE_TOWARDS: the W2 assay
  // (docs/word_attention_bias.md) trains a single-action catalog, so nothing
  // in it has to tell the two routes apart. Promoting CHASE into the
  // curriculum means giving it its own kind and raising routeKindCount.
  CHASE: 3,
};

export const ACTION_TOKEN: Readonly<Record<PolicyAction, number>> = {
  LOOK: fixtureTokenId("LOOK"),
  EAT: fixtureTokenId("EAT"),
  MOVE_TOWARDS: fixtureTokenId("MOVE_TOWARDS"),
  WAIT: fixtureTokenId("WAIT"),
  CRY: fixtureTokenId("CRY"),
  LAUGH: fixtureTokenId("LAUGH"),
  CHASE: fixtureTokenId("CHASE"),
};

export function catalogBankIndex(frame: v1_0_0.BrainFrameGpu, active: ActiveFrame, actionToken: number): number {
  return active.bankRecords.findIndex(
    (slot) => frame.schemaIds[slot] === ACTION_INTENT_SCHEMA_ID &&
      frame.tokenIds[slot * BRAIN_LIMITS.recordWidth] === actionToken,
  );
}

export function bankIndexOfRef(frame: v1_0_0.BrainFrameGpu, active: ActiveFrame, refToken: number): number {
  for (let j = 0; j < active.bankRecords.length; j++) {
    const slot = active.bankRecords[j]!;
    const packed = frame.runtimeRefs[slot * BRAIN_LIMITS.maxReferencesPerRecord]!;
    if (packed !== INVALID_U32 && unpackRuntimeHandle(packed).tokenId === refToken) return j;
  }
  throw new Error(`ref 0x${refToken.toString(16)} not found in the frame bank`);
}

export interface PreparedTrain {
  readonly frame: v1_0_0.BrainFrameGpu;
  readonly selection: SelectionMasks;
  readonly routeKinds: Uint32Array;
  readonly intentGold: Uint32Array;
  readonly argumentTargets: Uint32Array[];
}

export function prepareTrainFrame(
  frame: v1_0_0.BrainFrameGpu,
  gold: PolicyRawFrame["gold"],
  catalog: CompiledActionCatalog,
): PreparedTrain {
  const active = compileActiveFrame(frame);
  const goldIntentId = catalog.descriptors.find((d) => d.actionToken === ACTION_TOKEN[gold.action])!.intentId;
  const argumentTarget = gold.refToken === undefined ? INVALID_U32 : bankIndexOfRef(frame, active, gold.refToken);
  return {
    frame,
    selection: {
      intentMask: compileIntentMask(frame, active, ACTION_INTENT_SCHEMA_ID),
      argMask: argMaskFor(frame, active, catalog, goldIntentId),
    },
    routeKinds: Uint32Array.of(ROUTE_KIND[gold.action]),
    intentGold: Uint32Array.of(catalogBankIndex(frame, active, ACTION_TOKEN[gold.action])),
    argumentTargets: [Uint32Array.of(argumentTarget)],
  };
}

export interface PolicyPrediction {
  readonly intentId: number;
  readonly action: string;
  readonly refToken?: number;
}

export interface ProductionSelection {
  readonly frame: v1_0_0.BrainFrameGpu;
  readonly active: ActiveFrame;
  readonly intent: SelectionSlotResult;
  readonly argument: SelectionSlotResult;
}

export async function productionSelection(
  device: GPUDevice,
  runner: KrystalForward,
  frame: v1_0_0.BrainFrameGpu,
  catalog: CompiledActionCatalog,
  wordBias?: WordBias,
): Promise<ProductionSelection | null> {
  const active = compileActiveFrame(frame);
  const q = active.queryRecords.length;
  const r = active.bankRecords.length;
  const intentMask = compileIntentMask(frame, active, ACTION_INTENT_SCHEMA_ID);

  runner.forward(frame, { intentMask, argMask: intentMask }, wordBias);
  await device.queue.onSubmittedWorkDone();
  const first = await runner.readSelection(q, r, POLICY_CONFIG.hiddenSize);
  const bankIdx = first.intent.index[0]!;
  if (bankIdx >= r) return null;
  const slot = active.bankRecords[bankIdx]!;
  if (frame.schemaIds[slot] !== ACTION_INTENT_SCHEMA_ID) return null;
  const actionToken = frame.tokenIds[slot * BRAIN_LIMITS.recordWidth]!;
  const descriptor = catalog.descriptors.find((candidate) => candidate.actionToken === actionToken);
  if (!descriptor) return null;

  runner.forward(frame, {
    intentMask,
    argMask: argMaskFor(frame, active, catalog, descriptor.intentId),
  }, wordBias);
  await device.queue.onSubmittedWorkDone();
  const second = await runner.readSelection(q, r, POLICY_CONFIG.hiddenSize);
  return { frame, active, intent: first.intent, argument: second.argument };
}

export function emitPrediction(selection: ProductionSelection, catalog: CompiledActionCatalog): PolicyPrediction | null {
  const { intentSet } = emitIntentSet({
    frame: selection.frame,
    active: selection.active,
    catalog,
    intentSchemaId: ACTION_INTENT_SCHEMA_ID,
    intent: selection.intent,
    argument: selection.argument,
    tick: 0,
  });
  if (intentSet.count === 0) return null;
  const proposal = intentSet.proposals[0]!;
  const refToken = proposal.object.concept.handle.tokenId;
  return {
    intentId: proposal.intentId,
    action: FIXTURE_ACTION_INTENTS[proposal.intentId]!.name,
    refToken: refToken === 0 ? undefined : refToken,
  };
}
