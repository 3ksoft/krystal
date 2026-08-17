/**
 * Host-side IntentSet emission (concerns answer 27).
 *
 * The GPU selection heads emit per-slot softmax distributions P [Q, R] and
 * argmax record indices into the bank. The runtime never asks the network to
 * synthesize an arbitrary exact handle: the network selects a compatible
 * location, and this host resolver maps that location to the exact sidecar
 * binding (arch v2 §9, TRAINING_DESIGN.md §4/§5).
 *
 * v0 envelope (answer 27): at most one proposal per query row — `count`,
 * `lifecycle: 'start'`, resolved `intentId`, and typed arguments whose exact
 * handles are read from the selected records' runtime-ref sidecars.
 * Confidence and entropy are emitted as debug/diagnostic values. intentRef /
 * purposeGoal / controllerHint / topic are left invalid: the runtime assigns
 * an exact intentRef when a proposal is accepted.
 */
import {
  BRAIN_LIMITS,
  INVALID_U32,
} from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import { unpackRuntimeHandle } from "../frame/packer.ts";
import type { ActiveFrame } from "./masks.ts";
import type { SelectionSlotResult } from "./oracle.ts";
import type { CompiledActionCatalog } from "../fixtures/action-intents.ts";

const EPS = 1e-6;

function invalidRef(): v1_0_0.RuntimeRefHandle {
  return { tokenId: 0, generation: 0, kind: "none", status: "invalid" };
}

function emptySelector(): v1_0_0.SoftGatherResult {
  return {
    status: "empty",
    selectedRecord: INVALID_U32,
    selectedField: INVALID_U32,
    selectedReference: INVALID_U32,
    candidateCount: 0,
    probability: 0,
    entropy: 0,
    reserved0: 0,
  };
}

function emptyArgument(): v1_0_0.TypedArgumentValue {
  return {
    kind: "none",
    token: 0,
    flags: 0,
    reserved0: 0,
    handle: invalidRef(),
    selector: emptySelector(),
  };
}

/** A proposal with lifecycle 'empty' and all fields invalid/zero. */
export function emptyProposal(proposalSlot: number): v1_0_0.IntentProposal {
  return {
    proposalSlot,
    lifecycle: "empty",
    intentId: 0,
    flags: 0,
    intentRef: invalidRef(),
    purposeGoal: invalidRef(),
    controllerHint: invalidRef(),
    topic: invalidRef(),
    activation: 0,
    priority: 0,
    intensity: 0.5,
    persistence: 0,
    confidence: 0,
    arguments: Array.from({ length: BRAIN_LIMITS.maxActionArguments }, () => emptyArgument()),
  };
}

export interface IntentSetEmissionInput {
  /** The packed SoA frame (exact runtime-ref sidecars live here). */
  readonly frame: v1_0_0.BrainFrameGpu;
  /** Active record/query lists compiled from the packed frame. */
  readonly active: ActiveFrame;
  /** Compiled ActionIntent catalog: intent + argument descriptors. */
  readonly catalog: CompiledActionCatalog;
  /** Schema id of the ActionIntent catalog records (selector mask target). */
  readonly intentSchemaId: number;
  /** Per-slot selection head outputs (GPU readbacks or CPU oracle). */
  readonly intent: SelectionSlotResult;
  readonly argument: SelectionSlotResult;
  readonly tick: number;
  readonly revision?: number;
  /** Cap on emitted proposals (defaults to the schema transport capacity). */
  readonly maxProposals?: number;
}

export interface IntentSetEmissionResult {
  readonly intentSet: v1_0_0.IntentSet;
  /** Number of emitted proposals (== intentSet.count). */
  readonly emitted: number;
}

/**
 * Resolve the selection heads into a typed IntentSet.
 *
 * One proposal is emitted per query row whose argmax lands on a valid
 * ActionIntent catalog record (schema id check, then action-token lookup in
 * the catalog). For each argument of the selected intent, the shared argument
 * selector's argmax record is resolved through its runtime-ref sidecar; a
 * record that fails the accepted-schema check yields a masked/error argument
 * instead of a fabricated handle.
 */
export function emitIntentSet(input: IntentSetEmissionInput): IntentSetEmissionResult {
  const {
    frame,
    active,
    catalog,
    intentSchemaId,
    intent,
    argument,
    tick,
    revision = 0,
  } = input;
  const maxProposals = input.maxProposals ?? BRAIN_LIMITS.maxIntentProposals;

  const r = active.bankRecords.length;
  const q = active.queryRecords.length;
  const recordWidth = BRAIN_LIMITS.recordWidth;
  const maxRefs = BRAIN_LIMITS.maxReferencesPerRecord;

  if (intent.index.length < q || argument.index.length < q) {
    throw new Error(
      `emitIntentSet: selection indices must cover ${q} query rows ` +
        `(intent ${intent.index.length}, argument ${argument.index.length})`,
    );
  }
  if (intent.p.length < q * r || argument.p.length < q * r) {
    throw new Error(
      `emitIntentSet: selection P must be [${q}, ${r}] ` +
        `(intent ${intent.p.length}, argument ${argument.p.length})`,
    );
  }

  const proposals: v1_0_0.IntentProposal[] = [];
  for (let slot = 0; slot < maxProposals; slot++) {
    proposals.push(emptyProposal(slot));
  }

  let emitted = 0;
  for (let i = 0; i < Math.min(q, maxProposals); i++) {
    const bankIdx = intent.index[i]!;
    if (bankIdx >= r) continue;
    const slot = active.bankRecords[bankIdx]!;

    // The selector mask only leaves catalog records open; a record that is
    // not one (e.g. an all-masked row resolving to index 0) is not a proposal.
    if (frame.schemaIds[slot] !== intentSchemaId) continue;
    const actionToken = frame.tokenIds[slot * recordWidth]!;
    const descriptor = catalog.descriptors.find((candidate) => candidate.actionToken === actionToken);
    if (!descriptor) continue;

    const intentProb = intent.p[i * r + bankIdx]!;
    const rowStart = i * r;
    let candidateCount = 0;
    let entropy = 0;
    for (let j = 0; j < r; j++) {
      const p = intent.p[rowStart + j]!;
      if (p > EPS) {
        candidateCount++;
        entropy -= p * Math.log(p);
      }
    }

    const proposal: v1_0_0.IntentProposal = {
      proposalSlot: i,
      lifecycle: "start",
      intentId: descriptor.intentId,
      flags: 0,
      intentRef: invalidRef(), // runtime assigns an exact ref at acceptance
      purposeGoal: invalidRef(),
      controllerHint: invalidRef(),
      topic: invalidRef(),
      activation: intentProb,
      priority: 0,
      intensity: 0.5,
      persistence: 0,
      confidence: intentProb,
      arguments: Array.from({ length: BRAIN_LIMITS.maxActionArguments }, () => emptyArgument()),
    };

    // v0 shares one argument selector across slots (answer 26); each typed
    // argument of the selected intent resolves through that selection.
    for (let k = 0; k < descriptor.argumentCount && k < BRAIN_LIMITS.maxActionArguments; k++) {
      const argDesc = catalog.arguments[descriptor.argumentOffset + k]!;
      const argBankIdx = argument.index[i]!;
      const argSlot = argBankIdx < r ? active.bankRecords[argBankIdx]! : undefined;

      const argRowStart = i * r;
      let argCandidateCount = 0;
      let argEntropy = 0;
      for (let j = 0; j < r; j++) {
        const p = argument.p[argRowStart + j]!;
        if (p > EPS) {
          argCandidateCount++;
          argEntropy -= p * Math.log(p);
        }
      }

      let status: v1_0_0.SoftGatherStatus = "masked";
      let handle: v1_0_0.RuntimeRefHandle = invalidRef();
      let selectedRecord = INVALID_U32;
      let selectedReference = INVALID_U32;
      if (argSlot !== undefined && frame.schemaIds[argSlot] === argDesc.acceptedSchemaId) {
        // Exact handle from the record's runtime-ref sidecar (first binding).
        const packed = frame.runtimeRefs[argSlot * maxRefs] ?? INVALID_U32;
        if (packed !== INVALID_U32) {
          const unpacked = unpackRuntimeHandle(packed);
          handle = { ...unpacked, kind: "entity", status: "live" };
          selectedRecord = argSlot;
          selectedReference = 0;
          status = "selected";
        } else {
          status = "error";
        }
      }

      proposal.arguments[k] = {
        kind: argDesc.valueKind,
        token: handle.tokenId,
        flags: 0,
        reserved0: 0,
        handle,
        selector: {
          status,
          selectedRecord,
          selectedField: INVALID_U32,
          selectedReference,
          candidateCount: argCandidateCount,
          probability: argBankIdx < r ? argument.p[argRowStart + argBankIdx]! : 0,
          entropy: argEntropy,
          reserved0: 0,
        },
      };
    }

    proposals[i] = proposal;
    emitted++;
  }

  const intentSet: v1_0_0.IntentSet = {
    tick,
    count: emitted,
    revision,
    flags: 0,
    proposals,
  };
  return { intentSet, emitted };
}
