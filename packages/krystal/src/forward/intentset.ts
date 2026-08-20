/**
 * Host-side IntentSet emission.
 *
 * The selection heads emit per-slot softmax distributions P [Q, R] and argmax
 * record indices into the bank. The runtime never asks the network to
 * synthesize an exact handle: the network selects a compatible LOCATION, and
 * this resolver maps that location to the exact sidecar binding.
 *
 * Every proposal is a reified relation: one selection per role the relation
 * declares, resolved through the same path. There is no privileged subject any
 * more — the agent is scored against the bank like every other participant,
 * and `Self` is simply a candidate the agent role admits. A relation that
 * declares no patient is reflexive, and the flag records that the slot was
 * mirrored rather than chosen.
 *
 * intentRef / purposeGoal / controllerHint / topic are left invalid: the
 * runtime assigns an exact intentRef when a proposal is accepted.
 */
import {
  ACTION_INTENT_FLAGS,
  BRAIN_LIMITS,
  INTENT_PROPOSAL_FLAGS,
  INVALID_U32,
  RELATION_ROLES,
  RELATION_ROLE_FLAGS,
  RELATION_ROLE_INDEX,
  type RelationRoleName,
} from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import { bandIndex } from "../binary-layout-plan.ts";
import { unpackRuntimeHandle } from "../frame/packer.ts";
import { roleAdmitsRecord, roleFilterFor } from "./masks.ts";
import type { ActiveFrame } from "./masks.ts";
import type { SelectionSlotResult } from "./oracle.ts";
import type { CompiledCatalog } from "../bridge/agent.ts";

const EPS = 1e-6;

/**
 * Reaching into memory is wanting, not doing.
 *
 * A creature cannot act on what is not in front of it, so a proposal whose
 * participant came from the memory band is not executable as it stands — and
 * that is precisely what makes it a wanting rather than an act. The reading is
 * derived from where the selection landed; the creature never asserts it, and
 * so has nothing to gain by claiming to want something.
 */
const MEMORY_BAND_INDEX = bandIndex("memory");

/** One selection head per role. A role with no head is simply not filled. */
export type RoleSelections = Partial<Record<RelationRoleName, SelectionSlotResult>>;

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

function emptyConcept(): v1_0_0.ConceptRef {
  return { kind: "none", token: 0, flags: 0, reserved0: 0, handle: invalidRef() };
}

function emptySelectedConcept(): v1_0_0.SelectedConceptRef {
  return { concept: emptyConcept(), selector: emptySelector() };
}

/** A proposal with lifecycle 'empty' and all fields invalid/zero. */
export function emptyProposal(proposalSlot: number): v1_0_0.IntentProposal {
  return {
    proposalSlot,
    lifecycle: "empty",
    modality: "imperative",
    intentId: 0,
    flags: 0,
    intentRef: invalidRef(),
    purposeGoal: invalidRef(),
    controllerHint: invalidRef(),
    topic: invalidRef(),
    activation: 0,
    priority: 0,
    // Schema default: an empty slot carries no selection signal, so the ABI
    // default stands (real proposals get a head-computed value below).
    commitment: 0.5,
    intensity: 0,
    persistence: 0,
    confidence: 0,
    roles: RELATION_ROLES.map(() => emptySelectedConcept()),
  };
}

/**
 * Resolve one role of a relation from a chosen record slot.
 *
 * How a concept is identified depends on its declared value kind, and the two
 * cases are not interchangeable:
 *
 *   context_ref  A world entity. Identity is the exact handle in the record's
 *                runtime-ref sidecar, read from the frame and never synthesized
 *                by the network. No live binding means not executable.
 *   record_ref   A structural record — Self, a body part, a fixed slot. These
 *                are addressed by record index and legitimately carry no
 *                dynamic reference, so demanding a sidecar handle here would
 *                reject every structural participant.
 */
function resolveConcept(
  frame: v1_0_0.BrainFrameGpu,
  slot: number | undefined,
  valueKind: v1_0_0.BrainValueKind,
  maxRefs: number,
  selector: {
    status: v1_0_0.SoftGatherStatus;
    probability: number;
    candidateCount: number;
    entropy: number;
  },
): v1_0_0.SelectedConceptRef {
  let handle: v1_0_0.RuntimeRefHandle = invalidRef();
  let selectedRecord = INVALID_U32;
  let selectedReference = INVALID_U32;
  let status = selector.status;

  if (slot !== undefined) {
    const packed = frame.runtimeRefs[slot * maxRefs] ?? INVALID_U32;
    if (packed !== INVALID_U32) {
      handle = { ...unpackRuntimeHandle(packed), kind: "entity", status: "live" };
      selectedRecord = slot;
      selectedReference = 0;
      status = "selected";
    } else if (valueKind === "record_ref") {
      // The record slot is the identity; there is nothing to look up.
      selectedRecord = slot;
      status = "selected";
    } else {
      status = "error";
    }
  }

  return {
    concept: {
      kind: status === "selected" ? valueKind : "none",
      token: handle.tokenId,
      flags: 0,
      reserved0: 0,
      handle,
    },
    selector: {
      status,
      selectedRecord,
      selectedField: INVALID_U32,
      selectedReference,
      candidateCount: selector.candidateCount,
      probability: selector.probability,
      entropy: selector.entropy,
      reserved0: 0,
    },
  };
}

/** Peakedness and support of one row of a selection distribution. */
function rowStatistics(
  selection: SelectionSlotResult,
  row: number,
  r: number,
): { candidateCount: number; entropy: number } {
  let candidateCount = 0;
  let entropy = 0;
  const start = row * r;
  for (let j = 0; j < r; j++) {
    const p = selection.p[start + j]!;
    if (p > EPS) {
      candidateCount++;
      entropy -= p * Math.log(p);
    }
  }
  return { candidateCount, entropy };
}

export interface IntentSetEmissionInput {
  /** The packed SoA frame (exact runtime-ref sidecars live here). */
  readonly frame: v1_0_0.BrainFrameGpu;
  /** Active record/query lists compiled from the packed frame. */
  readonly active: ActiveFrame;
  /** Compiled relation catalog: one descriptor per declared relation. */
  readonly catalog: CompiledCatalog;
  /** Schema id of the catalog records (selector mask target). */
  readonly intentSchemaId: number;
  /** Per-slot selection head outputs (GPU readbacks or CPU oracle). */
  readonly intent: SelectionSlotResult;
  /** One selection per role; a role with no entry is left unbound. */
  readonly roleSelections: RoleSelections;
  readonly tick: number;
  readonly revision?: number;
  /** Cap on emitted proposals (defaults to the schema transport capacity). */
  readonly maxProposals?: number;
}

export interface IntentSetEmissionResult {
  readonly intentSet: v1_0_0.IntentSet;
  /** Number of emitted proposals (== intentSet.count). */
  readonly emitted: number;
  /**
   * Why proposals did not survive.
   *
   * An empty intent set has several quite different causes and they are
   * indistinguishable from outside: a creature with nothing to propose looks
   * exactly like one whose agent could not be resolved, or whose catalog is
   * empty. The first is a policy that has not committed; the others are
   * malformed input. Counting them apart is what turns "it does nothing" into
   * a question with an answer.
   */
  readonly droppedNoAgent: number;
  readonly droppedNoPatient: number;
}

export function emitIntentSet(input: IntentSetEmissionInput): IntentSetEmissionResult {
  const {
    frame,
    active,
    catalog,
    intentSchemaId,
    intent,
    roleSelections,
    tick,
    revision = 0,
  } = input;
  const maxProposals = input.maxProposals ?? BRAIN_LIMITS.maxIntentProposals;

  const r = active.bankRecords.length;
  const q = active.queryRecords.length;
  const recordWidth = BRAIN_LIMITS.recordWidth;
  const maxRefs = BRAIN_LIMITS.maxReferencesPerRecord;

  if (intent.index.length < q) {
    throw new Error(
      `emitIntentSet: intent indices must cover ${q} query rows (got ${intent.index.length})`,
    );
  }
  if (intent.p.length < q * r) {
    throw new Error(`emitIntentSet: intent P must be [${q}, ${r}] (got ${intent.p.length})`);
  }
  for (const [role, selection] of Object.entries(roleSelections)) {
    if (!selection) continue;
    if (selection.index.length < q || selection.p.length < q * r) {
      throw new Error(
        `emitIntentSet: role '${role}' selection must cover [${q}, ${r}] ` +
          `(index ${selection.index.length}, P ${selection.p.length})`,
      );
    }
  }

  const proposals: v1_0_0.IntentProposal[] = [];
  for (let slot = 0; slot < maxProposals; slot++) {
    proposals.push(emptyProposal(slot));
  }

  let emitted = 0;
  let droppedNoAgent = 0;
  let droppedNoPatient = 0;

  for (let i = 0; i < Math.min(q, maxProposals); i++) {
    const bankIdx = intent.index[i]!;
    if (bankIdx >= r) continue;
    const slot = active.bankRecords[bankIdx]!;

    // The selector mask only leaves catalog records open; a record that is not
    // one (e.g. an all-masked row resolving to index 0) is not a proposal.
    if (frame.schemaIds[slot] !== intentSchemaId) continue;
    const actionToken = frame.tokenIds[slot * recordWidth]!;
    const descriptor = catalog.descriptors.find((candidate) => candidate.actionToken === actionToken);
    if (!descriptor) continue;

    const intentProb = intent.p[i * r + bankIdx]!;
    const intentStats = rowStatistics(intent, i, r);
    // Peakedness of the intent distribution: 1 - normalized entropy, so a
    // single unambiguous candidate scores 1 and a uniform row scores 0.
    const peakedness =
      intentStats.candidateCount <= 1
        ? 1
        : Math.max(0, 1 - intentStats.entropy / Math.log(intentStats.candidateCount));

    const proposal: v1_0_0.IntentProposal = {
      ...emptyProposal(i),
      lifecycle: "start",
      intentId: descriptor.intentId,
      activation: intentProb,
      commitment: 0, // overwritten below once role support is known
      // No magnitude head exists yet, so this stays 0 rather than borrowing
      // `commitment`'s value. Emitting a fabricated magnitude would train the
      // motor layer on a number that means "how sure", not "how hard".
      intensity: 0,
      confidence: intentProb,
    };

    // Every declared role goes through the same path. Nothing is resolved
    // structurally any more: a role with exactly one legal filler simply has a
    // one-candidate distribution, which is a fact about the world rather than
    // something the emitter should special-case.
    const bound = new Set<RelationRoleName>();
    let roleSupport = 1;

    for (const role of RELATION_ROLES) {
      const roleDesc = descriptor.roles[RELATION_ROLE_INDEX[role]];
      if (!roleDesc || (roleDesc.flags & RELATION_ROLE_FLAGS.present) === 0) continue;
      const selection = roleSelections[role];
      if (!selection) continue;

      const chosenBank = selection.index[i]!;
      const chosenSlot = chosenBank < r ? active.bankRecords[chosenBank]! : undefined;
      const filter = roleFilterFor(catalog, descriptor.intentId, role, roleDesc);
      const admitted =
        chosenSlot !== undefined &&
        roleAdmitsRecord(frame, chosenSlot, filter)
          ? chosenSlot
          : undefined;

      const stats = rowStatistics(selection, i, r);
      const probability = chosenBank < r ? selection.p[i * r + chosenBank]! : 0;
      const resolved = resolveConcept(frame, admitted, roleDesc.valueKind, maxRefs, {
        // Starts unresolved. An optimistic default made a role that resolved to
        // nothing report itself as resolved, and the executability check then
        // let it through with no handle.
        status: "masked",
        probability,
        candidateCount: stats.candidateCount,
        entropy: stats.entropy,
      });

      proposal.roles[RELATION_ROLE_INDEX[role]] = resolved;
      if (resolved.selector.status === "selected") {
        bound.add(role);
        roleSupport *= probability;
        if (admitted !== undefined && frame.bandIds[admitted] === MEMORY_BAND_INDEX) {
          proposal.flags |= INTENT_PROPOSAL_FLAGS.volitive;
        }
      }
    }

    // A relation with no agent is not executable: there is nobody to act.
    if (!bound.has("agent")) {
      droppedNoAgent++;
      continue;
    }

    const reflexive = (descriptor.flags & ACTION_INTENT_FLAGS.canonicallyReflexive) !== 0;
    if (reflexive) {
      // The patient mirrors the agent. LAUGH() is "I rejoice myself" — the
      // reflexive reading, not a missing argument. The flag is what lets a
      // later reader tell this pair from one the head genuinely chose.
      const agent = proposal.roles[RELATION_ROLE_INDEX.agent]!;
      proposal.roles[RELATION_ROLE_INDEX.patient] = {
        concept: { ...agent.concept },
        selector: { ...agent.selector },
      };
      proposal.flags |= INTENT_PROPOSAL_FLAGS.reflexive;
    } else if (!bound.has("patient")) {
      droppedNoPatient++;
      continue;
    }

    // Commitment is the product of the heads: a decisive, well-supported set of
    // selections is committed; a flat or poorly supported one is not.
    proposal.commitment = intentProb * peakedness * roleSupport;

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
  return { intentSet, emitted, droppedNoAgent, droppedNoPatient };
}
