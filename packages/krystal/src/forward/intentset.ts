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
 * `lifecycle: 'start'`, resolved `intentId`, and a subject/object pair whose
 * exact handles are read from the selected records' runtime-ref sidecars.
 * Confidence and entropy are emitted as debug/diagnostic values, and
 * `commitment` is computed from the selection heads (intent top-1 probability
 * × distribution peakedness × object support) rather than a constant.
 * intentRef / purposeGoal / controllerHint / topic are left invalid: the
 * runtime assigns an exact intentRef when a proposal is accepted.
 *
 * Every proposal is a binary relation, so both sides are always populated:
 *
 *   subject  Resolved structurally, not selected. In this catalog the subject
 *            is always Self, and letting a head "choose" a value with exactly
 *            one legal filler would teach it nothing.
 *   object   Selected by the argument head — except for a canonically
 *            reflexive (authored-unary) intent, where the subject is mirrored
 *            in and INTENT_PROPOSAL_FLAGS.objectFromSubject records that the
 *            pair was filled rather than chosen.
 */
import {
  ACTION_INTENT_FLAGS,
  BRAIN_LIMITS,
  INTENT_PROPOSAL_FLAGS,
  INVALID_U32,
} from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import { unpackRuntimeHandle } from "../frame/packer.ts";
import { roleAdmitsRecord, roleFilterFor } from "./masks.ts";
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
    subject: emptySelectedConcept(),
    object: emptySelectedConcept(),
  };
}

/**
 * Resolve one side of a relation from a chosen record slot.
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
 *                reject every subject in the fixture catalog.
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
  /**
   * Why proposals did not survive.
   *
   * An empty intent set has several quite different causes and they are
   * indistinguishable from outside: a creature with nothing to propose looks
   * exactly like one whose actor record is missing, or whose catalog is empty.
   * The first is a policy that has not committed; the others are malformed
   * input. Counting them apart is what turns "it does nothing" into a question
   * with an answer.
   */
  readonly droppedNoSubject: number;
  readonly droppedNoObject: number;
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
  let droppedNoSubject = 0;
  let droppedNoObject = 0;
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
    // Peakedness of the intent distribution: 1 - normalized entropy, so a
    // single unambiguous candidate scores 1 and a uniform row scores 0.
    const peakedness = candidateCount <= 1 ? 1 : Math.max(0, 1 - entropy / Math.log(candidateCount));

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
      commitment: 0, // overwritten below once object support is known
      // No magnitude head exists yet, so this stays 0 rather than borrowing
      // `commitment`'s value. Emitting a fabricated magnitude would train the
      // motor layer on a number that means "how sure", not "how hard".
      intensity: 0,
      persistence: 0,
      confidence: intentProb,
      subject: emptySelectedConcept(),
      object: emptySelectedConcept(),
    };

    // The subject is resolved structurally: the first active bank record whose
    // schema the subject role accepts and whose band the role admits. It is not
    // put through a selection head because in this catalog exactly one record
    // (Self) is ever legal.
    const subjectFilter = roleFilterFor(
      catalog,
      descriptor.intentId,
      "subject",
      descriptor.subjectRole,
    );
    let subjectSlot: number | undefined;
    for (const slot of active.bankRecords) {
      if (!roleAdmitsRecord(frame, slot, subjectFilter)) continue;
      subjectSlot = slot;
      break;
    }
    const subject = resolveConcept(
      frame,
      subjectSlot,
      descriptor.subjectRole.valueKind,
      maxRefs,
      // Starts unresolved. Passing "selected" in as an optimistic default made
      // a subject that resolved to nothing report itself as resolved, and the
      // executability check downstream then let it through with no handle.
      { status: "masked", probability: 1, candidateCount: 1, entropy: 0 },
    );
    if (subject.selector.status !== "selected") {
      droppedNoSubject++;
      continue;
    }
    proposal.subject = subject;

    const unary = (descriptor.flags & ACTION_INTENT_FLAGS.canonicallyReflexive) !== 0;
    if (unary) {
      // Object mirrors subject. LAUGH() is "I rejoice myself" — the reflexive
      // reading, not a missing argument. The flag is what lets a later reader
      // tell this pair from one the head genuinely chose.
      proposal.object = {
        concept: { ...subject.concept },
        selector: { ...subject.selector },
      };
      proposal.flags |= INTENT_PROPOSAL_FLAGS.objectFromSubject;
      // Object support is not a free parameter here: nothing was selected, so
      // commitment rests on the intent head alone.
      proposal.commitment = intentProb * peakedness;
      proposals[i] = proposal;
      emitted++;
      continue;
    }

    // v0 shares one argument selector across slots (answer 26). Acceptance is
    // the same predicate the mask used, over the record's tokens: a role that
    // names a category (`category:Edible`) admits every record carrying it,
    // so EAT covers Apple/Berry/Bread — and a berry never seen before —
    // without a hardcoded identity list anywhere.
    {
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

      const objectFilter = roleFilterFor(
        catalog,
        descriptor.intentId,
        "object",
        descriptor.objectRole,
      );
      const objectSlot =
        argSlot !== undefined && roleAdmitsRecord(frame, argSlot, objectFilter)
          ? argSlot
          : undefined;
      const probability = argBankIdx < r ? argument.p[argRowStart + argBankIdx]! : 0;
      proposal.object = resolveConcept(
        frame,
        objectSlot,
        descriptor.objectRole.valueKind,
        maxRefs,
        { status: "masked", probability, candidateCount: argCandidateCount, entropy: argEntropy },
      );

      // A relation whose object did not resolve is not executable: no
      // fabricated handle and no emitted proposal. The slot stays empty,
      // exactly like a masked intent row, so `count` reflects executable
      // proposals only.
      if (proposal.object.selector.status !== "selected") {
        droppedNoObject++;
        continue;
      }

      // Commitment is the product of the two heads: a decisive, well-supported
      // selection is committed; a flat or poorly supported one is not.
      proposal.commitment = intentProb * peakedness * probability;
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
  return { intentSet, emitted, droppedNoSubject, droppedNoObject };
}
