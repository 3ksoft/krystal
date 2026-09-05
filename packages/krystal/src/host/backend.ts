/**
 * Where the encoding happens — and, when a backend can, where the gradients do.
 *
 * Reading a frame is the expensive half of thinking and asking questions of it
 * is the cheap one — measured in the simulation at 20 s against 0.17 s over
 * sixty ticks. So this is the seam: a backend turns a packed frame into the
 * three small matrices every question reads, and nothing else moves.
 *
 * Learning has the same shape. Differentiating a frame was 68% of a learning
 * tick in scalar JS; what the update does with the gradients — standardise an
 * advantage across a batch, move only the unfrozen parts, roll the whole thing
 * back if it ran away — is a few thousand multiplies. So a backend may also
 * differentiate, and it hands back exactly the arrays the update consumes; the
 * update itself stays here, on the host's own weights, whoever computed them.
 *
 * It is INJECTED rather than reached for. A brain that could construct its own
 * device backend would have to import the WebGPU package, which imports this
 * one — and every browser bundle that only ever meant to think on the CPU
 * would carry a shader artifact it never runs. Passing one in keeps the
 * dependency pointing the way it already does.
 */
import type { BrainForwardConfig, BrainForwardWeights } from "../forward/model.ts";
import type { HostFrame } from "./frame.ts";

/**
 * A frame, encoded: the mixed query output and the pooled record bank.
 *
 * Everything downstream — every question, its distribution, its choice — is a
 * function of these three and the host's mask. Which is why the seam is here
 * and not further in: this is the last point where the two backends still have
 * to agree, and it is three matrices wide.
 */
export interface EncodedFrame {
  /** [Q, H] mixed query output, after the mixer blocks. */
  readonly queryOutput: Float32Array;
  /** [R, H] pooled bank keys and values. */
  readonly bankKeys: Float32Array;
  readonly bankValues: Float32Array;
}

/** No target for this question: it was asked, and nothing is pushed toward. */
export const NO_TARGET = 0xffff_ffff;

/**
 * Set on a target's top bit: this row is what the question is pushed AWAY
 * from, not toward. "Not this" — the same pointer, the other sign.
 *
 * Pushed away by the unlikelihood loss, `-log(1 - p)`: its gradient is
 * `p/(1-p) · (onehot - p)`, so a choice the policy barely makes is barely
 * touched and one it is sure of is pushed as hard as a demonstration pulls.
 * Where the mass goes is up to the rest of the distribution — a "no" names
 * nothing to do instead, and this does not pretend it did.
 *
 * Kept in the target word rather than beside it so that every reader of a
 * target — the CPU oracle, the device shader, the argument slot's invalid-row
 * check — sees one array with one meaning per entry.
 */
export const AWAY = 0x8000_0000;
/** A target that pushes away from `row`. */
export const awayFrom = (row: number): number => (row | AWAY) >>> 0;
/** The bank row a target names, whichever way it pushes. */
export const targetRow = (target: number): number => (target & 0x7fff_ffff) >>> 0;
/** Whether a target pushes away rather than toward. NO_TARGET pushes nowhere. */
export const isAway = (target: number): boolean => target !== NO_TARGET && (target & AWAY) !== 0;

/**
 * One frame, to be differentiated.
 *
 * The same three things every update is made of, whether it reinforces a
 * choice or shows one: the grammar the question was asked under, what each
 * question is pushed toward, and — for a turn that had consequences — what
 * they were.
 */
export interface BackwardRequest {
  /**
   * The host's grammar as the selector reads it: [Q, R], NEG_INF on every
   * record a question may not choose. Built by `selectionMask`, so it is the
   * same mask the choice was made under.
   */
  readonly selection: Float32Array;
  /**
   * The bank row each question is pushed toward, [Q]; NO_TARGET where nothing
   * is; `awayFrom(row)` where it is pushed away from that row instead. Absent
   * when no question in the frame was answered at all, and then the pointer
   * loss does not run — the frame still teaches the critic.
   */
  readonly targets?: readonly number[];
  /**
   * Observed change in valence: the value head's target. Absent when there is
   * nothing to difference against, and then the critic contributes no gradient
   * rather than being trained toward zero.
   */
  readonly valenceTarget?: number;
}

/**
 * What comes back: the gradients of the parts the host may move, and the
 * forward's own reading of the frame, which the report is made of.
 *
 * Nothing about the encoder or the mixer blocks. Their gradients exist on a
 * device — the composed backward computes every one of them — but the host's
 * update keeps those blocks frozen on purpose, and a gradient nobody applies is
 * a readback nobody needed.
 */
export interface BackwardResult {
  /** [Q, R] the policy under the selection mask — what the update differentiates. */
  readonly policy: Float32Array;
  /** [Q] what the critic predicted, the baseline an advantage is measured against. */
  readonly valuePrediction: Float32Array;
  /** Squared-error loss of the value head; 0 when the frame carried no target. */
  readonly valueLoss: number;
  /** [H, H] */
  readonly dSelectorWq: Float32Array;
  /** [H, H] */
  readonly dSelectorWk: Float32Array;
  /** [2, H] */
  readonly dPool: Float32Array;
  /** [T, H] in active-token order: the gradient at each token's summed
   *  embedding, which the host scatters into the rows the frame read. */
  readonly dFieldStates: Float32Array;
  /** [3H] */
  readonly dValueWv: Float32Array;
}

/**
 * Which of the host's arrays an update wrote.
 *
 * Every device backend holds a COPY of the weights, and after `learn` or
 * `teach` the two disagree until it is told. Telling it everything cost 7 MB a
 * batch for a change to two projections and a few embedding rows — so an update
 * says what it touched, and the backend takes up that and nothing else. Absent
 * altogether means all of it, which is what a restored checkpoint needs.
 */
export interface WeightChanges {
  readonly selector?: boolean;
  readonly pool?: boolean;
  readonly valueHead?: boolean;
  readonly decisionHead?: boolean;
  /** The encoder and mixer blocks, all of them. */
  readonly blocks?: boolean;
  /** Element offsets into the embeddings page of the rows that moved; each
   *  runs `hiddenSize` elements from there. */
  readonly embeddingRows?: readonly number[];
}

export interface BrainBackend {
  /** Encode one packed frame. Asynchronous because a device answer has to come
   *  back before any question can be asked of it. */
  encode(frame: HostFrame): Promise<EncodedFrame>;
  /**
   * Differentiate one frame, against the weights this backend was last synced
   * with — which the session keeps equal to its own, so the gradient is of the
   * brain the creature is actually thinking with.
   *
   * Optional. A backend that only encodes leaves this to the CPU oracle, and
   * the session still syncs it afterwards; the update is the same either way.
   */
  backward?(frame: HostFrame, request: BackwardRequest): Promise<BackwardResult>;
  /**
   * Take up weights the host has just changed.
   *
   * A backend that holds a COPY — every device backend does — is otherwise
   * still thinking with the brain the creature had before it was taught, and
   * both sides stay individually consistent while they disagree. `changes`
   * says which arrays moved; absent means all of them.
   */
  sync(weights: BrainForwardWeights, changes?: WeightChanges): void;
  destroy?(): void;
}

/**
 * How a session obtains its backend: it hands over the geometry and the weight
 * arrays it owns, and gets something that can encode with them. The session
 * keeps ownership of the weights — a backend that allocated its own would make
 * "which of these two is the creature" a question with no answer.
 */
export type BrainBackendFactory = (
  config: BrainForwardConfig,
  weights: BrainForwardWeights,
) => BrainBackend;
