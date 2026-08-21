/**
 * One brain, on the CPU.
 *
 * The whole surface a simulation needs to think with: give it the rows its
 * symbols occupy, hand it records, get back what it chose for each question it
 * was asked. It knows nothing about the world those records describe — not what
 * a band means, not which record is an apple, not what the chosen one will be
 * used for. That knowledge stays where the world is.
 *
 * Weights are created deterministically from a seed when none are supplied, so
 * an untrained brain is a real brain with opinions worth nothing rather than a
 * special case to code around: the loop can be closed end to end before there
 * is anything to say about it.
 */
import {
  BRAIN_FORWARD_CONFIG,
  createBrainForwardWeights,
  validateBrainForwardWeights,
  type BrainForwardConfig,
  type BrainForwardWeights,
} from "../forward/model.ts";
import { brainForwardOracle, selectorOracle } from "../forward/oracle.ts";
import { decodeCheckpoint, encodeCheckpoint, type CheckpointRefusal } from "./checkpoint.ts";
import { teachFromDemonstration, type HostDemonstration, type TeachOptions, type TeachReport } from "./teach.ts";
import { packHostFrame, type HostFrame, type HostRecord } from "./frame.ts";
import {
  learnFromExperience,
  selectionMask,
  type HostExperience,
  type LearnOptions,
  type LearnReport,
} from "./learn.ts";

export interface HostSessionOptions {
  /**
   * Token id → embedding row, for the vocabulary this brain was built against.
   *
   * A brain and a manifest are only meaningful together: the row is where the
   * meaning was learned, so the same weights under a different mapping denote
   * something else entirely, with nothing to signal it.
   */
  readonly tokenRows: Uint32Array;
  readonly seed?: number;
  readonly weights?: BrainForwardWeights;
  readonly config?: BrainForwardConfig;
}

export interface HostSelection {
  /** Index of the question, in the order the query records were sent. */
  readonly query: number;
  /** The record slot it chose, as the host numbered it. */
  readonly record: number;
  readonly probability: number;
  /** Over the whole bank, in slot order. Kept because a selection is only as
   *  trustworthy as the distribution it came from — an argmax at 0.02 and one
   *  at 0.9 are different answers to the same question. */
  readonly distribution: Float32Array;
}

export interface ThinkResult {
  readonly selections: readonly HostSelection[];
  readonly frame: HostFrame;
}

/**
 * A frame that has been read, waiting to be asked about.
 *
 * The encoder and the mixer depend only on the RECORDS; the mask depends only
 * on the question. So a creature deciding "eat what?" after deciding "eat" is
 * looking at the same world through a different grammar, and re-encoding it
 * would compute the identical numbers a second time — most of the cost of
 * thinking, spent to arrive where it already was.
 */
export interface Deliberation {
  readonly frame: HostFrame;
  /** Ask this frame a question. Cheap: only the selector runs. */
  choose(options?: ThinkOptions): ThinkResult;
}

export interface ThinkOptions {
  /**
   * Which records a given question may choose. Called per (question, record);
   * `false` removes the record from that row's distribution entirely.
   *
   * This is where the host's grammar lands, and it stays the host's: Krystal
   * applies the answer and never asks why a record was refused.
   */
  readonly allows?: (query: number, record: number) => boolean;
  /** Draw from the distribution instead of taking its mode. Called once per
   *  question; must return a uniform in [0, 1). */
  readonly sample?: (query: number) => number;
}

export class BrainSession {
  readonly config: BrainForwardConfig;
  readonly weights: BrainForwardWeights;

  constructor(options: HostSessionOptions) {
    this.config = { ...(options.config ?? BRAIN_FORWARD_CONFIG), tokenRows: options.tokenRows };
    this.weights = options.weights ?? createBrainForwardWeights(this.config, options.seed ?? 1337);
    validateBrainForwardWeights(this.config, this.weights);
  }

  /**
   * Read a frame once, and keep it readable.
   *
   * The expensive half of thinking happens here and happens once; every
   * question asked of the result costs a selector pass and nothing else.
   */
  consider(records: readonly HostRecord[]): Deliberation {
    const frame = packHostFrame(records);
    const { active } = frame;
    const queries = active.queryRecords.length;
    const bank = active.bankRecords.length;
    if (!queries || !bank) return { frame, choose: () => ({ selections: [], frame }) };

    // The mixer is unconstrained here: what a question may attend to while it
    // thinks is not the same as what it may CHOOSE, and only the second is the
    // host's grammar.
    const forward = brainForwardOracle(
      frame.gpu,
      active,
      this.weights,
      this.config,
      frame.recordMask,
      new Float32Array(queries * bank),
    );

    const choose = (options: ThinkOptions = {}): ThinkResult => {
      // Built by the same function the update uses, because a mask that
      // differed between choosing and learning would compute the gradient of a
      // distribution the creature never sampled from.
      const selection = selectorOracle(
        forward.queryOutput,
        forward.bankKeys,
        forward.bankValues,
        selectionMask(active, options.allows),
        this.weights.selector,
        this.config.hiddenSize,
        options.sample,
      );
      const selections: HostSelection[] = [];
      for (let q = 0; q < queries; q++) {
        const chosen = selection.index[q]!;
        const distribution = selection.p.slice(q * bank, (q + 1) * bank);
        selections.push({
          query: q,
          record: active.bankRecords[chosen]!,
          probability: distribution[chosen] ?? 0,
          distribution,
        });
      }
      return { selections, frame };
    };
    return { frame, choose };
  }

  /** One pass: records in, one choice per question out. */
  think(records: readonly HostRecord[], options: ThinkOptions = {}): ThinkResult {
    return this.consider(records).choose(options);
  }

  /**
   * Live with what happened: one pass over remembered turns.
   *
   * Mutates this session's weights in place — a brain is not a value, and
   * copying one to hand back a new brain would make every update a decision
   * about which of two brains the creature is.
   */
  learn(experiences: readonly HostExperience[], options: LearnOptions = {}): LearnReport {
    return learnFromExperience(experiences, this.weights, this.config, options);
  }

  /**
   * Be shown what can be said here.
   *
   * A separate mechanism from `learn` on purpose: this teaches what the frame
   * ADMITS, not what turned out well. No reward, no baseline, no critic.
   */
  teach(demonstrations: readonly HostDemonstration[], options: TeachOptions = {}): TeachReport {
    return teachFromDemonstration(demonstrations, this.weights, this.config, options);
  }

  /**
   * This brain, written down: its weights, the geometry they were shaped for,
   * and the token→row mapping they learned their meanings under.
   *
   * All three, because weights alone can be loaded into the wrong brain without
   * failing — they would simply denote something else for the rest of that
   * creature's life.
   */
  snapshot(): Uint8Array {
    return encodeCheckpoint(this.weights, this.config, this.config.tokenRows);
  }

  /**
   * Take up a written-down brain, or say why it does not fit.
   *
   * In place, so anything already holding these weights is training the brain
   * the creature is now thinking with. Null means it was taken up.
   */
  restore(bytes: Uint8Array): CheckpointRefusal | null {
    return decodeCheckpoint(bytes, this.weights, this.config, this.config.tokenRows);
  }
}

export { createBrainForwardWeights, type BrainForwardWeights, type BrainForwardConfig };
