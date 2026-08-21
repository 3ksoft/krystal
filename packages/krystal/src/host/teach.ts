/**
 * Being shown, as opposed to finding out.
 *
 * A separate mechanism from `learn`, deliberately. The two answer different
 * questions and must not be confused for one:
 *
 *   teach   what CAN be said here — this frame admits this answer
 *   learn   what is WORTH saying  — that answer turned out well or badly
 *
 * A creature that has never been shown anything has to discover, by sampling,
 * that a verb takes an actor and a thing, that the thing is one of the ones in
 * front of it, and only THEN whether eating stones is a good idea. The first
 * two are facts about the grammar of the world, not preferences, and they are
 * cheap to demonstrate and expensive to stumble into: the measured policy sits
 * at full entropy for hundreds of batches before anything moves.
 *
 * So this applies the pointer loss and nothing else. No baseline, no advantage,
 * no critic, no reward — a demonstration is not "this went well", it is "this
 * is what one does". Folding it into the reward path would make a shown act
 * compete with a felt one on the same scale, and the creature would learn that
 * being taught feels good.
 */
import { brainBackwardOracle } from "../forward/backward.ts";
import { embeddingTableBases, type BrainForwardConfig, type BrainForwardWeights } from "../forward/model.ts";
import { packHostFrame, type HostRecord } from "./frame.ts";
import { selectionMask } from "./learn.ts";

/** One frame, and the answer it is being shown. */
export interface HostDemonstration {
  readonly records: readonly HostRecord[];
  /** The record slot each question should have chosen, by question index.
   *  `undefined` for a question this demonstration says nothing about. */
  readonly gold: readonly (number | undefined)[];
  /** The grammar in force, so a demonstration is shown under the same rules a
   *  choice would have been made under. */
  readonly allows?: (query: number, record: number) => boolean;
}

export interface TeachOptions {
  readonly learningRate?: number;
  /** How fast the semantic tables follow, relative to the selector's step. */
  readonly tokenRate?: number;
  /**
   * Whether the token and field-role tables move. ON by default here, unlike
   * in `learn`, and the difference is not a preference — it is the whole
   * mechanism.
   *
   * Measured on two records and one question: frozen, 200 showings move the
   * policy from `0.498` to `0.510`. Unfrozen at the same rate it reaches
   * `0.996` in fifty, and at `lr 0.2` in about twenty. A selector is one shared
   * projection; it cannot pull apart two records whose representations are
   * fixed random vectors. Teaching with the tables frozen is not slow teaching,
   * it is a no-op that looks like slow teaching.
   */
  readonly unfreezeTokens?: boolean;
}

export interface TeachReport {
  readonly framesSeen: number;
  /** Questions that carried an answer to be shown. */
  readonly shown: number;
  /**
   * How much probability the policy already put on what it was being shown,
   * averaged — before this pass changed anything.
   *
   * The whole measurement of teaching: it starts near `1/candidates` and should
   * climb. Loss would say the same thing less legibly.
   */
  readonly meanAgreement: number;
  /** Largest absolute value in anything this pass wrote. Reported rather than
   *  enforced: a demonstration's gradient dies as agreement approaches one, so
   *  there is no runaway to guard against — but a host that sees this climb has
   *  learned something is wrong before its creature has. */
  readonly maxParameterAbs: number;
}

const NO_TARGET = 0xffff_ffff;

export function teachFromDemonstration(
  demonstrations: readonly HostDemonstration[],
  weights: BrainForwardWeights,
  config: BrainForwardConfig,
  options: TeachOptions = {},
): TeachReport {
  const lr = options.learningRate ?? 0.2;
  const tokenRate = options.tokenRate ?? 0.5;
  const unfreezeTokens = options.unfreezeTokens ?? true;
  const h = config.hiddenSize;
  const bases = embeddingTableBases(config);

  let seen = 0;
  let shown = 0;
  let agreementTotal = 0;
  let touched = 0;

  for (const demonstration of demonstrations) {
    if (!demonstration.records.length) continue;
    const frame = packHostFrame(demonstration.records);
    const { active } = frame;
    const queries = active.queryRecords.length;
    const bank = active.bankRecords.length;
    if (!queries || !bank) continue;

    // Slots back to bank rows, exactly as the update does: the host numbers
    // records, the selector numbers the ones it may choose from.
    const targets = new Array<number>(queries).fill(NO_TARGET);
    let answered = false;
    for (let q = 0; q < queries; q++) {
      const slot = demonstration.gold[q];
      if (slot === undefined || slot < 0) continue;
      const row = active.bankRecords.indexOf(slot);
      if (row < 0) continue;
      targets[q] = row;
      answered = true;
    }
    if (!answered) continue;

    const mask = selectionMask(active, demonstration.allows);
    const result = brainBackwardOracle({
      frame: frame.gpu,
      active,
      weights,
      config,
      recordMask: frame.recordMask,
      mixerMask: new Float32Array(queries * bank),
      intentMask: mask,
      argMask: new Float32Array(queries * bank),
      intentTargets: targets,
      context: "available",
      // No target, so the value head contributes nothing at all. Being shown
      // something is not an outcome and there is nothing here to predict.
    });

    for (let q = 0; q < queries; q++) {
      if (targets[q] === NO_TARGET) continue;
      agreementTotal += result.policy[q * bank + targets[q]!]!;
      shown++;
    }

    // Straight down the gradient. No advantage to scale by: a demonstration is
    // not more or less true depending on how the creature felt about it.
    for (let i = 0; i < weights.selector.wq.length; i++) {
      weights.selector.wq[i] = weights.selector.wq[i]! - lr * result.dSelectorWq[i]!;
      touched = Math.max(touched, Math.abs(weights.selector.wq[i]!));
    }
    for (let i = 0; i < weights.selector.wk.length; i++) {
      weights.selector.wk[i] = weights.selector.wk[i]! - lr * result.dSelectorWk[i]!;
      touched = Math.max(touched, Math.abs(weights.selector.wk[i]!));
    }

    if (unfreezeTokens) {
      const step = lr * tokenRate;
      const table = weights.embeddings;
      for (let t = 0; t < active.activeTokens.length; t++) {
        const frameTok = active.activeTokens[t]!;
        const token = bases.token + config.tokenRows[frame.gpu.tokenIds[frameTok]!]! * h;
        const role = bases.field + config.tokenRows[frame.gpu.fieldRoles[frameTok]!]! * h;
        for (let d = 0; d < h; d++) {
          const gradient = step * result.dFieldStates[t * h + d]!;
          table[token + d] = table[token + d]! - gradient;
          table[role + d] = table[role + d]! - gradient;
          touched = Math.max(touched, Math.abs(table[token + d]!), Math.abs(table[role + d]!));
        }
      }
    }
    seen++;
  }

  return {
    framesSeen: seen,
    shown,
    meanAgreement: shown ? agreementTotal / shown : 0,
    maxParameterAbs: touched,
  };
}
