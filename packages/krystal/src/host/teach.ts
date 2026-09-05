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
import { embeddingTableBases, type BrainForwardConfig, type BrainForwardWeights } from "../forward/model.ts";
import { awayFrom, isAway, NO_TARGET, targetRow } from "./backend.ts";
import { packHostFrame, type HostRecord } from "./frame.ts";
import { cpuGradients, type GradientSource } from "./gradients.ts";
import { selectionMask } from "./learn.ts";

/** One frame, and the answer it is being shown. */
export interface HostDemonstration {
  readonly records: readonly HostRecord[];
  /** The record slot each question should have chosen, by question index.
   *  `undefined` for a question this demonstration says nothing about. */
  readonly gold: readonly (number | undefined)[];
  /**
   * The record slot each question is being shown is NOT the answer here, by
   * question index — "not this", pushed away from the way `gold` is pushed
   * toward. A question with both keeps its gold: what to do says more than
   * what not to. Its agreement is how much of the policy was already off it.
   */
  readonly forbidden?: readonly (number | undefined)[];
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
  /**
   * The largest step any one weight may take in a showing. A gradient asking
   * for more is scaled down whole, direction kept.
   *
   * Without it teaching runs away: the tables and the selector are pushed
   * together, each one's gradient grows with the other's size, and past a
   * point one showing multiplies the brain. Measured on a five-lesson
   * curriculum, second round: the largest weight went 0.69 → 5.6 → 441 →
   * 23 million in three showings, then every number was NaN.
   */
  readonly maxUpdateAbs?: number;
  /**
   * How much of every touched weight a showing forgets, as a fraction. A
   * cross-entropy fixed point lies at infinity — the logits are always pushed
   * further apart — and this is what makes the brain settle somewhere finite
   * instead: the pull toward zero grows with the weight, the push does not.
   */
  readonly decay?: number;
  /**
   * Reject a showing that would leave any touched weight past this. A last
   * line, not the first: with the step capped and decay on, an update that
   * still reaches it is one the arithmetic behind it got wrong.
   */
  readonly maxParameterAbs?: number;
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
  /**
   * Showings whose update was thrown away because it was not finite: a source
   * that answered with NaN or infinity in it — a lost device, a frame nothing
   * could be read from. The brain is exactly as it was before each of them.
   * One such showing would otherwise poison every weight it touched, and a
   * brain with NaN in it answers every question with noise from then on.
   */
  readonly rejected: number;
  /** Showings whose step was scaled down to `maxUpdateAbs`. Many of these in a
   *  row is teaching at the cap: slower than asked, and stable. */
  readonly clipped: number;
}

/**
 * Asynchronous for the same reason `learn` is: the gradient of a showing may
 * come from a device. Each demonstration is applied before the next one is
 * differentiated, and the source is told in between, so the second showing
 * meets the brain the first one left — wherever that brain's copy lives.
 */
export async function teachFromDemonstration(
  demonstrations: readonly HostDemonstration[],
  weights: BrainForwardWeights,
  config: BrainForwardConfig,
  options: TeachOptions = {},
  source: GradientSource = cpuGradients(weights, config),
): Promise<TeachReport> {
  const lr = options.learningRate ?? 0.2;
  const tokenRate = options.tokenRate ?? 0.5;
  const unfreezeTokens = options.unfreezeTokens ?? true;
  const maxUpdateAbs = options.maxUpdateAbs ?? 0.05;
  const keep = 1 - (options.decay ?? 1e-3);
  const maxParameterAbs = options.maxParameterAbs ?? 1;
  const h = config.hiddenSize;
  const bases = embeddingTableBases(config);

  let seen = 0;
  let shown = 0;
  let agreementTotal = 0;
  let touched = 0;
  let rejected = 0;
  let clipped = 0;

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
      const not = demonstration.forbidden?.[q];
      const toward = slot !== undefined && slot >= 0;
      const chosen = toward ? slot : not;
      if (chosen === undefined || chosen < 0) continue;
      const row = active.bankRecords.indexOf(chosen);
      if (row < 0) continue;
      targets[q] = toward ? row : awayFrom(row);
      answered = true;
    }
    if (!answered) continue;

    const mask = selectionMask(active, demonstration.allows);
    // No valence target, so the value head contributes nothing at all. Being
    // shown something is not an outcome and there is nothing here to predict.
    const result = await source.backward(frame, { selection: mask, targets });

    for (let q = 0; q < queries; q++) {
      const target = targets[q]!;
      if (target === NO_TARGET) continue;
      // Toward: how much was already on it. Away: how much was already off it.
      const on = result.policy[q * bank + targetRow(target)]!;
      // A reading that is not a number is not a reading; the showing below
      // is thrown away for the same reason, and the report says so.
      if (!Number.isFinite(on)) continue;
      agreementTotal += isAway(target) ? 1 - on : on;
      shown++;
    }

    // What this showing is about to move, kept so that a showing whose
    // arithmetic came back broken can be undone to the last weight.
    const rows = new Set<number>();
    if (unfreezeTokens)
      for (let t = 0; t < active.activeTokens.length; t++) {
        const frameTok = active.activeTokens[t]!;
        rows.add(bases.token + config.tokenRows[frame.gpu.tokenIds[frameTok]!]! * h);
        rows.add(bases.field + config.tokenRows[frame.gpu.fieldRoles[frameTok]!]! * h);
      }
    const before = {
      wq: Float32Array.from(weights.selector.wq),
      wk: Float32Array.from(weights.selector.wk),
      rows: [...rows].map((start) => [start, weights.embeddings.slice(start, start + h)] as const),
    };
    let written = 0;

    // The step, capped: the largest move any weight would make, and the
    // factor that brings it under the cap. One factor for everything, so the
    // update keeps its direction and only its length changes.
    let asked = 0;
    for (let i = 0; i < result.dSelectorWq.length; i++) asked = Math.max(asked, Math.abs(lr * result.dSelectorWq[i]!));
    for (let i = 0; i < result.dSelectorWk.length; i++) asked = Math.max(asked, Math.abs(lr * result.dSelectorWk[i]!));
    if (unfreezeTokens)
      for (let t = 0; t < active.activeTokens.length; t++)
        for (let d = 0; d < h; d++) asked = Math.max(asked, Math.abs(lr * tokenRate * result.dFieldStates[t * h + d]!));
    const scale = asked > maxUpdateAbs ? maxUpdateAbs / asked : 1;
    if (scale < 1) clipped++;

    // Straight down the gradient. No advantage to scale by: a demonstration is
    // not more or less true depending on how the creature felt about it.
    for (let i = 0; i < weights.selector.wq.length; i++) {
      weights.selector.wq[i] = weights.selector.wq[i]! * keep - scale * lr * result.dSelectorWq[i]!;
      written = Math.max(written, Math.abs(weights.selector.wq[i]!));
    }
    for (let i = 0; i < weights.selector.wk.length; i++) {
      weights.selector.wk[i] = weights.selector.wk[i]! * keep - scale * lr * result.dSelectorWk[i]!;
      written = Math.max(written, Math.abs(weights.selector.wk[i]!));
    }
    if (unfreezeTokens) {
      const step = scale * lr * tokenRate;
      const table = weights.embeddings;
      // Decayed once per row, however many tokens of the frame read it.
      for (const start of rows) for (let d = 0; d < h; d++) table[start + d] = table[start + d]! * keep;
      for (let t = 0; t < active.activeTokens.length; t++) {
        const frameTok = active.activeTokens[t]!;
        const token = bases.token + config.tokenRows[frame.gpu.tokenIds[frameTok]!]! * h;
        const role = bases.field + config.tokenRows[frame.gpu.fieldRoles[frameTok]!]! * h;
        for (let d = 0; d < h; d++) {
          const gradient = step * result.dFieldStates[t * h + d]!;
          table[token + d] = table[token + d]! - gradient;
          table[role + d] = table[role + d]! - gradient;
          written = Math.max(written, Math.abs(table[token + d]!), Math.abs(table[role + d]!));
        }
      }
    }
    // Math.max carries NaN through, so one bad number anywhere shows here —
    // and a finite number past the ceiling is refused the same way.
    if (!Number.isFinite(written) || written > maxParameterAbs) {
      weights.selector.wq.set(before.wq);
      weights.selector.wk.set(before.wk);
      for (const [start, values] of before.rows) weights.embeddings.set(values, start);
      rejected++;
      // Nothing is said to the source: the arrays are as it last saw them.
      continue;
    }
    touched = Math.max(touched, written);
    seen++;
    // Before the next frame is differentiated, not after the loop: a source
    // holding a copy would otherwise take the second showing's gradient against
    // the brain the creature had before the first.
    source.wrote?.({ selector: true, ...(rows.size ? { embeddingRows: [...rows] } : {}) });
  }

  return {
    framesSeen: seen,
    shown,
    meanAgreement: shown ? agreementTotal / shown : 0,
    maxParameterAbs: touched,
    rejected,
    clipped,
  };
}
