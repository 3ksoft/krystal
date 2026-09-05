/**
 * Learning from what followed.
 *
 * The same reduction the forward already went through. `policy-training.ts`
 * knows about experience buffers, compiled catalogs, intent schema ids and the
 * fixed frame packer — all of it the simulation bridge's vocabulary, none of it
 * the math's. What the update actually needs is far smaller: the records the
 * creature was shown, which one it picked for each question it was asked, the
 * grammar that limited the picking, and what became of things afterwards.
 *
 * REINFORCE with the value head as baseline. The trick that makes it reuse the
 * supervised machinery unchanged: the gradient of cross-entropy toward the
 * choice that was ACTUALLY MADE is exactly the direction that makes that choice
 * more likely, so scaling it by the advantage turns "push toward the right
 * answer" into "push toward what was done, in proportion to how much better
 * than expected it turned out". A negative advantage flips the sign.
 *
 * The baseline is what stops this from being useless. Reinforcing by the raw
 * outcome credits every choice made in a good moment equally, including the
 * ones that had nothing to do with it; subtracting what was expected leaves
 * only the part the choice is answerable for.
 *
 * Actor and critic move together on purpose. A baseline that lags behind the
 * policy is worse than no baseline: it reports stale expectations as surprise,
 * and the policy chases its own shadow.
 */
import type { ActiveFrame } from "../forward/masks.ts";
import { embeddingTableBases, type BrainForwardConfig, type BrainForwardWeights } from "../forward/model.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import { NO_TARGET } from "./backend.ts";
import { packHostFrame, type HostRecord } from "./frame.ts";
import { cpuGradients, type GradientSource } from "./gradients.ts";

/**
 * One remembered turn: what was shown, what was chosen, and what followed.
 *
 * `chosen` is in the same order and the same numbering as `think` answered in —
 * record slots, as the host sent them — because the two have to describe the
 * same event or the update trains against a frame that never happened.
 */
export interface HostExperience {
  readonly records: readonly HostRecord[];
  /** The record slot each question chose, by question index. `undefined` for a
   *  question the creature did not answer; the frame still teaches the critic. */
  readonly chosen?: readonly (number | undefined)[];
  /**
   * The grammar that was in force when the choice was made — the SAME predicate
   * `think` was given. A different one here would compute the gradient of a
   * distribution the creature never sampled from.
   */
  readonly allows?: (query: number, record: number) => boolean;
  /**
   * What became of things: the change in how the creature was doing. Omitted
   * when there is nothing to difference against — the first frame of a life —
   * and then the frame is skipped rather than counted as a zero, which is a
   * claim that nothing happened.
   */
  readonly reward?: number;
}

export interface LearnOptions {
  readonly learningRate?: number;
  /**
   * Reject the whole update if any mutable parameter exceeds this absolute
   * value. Infinity by default; a host enabling representation learning should
   * choose a ceiling and treat a rejection as a diagnostic, not silently keep
   * a half-updated actor.
   */
  readonly maxParameterAbs?: number;
  /**
   * Scale on the policy step relative to the value step.
   *
   * The critic can afford a steady rate; the actor cannot, because every update
   * changes the distribution the next batch is drawn from.
   */
  readonly policyScale?: number;
  /**
   * Cap on a standardised advantage, in batch deviations.
   *
   * One unusually good outcome should shift the policy, not rewrite it. Without
   * this the batch's own outlier sets the step size, which is how an actor and
   * its baseline chase each other into divergence.
   */
  readonly advantageClip?: number;
  /**
   * Which of the frozen parts move with the actor.
   *
   * Off, everything below the selector is a fixed random projection: a query
   * row cannot shape its own representation, so one shared `Wq`/`Wk` has to
   * serve every question at once and two rows with opposing targets cancel.
   *
   * The semantic token/role tables are the first useful part to unfreeze. Two
   * questions about one relation share schema, band, stream, position and the
   * relation token; only the field token initially tells them apart. The pool
   * is shared by every record, so moving it cannot create that missing
   * difference. `pool` remains an explicit experiment, not the first stage of
   * the schedule.
   *
   * Gradual on purpose. Each part unfrozen is a part that can now diverge, and
   * finding out WHICH one did requires having moved one at a time.
   */
  readonly unfreeze?: {
    /** The record/query pooling — the vector each row is scored as. */
    readonly pool?: boolean;
    /**
     * The token and field-role embedding tables.
     *
     * The one lever that can make two questions look different. A row's vector
     * is the SUM of six lookups — token, role, schema, band, stream, position —
     * and two questions about one relation share five of them, so what tells
     * them apart starts as about a hundredth of the vector. Nothing downstream
     * can amplify a difference that small; the difference itself has to grow.
     *
     * Structural tables (schema, band, stream, position) stay frozen: every
     * record in a group shares those rows, so moving them adds a bias to
     * everything at once and tells nothing apart.
     *
     * OFF by library default: representation learning is a host decision. A
     * host that enables it should choose its own smaller rate, monitor the
     * mutable groups and reject a runaway batch. The first apparent
     * instability was not an on-policy run: it replayed the same positive and
     * negative choices hundreds of times after the policy had moved away from
     * them. Fresh samples from the current policy remain finite and separate
     * the two questions. Do not hide stale replay with an optimiser; replay
     * needs an importance ratio or an explicitly off-policy objective.
     */
    readonly tokens?: boolean;
    /**
     * How fast the tables move, relative to the actor's own step.
     *
     * A row is written once per token that read it, so a symbol the frame
     * repeats legitimately collects a gradient many times while a rare one
     * collects it once. This separate rate changes the speed of the semantic
     * tables without dividing away that frequency signal or changing the
     * balance between shared and distinguishing tokens.
     */
    readonly tokenRate?: number;
  };
}

export interface LearnReport {
  readonly framesSeen: number;
  readonly meanAdvantage: number;
  readonly meanValueLoss: number;
  /** Choices that beat expectation, and were pushed toward. */
  readonly reinforced: number;
  /** Choices that fell short, and were pushed away from. */
  readonly discouraged: number;
  /**
   * How undecided the policy still is, 0..1: its entropy over what the grammar
   * allowed, against the entropy of guessing uniformly among the same records.
   *
   * 1 is a creature choosing at random. Falling is the only visible sign that
   * anything has been learned at all — reward can improve because the world got
   * easier, and an argmax looks the same however sure of itself it is.
   */
  readonly meanEntropy: number;
  /** How much probability the policy put on what it actually chose. */
  readonly meanConfidence: number;
  /** False when the attempted update was rolled back as one transaction. */
  readonly updateApplied: boolean;
  readonly rejected?: "non-finite" | "parameter-limit";
  readonly health: ParameterHealth;
}

export interface ParameterHealth {
  readonly finite: boolean;
  /** Token + field-role rows touched by this batch. */
  readonly semanticEmbeddingMaxAbs: number;
  readonly selectorMaxAbs: number;
  readonly valueMaxAbs: number;
  readonly poolMaxAbs: number;
}

const NEG_INF = -1e30;

function maxAbs(values: Float32Array, start = 0, end = values.length): number {
  let max = 0;
  for (let index = start; index < end; index++) {
    const value = Math.abs(values[index]!);
    if (!Number.isFinite(value)) return Number.POSITIVE_INFINITY;
    if (value > max) max = value;
  }
  return max;
}

function parameterHealth(
  weights: BrainForwardWeights,
  embeddingRows: readonly number[],
  hiddenSize: number,
): ParameterHealth {
  let semanticEmbeddingMaxAbs = 0;
  for (const start of embeddingRows)
    semanticEmbeddingMaxAbs = Math.max(semanticEmbeddingMaxAbs, maxAbs(weights.embeddings, start, start + hiddenSize));
  const selectorMaxAbs = Math.max(maxAbs(weights.selector.wq), maxAbs(weights.selector.wk));
  const valueMaxAbs = maxAbs(weights.valueHeadWv);
  const poolMaxAbs = maxAbs(weights.pool);
  return {
    finite: [semanticEmbeddingMaxAbs, selectorMaxAbs, valueMaxAbs, poolMaxAbs].every(Number.isFinite),
    semanticEmbeddingMaxAbs,
    selectorMaxAbs,
    valueMaxAbs,
    poolMaxAbs,
  };
}

/**
 * The host's grammar as the selector reads it: one row per question, NEG_INF on
 * every record that question may not choose.
 *
 * A question nothing answers is left OPEN rather than uniformly forbidden — a
 * row of NEG_INF comes back as a uniform over the impossible, which looks like
 * an answer and is not one.
 */
export function selectionMask(
  active: ActiveFrame,
  allows: ((query: number, record: number) => boolean) | undefined,
): Float32Array {
  const queries = active.queryRecords.length;
  const bank = active.bankRecords.length;
  const mask = new Float32Array(queries * bank);
  if (!allows) return mask;
  for (let q = 0; q < queries; q++) {
    let admitted = 0;
    for (let r = 0; r < bank; r++) {
      if (allows(q, active.bankRecords[r]!)) admitted++;
      else mask[q * bank + r] = NEG_INF;
    }
    if (!admitted) for (let r = 0; r < bank; r++) mask[q * bank + r] = 0;
  }
  return mask;
}

/**
 * Push a frame's token gradient back into the rows it read.
 *
 * A token state is the sum of six table rows, so the gradient at that state is
 * the gradient of each of them — no chain rule to do, just a scatter. Only the
 * two tables that say WHAT this is are written: the rest are shared by every
 * record of a kind and moving them shifts everything together.
 */
function scatterEmbeddings(
  saved: { frame: v1_0_0.BrainFrameGpu; active: ActiveFrame; dStates: Float32Array },
  weights: BrainForwardWeights,
  config: BrainForwardConfig,
  step: number,
): void {
  const h = config.hiddenSize;
  const bases = embeddingTableBases(config);
  const table = weights.embeddings;
  const { frame, active, dStates } = saved;

  for (let t = 0; t < active.activeTokens.length; t++) {
    const frameTok = active.activeTokens[t]!;
    const token = bases.token + config.tokenRows[frame.tokenIds[frameTok]!]! * h;
    const role = bases.field + config.tokenRows[frame.fieldRoles[frameTok]!]! * h;
    for (let d = 0; d < h; d++) {
      const gradient = step * dStates[t * h + d]!;
      table[token + d] = table[token + d]! - gradient;
      table[role + d] = table[role + d]! - gradient;
    }
  }
}

/**
 * One pass over remembered turns, updating the actor and the critic.
 *
 * Gradients are collected before any of them is applied. A per-frame step would
 * make each update depend on the ones before it inside the same batch and —
 * worse — leaves the advantage on whatever scale the world's own sense of
 * well-being happens to use. Standardising across the batch is what makes one
 * learning rate work for a world where eating is worth 0.35 and for one where
 * it is worth 3.5.
 *
 * Asynchronous because the gradients may come from a device, and the update
 * cannot be applied before they are back. On the CPU every await resolves at
 * once; the shape of the call is the same either way, which is what lets a
 * host choose a backend without choosing a different `learn`.
 */
export async function learnFromExperience(
  experiences: readonly HostExperience[],
  weights: BrainForwardWeights,
  config: BrainForwardConfig,
  options: LearnOptions = {},
  /**
   * Where the gradients come from: the CPU oracle on these very arrays unless
   * a session hands over its backend. The update is the same either way; only
   * who differentiates the frame changes.
   */
  source: GradientSource = cpuGradients(weights, config),
): Promise<LearnReport> {
  const lr = options.learningRate ?? 0.05;
  const policyScale = options.policyScale ?? 1;
  const clip = options.advantageClip ?? 3;
  const maxParameterAbs = options.maxParameterAbs ?? Number.POSITIVE_INFINITY;

  const unfreezePool = options.unfreeze?.pool ?? false;
  const unfreezeTokens = options.unfreeze?.tokens ?? false;
  const tokenRate = options.unfreeze?.tokenRate ?? 0.1;
  interface Pending {
    readonly advantage: number;
    readonly dSelectorWq: Float32Array;
    readonly dSelectorWk: Float32Array;
    readonly dPool: Float32Array;
    /** Kept only when the tables can move: it is [T, H] per frame. */
    readonly embedding?: { frame: v1_0_0.BrainFrameGpu; active: ActiveFrame; dStates: Float32Array };
  }
  const pending: Pending[] = [];
  const dValue = new Float32Array(weights.valueHeadWv.length);
  let seen = 0;
  let advantageTotal = 0;
  let lossTotal = 0;
  let entropyTotal = 0;
  let entropyRows = 0;
  let confidenceTotal = 0;
  let confidenceRows = 0;

  for (const experience of experiences) {
    if (experience.reward === undefined || !experience.records.length) continue;
    const frame = packHostFrame(experience.records);
    const { active } = frame;
    const queries = active.queryRecords.length;
    const bank = active.bankRecords.length;
    if (!queries || !bank) continue;

    // Slots back to bank rows: the host numbers records, the selector numbers
    // the ones it may choose from, and the target is in the second numbering.
    const targets = new Array<number>(queries).fill(NO_TARGET);
    let answered = false;
    for (let q = 0; q < queries; q++) {
      const slot = experience.chosen?.[q];
      if (slot === undefined || slot < 0) continue;
      const row = active.bankRecords.indexOf(slot);
      if (row < 0) continue;
      targets[q] = row;
      answered = true;
    }

    const mask = selectionMask(active, experience.allows);
    const result = await source.backward(frame, {
      selection: mask,
      ...(answered ? { targets } : {}),
      valenceTarget: experience.reward,
    });

    let baseline = 0;
    for (const prediction of result.valuePrediction) baseline += prediction;
    baseline = result.valuePrediction.length ? baseline / result.valuePrediction.length : 0;
    const advantage = experience.reward - baseline;

    // The critic's gradient is accumulated, not applied here.
    //
    // Applying it per frame would move the baseline in the middle of the batch,
    // so a later frame's advantage would be measured against expectations an
    // earlier frame had just changed — the batch would depend on the order it
    // happened to be in, and two identical outcomes would come out looking
    // different from each other.
    for (let i = 0; i < dValue.length; i++) dValue[i]! += result.dValueWv[i]!;

    if (answered && Number.isFinite(advantage))
      pending.push({
        advantage,
        dSelectorWq: result.dSelectorWq,
        dSelectorWk: result.dSelectorWk,
        dPool: result.dPool,
        ...(unfreezeTokens
          ? { embedding: { frame: frame.gpu, active, dStates: result.dFieldStates } }
          : {}),
      });

    // What the policy believed as it chose, which is what says whether it has
    // learned anything: read off the same distribution the update differentiates.
    for (let q = 0; q < queries; q++) {
      let allowed = 0;
      let entropy = 0;
      for (let j = 0; j < bank; j++) {
        const p = result.policy[q * bank + j]!;
        if (mask[q * bank + j]! <= NEG_INF / 2) continue;
        allowed++;
        if (p > 0) entropy -= p * Math.log(p);
      }
      if (allowed > 1) {
        entropyTotal += entropy / Math.log(allowed);
        entropyRows++;
      }
      const target = targets[q]!;
      if (target !== NO_TARGET) {
        confidenceTotal += result.policy[q * bank + target]!;
        confidenceRows++;
      }
    }

    advantageTotal += advantage;
    lossTotal += result.valueLoss;
    seen++;
  }

  // Only rows this batch can mutate belong in its transaction and health
  // report. Copying/scanning the whole semantic page would charge every update
  // for thousands of untouched vocabulary rows.
  const embeddingRows = new Set<number>();
  if (unfreezeTokens) {
    const bases = embeddingTableBases(config);
    const h = config.hiddenSize;
    for (const item of pending) {
      if (!item.embedding) continue;
      const { frame, active } = item.embedding;
      for (const frameTok of active.activeTokens) {
        embeddingRows.add(bases.token + config.tokenRows[frame.tokenIds[frameTok]!]! * h);
        embeddingRows.add(bases.field + config.tokenRows[frame.fieldRoles[frameTok]!]! * h);
      }
    }
  }
  const touchedEmbeddingRows = [...embeddingRows];
  // One update is one transaction. Representation, actor and critic are
  // coupled; keeping two after the third diverged creates a checkpoint that no
  // longer corresponds to any policy that produced the batch.
  const before = {
    semanticRows: touchedEmbeddingRows.map((start) => ({
      start,
      values: Float32Array.from(weights.embeddings.subarray(start, start + config.hiddenSize)),
    })),
    pool: unfreezePool ? Float32Array.from(weights.pool) : undefined,
    selectorWq: Float32Array.from(weights.selector.wq),
    selectorWk: Float32Array.from(weights.selector.wk),
    value: Float32Array.from(weights.valueHeadWv),
  };

  // Critic first, and on its own rate: it is a regression onto a number that
  // exists whether or not the creature chose anything, so it is safe to move
  // steadily. One step per batch, so the rate does not secretly scale with how
  // much was remembered.
  if (seen)
    for (let i = 0; i < weights.valueHeadWv.length; i++)
      weights.valueHeadWv[i] = weights.valueHeadWv[i]! - (lr * dValue[i]!) / seen;

  let reinforced = 0;
  let discouraged = 0;
  let actorMoved = false;
  if (pending.length) {
    const mean = pending.reduce((sum, item) => sum + item.advantage, 0) / pending.length;
    const variance = pending.reduce((sum, item) => sum + (item.advantage - mean) ** 2, 0) / pending.length;
    const deviation = Math.sqrt(variance);
    // A batch where everything went equally well says nothing about which choice
    // was responsible, so it must not push at all — a guard rather than a small
    // epsilon, which would amplify noise into confidence.
    if (deviation > 1e-6) {
      actorMoved = true;
      for (const item of pending) {
        const standardised = Math.max(-clip, Math.min(clip, (item.advantage - mean) / deviation));
        const step = (lr * policyScale * standardised) / pending.length;
        for (let i = 0; i < weights.selector.wq.length; i++)
          weights.selector.wq[i] = weights.selector.wq[i]! - step * item.dSelectorWq[i]!;
        for (let i = 0; i < weights.selector.wk.length; i++)
          weights.selector.wk[i] = weights.selector.wk[i]! - step * item.dSelectorWk[i]!;
        // The pool carries the value head's gradient as well as the policy's —
        // the oracle sums them into one trunk gradient — so scaling it by the
        // advantage scales a little of the critic's signal too. An
        // approximation, and a knowingly cheap one: separating them means a
        // second backward per frame, which doubles the cost of every turn to
        // recover a term the value head is already applying directly.
        if (unfreezePool)
          for (let i = 0; i < weights.pool.length; i++) weights.pool[i] = weights.pool[i]! - step * item.dPool[i]!;
        if (item.embedding) scatterEmbeddings(item.embedding, weights, config, step * tokenRate);
        if (standardised > 0) reinforced++;
        else if (standardised < 0) discouraged++;
      }
    }
  }

  const report = {
    framesSeen: seen,
    meanAdvantage: seen ? advantageTotal / seen : 0,
    meanValueLoss: seen ? lossTotal / seen : 0,
    reinforced,
    discouraged,
    meanEntropy: entropyRows ? entropyTotal / entropyRows : 0,
    meanConfidence: confidenceRows ? confidenceTotal / confidenceRows : 0,
  };
  const attempted = parameterHealth(weights, touchedEmbeddingRows, config.hiddenSize);
  const overLimit = [
    attempted.semanticEmbeddingMaxAbs,
    attempted.selectorMaxAbs,
    attempted.valueMaxAbs,
    attempted.poolMaxAbs,
  ].some((value) => value > maxParameterAbs);
  const rejected = !attempted.finite ? "non-finite" as const : overLimit ? "parameter-limit" as const : undefined;
  if (rejected) {
    for (const row of before.semanticRows) weights.embeddings.set(row.values, row.start);
    if (before.pool) weights.pool.set(before.pool);
    weights.selector.wq.set(before.selectorWq);
    weights.selector.wk.set(before.selectorWk);
    weights.valueHeadWv.set(before.value);
  } else if (seen) {
    // Exactly what this transaction wrote, and nothing it left alone: a backend
    // holding a copy takes up two projections and a handful of rows, not the
    // whole brain. Nothing is said after a rollback — the arrays are as the
    // backend last saw them.
    source.wrote?.({
      valueHead: true,
      ...(actorMoved ? { selector: true } : {}),
      ...(actorMoved && unfreezePool ? { pool: true } : {}),
      ...(actorMoved && touchedEmbeddingRows.length ? { embeddingRows: touchedEmbeddingRows } : {}),
    });
  }
  return {
    ...report,
    updateApplied: seen > 0 && !rejected,
    ...(rejected ? { rejected } : {}),
    health: parameterHealth(weights, touchedEmbeddingRows, config.hiddenSize),
  };
}
