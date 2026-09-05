// M3 close (docs/archive/WEBGPU_BACKWARD_PLAN.md §17 item 10): the composed Krystal
// backward runner. One submit per trainStep: the full forward (with per-block
// saved activations), the route-kind cross-entropy loss, every backward
// operator from the M3 slices, and plain SGD on the trainable pages — all
// GPU-resident. The only readback in the normal path is the compact scalar
// loss telemetry when `telemetry` is set.
//
// Backward flow (mirrors the forward graph in reverse):
//   CE(dLogits over route kinds)
//   <- decision head backward (dQueryOutput/dIntentGather/dArgGather/dWh)
//   <- selector backward x2 (soft gather + pointer loss) -> dQueryValues/dBankKeys/dBankValues/dSelectorWq/Wk
//   <- mixer blocks reverse (attention + FFN, interleaved per-block SGD)
//   <- pool backward x2 (bank + query) -> dFieldStates/dPool
//   <- encoder blocks reverse (interleaved SGD)
//   <- field embedding backward -> dEmbedding
//
// Deliberate shortcuts (documented in the plan): plain SGD without momentum,
// gradient buffers fully overwritten or accumulated deterministically (no
// atomics), and each block's weight gradients consumed by SGD before the next
// block reuses the same dW regions (block-local weights, so interleaved
// updates are exact for one step).
import {
  KRYSTAL_FORWARD_ARENA,
  KRYSTAL_FORWARD_ARENA_BASE,
  KRYSTAL_BACKWARD_ARENA,
  KRYSTAL_BACKWARD_ARENA_BASE,
  KRYSTAL_MAX_FFN,
  KRYSTAL_MAX_H,
  KRYSTAL_MAX_HEADS,
  KRYSTAL_MAX_QUERIES,
  KRYSTAL_MAX_RECORDS,
  KRYSTAL_MAX_TOKENS,
  KRYSTAL_TRAINING_ARENA,
  TRAINING_ARENA_BASE,
} from "./krystal-layout";
import { type KrystalDefinition } from "./krystal";
import {
  KrystalForward,
  type BrainForwardWeightPages,
  type KrystalMasks,
  type PreparedForward,
} from "./krystal-forward";
import type { WordBias } from "../../krystal/src/forward/masks";
import { EMBEDDING_TABLES } from "../../krystal/src/forward/model";
import type { BackwardResult } from "../../krystal/src/host/backend";
import type { v1_0_0 } from "../../schema/generated/krystal.types";

function validate(condition: boolean, message: string): void {
  if (!condition) throw new Error(`KrystalBackward: ${message}`);
}

export interface KrystalTrainStepOptions {
  readonly frame: v1_0_0.BrainFrameGpu;
  /** Every mask the graph reads, from the host (see KrystalMasks). */
  readonly masks?: KrystalMasks;
  /**
   * Route-kind gold labels [Q] for the decision-head cross-entropy loss.
   *
   * Optional, and leaving it out is not the same as passing zeros: the head is
   * a supervised classifier, so a label nobody has means training it toward
   * class 0 and pushing that gradient through the shared gather into the
   * selector — an auxiliary task made of noise, shaping the actor. Left out,
   * the head contributes nothing and the value head still reads the same
   * context.
   */
  readonly routeKinds?: readonly number[] | Uint32Array;
  /**
   * Observed change in valence for the tick this frame produced — the value
   * head's target. Omitted when there is nothing to difference against (the
   * first frame of a life), and then the head contributes no gradient at all
   * rather than being trained toward zero.
   */
  readonly valenceTarget?: number;
  /**
   * Optional pointer-loss targets for the argument selector, one [Q] array
   * per argument index (`argumentTarget[q][argument]`, S2-S10 contract):
   * bank indices of the gold records, 0xffffffff = no pointer loss for that
   * row. Arity-0 intents and unlabelled rows carry 0xffffffff. The temporary
   * implementation has one pointer head, so only `argumentTargets[0]` is
   * consumed; the shape is per-argument for future arity (default: none).
   */
  readonly argumentTargets?: readonly (readonly number[] | Uint32Array)[];
  /**
   * Optional pointer-loss targets [Q] for the main selector slot: bank indices
   * of the records the question should have chosen, 0xffffffff = no pointer
   * loss for that row (default: none). This is what a demonstration teaches
   * and what a reinforced choice is pushed toward. An index with its top bit
   * set (0x80000000 | row) is pushed AWAY from instead — see `AWAY` in the
   * host backend seam.
   */
  readonly selectionTargets?: readonly number[] | Uint32Array;
  readonly learningRate: number;
  /**
   * What to do with the gradients once they exist.
   *
   * `sgd` applies plain SGD to every page in the same submit, as the training
   * slice always did. `none` computes and leaves them, which is what a host
   * doing its own update needs: the actor's step is scaled by an advantage
   * standardised across a batch, only some parts are unfrozen, and the whole
   * thing is one transaction that may be rolled back. None of that can be
   * expressed as "subtract lr times the gradient, now, per frame".
   *
   * With `none` the fused sparse embedding update does not run either; the
   * exposed product for the tables is dFieldStates, which the host scatters
   * into the rows this frame actually read.
   */
  readonly optimizer?: "sgd" | "none";
  /**
   * Optional same-word attention bias (docs/archive/word_attention_bias.md). It rides
   * in the record mask, so it needs no device change and no gradient.
   */
  readonly wordBias?: WordBias;
  /** Read back the scalar mean loss (compact telemetry); off by default. */
  readonly telemetry?: boolean;
}

export interface KrystalTrainStepResult {
  readonly step: number;
  /** Mean cross-entropy over route kinds; absent when no labels were given. */
  readonly loss?: number;
  /** Mean squared error of the value head; absent without a valence target. */
  readonly valueLoss?: number;
}

/** Shared arena capacity for the save slices (must match the forward's dims). */
const FWD = KRYSTAL_FORWARD_ARENA;

export class KrystalBackward {
  private readonly forward: KrystalForward;
  private readonly definition: KrystalDefinition;
  private readonly pages: BrainForwardWeightPages;
  private step = 0;

  constructor(forward: KrystalForward) {
    this.forward = forward;
    this.definition = forward.getDefinition();
    this.pages = forward.weightPages;
  }

  private fwd(offset: number, elements: number): number {
    validate(offset + elements <= FWD.elements, "forward arena region overflows capacity");
    return KRYSTAL_FORWARD_ARENA_BASE + offset;
  }

  private bwd(offset: number, elements: number): number {
    validate(
      offset + elements <= KRYSTAL_BACKWARD_ARENA.elements,
      "backward arena region overflows capacity",
    );
    return KRYSTAL_BACKWARD_ARENA_BASE + offset;
  }

  private train(offset: number, elements: number): number {
    validate(offset + elements <= KRYSTAL_TRAINING_ARENA.elements, "training arena region overflows capacity");
    return TRAINING_ARENA_BASE + offset;
  }

  /** Run one GPU-resident training step: forward + loss + backward + SGD. */
  async trainStep(options: KrystalTrainStepOptions): Promise<KrystalTrainStepResult> {
    const config = this.forward.getConfig();
    const { hiddenSize: h, ffnSize: ffn, headCount: heads, headDim, encoderBlocks, mixerBlocks, routeKindCount: C } = config;
    const A = FWD;
    const B = KRYSTAL_BACKWARD_ARENA;
    const T = KRYSTAL_TRAINING_ARENA;

    const prepared = this.forward.prepare(options.frame, options.masks, options.wordBias);
    const { active, context, selects, t, r, q } = prepared;
    validate(selects, "training needs a selection mask: there is nothing to push toward without one");
    const routeKinds = options.routeKinds === undefined
      ? undefined
      : options.routeKinds instanceof Uint32Array ? options.routeKinds : Uint32Array.from(options.routeKinds);
    if (routeKinds) validate(routeKinds.length === q, `routeKinds must be [Q] = ${q}`);
    const applySgd = (options.optimizer ?? "sgd") === "sgd";
    validate(Number.isFinite(options.learningRate) && options.learningRate > 0, "learningRate must be > 0");

    // --- Forward arena regions ---
    const fieldStates = this.fwd(A.fieldStates, t * h);
    const bankKeys = this.fwd(A.bankKeys, r * h);
    const bankValues = this.fwd(A.bankValues, r * h);
    const queryKeys = this.fwd(A.queryKeys, q * h);
    const queryValues = this.fwd(A.queryValues, q * h);
    const selectorQ = this.fwd(A.selectorQ, q * h);
    const selectorK = this.fwd(A.selectorK, r * h);
    const intentP = this.fwd(A.intentP, q * r);
    const argP = this.fwd(A.argP, q * r);
    const intentGather = this.fwd(A.intentGather, q * h);
    const argGather = this.fwd(A.argGather, q * h);
    const decisionLogits = this.fwd(A.decisionLogits, q * C);
    const valuePrediction = this.fwd(A.valuePrediction, q);
    const availableP = this.fwd(A.availableP, q * r);
    const availableGather = this.fwd(A.availableGather, q * h);
    // The critic's third context block: what was on offer, or the second
    // slot's gather. The forward chose the same one.
    const valueContext = context === "available" ? availableGather : argGather;
    const tokenIds = this.fwd(A.tokenIds, 1024);
    const fieldRoles = this.fwd(A.fieldRoles, 1024);
    const schemaIds = this.fwd(A.schemaIds, 128);
    const bandIds = this.fwd(A.bandIds, 128);
    const streamIds = this.fwd(A.streamIds, 128);
    const activeTokens = this.fwd(A.activeTokens, t);
    const bankIndices = this.fwd(A.bankIndices, r);
    const queryIndices = this.fwd(A.queryIndices, q);
    const recordCompactOffset = this.fwd(A.recordCompactOffset, 128);
    const recordCompactCount = this.fwd(A.recordCompactCount, 128);

    // Per-block saved activations (written by the forward in save mode). The
    // block strides are the layout's max-capacity strides (identical to the
    // forward's save slices), not the actual dims.
    const th = KRYSTAL_MAX_TOKENS * KRYSTAL_MAX_H;
    const tf = KRYSTAL_MAX_TOKENS * KRYSTAL_MAX_FFN;
    const hmEnc = KRYSTAL_MAX_HEADS * KRYSTAL_MAX_TOKENS * KRYSTAL_MAX_TOKENS;
    const qh = KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_H;
    const rh = KRYSTAL_MAX_RECORDS * KRYSTAL_MAX_H;
    const qf = KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_FFN;
    const hmMix = KRYSTAL_MAX_HEADS * KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_RECORDS;
    const encStride = {
      in: A.encSavedIn, ffnIn: A.encSavedFfnIn, q: A.encSavedQ, k: A.encSavedK, v: A.encSavedV,
    } as const;
    const mixerStride = {
      in: A.mixerSavedIn, ffnIn: A.mixerSavedFfnIn, q: A.mixerSavedQ, k: A.mixerSavedK, v: A.mixerSavedV,
    } as const;

    // --- Backward arena regions ---
    const dDecisionLogits = this.bwd(B.dDecisionLogits, q * C);
    const dDecisionQuery = this.bwd(B.dDecisionQuery, qh);
    const dDecisionIntent = this.bwd(B.dDecisionIntent, qh);
    const dDecisionArg = this.bwd(B.dDecisionArg, qh);
    const dDecisionWh = this.bwd(B.dDecisionWh, C * 3 * h);
    const dValuePrediction = this.bwd(B.dValuePrediction, q);
    const valueLossRows = this.bwd(B.valueLossRows, q);
    const dValueQuery = this.bwd(B.dValueQuery, qh);
    const dValueIntent = this.bwd(B.dValueIntent, qh);
    const dValueArg = this.bwd(B.dValueArg, qh);
    const dValueWv = this.bwd(B.dValueWv, 3 * h);
    const dSelectorScores = this.bwd(B.dSelectorScores, q * r);
    const dSelectorQProj = this.bwd(B.dSelectorQProj, qh);
    const dSelectorKProj = this.bwd(B.dSelectorKProj, rh);
    const dSelectorValue = this.bwd(B.dSelectorValue, rh);
    const dSelectorWq = this.bwd(B.dSelectorWq, h * h);
    const dSelectorWk = this.bwd(B.dSelectorWk, h * h);
    const selectorGold = this.bwd(B.selectorGold, q);
    const argGoldOff = this.bwd(B.argGold, q);
    const dPoolPartial = this.bwd(B.dPoolPartial, r * 2 * h);
    const dPool = this.bwd(B.dPool, 2 * h);
    const dPool2 = this.bwd(B.dPool2, 2 * h);
    const dFieldStates = this.bwd(B.dFieldStates, th);
    const dEncQ = this.bwd(B.dEncQ, th);
    const dEncK = this.bwd(B.dEncK, th);
    const dEncV = this.bwd(B.dEncV, th);
    const dScoresEnc = this.bwd(B.dScoresEnc, hmEnc);
    const dHiddenQ = this.bwd(B.dHiddenQ, th);
    const dHiddenK = this.bwd(B.dHiddenK, th);
    const dHiddenV = this.bwd(B.dHiddenV, th);
    const dWq = this.bwd(B.dWq, h * h);
    const dWk = this.bwd(B.dWk, h * h);
    const dWv = this.bwd(B.dWv, h * h);
    const dH1 = this.bwd(B.dH1, tf);
    const dW1 = this.bwd(B.dW1, ffn * h);
    const dW2 = this.bwd(B.dW2, h * ffn);
    const dBankKeys = this.bwd(B.dBankKeys, rh);
    const dBankValues = this.bwd(B.dBankValues, rh);
    const dQueryKeys = this.bwd(B.dQueryKeys, qh);
    // The mixed-query gradient accumulator aliases the decision-head output
    // region: decisionHeadBackward seeds it with dQueryOutput, then the
    // selector routing and mixer loop residual-add into the same buffer. The
    // standalone B.dQueryValues region stays reserved for parity reads.
    const dQueryValues = this.bwd(B.dDecisionQuery, qh);

    // Training regions (shared with the M1 trainer; same arena buffer).
    const lossRows = this.train(T.lossRows, q);
    const scalarLoss = this.train(T.scalarLoss, 1);

    const device = this.definition.engine.device;
    const arena = this.definition.resources.arena.gpu;
    const NO_TARGET = 0xffff_ffff;
    const noTargets = new Uint32Array(q).fill(NO_TARGET);
    if (routeKinds) device.queue.writeBuffer(this.definition.resources.targets.gpu, 0, routeKinds);
    const selectionTargets = options.selectionTargets
      ? (options.selectionTargets instanceof Uint32Array
          ? options.selectionTargets
          : Uint32Array.from(options.selectionTargets))
      : noTargets;
    validate(selectionTargets.length === q, `selectionTargets must be [Q] = ${q}`);
    device.queue.writeBuffer(arena, selectorGold * 4, selectionTargets);
    const argTargets = options.argumentTargets?.[0]
      ? (options.argumentTargets[0] instanceof Uint32Array
          ? options.argumentTargets[0]
          : Uint32Array.from(options.argumentTargets[0]))
      : noTargets;
    validate(argTargets.length === q, `argumentTargets[0] must be [Q] = ${q}`);
    device.queue.writeBuffer(arena, argGoldOff * 4, argTargets);

    // Cumulative embedding-table row counts (field embed backward u0..u5).
    const embCum: number[] = [];
    {
      let cursor = 0;
      for (const table of EMBEDDING_TABLES) {
        cursor += table.rows;
        embCum.push(cursor);
      }
    }
    const embRows = embCum[embCum.length - 1]!;
    const dEmbeddingElems = embRows * h;
    validate(dEmbeddingElems <= KRYSTAL_BACKWARD_ARENA.elements, "dEmbedding overflow");

    // The dense reference embedding backward owns all 8,469 * H output
    // elements and makes each one scan every active token, even though only a
    // small subset of rows can receive a gradient in one frame. Build that
    // subset once on the host and let the fused sparse pass preserve the exact
    // token-order reduction for just those rows. B.dEmbedding is otherwise
    // dead in the composed runner, so its prefix safely stores the row list.
    const embeddingRows = new Set<number>();
    const embBases = [0, embCum[0]!, embCum[1]!, embCum[2]!, embCum[3]!, embCum[4]!];
    for (const frameTok of active.activeTokens) {
      const slot = frameTok >> 3;
      const indices = [
        // Same projection the forward upload applies: the tables are indexed by
        // embedding row, and a raw token id is not one.
        this.forward.config.tokenRows[options.frame.tokenIds[frameTok]!]!,
        this.forward.config.tokenRows[options.frame.fieldRoles[frameTok]!]!,
        options.frame.schemaIds[slot]!,
        options.frame.bandIds[slot]!,
        active.streamIds[slot]!,
        frameTok & 7,
      ];
      for (let table = 0; table < indices.length; table++) {
        const row = embBases[table]! + indices[table]!;
        validate(row >= embBases[table]! && row < embCum[table]!, `embedding row ${row} outside table ${table}`);
        embeddingRows.add(row);
      }
    }
    const sparseEmbeddingRows = Uint32Array.from(embeddingRows);
    const sparseEmbeddingRowsOff = this.bwd(B.dEmbedding, sparseEmbeddingRows.length);
    device.queue.writeBuffer(arena, sparseEmbeddingRowsOff * 4, sparseEmbeddingRows);

    const lr = options.learningRate;
    const pages = this.pages;
    const step = ++this.step;

    this.forward.submitPrepared(prepared, true, (encoder) => {
      // 1. Route-kind cross-entropy, when the frame carries labels. Without
      //    them the head stays inert: its gradient is explicitly zeroed rather
      //    than left as whatever the previous step wrote, so nothing trains
      //    toward a class nobody chose.
      if (routeKinds) {
        encoder.compute((pass) => pass.run("cross_entropy_forward_backward", {
          inputOffset: decisionLogits, outputOffset: dDecisionLogits, auxOffset: lossRows,
          tokenCount: q, outputDim: C, u1: 0,
        }));
        encoder.compute((pass) => pass.run("loss_reduce", {
          inputOffset: lossRows, outputOffset: scalarLoss, tokenCount: q,
        }));
      } else {
        encoder.compute((pass) => pass.run("zero_f32", { outputOffset: dDecisionLogits, tokenCount: q * C }));
      }

      // 2. Decision head backward (+ SGD on Wh when this runner optimizes).
      encoder.compute((pass) => pass.run("krystal_decision_head_backward", {
        inputOffset: dDecisionLogits, auxOffset: queryValues, aux2Offset: intentGather, aux3Offset: argGather,
        outputOffset: dDecisionQuery, aux4Offset: dDecisionIntent, aux5Offset: dDecisionArg, aux6Offset: dDecisionWh,
        tokenCount: q, inputDim: h, outputDim: C,
      }, pages.decisionHead));
      if (applySgd) {
        encoder.compute((pass) => pass.run("sgd_step", {
          inputOffset: dDecisionWh, tokenCount: C * 3 * h, f0: lr,
        }, pages.decisionHead));
      }

      // 2b. Value head: the same head with one class, its own squared-error
      //     loss, and the same first two context blocks. Its gradients ADD to
      //     the decision head's there, which is how the value signal shapes
      //     the trunk instead of only its own head. The third block is the
      //     one place they differ: with the `available` context the critic
      //     reads what was on offer, not what was chosen.
      encoder.compute((pass) => pass.run("krystal_value_head_loss", {
        inputOffset: valuePrediction, outputOffset: dValuePrediction, auxOffset: valueLossRows,
        tokenCount: q, f0: options.valenceTarget ?? 0, u0: options.valenceTarget === undefined ? 0 : 1,
      }));
      encoder.compute((pass) => pass.run("krystal_decision_head_backward", {
        inputOffset: dValuePrediction, auxOffset: queryValues, aux2Offset: intentGather,
        aux3Offset: valueContext,
        outputOffset: dValueQuery, aux4Offset: dValueIntent, aux5Offset: dValueArg, aux6Offset: dValueWv,
        tokenCount: q, inputDim: h, outputDim: 1,
      }, pages.valueHead));
      encoder.compute((pass) => pass.run("residual_add", {
        inputOffset: dDecisionQuery, auxOffset: dValueQuery, outputOffset: dDecisionQuery,
        tokenCount: q, inputDim: h,
      }));
      encoder.compute((pass) => pass.run("residual_add", {
        inputOffset: dDecisionIntent, auxOffset: dValueIntent, outputOffset: dDecisionIntent,
        tokenCount: q, inputDim: h,
      }));
      // Under the `available` context the critic's third block is a mean, not
      // a soft gather, so its gradient is not the argument slot's — it is
      // shared evenly over the rows that were averaged, below.
      if (context === "argument") {
        encoder.compute((pass) => pass.run("residual_add", {
          inputOffset: dDecisionArg, auxOffset: dValueArg, outputOffset: dDecisionArg,
          tokenCount: q, inputDim: h,
        }));
      }
      if (applySgd) {
        encoder.compute((pass) => pass.run("sgd_step", {
          inputOffset: dValueWv, tokenCount: 3 * h, f0: lr,
        }, pages.valueHead));
      }

      // 3. Zero the bank gradient accumulators (selector + mixer accumulate).
      encoder.compute((pass) => pass.run("zero_f32", { outputOffset: dBankKeys, tokenCount: rh }));
      encoder.compute((pass) => pass.run("zero_f32", { outputOffset: dBankValues, tokenCount: rh }));

      // 4. Selector backward. The main slot writes the accumulators directly;
      //    the second one writes scratch (dEncQ/K/V) and residual-adds.
      const slots: readonly (readonly [number, number, number, boolean])[] = context === "argument"
        ? [[dDecisionIntent, intentP, selectorGold, false], [dDecisionArg, argP, argGoldOff, true]]
        : [[dDecisionIntent, intentP, selectorGold, false]];
      for (const [dGather, pOff, goldOff, scratch] of slots) {
        const dQOut = scratch ? dEncQ : dSelectorQProj;
        const dKOut = scratch ? dEncK : dSelectorKProj;
        const dVOut = scratch ? dEncV : dSelectorValue;
        encoder.compute((pass) => pass.run("krystal_selector_backward_scores", {
          inputOffset: dGather, auxOffset: bankValues, aux2Offset: pOff, aux3Offset: goldOff,
          outputOffset: dSelectorScores,
          tokenCount: q, inputDim: h, u0: r, u1: scratch ? 1 : 0,
        }));
        encoder.compute((pass) => pass.run("krystal_selector_backward_qkv", {
          inputOffset: dSelectorScores, auxOffset: selectorQ, aux2Offset: selectorK,
          aux3Offset: pOff, aux4Offset: dGather,
          outputOffset: dQOut, aux5Offset: dKOut, aux6Offset: dVOut,
          tokenCount: q, inputDim: h, u0: r,
          // FOLLOW_UP2 Fix A: the argument slot's INVALID (arity-0 / unlabelled)
          // rows must contribute zero gradient everywhere; the scores pass
          // zeroes their dScore, this pass skips them in the dValue sum.
          u1: scratch ? argGoldOff : 0, u2: scratch ? 1 : 0,
        }));
        if (scratch) {
          encoder.compute((pass) => pass.run("residual_add", {
            inputOffset: dSelectorQProj, auxOffset: dEncQ, outputOffset: dSelectorQProj, tokenCount: q, inputDim: h,
          }));
          encoder.compute((pass) => pass.run("residual_add", {
            inputOffset: dSelectorKProj, auxOffset: dEncK, outputOffset: dSelectorKProj, tokenCount: r, inputDim: h,
          }));
          encoder.compute((pass) => pass.run("residual_add", {
            inputOffset: dSelectorValue, auxOffset: dEncV, outputOffset: dSelectorValue, tokenCount: r, inputDim: h,
          }));
        }
      }

      // 4b. The critic's mean context has no softmax to push back through: its
      //     gradient is shared evenly over the records it averaged. That is
      //     the same sum the selector's qkv backward already computes for the
      //     values — sum_i p[i,j] * dGather[i] — with p the uniform-over-open
      //     distribution the forward produced, so it runs that pass with a
      //     zeroed dScore. The projections then receive exactly nothing, which
      //     is right: the mean is not scored, so it trains no scoring.
      if (context === "available") {
        encoder.compute((pass) => pass.run("zero_f32", { outputOffset: dSelectorScores, tokenCount: q * r }));
        encoder.compute((pass) => pass.run("krystal_selector_backward_qkv", {
          inputOffset: dSelectorScores, auxOffset: selectorQ, aux2Offset: selectorK,
          aux3Offset: availableP, aux4Offset: dValueArg,
          outputOffset: dEncQ, aux5Offset: dEncK, aux6Offset: dEncV,
          tokenCount: q, inputDim: h, u0: r, u1: 0, u2: 0,
        }));
        encoder.compute((pass) => pass.run("residual_add", {
          inputOffset: dSelectorValue, auxOffset: dEncV, outputOffset: dSelectorValue, tokenCount: r, inputDim: h,
        }));
      }

      // 5. Selector weight gradients + route the selector gradients into the
      //    residual streams (dQueryValues already holds dDecisionQuery). SGD
      //    on the selector pages lands after the input-gradient passes so
      //    those read the pre-update weights.
      encoder.compute((pass) => pass.run("matmul_backward_weight", {
        inputOffset: dSelectorQProj, auxOffset: queryValues, outputOffset: dSelectorWq,
        tokenCount: q, inputDim: h, outputDim: h,
      }));
      encoder.compute((pass) => pass.run("matmul_backward_weight", {
        inputOffset: dSelectorKProj, auxOffset: bankKeys, outputOffset: dSelectorWk,
        tokenCount: r, inputDim: h, outputDim: h,
      }));
      encoder.compute((pass) => pass.run("matmul_backward_input", {
        inputOffset: dSelectorQProj, outputOffset: dEncQ,
        tokenCount: q, inputDim: h, outputDim: h,
      }, pages.selectorWq));
      encoder.compute((pass) => pass.run("residual_add", {
        inputOffset: dQueryValues, auxOffset: dEncQ, outputOffset: dQueryValues, tokenCount: q, inputDim: h,
      }));
      encoder.compute((pass) => pass.run("matmul_backward_input", {
        inputOffset: dSelectorKProj, outputOffset: dEncK,
        tokenCount: r, inputDim: h, outputDim: h,
      }, pages.selectorWk));
      encoder.compute((pass) => pass.run("residual_add", {
        inputOffset: dBankKeys, auxOffset: dEncK, outputOffset: dBankKeys, tokenCount: r, inputDim: h,
      }));
      encoder.compute((pass) => pass.run("residual_add", {
        inputOffset: dBankValues, auxOffset: dSelectorValue, outputOffset: dBankValues, tokenCount: r, inputDim: h,
      }));
      if (applySgd) {
        encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dSelectorWq, tokenCount: h * h, f0: lr }, pages.selectorWq));
        encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dSelectorWk, tokenCount: h * h, f0: lr }, pages.selectorWk));
      }

      // 6. Mixer blocks reverse (attention + FFN), interleaved SGD. dQueryKeys
      //    is the [Q,H] scratch here; it is re-zeroed before the pool pass.
      for (let b = mixerBlocks - 1; b >= 0; b--) {
        const block = pages.mixer[b]!;
        const sIn = this.fwd(mixerStride.in + b * qh, qh);
        const sFfnIn = this.fwd(mixerStride.ffnIn + b * qh, qh);
        const sQ = this.fwd(mixerStride.q + b * qh, qh);
        const sK = this.fwd(mixerStride.k + b * rh, rh);
        const sV = this.fwd(mixerStride.v + b * rh, rh);
        const sP = this.fwd(A.mixerSavedP + b * hmMix, hmMix);
        const sH1 = this.fwd(A.mixerSavedH1 + b * qf, qf);

        // FFN branch: dFfnOut = current dQueryValues (block-exit gradient).
        //   dW2 = dFfnOut^T @ h1;  dH1r = dFfnOut @ W2;  dH1p = relu'(h1)*dH1r
        //   dW1 = dH1p^T @ x1;     dFfnIn = dH1p @ W1
        // All weight-gradient and input-gradient passes run on the pre-update
        // weights; SGD lands at the end of the block so matmul_backward_input
        // reads the same W the forward used (gradients stay exact vs the CPU
        // oracle). The dW regions are then consumed before the next block
        // reuses them.
        encoder.compute((pass) => pass.run("matmul_backward_weight", {
          inputOffset: dQueryValues, auxOffset: sH1, outputOffset: dW2,
          tokenCount: q, inputDim: h, outputDim: ffn,
        }));
        encoder.compute((pass) => pass.run("matmul_backward_input", {
          inputOffset: dQueryValues, outputOffset: dH1,
          tokenCount: q, inputDim: h, outputDim: ffn,
        }, block.w2));
        encoder.compute((pass) => pass.run("relu_backward", {
          inputOffset: sH1, auxOffset: dH1, outputOffset: dH1, tokenCount: qf,
        }));
        encoder.compute((pass) => pass.run("matmul_backward_weight", {
          inputOffset: dH1, auxOffset: sFfnIn, outputOffset: dW1,
          tokenCount: q, inputDim: ffn, outputDim: h,
        }));
        encoder.compute((pass) => pass.run("matmul_backward_input", {
          inputOffset: dH1, outputOffset: dQueryKeys,
          tokenCount: q, inputDim: ffn, outputDim: h,
        }, block.w1));
        // dQueryValues = dFfnOut + dFfnIn = d(x1) = dAttnOut.
        encoder.compute((pass) => pass.run("residual_add", {
          inputOffset: dQueryValues, auxOffset: dQueryKeys, outputOffset: dQueryValues, tokenCount: q, inputDim: h,
        }));

        // Attention branch.
        encoder.compute((pass) => pass.run("krystal_attention_backward_scores", {
          inputOffset: dQueryValues, auxOffset: sV, aux2Offset: sP, outputOffset: dScoresEnc,
          tokenCount: q, inputDim: h, outputDim: headDim, u0: r, u1: heads,
        }));
        encoder.compute((pass) => pass.run("krystal_attention_backward_qkv", {
          inputOffset: dScoresEnc, auxOffset: sQ, aux2Offset: sK, aux3Offset: sP, aux4Offset: dQueryValues,
          outputOffset: dQueryKeys, aux5Offset: dEncK, aux6Offset: dEncV,
          tokenCount: q, inputDim: h, outputDim: headDim, u0: r, u1: heads,
        }));
        encoder.compute((pass) => pass.run("matmul_backward_weight", {
          inputOffset: dQueryKeys, auxOffset: sIn, outputOffset: dWq,
          tokenCount: q, inputDim: h, outputDim: h,
        }));
        encoder.compute((pass) => pass.run("matmul_backward_weight", {
          inputOffset: dEncK, auxOffset: bankKeys, outputOffset: dWk,
          tokenCount: r, inputDim: h, outputDim: h,
        }));
        encoder.compute((pass) => pass.run("matmul_backward_weight", {
          inputOffset: dEncV, auxOffset: bankValues, outputOffset: dWv,
          tokenCount: r, inputDim: h, outputDim: h,
        }));

        // Input gradients (read the pre-update weight pages).
        encoder.compute((pass) => pass.run("matmul_backward_input", {
          inputOffset: dQueryKeys, outputOffset: dHiddenQ,
          tokenCount: q, inputDim: h, outputDim: h,
        }, block.wq));
        encoder.compute((pass) => pass.run("residual_add", {
          inputOffset: dQueryValues, auxOffset: dHiddenQ, outputOffset: dQueryValues, tokenCount: q, inputDim: h,
        }));
        encoder.compute((pass) => pass.run("matmul_backward_input", {
          inputOffset: dEncK, outputOffset: dHiddenV,
          tokenCount: r, inputDim: h, outputDim: h,
        }, block.wk));
        encoder.compute((pass) => pass.run("residual_add", {
          inputOffset: dBankKeys, auxOffset: dHiddenV, outputOffset: dBankKeys, tokenCount: r, inputDim: h,
        }));
        encoder.compute((pass) => pass.run("matmul_backward_input", {
          inputOffset: dEncV, outputOffset: dHiddenK,
          tokenCount: r, inputDim: h, outputDim: h,
        }, block.wv));
        encoder.compute((pass) => pass.run("residual_add", {
          inputOffset: dBankValues, auxOffset: dHiddenK, outputOffset: dBankValues, tokenCount: r, inputDim: h,
        }));

        // SGD after the whole block's backward: all gradients used the
        // pre-update weights, and the dW regions are consumed before the next
        // (lower) block reuses them.
        if (applySgd) {
          encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dW2, tokenCount: h * ffn, f0: lr }, block.w2));
          encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dW1, tokenCount: ffn * h, f0: lr }, block.w1));
          encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dWq, tokenCount: h * h, f0: lr }, block.wq));
          encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dWk, tokenCount: h * h, f0: lr }, block.wk));
          encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dWv, tokenCount: h * h, f0: lr }, block.wv));
        }
      }

      // 7. Learned-query pooling backward (bank + query) + SGD on the shared
      //    pool queries. dFieldStates is zeroed first (both pools accumulate
      //    into it, then the encoder loop continues the accumulation).
      encoder.compute((pass) => pass.run("zero_f32", { outputOffset: dQueryKeys, tokenCount: qh }));
      encoder.compute((pass) => pass.run("zero_f32", { outputOffset: dFieldStates, tokenCount: th }));
      encoder.compute((pass) => pass.run("krystal_pool_backward", {
        inputOffset: fieldStates, auxOffset: bankIndices,
        aux2Offset: recordCompactOffset, aux3Offset: recordCompactCount,
        aux4Offset: dBankKeys, aux5Offset: dBankValues,
        outputOffset: dFieldStates, aux6Offset: dPoolPartial,
        tokenCount: r, inputDim: h,
      }, pages.pool));
      encoder.compute((pass) => pass.run("krystal_pool_dpool", {
        inputOffset: dPoolPartial, outputOffset: dPool, tokenCount: r, inputDim: h,
      }));
      encoder.compute((pass) => pass.run("krystal_pool_backward", {
        inputOffset: fieldStates, auxOffset: queryIndices,
        aux2Offset: recordCompactOffset, aux3Offset: recordCompactCount,
        aux4Offset: dQueryKeys, aux5Offset: dQueryValues,
        outputOffset: dFieldStates, aux6Offset: dPoolPartial,
        tokenCount: q, inputDim: h,
      }, pages.pool));
      encoder.compute((pass) => pass.run("krystal_pool_dpool", {
        inputOffset: dPoolPartial, outputOffset: dPool2, tokenCount: q, inputDim: h,
      }));
      encoder.compute((pass) => pass.run("residual_add", {
        inputOffset: dPool, auxOffset: dPool2, outputOffset: dPool, tokenCount: 2, inputDim: h,
      }));
      if (applySgd) {
        encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dPool, tokenCount: 2 * h, f0: lr }, pages.pool));
      }

      // 8. Encoder blocks reverse (attention + FFN), interleaved SGD. The
      //    dFieldStates accumulator already holds the pool contributions.
      for (let b = encoderBlocks - 1; b >= 0; b--) {
        const block = pages.enc[b]!;
        const sIn = this.fwd(encStride.in + b * th, th);
        const sFfnIn = this.fwd(encStride.ffnIn + b * th, th);
        const sQ = this.fwd(encStride.q + b * th, th);
        const sK = this.fwd(encStride.k + b * th, th);
        const sV = this.fwd(encStride.v + b * th, th);
        const sP = this.fwd(A.encSavedP + b * hmEnc, hmEnc);
        const sH1 = this.fwd(A.encSavedH1 + b * tf, tf);

        // FFN branch (same shape as the mixer loop; SGD deferred to the end
        // of the block so every gradient uses the pre-update weights).
        encoder.compute((pass) => pass.run("matmul_backward_weight", {
          inputOffset: dFieldStates, auxOffset: sH1, outputOffset: dW2,
          tokenCount: t, inputDim: h, outputDim: ffn,
        }));
        encoder.compute((pass) => pass.run("matmul_backward_input", {
          inputOffset: dFieldStates, outputOffset: dH1,
          tokenCount: t, inputDim: h, outputDim: ffn,
        }, block.w2));
        encoder.compute((pass) => pass.run("relu_backward", {
          inputOffset: sH1, auxOffset: dH1, outputOffset: dH1, tokenCount: tf,
        }));
        encoder.compute((pass) => pass.run("matmul_backward_weight", {
          inputOffset: dH1, auxOffset: sFfnIn, outputOffset: dW1,
          tokenCount: t, inputDim: ffn, outputDim: h,
        }));
        encoder.compute((pass) => pass.run("matmul_backward_input", {
          inputOffset: dH1, outputOffset: dHiddenQ,
          tokenCount: t, inputDim: ffn, outputDim: h,
        }, block.w1));
        encoder.compute((pass) => pass.run("residual_add", {
          inputOffset: dFieldStates, auxOffset: dHiddenQ, outputOffset: dFieldStates, tokenCount: t, inputDim: h,
        }));

        // Attention branch.
        encoder.compute((pass) => pass.run("krystal_attention_backward_scores", {
          inputOffset: dFieldStates, auxOffset: sV, aux2Offset: sP, outputOffset: dScoresEnc,
          aux5Offset: activeTokens, aux6Offset: recordCompactOffset,
          tokenCount: t, inputDim: h, outputDim: headDim,
          u0: t, u1: heads, u2: recordCompactCount, u3: 1,
        }));
        encoder.compute((pass) => pass.run("krystal_attention_backward_qkv", {
          inputOffset: dScoresEnc, auxOffset: sQ, aux2Offset: sK, aux3Offset: sP, aux4Offset: dFieldStates,
          outputOffset: dEncQ, aux5Offset: dEncK, aux6Offset: dEncV,
          tokenCount: t, inputDim: h, outputDim: headDim,
          u0: t, u1: heads, u2: activeTokens, u3: recordCompactOffset, u4: recordCompactCount, u5: 1,
        }));
        encoder.compute((pass) => pass.run("matmul_backward_weight", {
          inputOffset: dEncQ, auxOffset: sIn, outputOffset: dWq,
          tokenCount: t, inputDim: h, outputDim: h,
        }));
        encoder.compute((pass) => pass.run("matmul_backward_weight", {
          inputOffset: dEncK, auxOffset: sIn, outputOffset: dWk,
          tokenCount: t, inputDim: h, outputDim: h,
        }));
        encoder.compute((pass) => pass.run("matmul_backward_weight", {
          inputOffset: dEncV, auxOffset: sIn, outputOffset: dWv,
          tokenCount: t, inputDim: h, outputDim: h,
        }));

        // Input gradients (read the pre-update weight pages).
        encoder.compute((pass) => pass.run("matmul_backward_input", {
          inputOffset: dEncQ, outputOffset: dHiddenK,
          tokenCount: t, inputDim: h, outputDim: h,
        }, block.wq));
        encoder.compute((pass) => pass.run("residual_add", {
          inputOffset: dFieldStates, auxOffset: dHiddenK, outputOffset: dFieldStates, tokenCount: t, inputDim: h,
        }));
        encoder.compute((pass) => pass.run("matmul_backward_input", {
          inputOffset: dEncK, outputOffset: dHiddenV,
          tokenCount: t, inputDim: h, outputDim: h,
        }, block.wk));
        encoder.compute((pass) => pass.run("residual_add", {
          inputOffset: dFieldStates, auxOffset: dHiddenV, outputOffset: dFieldStates, tokenCount: t, inputDim: h,
        }));
        encoder.compute((pass) => pass.run("matmul_backward_input", {
          inputOffset: dEncV, outputOffset: dHiddenQ,
          tokenCount: t, inputDim: h, outputDim: h,
        }, block.wv));
        encoder.compute((pass) => pass.run("residual_add", {
          inputOffset: dFieldStates, auxOffset: dHiddenQ, outputOffset: dFieldStates, tokenCount: t, inputDim: h,
        }));

        // SGD after the whole block's backward (pre-update-weight gradients).
        if (applySgd) {
          encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dW2, tokenCount: h * ffn, f0: lr }, block.w2));
          encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dW1, tokenCount: ffn * h, f0: lr }, block.w1));
          encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dWq, tokenCount: h * h, f0: lr }, block.wq));
          encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dWk, tokenCount: h * h, f0: lr }, block.wk));
          encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dWv, tokenCount: h * h, f0: lr }, block.wv));
        }
      }

      // 9. Fused sparse field-embedding backward + SGD. The standalone dense
      //    gradient pass remains available for parity/debug tests, while the
      //    composed runner touches only rows referenced by this frame. A host
      //    doing its own update takes dFieldStates instead and scatters it
      //    into those rows itself, so the fused pass does not run.
      if (applySgd) {
        encoder.compute((pass) => pass.run("krystal_field_embed_sgd", {
          inputOffset: dFieldStates, outputOffset: sparseEmbeddingRowsOff,
          auxOffset: tokenIds, aux2Offset: fieldRoles, aux3Offset: schemaIds,
          aux4Offset: bandIds, aux5Offset: streamIds, aux6Offset: activeTokens,
          tokenCount: sparseEmbeddingRows.length, inputDim: h, outputDim: t, f0: lr,
          u0: embCum[0], u1: embCum[1], u2: embCum[2], u3: embCum[3], u4: embCum[4], u5: embCum[5],
        }, pages.embeddings));
      }
    });

    if (!options.telemetry) return { step };
    await device.queue.onSubmittedWorkDone();
    // Only what this step actually computed. A cross-entropy over labels
    // nobody supplied would be a number about nothing, and reporting the
    // previous step's reduction as this one's is worse than reporting none.
    const loss = routeKinds ? await this.readLoss() : undefined;
    const valueLoss = options.valenceTarget === undefined
      ? undefined
      : (await this.readRegion(valueLossRows, q)).reduce((sum, value) => sum + value, 0) / q;
    return { step, ...(loss === undefined ? {} : { loss }), ...(valueLoss === undefined ? {} : { valueLoss }) };
  }

  /** Mean of a small row vector, for the losses this runner reduces on the host. */
  private readRegion(offset: number, elements: number): Promise<Float32Array> {
    return this.forward.readRegions([[offset, elements]]);
  }

  /**
   * The gradients a host applying its own update needs, and the ones parity is
   * checked on. Read after a step; SGD writes the weight PAGES, never these
   * regions, so they hold this step's gradient whether or not it was applied.
   */
  async readGradients(dims: { readonly t: number; readonly r: number; readonly q: number }): Promise<{
    readonly dSelectorWq: Float32Array;
    readonly dSelectorWk: Float32Array;
    readonly dPool: Float32Array;
    readonly dFieldStates: Float32Array;
    readonly dValueWv: Float32Array;
    readonly dBankKeys: Float32Array;
    readonly dBankValues: Float32Array;
    readonly dQueryValues: Float32Array;
  }> {
    const h = this.forward.getConfig().hiddenSize;
    const B = KRYSTAL_BACKWARD_ARENA;
    const { t, r, q } = dims;
    const sizes = [h * h, h * h, 2 * h, t * h, 3 * h, r * h, r * h, q * h];
    // The mixed-query gradient accumulator aliases the decision-head region.
    const offsets = [B.dSelectorWq, B.dSelectorWk, B.dPool, B.dFieldStates, B.dValueWv,
      B.dBankKeys, B.dBankValues, B.dDecisionQuery];
    const raw = await this.forward.readRegions(
      offsets.map((offset, index) => [this.bwd(offset, sizes[index]!), sizes[index]!] as const),
    );
    let at = 0;
    const next = (elements: number): Float32Array => raw.subarray(at, (at += elements));
    return {
      dSelectorWq: next(sizes[0]!), dSelectorWk: next(sizes[1]!), dPool: next(sizes[2]!),
      dFieldStates: next(sizes[3]!), dValueWv: next(sizes[4]!), dBankKeys: next(sizes[5]!),
      dBankValues: next(sizes[6]!), dQueryValues: next(sizes[7]!),
    };
  }

  /**
   * What a host applying its own update needs from one step, in one map: the
   * gradients of the parts it may move, and the forward's reading of the frame
   * — the policy it differentiated and the critic's prediction — which are what
   * its report is made of. Read after a `trainStep` with `optimizer: "none"`;
   * the value loss is the mean of the per-row losses that step wrote, and zero
   * when the frame carried no target, exactly as the CPU oracle reports it.
   */
  async readHostGradients(dims: { readonly t: number; readonly r: number; readonly q: number }): Promise<BackwardResult> {
    const h = this.forward.getConfig().hiddenSize;
    const A = FWD;
    const B = KRYSTAL_BACKWARD_ARENA;
    const { t, r, q } = dims;
    const copies: (readonly [number, number])[] = [
      [this.fwd(A.intentP, q * r), q * r],
      [this.fwd(A.valuePrediction, q), q],
      [this.bwd(B.valueLossRows, q), q],
      [this.bwd(B.dSelectorWq, h * h), h * h],
      [this.bwd(B.dSelectorWk, h * h), h * h],
      [this.bwd(B.dPool, 2 * h), 2 * h],
      [this.bwd(B.dFieldStates, t * h), t * h],
      [this.bwd(B.dValueWv, 3 * h), 3 * h],
    ];
    const raw = await this.forward.readRegions(copies);
    let at = 0;
    const next = (elements: number): Float32Array => raw.subarray(at, (at += elements));
    const policy = next(q * r);
    const valuePrediction = next(q);
    const lossRows = next(q);
    let valueLoss = 0;
    for (const row of lossRows) valueLoss += row;
    valueLoss = q ? valueLoss / q : 0;
    return {
      policy,
      valuePrediction,
      valueLoss,
      dSelectorWq: next(h * h),
      dSelectorWk: next(h * h),
      dPool: next(2 * h),
      dFieldStates: next(t * h),
      dValueWv: next(3 * h),
    };
  }

  /** Read the scalar mean loss produced by the last trainStep. Debug/test-only. */
  async readLoss(): Promise<number> {
    const value = await this.definition.resources.lossTelemetry.readback();
    if (typeof value === "number") return value;
    return (value as unknown as Float32Array)[0]!;
  }

  get currentStep(): number {
    return this.step;
  }
}
