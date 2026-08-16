// M3 close (WEBGPU_BACKWARD_PLAN.md §17 item 10): the composed Krystal
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
  LFM2_TRAINING_ARENA,
  TRAINING_ARENA_BASE,
} from "./lfm2-layout";
import { type Lfm2Definition } from "./lfm2";
import {
  KrystalForward,
  type BrainForwardWeightPages,
  type PreparedForward,
  type SelectionMasks,
} from "./krystal-forward";
import { EMBEDDING_TABLES } from "../../krystal/src/forward/model";
import type { v1_0_0 } from "../../schema/generated/krystal.types";

function validate(condition: boolean, message: string): void {
  if (!condition) throw new Error(`KrystalBackward: ${message}`);
}

export interface KrystalTrainStepOptions {
  readonly frame: v1_0_0.BrainFrameGpu;
  readonly selection?: SelectionMasks;
  /** Route-kind gold labels [Q] for the decision-head cross-entropy loss. */
  readonly routeKinds: readonly number[] | Uint32Array;
  /**
   * Optional pointer-loss targets [Q] for the argument selector slot
   * (0xffffffff = no pointer loss for that row; default: none).
   */
  readonly argGold?: readonly number[] | Uint32Array;
  readonly learningRate: number;
  /** Read back the scalar mean loss (compact telemetry); off by default. */
  readonly telemetry?: boolean;
}

export interface KrystalTrainStepResult {
  readonly step: number;
  readonly loss?: number;
}

/** Shared arena capacity for the save slices (must match the forward's dims). */
const FWD = KRYSTAL_FORWARD_ARENA;

export class KrystalBackward {
  private readonly forward: KrystalForward;
  private readonly definition: Lfm2Definition;
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
    validate(offset + elements <= LFM2_TRAINING_ARENA.elements, "training arena region overflows capacity");
    return TRAINING_ARENA_BASE + offset;
  }

  /** Run one GPU-resident training step: forward + loss + backward + SGD. */
  async trainStep(options: KrystalTrainStepOptions): Promise<KrystalTrainStepResult> {
    const config = this.forward.getConfig();
    const { hiddenSize: h, ffnSize: ffn, headCount: heads, headDim, encoderBlocks, mixerBlocks, routeKindCount: C } = config;
    const A = FWD;
    const B = KRYSTAL_BACKWARD_ARENA;
    const T = LFM2_TRAINING_ARENA;

    const prepared = this.forward.prepare(options.frame, options.selection);
    const { t, r, q } = prepared;
    const routeKinds = options.routeKinds instanceof Uint32Array
      ? options.routeKinds
      : Uint32Array.from(options.routeKinds);
    validate(routeKinds.length === q, `routeKinds must be [Q] = ${q}`);
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
    device.queue.writeBuffer(this.definition.resources.targets.gpu, 0, routeKinds);
    device.queue.writeBuffer(arena, selectorGold * 4, noTargets);
    const argTargets = options.argGold
      ? (options.argGold instanceof Uint32Array ? options.argGold : Uint32Array.from(options.argGold))
      : noTargets;
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

    const lr = options.learningRate;
    const pages = this.pages;
    const step = ++this.step;

    this.forward.submitPrepared(prepared, true, (encoder) => {
      // 1. Cross-entropy over route-kind logits + telemetry reduction.
      encoder.compute((pass) => pass.run("cross_entropy_forward_backward", {
        inputOffset: decisionLogits, outputOffset: dDecisionLogits, auxOffset: lossRows,
        tokenCount: q, outputDim: C, u1: 0,
      }));
      encoder.compute((pass) => pass.run("loss_reduce", {
        inputOffset: lossRows, outputOffset: scalarLoss, tokenCount: q,
      }));

      // 2. Decision head backward + SGD on Wh.
      encoder.compute((pass) => pass.run("krystal_decision_head_backward", {
        inputOffset: dDecisionLogits, auxOffset: queryValues, aux2Offset: intentGather, aux3Offset: argGather,
        outputOffset: dDecisionQuery, aux4Offset: dDecisionIntent, aux5Offset: dDecisionArg, aux6Offset: dDecisionWh,
        tokenCount: q, inputDim: h, outputDim: C,
      }, pages.decisionHead));
      encoder.compute((pass) => pass.run("sgd_step", {
        inputOffset: dDecisionWh, tokenCount: C * 3 * h, f0: lr,
      }, pages.decisionHead));

      // 3. Zero the bank gradient accumulators (selector + mixer accumulate).
      encoder.compute((pass) => pass.run("zero_f32", { outputOffset: dBankKeys, tokenCount: rh }));
      encoder.compute((pass) => pass.run("zero_f32", { outputOffset: dBankValues, tokenCount: rh }));

      // 4. Selector backward, both slots (soft gather + optional pointer loss).
      //    Intent slot writes the accumulators directly; the argument slot
      //    writes scratch (dEncQ/K/V) then residual-adds into them.
      for (const [dGather, pOff, goldOff, scratch] of [
        [dDecisionIntent, intentP, selectorGold, false],
        [dDecisionArg, argP, argGoldOff, true],
      ] as const) {
        const dQOut = scratch ? dEncQ : dSelectorQProj;
        const dKOut = scratch ? dEncK : dSelectorKProj;
        const dVOut = scratch ? dEncV : dSelectorValue;
        encoder.compute((pass) => pass.run("krystal_selector_backward_scores", {
          inputOffset: dGather, auxOffset: bankValues, aux2Offset: pOff, aux3Offset: goldOff,
          outputOffset: dSelectorScores,
          tokenCount: q, inputDim: h, u0: r,
        }));
        encoder.compute((pass) => pass.run("krystal_selector_backward_qkv", {
          inputOffset: dSelectorScores, auxOffset: selectorQ, aux2Offset: selectorK,
          aux3Offset: pOff, aux4Offset: dGather,
          outputOffset: dQOut, aux5Offset: dKOut, aux6Offset: dVOut,
          tokenCount: q, inputDim: h, u0: r,
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
      encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dSelectorWq, tokenCount: h * h, f0: lr }, pages.selectorWq));
      encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dSelectorWk, tokenCount: h * h, f0: lr }, pages.selectorWk));

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
        encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dW2, tokenCount: h * ffn, f0: lr }, block.w2));
        encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dW1, tokenCount: ffn * h, f0: lr }, block.w1));
        encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dWq, tokenCount: h * h, f0: lr }, block.wq));
        encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dWk, tokenCount: h * h, f0: lr }, block.wk));
        encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dWv, tokenCount: h * h, f0: lr }, block.wv));
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
      encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dPool, tokenCount: 2 * h, f0: lr }, pages.pool));

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
          tokenCount: t, inputDim: h, outputDim: headDim, u0: t, u1: heads,
        }));
        encoder.compute((pass) => pass.run("krystal_attention_backward_qkv", {
          inputOffset: dScoresEnc, auxOffset: sQ, aux2Offset: sK, aux3Offset: sP, aux4Offset: dFieldStates,
          outputOffset: dEncQ, aux5Offset: dEncK, aux6Offset: dEncV,
          tokenCount: t, inputDim: h, outputDim: headDim, u0: t, u1: heads,
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
        encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dW2, tokenCount: h * ffn, f0: lr }, block.w2));
        encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dW1, tokenCount: ffn * h, f0: lr }, block.w1));
        encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dWq, tokenCount: h * h, f0: lr }, block.wq));
        encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dWk, tokenCount: h * h, f0: lr }, block.wk));
        encoder.compute((pass) => pass.run("sgd_step", { inputOffset: dWv, tokenCount: h * h, f0: lr }, block.wv));
      }

      // 9. Field embedding backward (scatter-add over the whole concatenated
      //    page) + SGD on the embeddings.
      encoder.compute((pass) => pass.run("krystal_field_embed_backward", {
        inputOffset: dFieldStates, outputOffset: this.bwd(B.dEmbedding, dEmbeddingElems),
        auxOffset: tokenIds, aux2Offset: fieldRoles, aux3Offset: schemaIds,
        aux4Offset: bandIds, aux5Offset: streamIds, aux6Offset: activeTokens,
        tokenCount: embRows, inputDim: h, outputDim: t,
        u0: embCum[0], u1: embCum[1], u2: embCum[2], u3: embCum[3], u4: embCum[4], u5: embCum[5],
      }));
      encoder.compute((pass) => pass.run("sgd_step", {
        inputOffset: this.bwd(B.dEmbedding, dEmbeddingElems), tokenCount: dEmbeddingElems, f0: lr,
      }, pages.embeddings));
    });

    if (!options.telemetry) return { step };
    await device.queue.onSubmittedWorkDone();
    const loss = await this.readLoss();
    return { step, loss };
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

