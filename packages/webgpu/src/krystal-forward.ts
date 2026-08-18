// M2b Krystal forward runner: wires the packed SoA BrainFrameGpu (M2a) into
// the record/query encoder and query-to-record mixer, GPU-resident in one
// submit (WEBGPU_BACKWARD_PLAN.md §17 item 7, first half).
//
//   field embed (6 additive tables) -> fieldStates
//   -> 2 encoder blocks: local masked self-attention + ReLU FFN
//   -> learned-query pooling -> bank keys/values + query states
//   -> 2 mixer blocks: query -> bank cross-attention + ReLU FFN
//   -> mixed query output + record bank
//
// Dispatch mirrors the oracle in packages/krystal/src/forward/oracle.ts; the
// parity test compares the two on identical packed frames. Masks and active
// lists are compiled on the host (concerns answers 15/16) from the frame's
// ABI metadata. Forward-only: M3 adds backward for these ops.
//
// The SoA frame payloads are u32 and are uploaded into the shared f32 arena
// (bitcast inside the shaders). Weight pages are created once in the
// constructor from the deterministic host initialization.
import {
  KRYSTAL_FORWARD_ARENA,
  KRYSTAL_FORWARD_ARENA_BASE,
  KRYSTAL_MAX_FFN,
  KRYSTAL_MAX_HEADS,
  KRYSTAL_MAX_H,
  KRYSTAL_MAX_QUERIES,
  KRYSTAL_MAX_RECORDS,
  KRYSTAL_MAX_TOKENS,
  TRAINING_READBACK_ELEMENTS,
} from "./krystal-layout";
import { krystal, type KrystalDefinition } from "./krystal";
import { KrystalExecutor, type KrystalCommandEncoder } from "./pass";
import {
  compileActiveFrame,
  compileMixerMask,
  compileRecordMask,
  type WordBias,
  type ActiveFrame,
} from "../../krystal/src/forward/masks";
import {
  BRAIN_FORWARD_CONFIG,
  embeddingTableBases,
  validateBrainForwardWeights,
  type BrainForwardConfig,
  type BrainForwardWeights,
} from "../../krystal/src/forward/model";
import type { v1_0_0 } from "../../schema/generated/krystal.types";

function validate(condition: boolean, message: string): void {
  if (!condition) throw new Error(`KrystalForward: ${message}`);
}

/**
 * Host-compiled selector masks ([Q, R] each; architecture v2 §7). When both
 * are provided, the intent and argument selectors run after the mixer;
 * otherwise selection is skipped (encoder + mixer only).
 */
export interface SelectionMasks {
  readonly intentMask: Float32Array;
  readonly argMask: Float32Array;
}

/**
 * Host-side state of one prepared frame: SoA payloads + masks already
 * uploaded into the arena, plus the compiled active dimensions. The composed
 * backward runner prepares once and reuses the same uploads for its forward.
 */
export interface PreparedForward {
  readonly frame: v1_0_0.BrainFrameGpu;
  readonly active: ActiveFrame;
  readonly selection?: SelectionMasks;
  readonly t: number; // active tokens
  readonly r: number; // bank records
  readonly q: number; // query records
}

/** Trainable weight pages owned by the forward runner (shared with backward). */
export interface BrainForwardWeightPages {
  readonly embeddings: GPUBuffer;
  readonly enc: { wq: GPUBuffer; wk: GPUBuffer; wv: GPUBuffer; w1: GPUBuffer; w2: GPUBuffer }[];
  readonly pool: GPUBuffer;
  readonly mixer: { wq: GPUBuffer; wk: GPUBuffer; wv: GPUBuffer; w1: GPUBuffer; w2: GPUBuffer }[];
  readonly selectorWq: GPUBuffer;
  readonly selectorWk: GPUBuffer;
  readonly decisionHead: GPUBuffer;
}

export class KrystalForward {
  private readonly definition: KrystalDefinition;
  private readonly config: BrainForwardConfig;
  private readonly executor: KrystalExecutor;

  private readonly embeddingsPage: GPUBuffer;
  private readonly encPages: { wq: GPUBuffer; wk: GPUBuffer; wv: GPUBuffer; w1: GPUBuffer; w2: GPUBuffer }[];
  private readonly poolPage: GPUBuffer;
  private readonly mixerPages: { wq: GPUBuffer; wk: GPUBuffer; wv: GPUBuffer; w1: GPUBuffer; w2: GPUBuffer }[];
  private readonly selectorWqPage: GPUBuffer;
  private readonly selectorWkPage: GPUBuffer;
  private readonly decisionHeadPage: GPUBuffer;

  constructor(
    weights: BrainForwardWeights,
    config: BrainForwardConfig = BRAIN_FORWARD_CONFIG,
    definition: KrystalDefinition = krystal,
  ) {
    validateBrainForwardWeights(config, weights);
    validate(config.hiddenSize <= KRYSTAL_MAX_H, `hiddenSize ${config.hiddenSize} exceeds capacity`);
    validate(config.ffnSize <= KRYSTAL_MAX_FFN, `ffnSize ${config.ffnSize} exceeds capacity`);
    validate(config.headCount <= KRYSTAL_MAX_HEADS, `headCount ${config.headCount} exceeds capacity`);
    this.config = config;
    this.definition = definition;
    this.executor = new KrystalExecutor(definition);

    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const device = definition.engine.device;
    const page = (label: string, values: Float32Array): GPUBuffer => {
      const buffer = device.createBuffer({ label, size: Math.max(4, values.byteLength), usage });
      device.queue.writeBuffer(buffer, 0, values);
      return buffer;
    };
    const blockPages = (label: string, block: BrainForwardWeights["enc"][number]) => ({
      wq: page(`${label}.wq`, block.wq),
      wk: page(`${label}.wk`, block.wk),
      wv: page(`${label}.wv`, block.wv),
      w1: page(`${label}.w1`, block.w1),
      w2: page(`${label}.w2`, block.w2),
    });

    this.embeddingsPage = page("krystal.embeddings", weights.embeddings);
    this.encPages = weights.enc.map((block, b) => blockPages(`krystal.enc${b}`, block));
    this.poolPage = page("krystal.pool", weights.pool);
    this.mixerPages = weights.mixer.map((block, b) => blockPages(`krystal.mixer${b}`, block));
    this.selectorWqPage = page("krystal.selector.wq", weights.selector.wq);
    this.selectorWkPage = page("krystal.selector.wk", weights.selector.wk);
    this.decisionHeadPage = page("krystal.decision-head", weights.decisionHeadWh);
  }

  private region(offset: number, elements: number): number {
    validate(
      offset + elements <= KRYSTAL_FORWARD_ARENA.elements,
      "krystal forward arena region overflows the declared capacity",
    );
    return KRYSTAL_FORWARD_ARENA_BASE + offset;
  }

  /**
   * Upload the SoA payloads + host-compiled masks for one packed frame and
   * return the compiled dimensions (t/r/q). The uploads live in the shared
   * arena, so the composed backward runner prepares once and reuses them.
   */
  prepare(
    frame: v1_0_0.BrainFrameGpu,
    selection?: SelectionMasks,
    wordBias?: WordBias,
  ): PreparedForward {
    const { hiddenSize: h, ffnSize: ffn, headCount: heads, headDim, encoderBlocks, mixerBlocks } = this.config;
    const A = KRYSTAL_FORWARD_ARENA;
    const active = compileActiveFrame(frame);
    const t = active.activeTokens.length;
    const r = active.bankRecords.length;
    const q = active.queryRecords.length;
    validate(t <= KRYSTAL_MAX_TOKENS, `active tokens ${t} exceed capacity`);
    validate(r <= KRYSTAL_MAX_RECORDS, `bank records ${r} exceed capacity`);
    validate(q <= KRYSTAL_MAX_QUERIES, `query records ${q} exceed capacity`);
    validate(q > 0, "the frame must contain at least one query record for the mixer");

    const { mask: recordMask } = compileRecordMask(active.activeTokens, wordBias);
    const mixerMask = compileMixerMask(frame, active);
    if (selection) {
      validate(
        selection.intentMask.length === q * r && selection.argMask.length === q * r,
        `selector masks must be [${q}, ${r}]`,
      );
    }

    const device = this.definition.engine.device;
    const arena = this.definition.resources.arena.gpu;
    const uploadU32 = (offset: number, values: Uint32Array): void => {
      device.queue.writeBuffer(arena, this.region(offset, values.length) * 4, values);
    };
    const uploadF32 = (offset: number, values: Float32Array): void => {
      device.queue.writeBuffer(arena, this.region(offset, values.length) * 4, values);
    };

    // SoA frame payloads + host-compiled active lists (u32 in the f32 arena).
    uploadU32(A.tokenIds, Uint32Array.from(frame.tokenIds));
    uploadU32(A.fieldRoles, Uint32Array.from(frame.fieldRoles));
    uploadU32(A.schemaIds, Uint32Array.from(frame.schemaIds));
    uploadU32(A.bandIds, Uint32Array.from(frame.bandIds));
    uploadU32(A.streamIds, active.streamIds);
    uploadU32(A.activeTokens, active.activeTokens);
    uploadU32(A.recordCompactOffset, active.recordCompactOffset);
    uploadU32(A.recordCompactCount, active.recordCompactCount);
    uploadU32(A.bankIndices, active.bankRecords);
    uploadU32(A.queryIndices, active.queryRecords);
    uploadF32(A.encMask, recordMask);
    uploadF32(A.mixerMask, mixerMask);
    if (selection) {
      uploadF32(A.intentMask, selection.intentMask);
      uploadF32(A.argMask, selection.argMask);
    }

    return { frame, active, selection, t, r, q };
  }

  /**
   * Dispatch the whole encoder + mixer + selection pipeline for a prepared
   * frame. When `save` is set, per-block activations (block inputs, Q/K/V
   * projections, attention probs, post-ReLU FFN states) are written into the
   * stacked save regions for the composed backward runner; otherwise the
   * scratch regions are reused per block as in the plain forward.
   */
  dispatchForward(encoder: KrystalCommandEncoder, prepared: PreparedForward, save: boolean): void {
    const { hiddenSize: h, ffnSize: ffn, headCount: heads, headDim, encoderBlocks, mixerBlocks } = this.config;
    const { frame, selection, t, r, q } = prepared;
    const A = KRYSTAL_FORWARD_ARENA;

    const fieldStates = this.region(A.fieldStates, t * h);
    const encQ = this.region(A.encQ, t * h);
    const encK = this.region(A.encK, t * h);
    const encV = this.region(A.encV, t * h);
    const encOut = this.region(A.encOut, t * h);
    const encH1 = this.region(A.encH1, t * ffn);
    const encMask = this.region(A.encMask, t * t);
    const bankKeys = this.region(A.bankKeys, r * h);
    const bankValues = this.region(A.bankValues, r * h);
    const queryKeys = this.region(A.queryKeys, q * h);
    const queryValues = this.region(A.queryValues, q * h);
    const mixerQ = this.region(A.mixerQ, q * h);
    const mixerK = this.region(A.mixerK, r * h);
    const mixerV = this.region(A.mixerV, r * h);
    const mixerH1 = this.region(A.mixerH1, q * ffn);
    const mixed = this.region(A.mixed, q * h);
    const mixerMaskOffset = this.region(A.mixerMask, q * r);
    const selectorQ = this.region(A.selectorQ, q * h);
    const selectorK = this.region(A.selectorK, r * h);
    const intentMaskOffset = this.region(A.intentMask, q * r);
    const argMaskOffset = this.region(A.argMask, q * r);
    const intentP = this.region(A.intentP, q * r);
    const intentGather = this.region(A.intentGather, q * h);
    const intentIndices = this.region(A.intentIndices, q);
    const argP = this.region(A.argP, q * r);
    const argGather = this.region(A.argGather, q * h);
    const argIndices = this.region(A.argIndices, q);
    const bankIndices = this.region(A.bankIndices, r);
    const queryIndices = this.region(A.queryIndices, q);
    const recordCompactOffset = this.region(A.recordCompactOffset, 128);
    const recordCompactCount = this.region(A.recordCompactCount, 128);

    // Per-block save slices for the composed backward runner (M3 close).
    const th = KRYSTAL_MAX_TOKENS * KRYSTAL_MAX_H;
    const tf = KRYSTAL_MAX_TOKENS * KRYSTAL_MAX_FFN;
    const hmEnc = KRYSTAL_MAX_HEADS * KRYSTAL_MAX_TOKENS * KRYSTAL_MAX_TOKENS;
    const qh = KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_H;
    const rh = KRYSTAL_MAX_RECORDS * KRYSTAL_MAX_H;
    const qf = KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_FFN;
    const hmMix = KRYSTAL_MAX_HEADS * KRYSTAL_MAX_QUERIES * KRYSTAL_MAX_RECORDS;
    const slice = (base: number, b: number, stride: number, elements: number): number =>
      this.region(base + b * stride, elements);
    const encSave = save
      ? {
          in: (b: number) => slice(A.encSavedIn, b, th, t * h),
          ffnIn: (b: number) => slice(A.encSavedFfnIn, b, th, t * h),
          q: (b: number) => slice(A.encSavedQ, b, th, t * h),
          k: (b: number) => slice(A.encSavedK, b, th, t * h),
          v: (b: number) => slice(A.encSavedV, b, th, t * h),
          p: (b: number) => slice(A.encSavedP, b, hmEnc, heads * t * t),
          h1: (b: number) => slice(A.encSavedH1, b, tf, t * ffn),
        }
      : undefined;
    const mixerSave = save
      ? {
          in: (b: number) => slice(A.mixerSavedIn, b, qh, q * h),
          ffnIn: (b: number) => slice(A.mixerSavedFfnIn, b, qh, q * h),
          q: (b: number) => slice(A.mixerSavedQ, b, qh, q * h),
          k: (b: number) => slice(A.mixerSavedK, b, rh, r * h),
          v: (b: number) => slice(A.mixerSavedV, b, rh, r * h),
          p: (b: number) => slice(A.mixerSavedP, b, hmMix, heads * q * r),
          h1: (b: number) => slice(A.mixerSavedH1, b, qf, q * ffn),
        }
      : undefined;

    const bases = embeddingTableBases(this.config);

    // 1. field embed -> fieldStates.
    encoder.compute((pass) => pass.run("krystal_field_embed", {
        inputOffset: this.region(A.tokenIds, 1024),
        auxOffset: this.region(A.fieldRoles, 1024),
        aux2Offset: this.region(A.schemaIds, 128),
        aux3Offset: this.region(A.bandIds, 128),
        aux4Offset: this.region(A.activeTokens, t),
        aux5Offset: this.region(A.streamIds, 128),
        outputOffset: fieldStates,
        tokenCount: t, inputDim: h,
        u0: bases.token, u1: bases.field, u2: bases.schema, u3: bases.band, u4: bases.stream, u5: bases.pos,
      }, this.embeddingsPage));

      // 2. encoder blocks.
      for (let b = 0; b < encoderBlocks; b++) {
        const block = this.encPages[b]!;
        const qOff = encSave ? encSave.q(b) : encQ;
        const kOff = encSave ? encSave.k(b) : encK;
        const vOff = encSave ? encSave.v(b) : encV;
        const pOff = encSave ? encSave.p(b) : this.region(A.encP, heads * t * t);
        const h1Off = encSave ? encSave.h1(b) : encH1;
        if (encSave) {
          encoder.compute((pass) => pass.run("arena_copy", {
            inputOffset: fieldStates, outputOffset: encSave.in(b),
            tokenCount: t, inputDim: h,
          }));
        }
        for (const [page, out] of [[block.wq, qOff], [block.wk, kOff], [block.wv, vOff]] as const) {
          encoder.compute((pass) => pass.run("matmul_f32", {
            inputOffset: fieldStates, outputOffset: out,
            tokenCount: t, inputDim: h, outputDim: h, rowStart: 0, rowCount: h,
          }, page));
        }
        encoder.compute((pass) => pass.run("krystal_attention_forward", {
          inputOffset: qOff, auxOffset: kOff, aux2Offset: vOff, aux3Offset: encMask,
          outputOffset: encOut, aux4Offset: pOff,
          // Encoder attention is block-local by construction. Supplying the
          // compact record ranges lets the shader skip all masked cross-record
          // keys while retaining dense mode for generic/cross attention.
          aux5Offset: this.region(A.activeTokens, t),
          aux6Offset: this.region(A.recordCompactOffset, 128),
          tokenCount: t, inputDim: h, outputDim: headDim,
          u0: t, u1: heads, u2: this.region(A.recordCompactCount, 128), u3: 1,
        }));
        encoder.compute((pass) => pass.run("residual_add", {
          inputOffset: fieldStates, auxOffset: encOut, outputOffset: fieldStates,
          tokenCount: t, inputDim: h,
        }));
        if (encSave) {
          encoder.compute((pass) => pass.run("arena_copy", {
            inputOffset: fieldStates, outputOffset: encSave.ffnIn(b),
            tokenCount: t, inputDim: h,
          }));
        }
        // FFN: relu(x @ W1^T) @ W2^T, residual.
        encoder.compute((pass) => pass.run("matmul_f32", {
          inputOffset: fieldStates, outputOffset: h1Off,
          tokenCount: t, inputDim: h, outputDim: ffn, rowStart: 0, rowCount: ffn,
        }, block.w1));
        encoder.compute((pass) => pass.run("relu", {
          inputOffset: h1Off, outputOffset: h1Off, tokenCount: t * ffn,
        }));
        encoder.compute((pass) => pass.run("matmul_f32", {
          inputOffset: h1Off, outputOffset: encOut,
          tokenCount: t, inputDim: ffn, outputDim: h, rowStart: 0, rowCount: h,
        }, block.w2));
        encoder.compute((pass) => pass.run("residual_add", {
          inputOffset: fieldStates, auxOffset: encOut, outputOffset: fieldStates,
          tokenCount: t, inputDim: h,
        }));
      }

      // 3. learned-query pooling -> bank keys/values + query states.
      encoder.compute((pass) => pass.run("krystal_pool", {
        inputOffset: fieldStates,
        auxOffset: bankIndices,
        aux2Offset: recordCompactOffset,
        aux3Offset: recordCompactCount,
        outputOffset: bankKeys,
        aux4Offset: bankValues,
        tokenCount: r, inputDim: h,
      }, this.poolPage));
      encoder.compute((pass) => pass.run("krystal_pool", {
        inputOffset: fieldStates,
        auxOffset: queryIndices,
        aux2Offset: recordCompactOffset,
        aux3Offset: recordCompactCount,
        outputOffset: queryKeys,
        aux4Offset: queryValues,
        tokenCount: q, inputDim: h,
      }, this.poolPage));

      // 4. mixer blocks (query -> bank cross-attention + ReLU FFN).
      for (let b = 0; b < mixerBlocks; b++) {
        const block = this.mixerPages[b]!;
        const qOff = mixerSave ? mixerSave.q(b) : mixerQ;
        const kOff = mixerSave ? mixerSave.k(b) : mixerK;
        const vOff = mixerSave ? mixerSave.v(b) : mixerV;
        const pOff = mixerSave ? mixerSave.p(b) : this.region(A.mixerP, heads * q * r);
        const h1Off = mixerSave ? mixerSave.h1(b) : mixerH1;
        if (mixerSave) {
          encoder.compute((pass) => pass.run("arena_copy", {
            inputOffset: queryValues, outputOffset: mixerSave.in(b),
            tokenCount: q, inputDim: h,
          }));
        }
        encoder.compute((pass) => pass.run("matmul_f32", {
          inputOffset: queryValues, outputOffset: qOff,
          tokenCount: q, inputDim: h, outputDim: h, rowStart: 0, rowCount: h,
        }, block.wq));
        encoder.compute((pass) => pass.run("matmul_f32", {
          inputOffset: bankKeys, outputOffset: kOff,
          tokenCount: r, inputDim: h, outputDim: h, rowStart: 0, rowCount: h,
        }, block.wk));
        encoder.compute((pass) => pass.run("matmul_f32", {
          inputOffset: bankValues, outputOffset: vOff,
          tokenCount: r, inputDim: h, outputDim: h, rowStart: 0, rowCount: h,
        }, block.wv));
        encoder.compute((pass) => pass.run("krystal_attention_forward", {
          inputOffset: qOff, auxOffset: kOff, aux2Offset: vOff, aux3Offset: mixerMaskOffset,
          outputOffset: mixed, aux4Offset: pOff,
          tokenCount: q, inputDim: h, outputDim: headDim, u0: r, u1: heads,
        }));
        encoder.compute((pass) => pass.run("residual_add", {
          inputOffset: queryValues, auxOffset: mixed, outputOffset: queryValues,
          tokenCount: q, inputDim: h,
        }));
        if (mixerSave) {
          encoder.compute((pass) => pass.run("arena_copy", {
            inputOffset: queryValues, outputOffset: mixerSave.ffnIn(b),
            tokenCount: q, inputDim: h,
          }));
        }
        encoder.compute((pass) => pass.run("matmul_f32", {
          inputOffset: queryValues, outputOffset: h1Off,
          tokenCount: q, inputDim: h, outputDim: ffn, rowStart: 0, rowCount: ffn,
        }, block.w1));
        encoder.compute((pass) => pass.run("relu", {
          inputOffset: h1Off, outputOffset: h1Off, tokenCount: q * ffn,
        }));
        encoder.compute((pass) => pass.run("matmul_f32", {
          inputOffset: h1Off, outputOffset: mixed,
          tokenCount: q, inputDim: ffn, outputDim: h, rowStart: 0, rowCount: h,
        }, block.w2));
        encoder.compute((pass) => pass.run("residual_add", {
          inputOffset: queryValues, auxOffset: mixed, outputOffset: queryValues,
          tokenCount: q, inputDim: h,
        }));
      }

      // 5. catalog selection + soft gather (§7, answer 26): project the mixed
      // query and the bank keys once, then one selector dispatch per slot.
      if (selection) {
        encoder.compute((pass) => pass.run("matmul_f32", {
          inputOffset: queryValues, outputOffset: selectorQ,
          tokenCount: q, inputDim: h, outputDim: h, rowStart: 0, rowCount: h,
        }, this.selectorWqPage));
        encoder.compute((pass) => pass.run("matmul_f32", {
          inputOffset: bankKeys, outputOffset: selectorK,
          tokenCount: r, inputDim: h, outputDim: h, rowStart: 0, rowCount: h,
        }, this.selectorWkPage));
        encoder.compute((pass) => pass.run("krystal_selector", {
          inputOffset: selectorQ, auxOffset: selectorK, aux2Offset: bankValues,
          aux3Offset: intentMaskOffset, outputOffset: intentGather,
          aux4Offset: intentP, aux5Offset: intentIndices,
          tokenCount: q, inputDim: h, u0: r,
        }));
        encoder.compute((pass) => pass.run("krystal_selector", {
          inputOffset: selectorQ, auxOffset: selectorK, aux2Offset: bankValues,
          aux3Offset: argMaskOffset, outputOffset: argGather,
          aux4Offset: argP, aux5Offset: argIndices,
          tokenCount: q, inputDim: h, u0: r,
        }));

        // 6. typed decision head: route-kind logits from the gathered context
        // (query output + intent gather + argument gather; §17 item 9).
        encoder.compute((pass) => pass.run("krystal_decision_head", {
          inputOffset: queryValues, auxOffset: intentGather, aux2Offset: argGather,
          outputOffset: this.region(A.decisionLogits, q * this.config.routeKindCount),
          tokenCount: q, inputDim: h, outputDim: this.config.routeKindCount,
        }, this.decisionHeadPage));
    }
  }

  /** Run the full encoder + mixer + selection forward, GPU-resident. */
  forward(
    frame: v1_0_0.BrainFrameGpu,
    selection?: SelectionMasks,
    wordBias?: WordBias,
  ): void {
    const prepared = this.prepare(frame, selection, wordBias);
    this.executor.submit((encoder) => this.dispatchForward(encoder, prepared, false));
  }

  /**
   * One-submit entry point for the composed backward runner: dispatch the
   * forward (optionally saving per-block activations) and then run the
   * caller's backward/optimizer dispatches in the same submit.
   */
  submitPrepared(
    prepared: PreparedForward,
    save: boolean,
    callback: (encoder: KrystalCommandEncoder) => void,
  ): void {
    this.executor.submit((encoder) => {
      this.dispatchForward(encoder, prepared, save);
      callback(encoder);
    });
  }

  /** Trainable weight pages; shared with the composed backward runner. */
  get weightPages(): BrainForwardWeightPages {
    return {
      embeddings: this.embeddingsPage,
      enc: this.encPages,
      pool: this.poolPage,
      mixer: this.mixerPages,
      selectorWq: this.selectorWqPage,
      selectorWk: this.selectorWkPage,
      decisionHead: this.decisionHeadPage,
    };
  }

  /** Expose the shared definition (composed runner readbacks). */
  getDefinition(): KrystalDefinition {
    return this.definition;
  }

  /** Expose the model config (composed runner dispatch dims). */
  getConfig(): BrainForwardConfig {
    return this.config;
  }

  /** Copy one arena region into the staging buffer and read it back. */
  private async readbackRegion(offset: number, elements: number): Promise<Float32Array> {
    validate(elements <= TRAINING_READBACK_ELEMENTS, `readback region ${elements} exceeds staging capacity`);
    const device = this.definition.engine.device;
    const arena = this.definition.resources.arena;
    const staging = this.definition.resources.trainingReadback;
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(arena.gpu, offset * 4, staging.gpu, 0, elements * 4);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const raw = (await staging.readback()) as unknown as ArrayLike<number>;
    return Float32Array.from(raw).slice(0, elements);
  }

  /** Read back the record bank keys [R, H]. Test-only. */
  async readBankKeys(r: number, h: number): Promise<Float32Array> {
    return this.readbackRegion(this.region(KRYSTAL_FORWARD_ARENA.bankKeys, r * h), r * h);
  }

  /** Read back the record bank values [R, H]. Test-only. */
  async readBankValues(r: number, h: number): Promise<Float32Array> {
    return this.readbackRegion(this.region(KRYSTAL_FORWARD_ARENA.bankValues, r * h), r * h);
  }

  /** Read back the mixed query output [Q, H] (post-mixer). Test-only. */
  async readQueryOutput(q: number, h: number): Promise<Float32Array> {
    return this.readbackRegion(this.region(KRYSTAL_FORWARD_ARENA.queryValues, q * h), q * h);
  }

  /** Read back the encoded field states [T, H]. Test-only. */
  async readFieldStates(t: number, h: number): Promise<Float32Array> {
    return this.readbackRegion(this.region(KRYSTAL_FORWARD_ARENA.fieldStates, t * h), t * h);
  }

  /** Read back the intent selector distribution [Q, R]. Test-only. */
  async readIntentP(q: number, r: number): Promise<Float32Array> {
    return this.readbackRegion(this.region(KRYSTAL_FORWARD_ARENA.intentP, q * r), q * r);
  }

  /** Read back the intent soft-gathered vector [Q, H]. Test-only. */
  async readIntentGather(q: number, h: number): Promise<Float32Array> {
    return this.readbackRegion(this.region(KRYSTAL_FORWARD_ARENA.intentGather, q * h), q * h);
  }

  /** Read back the intent selected bank indices [Q] (u32 payloads). Test-only. */
  async readIntentIndices(q: number): Promise<Uint32Array> {
    const raw = await this.readbackRegion(this.region(KRYSTAL_FORWARD_ARENA.intentIndices, q), q);
    return new Uint32Array(raw.buffer, raw.byteOffset, q);
  }

  /** Read back the argument selector distribution [Q, R]. Test-only. */
  async readArgP(q: number, r: number): Promise<Float32Array> {
    return this.readbackRegion(this.region(KRYSTAL_FORWARD_ARENA.argP, q * r), q * r);
  }

  /** Read back the argument soft-gathered vector [Q, H]. Test-only. */
  async readArgGather(q: number, h: number): Promise<Float32Array> {
    return this.readbackRegion(this.region(KRYSTAL_FORWARD_ARENA.argGather, q * h), q * h);
  }

  /** Read back the argument selected bank indices [Q] (u32 payloads). Test-only. */
  async readDecisionLogits(q: number, c: number): Promise<Float32Array> {
    return this.readbackRegion(this.region(KRYSTAL_FORWARD_ARENA.decisionLogits, q * c), q * c);
  }

  async readArgIndices(q: number): Promise<Uint32Array> {
    const raw = await this.readbackRegion(this.region(KRYSTAL_FORWARD_ARENA.argIndices, q), q);
    return new Uint32Array(raw.buffer, raw.byteOffset, q);
  }

  /**
   * Read back both selector slots ([Q, R] P, [Q, H] gather, [Q] argmax
   * indices) in one pass over the staging buffer. Test-only.
   */
  async readSelection(q: number, r: number, h: number): Promise<{
    readonly intent: {
      readonly p: Float32Array;
      readonly gather: Float32Array;
      readonly index: Uint32Array;
    };
    readonly argument: {
      readonly p: Float32Array;
      readonly gather: Float32Array;
      readonly index: Uint32Array;
    };
  }> {
    const A = KRYSTAL_FORWARD_ARENA;
    const intentP = await this.readbackRegion(this.region(A.intentP, q * r), q * r);
    const intentGather = await this.readbackRegion(this.region(A.intentGather, q * h), q * h);
    const intentIdxRaw = await this.readbackRegion(this.region(A.intentIndices, q), q);
    const argP = await this.readbackRegion(this.region(A.argP, q * r), q * r);
    const argGather = await this.readbackRegion(this.region(A.argGather, q * h), q * h);
    const argIdxRaw = await this.readbackRegion(this.region(A.argIndices, q), q);
    return {
      intent: {
        p: intentP,
        gather: intentGather,
        index: new Uint32Array(intentIdxRaw.buffer, intentIdxRaw.byteOffset, q),
      },
      argument: {
        p: argP,
        gather: argGather,
        index: new Uint32Array(argIdxRaw.buffer, argIdxRaw.byteOffset, q),
      },
    };
  }

  destroy(): void {
    this.embeddingsPage.destroy();
    this.poolPage.destroy();
    this.selectorWqPage.destroy();
    this.selectorWkPage.destroy();
    for (const block of [...this.encPages, ...this.mixerPages]) {
      block.wq.destroy();
      block.wk.destroy();
      block.wv.destroy();
      block.w1.destroy();
      block.w2.destroy();
    }
  }
}
