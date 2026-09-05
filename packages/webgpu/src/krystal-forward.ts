// M2b Krystal forward runner: wires the packed SoA BrainFrameGpu (M2a) into
// the record/query encoder and query-to-record mixer, GPU-resident in one
// submit (docs/archive/WEBGPU_BACKWARD_PLAN.md §17 item 7, first half).
//
//   field embed (6 additive tables) -> fieldStates
//   -> 2 encoder blocks: local masked self-attention + ReLU FFN
//   -> learned-query pooling -> bank keys/values + query states
//   -> 2 mixer blocks: query -> bank cross-attention + ReLU FFN
//   -> mixed query output + record bank
//
// Dispatch mirrors the oracle in packages/krystal/src/forward/oracle.ts; the
// parity test compares the two on identical packed frames. The active lists
// are compiled from the frame; every MASK comes from the host, including the
// mixer's — this runner used to compile that one itself and thereby disagreed
// with the CPU about what a question was looking at.
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
} from "./krystal-layout";
import { krystal, type KrystalDefinition } from "./krystal";
import { KrystalExecutor, type KrystalCommandEncoder } from "./pass";
import {
  compileActiveFrame,
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
import type { WeightChanges } from "../../krystal/src/host/backend";
import type { v1_0_0 } from "../../schema/generated/krystal.types";
import { BRAIN_LIMITS } from "../../schema/src/krystal-engine-schema";

function validate(condition: boolean, message: string): void {
  if (!condition) throw new Error(`KrystalForward: ${message}`);
}


/**
 * Every mask the graph reads, and all of them come from the host.
 *
 * The runner used to compile the mixer mask itself, from the frame's runtime
 * references. That was one layer deciding another's business, and it broke the
 * moment the host stopped shipping a reference table: every cell came back
 * blocked, the mixer output collapsed to zero, and the GPU quietly disagreed
 * with the CPU about what the creature was looking at. Masks are the host's
 * grammar; Krystal applies them and never asks why a record was refused.
 *
 * All [Q, R], in the compiled active order (queries x bank records).
 */
export interface KrystalMasks {
  /**
   * What a question may ATTEND to while it thinks. Absent means unconstrained,
   * which is how the host session runs it — what a question may attend to is
   * not what it may CHOOSE, and only the second is grammar.
   */
  readonly mixer?: Float32Array;
  /**
   * What a question may CHOOSE. Absent means selection does not run at all
   * (encoder + mixer only).
   */
  readonly selection?: Float32Array;
  /**
   * The second selector slot's own mask. Absent leaves that slot unconstrained
   * (it scores the whole bank), which is what the oracle computes when nothing
   * shapes it — and the reason the `available` context exists: an unshaped
   * slot averages records the question could never have chosen.
   */
  readonly argument?: Float32Array;
  /**
   * What fills the third block of the context the VALUE head reads.
   *
   * `argument` is the second selector slot's soft gather. `available` replaces
   * it with the mean bank value over what the grammar allows this question —
   * "what is on offer here". A state feature, and that matters: the value head
   * is a REINFORCE baseline, so it must not depend on which action was drawn.
   * It is also a selector projection cheaper, because a selector with a zero
   * query projection already computes that mean.
   */
  readonly context?: "argument" | "available";
}

/**
 * Host-side state of one prepared frame: SoA payloads + masks already
 * uploaded into the arena, plus the compiled active dimensions. The composed
 * backward runner prepares once and reuses the same uploads for its forward.
 */
export interface PreparedForward {
  readonly frame: v1_0_0.BrainFrameGpu;
  readonly active: ActiveFrame;
  readonly masks: KrystalMasks;
  /** Whether the selector slots run at all (the host supplied a grammar). */
  readonly selects: boolean;
  readonly context: "argument" | "available";
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
  readonly valueHead: GPUBuffer;
}

export class KrystalForward {
  private readonly definition: KrystalDefinition;
  /** The profile this runner was built for, including its embedding row table. */
  readonly config: BrainForwardConfig;
  private readonly executor: KrystalExecutor;

  private readonly embeddingsPage: GPUBuffer;
  private readonly encPages: { wq: GPUBuffer; wk: GPUBuffer; wv: GPUBuffer; w1: GPUBuffer; w2: GPUBuffer }[];
  private readonly poolPage: GPUBuffer;
  private readonly mixerPages: { wq: GPUBuffer; wk: GPUBuffer; wv: GPUBuffer; w1: GPUBuffer; w2: GPUBuffer }[];
  private readonly selectorWqPage: GPUBuffer;
  private readonly selectorWkPage: GPUBuffer;
  private readonly decisionHeadPage: GPUBuffer;
  private readonly valueHeadPage: GPUBuffer;
  /** Our own MAP_READ buffer; see readMapped for why not the shared one. */
  private mapped: GPUBuffer | undefined;
  /** One map at a time: a mapped buffer cannot be mapped again. */
  private reads: Promise<unknown> = Promise.resolve();

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
    this.valueHeadPage = page("krystal.value-head", weights.valueHeadWv);
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
    masks: KrystalMasks = {},
    wordBias?: WordBias,
  ): PreparedForward {
    const { hiddenSize: h, ffnSize: ffn, headCount: heads, headDim, encoderBlocks, mixerBlocks } = this.config;
    const A = KRYSTAL_FORWARD_ARENA;
    const active = compileActiveFrame(frame);
    const t = active.activeTokens.length;
    const r = active.bankRecords.length;
    const q = active.queryRecords.length;
    // Slots, not bank records: the per-slot arrays (schema, band, stream, the
    // compact ranges) are indexed by SLOT, and a query record occupies one like
    // any other. Checking the bank alone left `r + q` free to run past the end
    // of those regions and into the next one, silently, on a full frame.
    const slots = frame.schemaIds.length;
    validate(t <= KRYSTAL_MAX_TOKENS, `active tokens ${t} exceed capacity ${KRYSTAL_MAX_TOKENS}`);
    validate(slots <= KRYSTAL_MAX_RECORDS, `record slots ${slots} exceed capacity ${KRYSTAL_MAX_RECORDS}`);
    validate(q <= KRYSTAL_MAX_QUERIES, `query records ${q} exceed capacity ${KRYSTAL_MAX_QUERIES}`);
    validate(q > 0, "the frame must contain at least one query record for the mixer");

    // Uploaded every frame, though the shader reads only each token's own
    // record range and would find zeros everywhere else. Skipping the upload
    // would make correctness depend on nobody else ever writing to this arena
    // region — and the arena is shared by every runner and every test on one
    // engine. That assumption is the same one that made the mixer mask disagree
    // between CPU and GPU, so it stays paid: the mask carries the same-word
    // bias, and at a realistic frame it is 264 KB, not the 9.4 MB of a full one.
    const { mask: recordMask } = compileRecordMask(active.activeTokens, wordBias);
    // Unconstrained by default, and it is the host that decides otherwise.
    const mixerMask = masks.mixer ?? new Float32Array(q * r);
    const context = masks.context ?? "argument";
    for (const [label, mask] of [
      ["mixer", mixerMask],
      ["selection", masks.selection],
      ["argument", masks.argument],
    ] as const) {
      if (mask) validate(mask.length === q * r, `the ${label} mask must be [${q}, ${r}]`);
    }
    validate(
      !(masks.argument && context === "available"),
      "the `available` context REPLACES the second selector slot; supplying an argument mask as well would silently drop it",
    );

    const device = this.definition.engine.device;
    const arena = this.definition.resources.arena.gpu;
    const uploadU32 = (offset: number, values: Uint32Array): void => {
      device.queue.writeBuffer(arena, this.region(offset, values.length) * 4, values);
    };
    const uploadF32 = (offset: number, values: Float32Array): void => {
      device.queue.writeBuffer(arena, this.region(offset, values.length) * 4, values);
    };

    // SoA frame payloads + host-compiled active lists (u32 in the f32 arena).
    // The embed kernel indexes the table directly, so it receives embedding
    // ROWS, not token ids. Projecting on the host keeps the kernel free of the
    // semantic/reference split and means the shader never has to know that a
    // reference token shares a pooled row.
    uploadU32(A.tokenIds, projectRows(frame.tokenIds, this.config.tokenRows));
    uploadU32(A.fieldRoles, projectRows(frame.fieldRoles, this.config.tokenRows));
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
    if (masks.selection) uploadF32(A.intentMask, masks.selection);
    if (context === "argument") uploadF32(A.argMask, masks.argument ?? new Float32Array(q * r));

    return { frame, active, masks, selects: masks.selection !== undefined, context, t, r, q };
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
    const { frame, masks, selects, context, t, r, q } = prepared;
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
    const zeroQuery = this.region(A.zeroQuery, q * h);
    const availableGather = this.region(A.availableGather, q * h);
    const availableP = this.region(A.availableP, q * r);
    const valuePrediction = this.region(A.valuePrediction, q);
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
        inputOffset: this.region(A.tokenIds, BRAIN_LIMITS.frameTokens),
        auxOffset: this.region(A.fieldRoles, BRAIN_LIMITS.frameTokens),
        aux2Offset: this.region(A.schemaIds, BRAIN_LIMITS.frameRecordSlots),
        aux3Offset: this.region(A.bandIds, BRAIN_LIMITS.frameRecordSlots),
        aux4Offset: this.region(A.activeTokens, t),
        aux5Offset: this.region(A.streamIds, BRAIN_LIMITS.frameRecordSlots),
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

      // 5. selection + soft gather: project the mixed query and the bank keys
      // once, then one selector dispatch per slot the host asked for.
      if (selects) {
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

        // The second slot. In the `argument` context it always runs — with an
        // unconstrained mask when the host shaped none, which is what the CPU
        // oracle does with the same input. Under `available` it does not run at
        // all and its gather is explicitly zero, again mirroring the oracle,
        // rather than left as whatever the previous frame wrote.
        if (context === "argument") {
          encoder.compute((pass) => pass.run("krystal_selector", {
            inputOffset: selectorQ, auxOffset: selectorK, aux2Offset: bankValues,
            aux3Offset: argMaskOffset, outputOffset: argGather,
            aux4Offset: argP, aux5Offset: argIndices,
            tokenCount: q, inputDim: h, u0: r,
          }));
        } else {
          encoder.compute((pass) => pass.run("zero_f32", { outputOffset: argGather, tokenCount: q * h }));
          encoder.compute((pass) => pass.run("zero_f32", { outputOffset: argP, tokenCount: q * r }));
        }

        // "What is on offer here": the mean bank value over the records this
        // question is ALLOWED to choose. A selector with a zero query
        // projection computes exactly that — every score is 0 + mask, so the
        // softmax is uniform over the open positions — and an entirely blocked
        // row takes the shader's all-blocked path and gathers zero, which is
        // what the oracle does with an empty allowed set.
        if (context === "available") {
          encoder.compute((pass) => pass.run("zero_f32", { outputOffset: zeroQuery, tokenCount: q * h }));
          encoder.compute((pass) => pass.run("krystal_selector", {
            inputOffset: zeroQuery, auxOffset: selectorK, aux2Offset: bankValues,
            aux3Offset: intentMaskOffset, outputOffset: availableGather,
            // The pointer output goes to the second slot's index region: that
            // slot does not run under this context, and a mean has no choice
            // to record anyway.
            aux4Offset: availableP, aux5Offset: this.region(A.argIndices, q),
            tokenCount: q, inputDim: h, u0: r,
          }));
        }

        // 6. typed decision head: route-kind logits from the gathered context
        // (query output + intent gather + argument gather; §17 item 9).
        encoder.compute((pass) => pass.run("krystal_decision_head", {
          inputOffset: queryValues, auxOffset: intentGather, aux2Offset: argGather,
          outputOffset: this.region(A.decisionLogits, q * this.config.routeKindCount),
          tokenCount: q, inputDim: h, outputDim: this.config.routeKindCount,
        }, this.decisionHeadPage));

        // 7. value head: the same head with a single class, reading the same
        // context except for its third block. Sharing the first two is what
        // lets the value signal shape the representation rather than only its
        // own head — whatever the encoder and mixer produce has to serve both
        // "which one" and "how will this turn out".
        encoder.compute((pass) => pass.run("krystal_decision_head", {
          inputOffset: queryValues, auxOffset: intentGather,
          aux2Offset: context === "available" ? availableGather : argGather,
          outputOffset: valuePrediction,
          tokenCount: q, inputDim: h, outputDim: 1,
        }, this.valueHeadPage));
      }
  }


  /** Run the full encoder + mixer + selection forward, GPU-resident. */
  forward(
    frame: v1_0_0.BrainFrameGpu,
    masks: KrystalMasks = {},
    wordBias?: WordBias,
  ): void {
    const prepared = this.prepare(frame, masks, wordBias);
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
      valueHead: this.valueHeadPage,
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

  /**
   * Read arena regions back through a buffer we own and map ourselves.
   *
   * The shared `trainingReadback` resource goes through Sandblaster's generic
   * `readback()`, which copies a SECOND time into its own staging buffer, waits
   * on a second submit, maps the WHOLE resource — 524,288 elements — and then
   * builds a plain JS array one `codec.deserialize` call per element. Half a
   * million of those to fetch a hundred kilobytes. On Dawn in-process that cost
   * a few milliseconds; in a browser, where every submit and map crosses into
   * the GPU process, it was most of a frame.
   *
   * `copies` are (arenaOffset, elements) pairs laid out back to back in the
   * result. `dispatch`, when given, runs in the SAME submit as the copies —
   * which is the other half of the fix: one submit and one map per encode
   * instead of three and two.
   */
  private readMapped(
    copies: readonly (readonly [offset: number, elements: number])[],
    dispatch?: (encoder: KrystalCommandEncoder) => void,
  ): Promise<Float32Array> {
    const total = copies.reduce((sum, [, elements]) => sum + elements, 0);
    const bytes = total * 4;
    const device = this.definition.engine.device;
    const arena = this.definition.resources.arena.gpu;

    const run = async (): Promise<Float32Array> => {
      if (!this.mapped || this.mapped.size < bytes) {
        this.mapped?.destroy();
        this.mapped = device.createBuffer({
          label: "krystal.readback",
          size: Math.max(bytes, 4),
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
      }
      const target = this.mapped;
      const emit = (encoder: GPUCommandEncoder): void => {
        let at = 0;
        for (const [offset, elements] of copies) {
          encoder.copyBufferToBuffer(arena, offset * 4, target, at * 4, elements * 4);
          at += elements;
        }
      };
      if (dispatch) {
        this.executor.submit((encoder) => {
          dispatch(encoder);
          emit(encoder.gpu);
        });
      } else {
        const encoder = device.createCommandEncoder();
        emit(encoder);
        device.queue.submit([encoder.finish()]);
      }
      await target.mapAsync(GPUMapMode.READ, 0, bytes);
      const out = new Float32Array(total);
      out.set(new Float32Array(target.getMappedRange(0, bytes)));
      target.unmap();
      return out;
    };

    const task = this.reads.then(run, run);
    this.reads = task.catch(() => undefined);
    return task;
  }

  /** Copy one arena region into a mapped buffer and read it back. */
  private async readbackRegion(offset: number, elements: number): Promise<Float32Array> {
    return this.readMapped([[offset, elements]]);
  }

  /**
   * Read several arena regions in one submit and one map, laid out back to
   * back. The composed backward runner reads its gradients through this rather
   * than through the shared readback resource, for the reason readMapped gives.
   */
  readRegions(copies: readonly (readonly [offset: number, elements: number])[]): Promise<Float32Array> {
    return this.readMapped(copies);
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

  /**
   * Encode one prepared frame and bring back everything a question needs.
   *
   * One submit, one map. The three matrices are the encoder's whole product —
   * the mixed query output and the pooled bank — and every `choose` against
   * this frame is a function of them and the host's mask.
   */
  async encodeAndRead(prepared: PreparedForward): Promise<{
    readonly queryOutput: Float32Array;
    readonly bankKeys: Float32Array;
    readonly bankValues: Float32Array;
  }> {
    const A = KRYSTAL_FORWARD_ARENA;
    const h = this.config.hiddenSize;
    const qh = prepared.q * h;
    const rh = prepared.r * h;
    const raw = await this.readMapped(
      [
        [this.region(A.queryValues, qh), qh],
        [this.region(A.bankKeys, rh), rh],
        [this.region(A.bankValues, rh), rh],
      ],
      (encoder) => this.dispatchForward(encoder, prepared, false),
    );
    return {
      queryOutput: raw.subarray(0, qh),
      bankKeys: raw.subarray(qh, qh + rh),
      bankValues: raw.subarray(qh + rh),
    };
  }

  /**
   * Re-upload the weight pages from the host arrays they were built from.
   *
   * The pages are a COPY: `learn` and `teach` mutate the host's Float32Arrays
   * and the device knows nothing about it. Without this, a creature would keep
   * thinking with the brain it had before it was taught — and nothing would
   * report the divergence, because both sides are individually consistent.
   */
  uploadWeights(weights: BrainForwardWeights, changes?: WeightChanges): void {
    validateBrainForwardWeights(this.config, weights);
    const device = this.definition.engine.device;
    const write = (buffer: GPUBuffer, values: Float32Array): void => device.queue.writeBuffer(buffer, 0, values);
    const blocks = (): void => {
      weights.enc.forEach((block, b) => {
        const page = this.encPages[b]!;
        write(page.wq, block.wq); write(page.wk, block.wk); write(page.wv, block.wv);
        write(page.w1, block.w1); write(page.w2, block.w2);
      });
      weights.mixer.forEach((block, b) => {
        const page = this.mixerPages[b]!;
        write(page.wq, block.wq); write(page.wk, block.wk); write(page.wv, block.wv);
        write(page.w1, block.w1); write(page.w2, block.w2);
      });
    };
    if (!changes) {
      write(this.embeddingsPage, weights.embeddings);
      write(this.poolPage, weights.pool);
      write(this.selectorWqPage, weights.selector.wq);
      write(this.selectorWkPage, weights.selector.wk);
      write(this.decisionHeadPage, weights.decisionHeadWh);
      write(this.valueHeadPage, weights.valueHeadWv);
      blocks();
      return;
    }
    // Only what moved. An update to the selector and a handful of embedding
    // rows is a few hundred kilobytes; the whole brain is seven megabytes, and
    // it was being sent after every batch and every showing.
    if (changes.selector) {
      write(this.selectorWqPage, weights.selector.wq);
      write(this.selectorWkPage, weights.selector.wk);
    }
    if (changes.pool) write(this.poolPage, weights.pool);
    if (changes.valueHead) write(this.valueHeadPage, weights.valueHeadWv);
    if (changes.decisionHead) write(this.decisionHeadPage, weights.decisionHeadWh);
    if (changes.blocks) blocks();
    if (changes.embeddingRows) {
      const h = this.config.hiddenSize;
      for (const start of changes.embeddingRows)
        device.queue.writeBuffer(this.embeddingsPage, start * 4, weights.embeddings, start, h);
    }
  }

  /** Read back the value head's prediction [Q]. Test-only. */
  async readValuePrediction(q: number): Promise<Float32Array> {
    return this.readbackRegion(this.region(KRYSTAL_FORWARD_ARENA.valuePrediction, q), q);
  }

  /** Read back the mean bank value over what each question may choose [Q, H]. Test-only. */
  async readAvailableGather(q: number, h: number): Promise<Float32Array> {
    return this.readbackRegion(this.region(KRYSTAL_FORWARD_ARENA.availableGather, q * h), q * h);
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
    this.decisionHeadPage.destroy();
    this.valueHeadPage.destroy();
    for (const block of [...this.encPages, ...this.mixerPages]) {
      block.wq.destroy();
      block.wk.destroy();
      block.wv.destroy();
      block.w1.destroy();
      block.w2.destroy();
    }
  }
}

/** Project a token buffer into embedding rows using this profile's table. */
function projectRows(tokenIds: ArrayLike<number>, table: Uint32Array): Uint32Array {
  const rows = new Uint32Array(tokenIds.length);
  for (let i = 0; i < tokenIds.length; i++) rows[i] = table[tokenIds[i]!]!;
  return rows;
}
