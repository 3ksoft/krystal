// ArkType-backed LFM2 definition DSL.
//
// This is the build-time path: it declares the whole typed resource/program
// graph through Sandblaster's scope API and is used by the AOT scripts
// (scripts/build-lfm2-artifact.ts, scripts/validate-lfm2-shaders.ts) to link
// and serialize lfm2.artifact.generated.ts.
//
// The runtime never needs it. packages/webgpu/src/lfm2.ts builds the same
// definition from the already-linked artifact (lfm2-artifact.ts), so this
// module — and with it arktype and `$` — stays out of scriptc-compiled graphs.
import {
  Sandblaster,
  type AnyComputeHandle,
  type BufferResource,
  type BufferResourceUse,
} from "@sandblaster/core";
import { $ } from "../../schema/src/schema";
import {
  VOCAB,
  ARENA_ELEMENTS,
  KV_ELEMENTS,
  CONV_ELEMENTS,
  TOKEN_CAPACITY,
  TELEMETRY_CAPACITY,
  OP_PARAM_BUFFER_BYTES,
  CONSTRAINT_PROGRAM_WORD_CAPACITY,
  CONSTRAINT_TOKENIZER_WORD_CAPACITY,
  CONSTRAINT_MASK_WORDS,
  CONSTRAINT_MASK_WORKGROUP_SIZE,
  LFM2_DEFINITION_PLAIN,
  LFM2_SHADER_NAMES,
  TRAINING_READBACK_ELEMENTS,
  TRAINING_SHADER_NAMES,
  type Lfm2ProgramName,
  type Lfm2ShaderName,
  type TrainingShaderName,
  defineLfm2Passes,
} from "./lfm2-layout";

export * from "./lfm2-layout";

export const LFM2_INCLUDE_NAMES = [
  "arena",
  "attention-scores",
  "common",
  "matmul-rows",
  "matmul-rows-wide",
  "reduce-f32",
  "reduce-u32",
  "runtime",
  "sampling",
  "telemetry",
  "weights",
  "constraint-vm",
  "constraint-commit",
] as const;

export type Lfm2IncludeName = (typeof LFM2_INCLUDE_NAMES)[number];

export interface Lfm2ShaderBundle {
  readonly sources: Record<Lfm2ShaderName | TrainingShaderName, string>;
  readonly includes: Record<Lfm2IncludeName, string>;
}

function emptyRecord<const K extends readonly string[]>(keys: K): Record<K[number], string> {
  return Object.fromEntries(keys.map((key) => [key, ""])) as Record<K[number], string>;
}

export function emptyLfm2ShaderBundle(): Lfm2ShaderBundle {
  return {
    sources: emptyRecord([...LFM2_SHADER_NAMES, ...TRAINING_SHADER_NAMES]),
    includes: emptyRecord(LFM2_INCLUDE_NAMES),
  };
}

export function defineLfm2(bundle: Lfm2ShaderBundle = emptyLfm2ShaderBundle()) {
  const engine = Sandblaster.create($, {
    codec: "jit",
    schema: { autoSort: true },
  });
  const sources = bundle.sources;
  const shaderIncludes = bundle.includes;

  const OpParams = engine.type("OpParams");
  const LlmRuntime = engine.type("LlmRuntime");

  // One OpParams record per dispatch, selected by dynamic uniform offset.
  // Lfm2ParamWriter accumulates the whole submit's records and writes them in
  // one queue.writeBuffer before the command buffer runs, so the GPU buffer
  // must hold up to OP_PARAM_BUFFER_BYTES (not a small fixed ring). count
  // stays 1: Sandblaster sizes the buffer with `size`, and a scalar value is
  // only valid for count 1. Every record is overwritten before any pass reads
  // it, so no initial value is required.
  const op = engine.buffer(OpParams, {
    label: "lfm2.op",
    size: OP_PARAM_BUFFER_BYTES,
  });
  const runtime = engine.buffer(LlmRuntime, { label: "lfm2.runtime", value: LlmRuntime.assert({}), readback: true });
  const tokens = engine.buffer(engine.type(`u32[] == ${TOKEN_CAPACITY}`), { label: "lfm2.tokens", readback: true });
  // Training targets (cross-entropy ground truth), same capacity convention as
  // the token-id buffer. Only written by trainStep before a submit; no readback.
  const targets = engine.buffer(engine.type(`u32[] == ${TOKEN_CAPACITY}`), { label: "lfm2.targets" });
  // Compact loss telemetry: loss_reduce writes the mean scalar here (in
  // addition to the arena region) so trainStep can read back 4 bytes instead of
  // the whole arena. Debug/test-only readback, absent from the normal path.
  const lossTelemetry = engine.buffer(engine.type("f32[] == 1"), { label: "lfm2.loss-telemetry", readback: true });
  // Debug readback staging: tests copy one arena region (logits, a parameter
  // page) here with copyBufferToBuffer and read back a small slice. Sized for
  // the largest training region (a full V*H parameter page).
  const trainingReadback = engine.buffer(engine.type(`f32[] == ${TRAINING_READBACK_ELEMENTS}`), {
    label: "lfm2.training-readback",
    readback: true,
  });
  const arena = engine.buffer(engine.type(`f32[] == ${ARENA_ELEMENTS}`), { label: "lfm2.arena", readback: true });
  // Checkpoints snapshot these buffers with copyBufferToBuffer. readback=true
  // adds COPY_SRC without forcing any staging allocation until readback() is used.
  const kvCache = engine.buffer(engine.type(`f32[] == ${KV_ELEMENTS}`), { label: "lfm2.kv-cache", readback: true });
  const convCache = engine.buffer(engine.type(`f32[] == ${CONV_ELEMENTS}`), { label: "lfm2.conv-cache", readback: true });
  const candidateTokens = engine.buffer(engine.type(`u32[] == ${VOCAB}`), { label: "lfm2.candidate-tokens" });
  const decodeTelemetry = engine.buffer(engine.type(`u32[] == ${TELEMETRY_CAPACITY}`), { label: "lfm2.decode-telemetry" });
  const constraintProgram = engine.buffer(engine.type(`u32[] == ${CONSTRAINT_PROGRAM_WORD_CAPACITY}`), {
    label: "lfm2.constraint-program",
  });
  const constraintTokenizer = engine.buffer(engine.type(`u32[] == ${CONSTRAINT_TOKENIZER_WORD_CAPACITY}`), {
    label: "lfm2.constraint-tokenizer",
  });
  const constraintState = engine.buffer(engine.type("ConstraintDecoderState"), { label: "lfm2.constraint-state" });
  const constraintMask = engine.buffer(engine.type(`u32[] == ${CONSTRAINT_MASK_WORDS}`), {
    label: "lfm2.constraint-mask",
    readback: true,
  });
  // Placeholders overridden per dispatch with real tensor pages (pass.ts
  // `resources.weightRaw/weight32`). They must be declared as count>1 SCALAR
  // buffers so Sandblaster emits a runtime-sized `array<u32>`/`array<f32>` in
  // WGSL: a fixed-length array type either caps reads (a `u32[] == 2` buffer
  // lowers to vec2<u32>, so every read past word 1 is out of bounds) or forces
  // every bound page to be at least the declared type size. With a runtime
  // array, load_wq4/load_f16 can address pages of any size and the layout has
  // no minBindingSize. count=2 also keeps this placeholder at 8 bytes; no
  // initial value is required.
  const weightRaw = engine.buffer(engine.type("u32"), { label: "lfm2.probe-weight-raw", count: 2 });
  const weight32 = engine.buffer(engine.type("f32"), { label: "lfm2.probe-weight32", count: 2 });


  type Resource = BufferResource<any>;
  function nativeRead(resource: Resource, group = 0): BufferResourceUse {
    return {
      resource,
      group,
      buffer: { type: "read-only-storage" },
      representation: "native",
    };
  }
  function nativeWrite(resource: Resource, group = 0): BufferResourceUse {
    return {
      resource,
      group,
      buffer: { type: "storage" },
      representation: "native",
    };
  }

  /**
   * Canonical WGSL views. `op` is one 256-byte uniform record selected by a
   * dynamic offset; tensor pages live in group 1 so they can be overridden per
   * dispatch without rebuilding the long-lived runtime group.
   */
  function lfm2ResourceViews() {
    return {
      op: {
        resource: op,
        group: 0,
        buffer: {
          type: "uniform",
          hasDynamicOffset: true,
          minBindingSize: 80,
        },
        offset: 0,
        size: 80,
        representation: "native",
      } satisfies BufferResourceUse,
      runtime: nativeWrite(runtime),
      tokens: nativeWrite(tokens),
      targets: nativeWrite(targets),
      lossTelemetry: nativeWrite(lossTelemetry),
      trainingReadback: nativeWrite(trainingReadback),
      arena: nativeWrite(arena),
      kvCache: nativeWrite(kvCache),
      convCache: nativeWrite(convCache),
      candidateTokens: nativeRead(candidateTokens),
      decodeTelemetry: nativeWrite(decodeTelemetry),
      constraintProgram: nativeRead(constraintProgram),
      constraintTokenizer: nativeRead(constraintTokenizer),
      constraintState: nativeWrite(constraintState),
      constraintMask: nativeWrite(constraintMask),
      weightRaw: nativeRead(weightRaw, 1),
      weight32: nativeRead(weight32, 1),
    } as const;
  }

  const r = lfm2ResourceViews();

  /**
   * Define all current compute entry points against the per-program resource
   * subsets. This is sufficient for engine.link() and shader validation/Dawn.
   *
   * It is intentionally NOT yet the inference call scheduler: the legacy runtime
   * changes OpParams and concrete weight pages per dispatch, while Sandblaster's
   * current ComputePassRunner binds a program's resources statically.
   */
  const gid = engine.type({ gid: "global_invocation_id" })
  const wid = engine.type({ wid: "workgroup_id" });
  const lid = engine.type({ lid: "local_invocation_id" });
  const widLid = engine.type({
    wid: "workgroup_id",
    lid: "local_invocation_id",
  });

  const include = (name: keyof typeof shaderIncludes) => shaderIncludes[name];

  const commonIncludes = [include("common")];
  const weightIncludes = [...commonIncludes, include("weights")];
  const reduceF32Includes = [...commonIncludes, include("reduce-f32")];
  const weightReduceF32Includes = [...weightIncludes, include("reduce-f32")];
  const runtimeIncludes = [...commonIncludes, include("runtime")];
  const ropeIncludes = [...runtimeIncludes, include("arena"), include("reduce-f32")];
  const attentionIncludes = [...runtimeIncludes, include("attention-scores")];
  const argmaxIncludes = [
    ...commonIncludes,
    include("telemetry"),
    include("reduce-f32"),
    include("reduce-u32"),
  ];
  // Only the plain argmax kernel samples. The candidate and constraint kernels
  // stay greedy, so they must not pull in the sampling workgroup arrays.
  const samplingArgmaxIncludes = [...argmaxIncludes, include("sampling")];

  const programs = {
    embedding: engine.compute({
      label: "embedding",
      resources: { op: r.op, runtime: r.runtime, tokens: r.tokens, arena: r.arena, weightRaw: r.weightRaw },
      includes: weightIncludes,
      compute: { entryPoint: "embedding", params: gid, workgroupSize: 256, code: sources.embedding },
    }),

    embedding_wq4: engine.compute({
      label: "embedding_wq4",
      resources: { op: r.op, runtime: r.runtime, tokens: r.tokens, arena: r.arena, weightRaw: r.weightRaw },
      includes: weightIncludes,
      compute: { entryPoint: "embedding_wq4", params: gid, workgroupSize: 256, code: sources.embedding_wq4 },
    }),

    rms_norm: engine.compute({
      label: "rms_norm",
      resources: { op: r.op, arena: r.arena, weight32: r.weight32 },
      includes: reduceF32Includes,
      compute: { entryPoint: "rms_norm", params: widLid, workgroupSize: 64, code: sources.rms_norm },
    }),

    matmul_f16: engine.compute({
      label: "matmul_f16",
      resources: { op: r.op, arena: r.arena, weightRaw: r.weightRaw },
      includes: weightReduceF32Includes,
      compute: { entryPoint: "matmul_f16", params: widLid, workgroupSize: 64, code: sources.matmul_f16 },
    }),

    matmul_f32: engine.compute({
      label: "matmul_f32",
      resources: { op: r.op, arena: r.arena, weight32: r.weight32 },
      includes: reduceF32Includes,
      compute: { entryPoint: "matmul_f32", params: widLid, workgroupSize: 64, code: sources.matmul_f32 },
    }),

    // Two programs, one body. They differ only in the MATMUL_ROWS constant the
    // include supplies; the tiling that wins depends on the output width, and
    // no single value wins everywhere (see matmul-rows-wide.wgsl). The row
    // count cannot be a pipeline override: WGSL requires a const-expression for
    // the size of the function-scope `acc` array.
    matmul_wq4: engine.compute({
      label: "matmul_wq4",
      resources: { op: r.op, arena: r.arena, weightRaw: r.weightRaw },
      includes: [...reduceF32Includes, include("matmul-rows")],
      compute: { entryPoint: "matmul_wq4", params: widLid, workgroupSize: 64, code: sources.matmul_wq4 },
    }),

    matmul_wq4_wide: engine.compute({
      label: "matmul_wq4_wide",
      resources: { op: r.op, arena: r.arena, weightRaw: r.weightRaw },
      includes: [...reduceF32Includes, include("matmul-rows-wide")],
      compute: { entryPoint: "matmul_wq4", params: widLid, workgroupSize: 64, code: sources.matmul_wq4 },
    }),

    residual_add: engine.compute({
      label: "residual_add",
      resources: { op: r.op, arena: r.arena },
      compute: { entryPoint: "residual_add", params: gid, workgroupSize: 256, code: sources.residual_add },
    }),

    silu_mul: engine.compute({
      label: "silu_mul",
      resources: { op: r.op, arena: r.arena },
      compute: { entryPoint: "silu_mul", params: gid, workgroupSize: 256, code: sources.silu_mul },
    }),

    shortconv_prefill: engine.compute({
      label: "shortconv_prefill",
      resources: { op: r.op, arena: r.arena, convCache: r.convCache, weight32: r.weight32 },
      compute: { entryPoint: "shortconv_prefill", params: gid, workgroupSize: 256, code: sources.shortconv_prefill },
    }),

    shortconv_continue: engine.compute({
      label: "shortconv_continue",
      resources: { op: r.op, arena: r.arena, convCache: r.convCache, weight32: r.weight32 },
      compute: { entryPoint: "shortconv_continue", params: wid, workgroupSize: 1, code: sources.shortconv_continue },
    }),

    shortconv_decode: engine.compute({
      label: "shortconv_decode",
      resources: { op: r.op, arena: r.arena, convCache: r.convCache, weight32: r.weight32 },
      compute: { entryPoint: "shortconv_decode", params: gid, workgroupSize: 256, code: sources.shortconv_decode },
    }),

    qk_norm_rope: engine.compute({
      label: "qk_norm_rope",
      resources: { op: r.op, runtime: r.runtime, arena: r.arena, weight32: r.weight32 },
      includes: ropeIncludes,
      compute: { entryPoint: "qk_norm_rope", params: widLid, workgroupSize: 64, code: sources.qk_norm_rope },
    }),

    kv_store: engine.compute({
      label: "kv_store",
      resources: { op: r.op, runtime: r.runtime, arena: r.arena, kvCache: r.kvCache },
      includes: runtimeIncludes,
      compute: { entryPoint: "kv_store", params: gid, workgroupSize: 256, code: sources.kv_store },
    }),

    attention: engine.compute({
      label: "attention",
      resources: { op: r.op, runtime: r.runtime, arena: r.arena, kvCache: r.kvCache },
      includes: attentionIncludes,
      compute: { entryPoint: "attention", params: widLid, workgroupSize: 64, code: sources.attention },
    }),

    arena_copy: engine.compute({
      label: "arena_copy",
      resources: { op: r.op, arena: r.arena },
      compute: { entryPoint: "arena_copy", params: gid, workgroupSize: 256, code: sources.arena_copy },
    }),

    argmax_candidates: engine.compute({
      label: "argmax_candidates",
      resources: {
        op: r.op,
        runtime: r.runtime,
        tokens: r.tokens,
        arena: r.arena,
        candidateTokens: r.candidateTokens,
        decodeTelemetry: r.decodeTelemetry,
      },
      codecs: [engine.type("DecodeTelemetryEntry")],
      includes: argmaxIncludes,
      compute: { entryPoint: "argmax_candidates", params: lid, workgroupSize: 256, code: sources.argmax_candidates },
    }),

    argmax: engine.compute({
      label: "argmax",
      resources: { op: r.op, runtime: r.runtime, tokens: r.tokens, arena: r.arena, decodeTelemetry: r.decodeTelemetry },
      codecs: [engine.type("DecodeTelemetryEntry")],
      includes: samplingArgmaxIncludes,
      compute: { entryPoint: "argmax", params: lid, workgroupSize: 256, code: sources.argmax },
    }),

    constraint_mask: engine.compute({
      label: "constraint_mask",
      resources: {
        constraintProgram: r.constraintProgram,
        constraintTokenizer: r.constraintTokenizer,
        constraintState: r.constraintState,
        constraintMask: r.constraintMask,
      },
      includes: [include("constraint-vm")],
      compute: {
        entryPoint: "constraint_mask",
        params: gid,
        workgroupSize: CONSTRAINT_MASK_WORKGROUP_SIZE,
        code: sources.constraint_mask,
      },
    }),

    constraint_argmax: engine.compute({
      label: "constraint_argmax",
      resources: {
        op: r.op,
        runtime: r.runtime,
        tokens: r.tokens,
        arena: r.arena,
        decodeTelemetry: r.decodeTelemetry,
        constraintProgram: r.constraintProgram,
        constraintTokenizer: r.constraintTokenizer,
        constraintState: r.constraintState,
        constraintMask: r.constraintMask,
      },
      types: [engine.type("ConstraintDecoderState")],
      codecs: [engine.type("DecodeTelemetryEntry")],
      includes: [
        include("common"),
        include("telemetry"),
        include("reduce-f32"),
        include("reduce-u32"),
        include("constraint-vm"),
        include("constraint-commit"),
      ],

      compute: {
        entryPoint: "constraint_argmax",
        params: lid,
        workgroupSize: 256,
        code: sources.constraint_argmax,
      },
    }),

    // --- M1 training programs ---
    // Reuse the shared OpParams/arena conventions. Gradient and optimizer
    // dispatches are separate: no training shader updates parameters except
    // sgd_step, which binds weight32 as writable storage.
    embedding_f32: engine.compute({
      label: "embedding_f32",
      resources: { op: r.op, tokens: r.tokens, arena: r.arena, weight32: r.weight32 },
      includes: commonIncludes,
      compute: { entryPoint: "embedding_f32", params: gid, workgroupSize: 256, code: sources.embedding_f32 },
    }),

    zero_f32: engine.compute({
      label: "zero_f32",
      resources: { op: r.op, arena: r.arena },
      includes: commonIncludes,
      compute: { entryPoint: "zero_f32", params: gid, workgroupSize: 256, code: sources.zero_f32 },
    }),

    cross_entropy_forward_backward: engine.compute({
      label: "cross_entropy_forward_backward",
      resources: { op: r.op, targets: r.targets, arena: r.arena },
      includes: reduceF32Includes,
      compute: {
        entryPoint: "cross_entropy_forward_backward",
        params: widLid,
        workgroupSize: 64,
        code: sources.cross_entropy_forward_backward,
      },
    }),

    loss_reduce: engine.compute({
      label: "loss_reduce",
      resources: { op: r.op, arena: r.arena, lossTelemetry: r.lossTelemetry },
      includes: reduceF32Includes,
      compute: { entryPoint: "loss_reduce", params: lid, workgroupSize: 64, code: sources.loss_reduce },
    }),

    matmul_backward_input: engine.compute({
      label: "matmul_backward_input",
      resources: { op: r.op, arena: r.arena, weight32: r.weight32 },
      includes: commonIncludes,
      compute: { entryPoint: "matmul_backward_input", params: gid, workgroupSize: 256, code: sources.matmul_backward_input },
    }),

    matmul_backward_weight: engine.compute({
      label: "matmul_backward_weight",
      resources: { op: r.op, arena: r.arena },
      includes: commonIncludes,
      compute: { entryPoint: "matmul_backward_weight", params: gid, workgroupSize: 256, code: sources.matmul_backward_weight },
    }),

    embedding_backward: engine.compute({
      label: "embedding_backward",
      resources: { op: r.op, tokens: r.tokens, arena: r.arena },
      includes: commonIncludes,
      compute: { entryPoint: "embedding_backward", params: gid, workgroupSize: 256, code: sources.embedding_backward },
    }),

    // SGD writes the trainable parameter page in place, so weight32 is bound
    // as storage (read-write) here, unlike the read-only weight32 views above.
    sgd_step: engine.compute({
      label: "sgd_step",
      resources: {
        op: r.op,
        arena: r.arena,
        weight32: { resource: weight32, group: 1, buffer: { type: "storage" }, representation: "native" },
      },
      includes: commonIncludes,
      compute: { entryPoint: "sgd_step", params: gid, workgroupSize: 256, code: sources.sgd_step },
    }),

    // --- Attention (§17 item 6) ---
    // Krystal encoder semantics: bidirectional, host-masked, multi-head, no
    // KV cache, no GQA. All tensor regions live in the shared f32 arena; no
    // weight pages. attention_forward persists the softmax probs P so the two
    // backward shaders can reuse them (the mask never needs re-reading: masked
    // positions carry P == 0).
    attention_forward: engine.compute({
      label: "attention_forward",
      resources: { op: r.op, arena: r.arena },
      includes: [...commonIncludes, include("attention-scores")],
      compute: { entryPoint: "attention_forward", params: widLid, workgroupSize: 64, code: sources.attention_forward },
    }),

    attention_backward_scores: engine.compute({
      label: "attention_backward_scores",
      resources: { op: r.op, arena: r.arena },
      includes: [...commonIncludes, include("attention-scores"), include("reduce-f32")],
      compute: {
        entryPoint: "attention_backward_scores",
        params: widLid,
        workgroupSize: 64,
        code: sources.attention_backward_scores,
      },
    }),

    attention_backward_qkv: engine.compute({
      label: "attention_backward_qkv",
      resources: { op: r.op, arena: r.arena },
      includes: commonIncludes,
      compute: { entryPoint: "attention_backward_qkv", params: gid, workgroupSize: 256, code: sources.attention_backward_qkv },
    }),
  } satisfies Record<Lfm2ProgramName, AnyComputeHandle>;

  const passes = defineLfm2Passes(programs);

  const resources = {
    op,
    runtime,
    tokens,
    targets,
    arena,
    kvCache,
    convCache,
    candidateTokens,
    decodeTelemetry,
    constraintProgram,
    constraintTokenizer,
    constraintState,
    constraintMask,
    weightRaw,
    weight32,
    lossTelemetry,
    trainingReadback,
  } as const;


  return {
    ...LFM2_DEFINITION_PLAIN,
    engine,
    resources,
    programs,
    passes,
  } as const;
}
