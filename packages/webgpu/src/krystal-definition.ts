// ArkType-backed Krystal definition DSL.
//
// This is the build-time path: it declares the whole typed resource/program
// graph through Sandblaster's scope API and is used by the AOT scripts
// (scripts/build-krystal-artifact.ts, scripts/validate-krystal-shaders.ts) to link
// and serialize krystal.artifact.generated.ts.
//
// The runtime never needs it. packages/webgpu/src/krystal.ts builds the same
// definition from the already-linked artifact (krystal-artifact.ts), so this
// module — and with it arktype and `$` — stays out of scriptc-compiled graphs.
import {
  Sandblaster,
  type AnyComputeHandle,
  type BufferResource,
  type BufferResourceUse,
} from "@sandblaster/core";
import { $ } from "../../schema/src/schema";
import {
  ARENA_ELEMENTS,
  TOKEN_CAPACITY,
  OP_PARAM_BUFFER_BYTES,
  KRYSTAL_DEFINITION_PLAIN,
  KRYSTAL_SHADER_NAMES,
  TRAINING_READBACK_ELEMENTS,
  TRAINING_SHADER_NAMES,
  KRYSTAL_FORWARD_SHADER_NAMES,
  KRYSTAL_BACKWARD_SHADER_NAMES,
  type KrystalProgramName,
  type KrystalShaderName,
  type TrainingShaderName,
  type KrystalForwardShaderName,
  type KrystalBackwardShaderName,
  defineKrystalPasses,
} from "./krystal-layout";

export * from "./krystal-layout";

// Includes still referenced by the Krystal + M1 training programs. The LFM2
// model-inference includes (weights, matmul-rows, sampling, constraint-vm,
// telemetry, ...) were removed with the legacy runtime.
export const KRYSTAL_INCLUDE_NAMES = [
  "attention-scores",
  "common",
  "pool-scores",
  "pool-backward-scores",
  "reduce-f32",
] as const;

export type KrystalIncludeName = (typeof KRYSTAL_INCLUDE_NAMES)[number];

export interface KrystalShaderBundle {
  readonly sources: Record<
    KrystalShaderName | TrainingShaderName | KrystalForwardShaderName | KrystalBackwardShaderName,
    string
  >;
  readonly includes: Record<KrystalIncludeName, string>;
}

function emptyRecord<const K extends readonly string[]>(keys: K): Record<K[number], string> {
  return Object.fromEntries(keys.map((key) => [key, ""])) as Record<K[number], string>;
}

export function emptyKrystalShaderBundle(): KrystalShaderBundle {
  return {
    sources: emptyRecord([
      ...KRYSTAL_SHADER_NAMES,
      ...TRAINING_SHADER_NAMES,
      ...KRYSTAL_FORWARD_SHADER_NAMES,
      ...KRYSTAL_BACKWARD_SHADER_NAMES,
    ]),
    includes: emptyRecord(KRYSTAL_INCLUDE_NAMES),
  };
}

export function defineKrystal(bundle: KrystalShaderBundle = emptyKrystalShaderBundle()) {
  const engine = Sandblaster.create($, {
    codec: "jit",
    schema: { autoSort: true },
  });
  const sources = bundle.sources;
  const shaderIncludes = bundle.includes;

  const OpParams = engine.type("OpParams");

  // One OpParams record per dispatch, selected by dynamic uniform offset.
  // KrystalParamWriter accumulates the whole submit's records and writes them in
  // one queue.writeBuffer before the command buffer runs, so the GPU buffer
  // must hold up to OP_PARAM_BUFFER_BYTES (not a small fixed ring). count
  // stays 1: Sandblaster sizes the buffer with `size`, and a scalar value is
  // only valid for count 1. Every record is overwritten before any pass reads
  // it, so no initial value is required.
  const op = engine.buffer(OpParams, {
    label: "krystal.op",
    size: OP_PARAM_BUFFER_BYTES,
  });
  const tokens = engine.buffer(engine.type(`u32[] == ${TOKEN_CAPACITY}`), { label: "krystal.tokens", readback: true });
  // Training targets (cross-entropy ground truth), same capacity convention as
  // the token-id buffer. Only written by trainStep before a submit; no readback.
  const targets = engine.buffer(engine.type(`u32[] == ${TOKEN_CAPACITY}`), { label: "krystal.targets" });
  // Compact loss telemetry: loss_reduce writes the mean scalar here (in
  // addition to the arena region) so trainStep can read back 4 bytes instead of
  // the whole arena. Debug/test-only readback, absent from the normal path.
  const lossTelemetry = engine.buffer(engine.type("f32[] == 1"), { label: "krystal.loss-telemetry", readback: true });
  // Debug readback staging: tests copy one arena region (logits, a parameter
  // page) here with copyBufferToBuffer and read back a small slice. Sized for
  // the largest training region (a full V*H parameter page).
  const trainingReadback = engine.buffer(engine.type(`f32[] == ${TRAINING_READBACK_ELEMENTS}`), {
    label: "krystal.training-readback",
    readback: true,
  });
  const arena = engine.buffer(engine.type(`f32[] == ${ARENA_ELEMENTS}`), { label: "krystal.arena", readback: true });
  // Placeholder overridden per dispatch with real tensor pages (pass.ts
  // `resources.weight32`). It must be declared as a count>1 SCALAR buffer so
  // Sandblaster emits a runtime-sized `array<f32>` in WGSL: a fixed-length
  // array type either caps reads or forces every bound page to be at least the
  // declared type size. With a runtime array, shaders can address pages of any
  // size and the layout has no minBindingSize. count=2 keeps this placeholder
  // at 8 bytes; no initial value is required.
  const weight32 = engine.buffer(engine.type("f32"), { label: "krystal.probe-weight32", count: 2 });


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
  function krystalResourceViews() {
    return {
      op: {
        resource: op,
        group: 0,
        buffer: {
          type: "uniform",
          hasDynamicOffset: true,
          minBindingSize: 96,
        },
        offset: 0,
        size: 96,
        representation: "native",
      } satisfies BufferResourceUse,
      tokens: nativeWrite(tokens),
      targets: nativeWrite(targets),
      lossTelemetry: nativeWrite(lossTelemetry),
      trainingReadback: nativeWrite(trainingReadback),
      arena: nativeWrite(arena),
      weight32: nativeRead(weight32, 1),
    } as const;
  }

  const r = krystalResourceViews();

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
  const reduceF32Includes = [...commonIncludes, include("reduce-f32")];

  const programs = {
    matmul_f32: engine.compute({
      label: "matmul_f32",
      resources: { op: r.op, arena: r.arena, weight32: r.weight32 },
      includes: reduceF32Includes,
      compute: { entryPoint: "matmul_f32", params: widLid, workgroupSize: 64, code: sources.matmul_f32 },
    }),

    residual_add: engine.compute({
      label: "residual_add",
      resources: { op: r.op, arena: r.arena },
      compute: { entryPoint: "residual_add", params: gid, workgroupSize: 256, code: sources.residual_add },
    }),

    arena_copy: engine.compute({
      label: "arena_copy",
      resources: { op: r.op, arena: r.arena },
      compute: { entryPoint: "arena_copy", params: gid, workgroupSize: 256, code: sources.arena_copy },
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

    // --- M2b Krystal forward programs ---
    // The SoA frame and host-compiled active lists live in the shared arena
    // as u32 payloads (bitcast inside the shaders). krystal_field_embed binds
    // weight32 for the concatenated embedding tables; krystal_pool binds the
    // two learned pooling query vectors.
    krystal_field_embed: engine.compute({
      label: "krystal_field_embed",
      resources: { op: r.op, arena: r.arena, weight32: r.weight32 },
      includes: commonIncludes,
      compute: {
        entryPoint: "krystal_field_embed",
        params: gid,
        workgroupSize: 256,
        code: sources.krystal_field_embed,
      },
    }),

    krystal_attention_forward: engine.compute({
      label: "krystal_attention_forward",
      resources: { op: r.op, arena: r.arena },
      includes: [...commonIncludes, include("attention-scores")],
      compute: {
        entryPoint: "krystal_attention_forward",
        params: widLid,
        workgroupSize: 64,
        code: sources.krystal_attention_forward,
      },
    }),

    relu: engine.compute({
      label: "relu",
      resources: { op: r.op, arena: r.arena },
      includes: commonIncludes,
      compute: { entryPoint: "relu", params: gid, workgroupSize: 256, code: sources.relu },
    }),

    krystal_pool: engine.compute({
      label: "krystal_pool",
      resources: { op: r.op, arena: r.arena, weight32: r.weight32 },
      includes: [...commonIncludes, include("pool-scores")],
      compute: {
        entryPoint: "krystal_pool",
        params: widLid,
        workgroupSize: 64,
        code: sources.krystal_pool,
      },
    }),

    // Catalog selection + soft gather (§7, answer 26): masked scoring,
    // softmax distribution, gathered value vector and first-max argmax index.
    krystal_selector: engine.compute({
      label: "krystal_selector",
      resources: { op: r.op, arena: r.arena },
      includes: [...commonIncludes, include("attention-scores")],
      compute: {
        entryPoint: "krystal_selector",
        params: widLid,
        workgroupSize: 64,
        code: sources.krystal_selector,
      },
    }),

    // Typed decision head forward (§17 item 9): route-kind logits from the
    // three gathered-context regions, linear head over the weight page.
    krystal_decision_head: engine.compute({
      label: "krystal_decision_head",
      resources: { op: r.op, arena: r.arena, weight32: r.weight32 },
      includes: commonIncludes,
      compute: {
        entryPoint: "krystal_decision_head",
        params: widLid,
        workgroupSize: 64,
        code: sources.krystal_decision_head,
      },
    }),

    // M3 Krystal backward programs.
    relu_backward: engine.compute({
      label: "relu_backward",
      resources: { op: r.op, arena: r.arena },
      includes: commonIncludes,
      compute: { entryPoint: "relu_backward", params: gid, workgroupSize: 256, code: sources.relu_backward },
    }),

    krystal_attention_backward_scores: engine.compute({
      label: "krystal_attention_backward_scores",
      resources: { op: r.op, arena: r.arena },
      includes: [...commonIncludes, include("attention-scores"), include("reduce-f32")],
      compute: {
        entryPoint: "krystal_attention_backward_scores",
        params: widLid,
        workgroupSize: 64,
        code: sources.krystal_attention_backward_scores,
      },
    }),

    krystal_attention_backward_qkv: engine.compute({
      label: "krystal_attention_backward_qkv",
      resources: { op: r.op, arena: r.arena },
      includes: commonIncludes,
      compute: {
        entryPoint: "krystal_attention_backward_qkv",
        params: gid,
        workgroupSize: 256,
        code: sources.krystal_attention_backward_qkv,
      },
    }),

    krystal_field_embed_backward: engine.compute({
      label: "krystal_field_embed_backward",
      resources: { op: r.op, arena: r.arena },
      includes: commonIncludes,
      compute: {
        entryPoint: "krystal_field_embed_backward",
        params: gid,
        workgroupSize: 256,
        code: sources.krystal_field_embed_backward,
      },
    }),

    krystal_pool_backward: engine.compute({
      label: "krystal_pool_backward",
      resources: { op: r.op, arena: r.arena, weight32: r.weight32 },
      includes: [...commonIncludes, include("pool-backward-scores")],
      compute: {
        entryPoint: "krystal_pool_backward",
        params: widLid,
        workgroupSize: 64,
        code: sources.krystal_pool_backward,
      },
    }),

    krystal_pool_dpool: engine.compute({
      label: "krystal_pool_dpool",
      resources: { op: r.op, arena: r.arena },
      includes: commonIncludes,
      compute: {
        entryPoint: "krystal_pool_dpool",
        params: gid,
        workgroupSize: 256,
        code: sources.krystal_pool_dpool,
      },
    }),

    krystal_selector_backward_scores: engine.compute({
      label: "krystal_selector_backward_scores",
      resources: { op: r.op, arena: r.arena },
      includes: [...commonIncludes, include("attention-scores"), include("reduce-f32")],
      compute: {
        entryPoint: "krystal_selector_backward_scores",
        params: widLid,
        workgroupSize: 64,
        code: sources.krystal_selector_backward_scores,
      },
    }),

    krystal_selector_backward_qkv: engine.compute({
      label: "krystal_selector_backward_qkv",
      resources: { op: r.op, arena: r.arena },
      includes: commonIncludes,
      compute: {
        entryPoint: "krystal_selector_backward_qkv",
        params: gid,
        workgroupSize: 256,
        code: sources.krystal_selector_backward_qkv,
      },
    }),

    // Typed decision head backward (§17 item 9): reads dLogits + the three
    // gathered-context regions + the head weight page, writes the dCtx parts
    // and dWh. Needs the weight32 binding for Wh.
    krystal_decision_head_backward: engine.compute({
      label: "krystal_decision_head_backward",
      resources: { op: r.op, arena: r.arena, weight32: r.weight32 },
      includes: commonIncludes,
      compute: {
        entryPoint: "krystal_decision_head_backward",
        params: gid,
        workgroupSize: 256,
        code: sources.krystal_decision_head_backward,
      },
    }),
  } satisfies Record<KrystalProgramName, AnyComputeHandle>;

  const passes = defineKrystalPasses(programs);

  const resources = {
    op,
    tokens,
    targets,
    arena,
    weight32,
    lossTelemetry,
    trainingReadback,
  } as const;


  return {
    ...KRYSTAL_DEFINITION_PLAIN,
    engine,
    resources,
    programs,
    passes,
  } as const;
}
