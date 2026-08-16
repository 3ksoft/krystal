// Artifact-backed LFM2 definition (runtime path).
//
// `lfm2.artifact.generated.ts` is linked at build time: it carries every layout
// plan, binding manifest and linked WGSL string. `Sandblaster.fromArtifact()`
// creates the resource and program handles *from* that artifact, so the runtime
// never re-declares the graph through the arktype DSL (which would mean parsing
// the whole schema at startup only to throw the result away).
//
// This module — with lfm2-layout.ts — is the entire runtime definition surface.
// Nothing here imports `$`, arktype or @schema-pop, so it is what the native
// (scriptc) target can compile statically.
import { Sandblaster } from "@sandblaster/core";
import { lfm2Artifact } from "./lfm2.artifact.generated";
import {
  LFM2_DEFINITION_PLAIN,
  OP_PARAM_BUFFER_BYTES,
  type Lfm2ProgramName,
  type Lfm2OpParams,
  defineLfm2Passes,
} from "./lfm2-layout";

/**
 * Runtime-only buffer fields that serialization deliberately leaves out,
 * keyed by the artifact buffer's label (unknown keys are rejected).
 *
 * These mirror the `engine.buffer(...)` declarations in lfm2-definition.ts:
 * the OpParams buffer must be sized for the whole schema-derived dispatch
 * budget, and the readback-flagged buffers need COPY_SRC (checkpoint copies,
 * generation readback) without forcing staging until actually read back.
 *
 * The DSL also passes `value: LlmRuntime.assert({})` as the runtime buffer's
 * initial value. It is deliberately not repeated here: the defaults are all
 * zero (a zero-filled buffer is byte-identical) and Lfm2Forward.writeRuntime()
 * overwrites the whole record before anything reads it.
 */
const ARTIFACT_BUFFER_OPTIONS = {
  "lfm2.op": { size: OP_PARAM_BUFFER_BYTES },
  "lfm2.runtime": { readback: true },
  "lfm2.tokens": { readback: true },
  "lfm2.arena": { readback: true },
  "lfm2.targets": {},
  "lfm2.loss-telemetry": { readback: true },
  "lfm2.training-readback": { readback: true },
  "lfm2.kv-cache": { readback: true },
  "lfm2.conv-cache": { readback: true },
  "lfm2.constraint-mask": { readback: true },
} as const;

export function defineLfm2FromArtifact() {
  const engine = Sandblaster.fromArtifact(lfm2Artifact, {
    // codec: "jit" is also the sandblaster default, but state it for parity
    // with the DSL builder (the runtime readback decoders depend on it).
    codec: "jit",
    buffers: ARTIFACT_BUFFER_OPTIONS,
  });

  const resources = {
    op: engine.resource<Lfm2OpParams>("lfm2.op"),
    runtime: engine.resource("lfm2.runtime"),
    tokens: engine.resource("lfm2.tokens"),
    targets: engine.resource("lfm2.targets"),
    lossTelemetry: engine.resource("lfm2.loss-telemetry"),
    trainingReadback: engine.resource("lfm2.training-readback"),
    arena: engine.resource("lfm2.arena"),
    kvCache: engine.resource("lfm2.kv-cache"),
    convCache: engine.resource("lfm2.conv-cache"),
    candidateTokens: engine.resource("lfm2.candidate-tokens"),
    decodeTelemetry: engine.resource("lfm2.decode-telemetry"),
    constraintProgram: engine.resource("lfm2.constraint-program"),
    constraintTokenizer: engine.resource("lfm2.constraint-tokenizer"),
    constraintState: engine.resource("lfm2.constraint-state"),
    constraintMask: engine.resource("lfm2.constraint-mask"),
    weightRaw: engine.resource("lfm2.probe-weight-raw"),
    weight32: engine.resource("lfm2.probe-weight32"),
  } as const;

  const programs = {
    embedding: engine.computeProgram("embedding"),
    embedding_wq4: engine.computeProgram("embedding_wq4"),
    rms_norm: engine.computeProgram("rms_norm"),
    matmul_f16: engine.computeProgram("matmul_f16"),
    matmul_f32: engine.computeProgram("matmul_f32"),
    matmul_wq4: engine.computeProgram("matmul_wq4"),
    matmul_wq4_wide: engine.computeProgram("matmul_wq4_wide"),
    residual_add: engine.computeProgram("residual_add"),
    silu_mul: engine.computeProgram("silu_mul"),
    shortconv_prefill: engine.computeProgram("shortconv_prefill"),
    shortconv_continue: engine.computeProgram("shortconv_continue"),
    shortconv_decode: engine.computeProgram("shortconv_decode"),
    qk_norm_rope: engine.computeProgram("qk_norm_rope"),
    kv_store: engine.computeProgram("kv_store"),
    attention: engine.computeProgram("attention"),
    arena_copy: engine.computeProgram("arena_copy"),
    argmax_candidates: engine.computeProgram("argmax_candidates"),
    argmax: engine.computeProgram("argmax"),
    constraint_mask: engine.computeProgram("constraint_mask"),
    constraint_argmax: engine.computeProgram("constraint_argmax"),
    // M1 training programs (same names as the shader files).
    embedding_f32: engine.computeProgram("embedding_f32"),
    zero_f32: engine.computeProgram("zero_f32"),
    cross_entropy_forward_backward: engine.computeProgram("cross_entropy_forward_backward"),
    loss_reduce: engine.computeProgram("loss_reduce"),
    matmul_backward_input: engine.computeProgram("matmul_backward_input"),
    matmul_backward_weight: engine.computeProgram("matmul_backward_weight"),
    embedding_backward: engine.computeProgram("embedding_backward"),
    sgd_step: engine.computeProgram("sgd_step"),
    attention_forward: engine.computeProgram("attention_forward"),
    attention_backward_scores: engine.computeProgram("attention_backward_scores"),
    attention_backward_qkv: engine.computeProgram("attention_backward_qkv"),
    // M2b Krystal forward programs.
    krystal_field_embed: engine.computeProgram("krystal_field_embed"),
    krystal_attention_forward: engine.computeProgram("krystal_attention_forward"),
    relu: engine.computeProgram("relu"),
    krystal_pool: engine.computeProgram("krystal_pool"),
    krystal_selector: engine.computeProgram("krystal_selector"),
    // M3 Krystal backward programs.
    relu_backward: engine.computeProgram("relu_backward"),
    krystal_attention_backward_scores: engine.computeProgram("krystal_attention_backward_scores"),
    krystal_attention_backward_qkv: engine.computeProgram("krystal_attention_backward_qkv"),
    krystal_field_embed_backward: engine.computeProgram("krystal_field_embed_backward"),
    krystal_pool_backward: engine.computeProgram("krystal_pool_backward"),
    krystal_pool_dpool: engine.computeProgram("krystal_pool_dpool"),
  } satisfies Record<Lfm2ProgramName, ReturnType<typeof engine.computeProgram>>;

  const passes = defineLfm2Passes(programs);

  return {
    ...LFM2_DEFINITION_PLAIN,
    engine,
    resources,
    programs,
    passes,
  } as const;
}

/** The runtime definition shape (artifact-built engine, no arktype scope). */
export type Lfm2Definition = ReturnType<typeof defineLfm2FromArtifact>;
