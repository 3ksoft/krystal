// Artifact-backed Krystal definition (runtime path).
//
// `krystal.artifact.generated.ts` is linked at build time: it carries every layout
// plan, binding manifest and linked WGSL string. `Sandblaster.fromArtifact()`
// creates the resource and program handles *from* that artifact, so the runtime
// never re-declares the graph through the arktype DSL (which would mean parsing
// the whole schema at startup only to throw the result away).
//
// This module — with krystal-layout.ts — is the entire runtime definition surface.
// Nothing here imports `$`, arktype or @schema-pop, so it is what the native
// (scriptc) target can compile statically.
import { Sandblaster } from "@sandblaster/core";
import { krystalArtifact } from "./krystal.artifact.generated";
import {
  KRYSTAL_DEFINITION_PLAIN,
  OP_PARAM_BUFFER_BYTES,
  type KrystalProgramName,
  type KrystalOpParams,
  defineKrystalPasses,
} from "./krystal-layout";

/**
 * Runtime-only buffer fields that serialization deliberately leaves out,
 * keyed by the artifact buffer's label (unknown keys are rejected).
 *
 * These mirror the `engine.buffer(...)` declarations in krystal-definition.ts:
 * the OpParams buffer must be sized for the whole schema-derived dispatch
 * budget, and the readback-flagged buffers need COPY_SRC (checkpoint copies,
 * generation readback) without forcing staging until actually read back.
 *
 * The legacy LFM2 runtime buffer (with its `value: LlmRuntime.assert({})`
 * initial value) was removed with the legacy runtime; no remaining buffer needs
 * an initial value.
 */
const ARTIFACT_BUFFER_OPTIONS = {
  "krystal.op": { size: OP_PARAM_BUFFER_BYTES },
  "krystal.tokens": { readback: true },
  "krystal.arena": { readback: true },
  "krystal.targets": {},
  "krystal.loss-telemetry": { readback: true },
  "krystal.training-readback": { readback: true },
} as const;

export function defineKrystalFromArtifact() {
  const engine = Sandblaster.fromArtifact(krystalArtifact, {
    // codec: "jit" is also the sandblaster default, but state it for parity
    // with the DSL builder (the runtime readback decoders depend on it).
    codec: "jit",
    buffers: ARTIFACT_BUFFER_OPTIONS,
  });

  const resources = {
    op: engine.resource<KrystalOpParams>("krystal.op"),
    tokens: engine.resource("krystal.tokens"),
    targets: engine.resource("krystal.targets"),
    lossTelemetry: engine.resource("krystal.loss-telemetry"),
    trainingReadback: engine.resource("krystal.training-readback"),
    arena: engine.resource("krystal.arena"),
    weight32: engine.resource("krystal.probe-weight32"),
  } as const;

  const programs = {
    matmul_f32: engine.computeProgram("matmul_f32"),
    residual_add: engine.computeProgram("residual_add"),
    arena_copy: engine.computeProgram("arena_copy"),
    // M1 training programs (same names as the shader files).
    zero_f32: engine.computeProgram("zero_f32"),
    cross_entropy_forward_backward: engine.computeProgram("cross_entropy_forward_backward"),
    loss_reduce: engine.computeProgram("loss_reduce"),
    matmul_backward_input: engine.computeProgram("matmul_backward_input"),
    matmul_backward_weight: engine.computeProgram("matmul_backward_weight"),
    sgd_step: engine.computeProgram("sgd_step"),
    // M2b Krystal forward programs.
    krystal_field_embed: engine.computeProgram("krystal_field_embed"),
    krystal_attention_forward: engine.computeProgram("krystal_attention_forward"),
    relu: engine.computeProgram("relu"),
    krystal_pool: engine.computeProgram("krystal_pool"),
    krystal_selector: engine.computeProgram("krystal_selector"),
    krystal_decision_head: engine.computeProgram("krystal_decision_head"),
    // M3 Krystal backward programs.
    relu_backward: engine.computeProgram("relu_backward"),
    krystal_attention_backward_scores: engine.computeProgram("krystal_attention_backward_scores"),
    krystal_attention_backward_qkv: engine.computeProgram("krystal_attention_backward_qkv"),
    krystal_field_embed_backward: engine.computeProgram("krystal_field_embed_backward"),
    krystal_field_embed_sgd: engine.computeProgram("krystal_field_embed_sgd"),
    krystal_pool_backward: engine.computeProgram("krystal_pool_backward"),
    krystal_pool_dpool: engine.computeProgram("krystal_pool_dpool"),
    krystal_selector_backward_scores: engine.computeProgram("krystal_selector_backward_scores"),
    krystal_selector_backward_qkv: engine.computeProgram("krystal_selector_backward_qkv"),
    krystal_decision_head_backward: engine.computeProgram("krystal_decision_head_backward"),
    krystal_value_head_loss: engine.computeProgram("krystal_value_head_loss"),
  } satisfies Record<KrystalProgramName, ReturnType<typeof engine.computeProgram>>;

  const passes = defineKrystalPasses(programs);

  return {
    ...KRYSTAL_DEFINITION_PLAIN,
    engine,
    resources,
    programs,
    passes,
  } as const;
}

/** The runtime definition shape (artifact-built engine, no arktype scope). */
export type KrystalDefinition = ReturnType<typeof defineKrystalFromArtifact>;
