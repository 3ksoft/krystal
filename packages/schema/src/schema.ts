import { scope } from "arktype";
import { sandblaster } from "@sandblaster/core";

/**
 * CPU <-> GPU contract for one minimal inference request.
 *
 * Large tensors intentionally do NOT live in schema-pop. The codec owns the
 * small control/telemetry structure; activations, KV, conv state and weights
 * are raw GPU buffers.
 */
export const $ = scope({
  ...sandblaster.import(),

  LlmRuntime: {
    contextCapacity: "u32",
    maxNewTokens: "u32",
    eosToken: "u32",
    promptTokenCount: "u32",

    /** Position occupied by currentToken during the next decode step. */
    position: "u32",
    generatedCount: "u32",
    currentToken: "u32",
    status: "u32",

    /** Increased by the GPU whenever a new output token is committed. */
    telemetryRevision: "u32",
    lastToken: "u32",
    errorCode: "u32",
    pad0: "u32",
  },
});

export const LLM_STATUS = {
  IDLE: 0,
  RUNNING: 1,
  EOS: 2,
  DONE: 3,
  ERROR: 4,
} as const;

export type LlmRuntimeState = {
  contextCapacity: number;
  maxNewTokens: number;
  eosToken: number;
  promptTokenCount: number;
  position: number;
  generatedCount: number;
  currentToken: number;
  status: number;
  telemetryRevision: number;
  lastToken: number;
  errorCode: number;
  pad0: number;
};

/**
 * Field order mirrors runtime.wgsl's LlmRuntime struct exactly (12 consecutive
 * u32, no padding). The WGSL declaration is explicit so the runtime does not
 * depend on Sandblaster exposing its LayoutPlan; this CPU codec is the only
 * place that must stay in lock-step with the struct.
 */
const LLM_RUNTIME_FIELDS = [
  "contextCapacity",
  "maxNewTokens",
  "eosToken",
  "promptTokenCount",
  "position",
  "generatedCount",
  "currentToken",
  "status",
  "telemetryRevision",
  "lastToken",
  "errorCode",
  "pad0",
] as const satisfies readonly (keyof LlmRuntimeState)[];

/** Byte size of the LlmRuntime struct on the GPU (12 u32). */
export const LLM_RUNTIME_BYTES = LLM_RUNTIME_FIELDS.length * 4;

export function serializeLlmRuntime(view: DataView, state: LlmRuntimeState, offset = 0): void {
  let p = offset;
  for (const field of LLM_RUNTIME_FIELDS) {
    view.setUint32(p, state[field] >>> 0, true);
    p += 4;
  }
}

export function deserializeLlmRuntime(view: DataView, offset = 0): LlmRuntimeState {
  const state = {} as LlmRuntimeState;
  let p = offset;
  for (const field of LLM_RUNTIME_FIELDS) {
    state[field] = view.getUint32(p, true);
    p += 4;
  }
  return state;
}

export function createInitialRuntimeState(options: {
  contextCapacity: number;
  maxNewTokens: number;
  eosToken: number;
  promptTokenCount: number;
}): LlmRuntimeState {
  return {
    contextCapacity: options.contextCapacity,
    maxNewTokens: options.maxNewTokens,
    eosToken: options.eosToken,
    promptTokenCount: options.promptTokenCount,
    position: options.promptTokenCount,
    generatedCount: 0,
    currentToken: 0,
    status: LLM_STATUS.RUNNING,
    telemetryRevision: 0,
    lastToken: 0,
    errorCode: 0,
    pad0: 0,
  };
}
