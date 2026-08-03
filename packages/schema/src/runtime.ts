/**
 * CPU <-> GPU ABI for the small inference control/telemetry structure.
 *
 * This package is the interop boundary. The current codec is intentionally
 * kept explicit during the repository split; it is the next candidate to be
 * replaced by the schema-pop generated runtime codec/WGSL layout so the host
 * and shader declarations share one source of truth.
 */
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
 * u32, no padding). The WGSL declaration is still explicit during this mechanical repository split.
 * Migrating this contract to schema-pop generated codec + WGSL is intentionally
 * isolated to this package.
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
