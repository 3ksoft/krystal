import type { GpuWeightFormat } from "../model.ts";
import type { LlmRuntimeState } from "../../../schema/src/schema.ts";
import type { DecodeTelemetryEntry } from "../../../schema/dist/util/codec.ts";

export interface MatmulDispatchArgs {
  rowCount: number;
  tokenCount: number;
  inputDim: number;
  outputDim: number;
}

export interface MatmulKernelSpec {
  /** WGSL entry point compiled into the shared LFM runtime pipeline layout. */
  entryPoint: string;
  /** Optional WGSL appended to runtime.wgsl, useful for drop-in kernel experiments. */
  wgsl?: string;
  /** Override dispatch geometry if a kernel is not [row, token]. */
  workgroups?: (args: MatmulDispatchArgs) => [number, number?, number?];
}

export interface Lfm2RuntimeOptions {
  contextCapacity?: number;
  maxNewTokens?: number;
  /** Swap only matmul implementations without touching layer scheduling/cache code. */
  matmulKernels?: Partial<Record<GpuWeightFormat, MatmulKernelSpec>>;
}

export interface ResolvedLfm2RuntimeOptions {
  contextCapacity: number;
  maxNewTokens: number;
  matmulKernels: Record<GpuWeightFormat, MatmulKernelSpec>;
}

export interface GenerateOptions {
  maxNewTokens?: number;
  /**
   * Diagnostic mode: submit prefill and decode separately and synchronize once
   * between them so wall-clock timings can be reported independently. There is
   * still no token-by-token CPU/GPU roundtrip.
   */
  profile?: boolean;
}

export interface GenerateTimings {
  prefillMs: number;
  decodeMs: number;
  readbackMs: number;
  totalMs: number;
  promptTokens: number;
  scheduledDecodeSteps: number;
  /** Present for generateFromBlocks(): cache depth used by the prefill. Depth > 2 is approximate. */
  cacheDepth?: number;
  cachedBlocks?: number;
  cachedTokens?: number;
  liveQueryTokens?: number;
  repairedTokens?: number;
}

export interface DecodeTelemetryResult {
  /** Raw GPU cursor; may exceed entries.length if the fixed telemetry ring overflowed. */
  cursor: number;
  entries: DecodeTelemetryEntry[];
}

export interface GenerateResult {
  tokenIds: number[];
  state: LlmRuntimeState;
  decodeTelemetry: DecodeTelemetryResult;
  timings?: GenerateTimings;
}

/** Diagnostic-only hook used by the CPU constrained-decoding bring-up path. */
export interface CpuCandidateProcessor {
  candidates(
    context: { step: number; decode: boolean },
  ): Uint32Array | Promise<Uint32Array>;
  accept?(
    tokenId: number,
    context: { step: number; decode: boolean },
  ): void | Promise<void>;
  shouldStop?(): boolean;
}

/** Diagnostic-only full-logit oracle used by constrained-decoding bring-up. */
export interface CpuLogitProcessor {
  process(
    logits: Float32Array,
    context: { step: number; decode: boolean },
  ): void | Promise<void>;
  accept?(
    tokenId: number,
    context: { step: number; decode: boolean },
  ): void | Promise<void>;
  shouldStop?(): boolean;
}

export interface CacheBlockOptions {
  /** Maximum standalone prefix depth retained for this block. */
  depth?: number;
}
