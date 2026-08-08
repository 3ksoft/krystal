import { scope } from "arktype";
import { binary } from "@schema-pop/schema";
export { CHOMATO_BRIDGE_MAGIC, CHOMATO_BRIDGE_VERSION, NO_CHECKPOINT, MAX_CONTEXT_BLOCKS } from "./constants";

/**
 * Chomato's portable control ABI.
 *
 * There are deliberately no variable-size arrays or strings in this schema.
 * Bulk data is carried as the frame payload:
 *   - PutBlock: tokenCount little-endian u32 token IDs
 *   - CreateCheckpoint / Generate: context.blockCount little-endian u32 block IDs
 *   - Failed event: messageBytes UTF-8 bytes
 *
 * Keeping the control body fixed-size makes the generated C/C++ ABI small and
 * avoids forcing every tagged union variant to reserve space for model context.
 */
const primitives = binary.export();

export const $ = scope({
  u8: primitives.u8,
  u16: primitives.u16,
  u32: primitives.u32,
  f32: primitives.f32,

  FrameDirection: "'command' | 'event'",
  FrameHeader: {
    magic: "u32",
    version: "u16",
    direction: "FrameDirection",
    flags: "u8",
    bodyBytes: "u32",
    payloadBytes: "u32",
  },

  /** checkpoint=0 means no checkpoint. Ordered block IDs live in payload. */
  ContextRef: {
    checkpoint: "u32",
    blockCount: "u16",
    reserved: "u16",
  },

  PutBlock: {
    kind: "'PutBlock'",
    operation: "u32",
    block: "u32",
    tokenCount: "u32",
  },

  DropBlock: {
    kind: "'DropBlock'",
    operation: "u32",
    block: "u32",
  },

  CreateCheckpoint: {
    kind: "'CreateCheckpoint'",
    operation: "u32",
    checkpoint: "u32",
    context: "ContextRef",
  },

  DropCheckpoint: {
    kind: "'DropCheckpoint'",
    operation: "u32",
    checkpoint: "u32",
  },

  /**
   * Token selection for one Generate.
   *
   * 'argmax' is greedy and fully deterministic on its own. 'topk' draws from
   * the k highest logits at the given temperature using a seeded Gumbel-max,
   * so it is deterministic in (seed, context, options) rather than random —
   * replaying a generation is a matter of resending the same seed.
   */
  Sampler: "'argmax' | 'topk'",

  Generate: {
    kind: "'Generate'",
    operation: "u32",
    context: "ContextRef",
    maxTokens: "u32",
    /**
     * Softmax temperature; must be > 0 for sampler 'topk'.
     *
     * Field ORDER here is load-bearing. The generated C++ struct is not
     * packed, so it only agrees with this wire layout while every member also
     * lands on its natural alignment — widest first, and `reserved` paying for
     * the tail padding the compiler would add anyway. packages/bridge/test/
     * native-smoke.cpp asserts exactly that.
     */
    temperature: "f32",
    /** RNG seed. Backends must not substitute their own entropy for it. */
    seed: "u32",
    /** Candidate count, 1..64 (the GPU per-lane list capacity). */
    topK: "u16",
    /** Sampler for this generation. 'argmax' ignores the three fields above. */
    sampler: "Sampler",
    reserved: "u8",
  },

  Cancel: {
    kind: "'Cancel'",
    operation: "u32",
    target: "u32",
  },

  EngineCommand:
    "PutBlock | DropBlock | CreateCheckpoint | DropCheckpoint | Generate | Cancel",

  ErrorCode:
    "'InvalidCommand' | 'InvalidContext' | 'NotFound' | 'CapacityExceeded' | 'Cancelled' | 'InternalError'",

  Completed: {
    kind: "'Completed'",
    operation: "u32",
  },

  TokenEmitted: {
    kind: "'TokenEmitted'",
    operation: "u32",
    token: "u32",
  },

  /**
   * Backend-reported execution facts. May be emitted for Generate and
   * CreateCheckpoint operations. All byte/time counters are physical facts,
   * never inferred by engine-ts from the request shape.
   *
   * `checkpointRestoreUs === 0` means the backend did not measure restore
   * duration separately (the WebGPU backend intentionally avoids an extra
   * submit/wait solely for instrumentation).
   */
  ExecutionStats: {
    kind: "'ExecutionStats'",
    operation: "u32",
    prefillTokens: "u32",
    checkpointHits: "u32",
    checkpointMisses: "u32",
    restoredBytes: "u32",
    checkpointBytes: "u32",
    kvBytes: "u32",
    kvCapacityBytes: "u32",
    convBytes: "u32",
    hiddenBytes: "u32",
    checkpointCreateUs: "u32",
    checkpointRestoreUs: "u32",
  },

  Failed: {
    kind: "'Failed'",
    operation: "u32",
    messageBytes: "u16",
    code: "ErrorCode",
    reserved: "u8",
  },

  EngineEvent: "Completed | TokenEmitted | ExecutionStats | Failed",
});

export const bridge = $.export();
