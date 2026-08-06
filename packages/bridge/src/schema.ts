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

  Generate: {
    kind: "'Generate'",
    operation: "u32",
    context: "ContextRef",
    maxTokens: "u32",
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

  Failed: {
    kind: "'Failed'",
    operation: "u32",
    messageBytes: "u16",
    code: "ErrorCode",
    reserved: "u8",
  },

  EngineEvent: "Completed | TokenEmitted | Failed",
});

export const bridge = $.export();
