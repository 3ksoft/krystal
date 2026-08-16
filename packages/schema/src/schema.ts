import { } from "./env"
import { type, scope } from "arktype"
import { wgsl } from "@schema-pop/schema";

/**
 * Host/native contracts. These live in schema for one source of truth, but are
 * intentionally excluded from the schema-pop layout build: GPU objects and
 * callbacks have no portable binary representation.
 */
const GPUBufferType = type("object").as<GPUBuffer>();
const GPUDeviceType = type("object").as<GPUDevice>();


/**
 * Hard limits for the first GPU structured-decoder ABI.
 *
 * The physical WGSL representation intentionally stays u32-aligned. The
 * smaller limits are semantic/compiler guards and leave room for later packed
 * encodings without changing the public plan model.
 */
export const STRUCTURED_LIMITS = {
  maxNodes: 0xffff,
  maxSwitchEdges: 0xff,
  maxLiteralBytes: 0xffff,
  maxStringLength: 0xffff,
  maxNumberChars: 64,
  maxTokenBytes: 0xff,
  maxVocabSize: 0x1_0000,
} as const;

export const CONSTRAINT_INVALID_NODE = 0xffff_ffff;
export const CONSTRAINT_NUMBER_FLAGS = {
  integer: 1 << 0,
  hasMin: 1 << 1,
  hasMax: 1 << 2,
  hasStep: 1 << 3,
} as const;
export const CONSTRAINT_TOKEN_META = {
  lengthMask: 0xff,
  special: 1 << 8,
} as const;

export const schema = scope({
  ...wgsl.import(),

  // -----------------------------------------------------------------------
  // Structured decoder GPU ABI
  // -----------------------------------------------------------------------
  //
  // Host-only LayoutPlan concepts (object/enum/optional/reference/array) are
  // compiled away before upload. The device executes a deterministic byte VM:
  // literal runs, sparse byte switches/tries, bounded strings and numbers.
  // Choice/split nodes from the CPU oracle are linked into switch/trie nodes,
  // so one generation owns one compact decoder state instead of an NFA branch
  // set.

  ConstraintNodeKind: "'literal' | 'switch' | 'string' | 'number' | 'accept' | 'jump'",
  ConstraintDecoderStatus: "'running' | 'accept' | 'dead' | 'error'",

  /**
   * Header of one packed constraint-program blob. Offsets are measured in u32
   * words, except byteLength which is the logical byte-pool length.
   */
  ConstraintProgramHeader: {
    version: "u32 = 2",
    flags: "u32 = 0",
    entryNode: "u32 = 0",
    acceptNode: "u32 = 0",

    nodeWordOffset: "u32 = 0",
    nodeCount: "u32 = 0",
    edgeWordOffset: "u32 = 0",
    edgeCount: "u32 = 0",

    byteWordOffset: "u32 = 0",
    byteLength: "u32 = 0",
    reserved0: "u32 = 0",
    reserved1: "u32 = 0",
  },

  /**
   * Fixed 48-byte instruction record. Interpretation:
   *
   * literal: dataOffset=byte offset, dataCount=byte length, next=continuation
   * switch:  dataOffset=edge index, dataCount=edge count, next=terminal/default
   * string:  arg0=minLength, arg1=maxLength, next=continuation
   * number:  arg0=flags, arg1=maxChars,
   *          arg2/3=min byte offset/length, arg4/5=max offset/length,
   *          arg6/7=step offset/length (step rejected by GPU ABI v1)
   * accept:  payload ignored
   * jump:    epsilon control-flow barrier; next=continuation
   *
   * Number flags: bit0=integer, bit1=hasMin, bit2=hasMax, bit3=hasStep.
   * A switch `next` of 0xffffffff means that its current trie prefix is not a
   * terminal alternative.
   */
  ConstraintNode: {
    kind: "ConstraintNodeKind = 'accept'",
    next: "u32 = 0",
    dataOffset: "u32 = 0",
    dataCount: "u32 = 0",
    arg0: "u32 = 0",
    arg1: "u32 = 0",
    arg2: "u32 = 0",
    arg3: "u32 = 0",
    arg4: "u32 = 0",
    arg5: "u32 = 0",
    arg6: "u32 = 0",
    arg7: "u32 = 0",
  },

  /**
   * One packed sparse byte-trie edge:
   *   bits  0..7  = input byte
   *   bits  8..23 = target node (maxNodes=65535)
   *   bits 24..31 = flags (bit 24 = replay input byte at target)
   */
  ConstraintByteEdge: {
    word: "u32 = 0",
  },

  /**
   * Tokenizer metadata is model-global and independent from any one schema.
   * The byte pool contains raw token bytes, not canonical token IDs for text.
   */
  ConstraintTokenizerHeader: {
    tokenCount: "u32 = 0",
    eosToken: "u32 = 0",
    entryWordOffset: "u32 = 0",
    byteWordOffset: "u32 = 0",
    byteLength: "u32 = 0",
    reserved0: "u32 = 0",
    reserved1: "u32 = 0",
    reserved2: "u32 = 0",
  },

  /**
   * Two words per vocabulary item. meta bits 0..7 = byte length, bit 8 =
   * special token. maxTokenBytes=255 is therefore an ABI/compiler guard.
   */
  ConstraintTokenByteEntry: {
    byteOffset: "u32 = 0",
    meta: "u32 = 0",
  },

  /**
   * One deterministic decoder state per live generation. The local words are
   * interpreted by the current node:
   *   literal -> local0 = literal cursor
   *   string  -> local0 = phase, local1 = logical length (local2 unused since
   *              \uXXXX left the accepted language; the word stays for ABI)
   *   number  -> local0 = numeric text length, local1 = JSON-number lexer phase;
   *              numberText stores <=64 ASCII bytes
   *   switch/accept/jump -> locals unused
   *
   * Keeping the bounded number lexeme in-state avoids any per-candidate scan
   * through the generated token stream. 96 B total is cheap enough to clone
   * into private WGSL state for one-token simulation.
   */
  ConstraintDecoderState: {
    node: "u32 = 0",
    local0: "u32 = 0",
    local1: "u32 = 0",
    local2: "u32 = 0",
    status: "ConstraintDecoderStatus = 'running'",
    errorCode: "u32 = 0",
    reserved0: "u32 = 0",
    reserved1: "u32 = 0",
    numberText: ["u32[] == 16", "=", () => new Array(16).fill(0)],
  },

  Lfm2Mode: "'prefill' | 'decode' | 'continuation'",

  OpParams: {
    inputOffset: "u32 = 0",
    outputOffset: "u32 = 0",
    auxOffset: "u32 = 0",
    aux2Offset: "u32 = 0",

    // Extended offsets for ops that move more than four tensor regions
    // (training attention reads Q/K/V/mask and writes out/P/dQ/dK/dV).
    aux3Offset: "u32 = 0",
    aux4Offset: "u32 = 0",
    aux5Offset: "u32 = 0",
    aux6Offset: "u32 = 0",

    tokenCount: "u32 = 0",
    inputDim: "u32 = 0",
    outputDim: "u32 = 0",
    rowStart: "u32 = 0",

    rowCount: "u32 = 0",
    layerIndex: "u32 = 0",
    attentionSlot: "u32 = 0",
    mode: "Lfm2Mode = 'prefill'",

    f0: "f32 = 0",
    f1: "f32 = 0",
    u0: "u32 = 0",
    u1: "u32 = 0",

    // Extra scalar u32s for ops that carry more than two small integers
    // (krystal_field_embed passes six embedding-table bases).
    u2: "u32 = 0",
    u3: "u32 = 0",
    u4: "u32 = 0",
    u5: "u32 = 0",
  },

  LlmRuntimeStatus: "'idle' | 'running' | 'eos' | 'done' | 'error'",

  LlmRuntime: {
    contextCapacity: "u32 = 0",
    maxNewTokens: "u32 = 0",
    eosToken: "u32 = 0",
    promptTokenCount: "u32 = 0",

    position: "u32 = 0",
    generatedCount: "u32 = 0",
    currentToken: "u32 = 0",
    status: "LlmRuntimeStatus = 'idle'",

    telemetryRevision: "u32 = 0",
    lastToken: "u32 = 0",
    errorCode: "u32 = 0",
    pad0: "u32 = 0",
  },

  DecodeTelemetryEntry: {
    position: "u8 = 0",
    status: "u4 = 0",
    tokenId: "u16 = 0",
  },

  GenerateOptions: {
    "maxNewTokens?": "number",
    "profile?": "boolean",
  },

  GenerateTimings: {
    prefillMs: "number",
    decodeMs: "number",
    readbackMs: "number",
    totalMs: "number",
    promptTokens: "number",
    scheduledDecodeSteps: "number",

    "cacheDepth?": "number",
    "cachedBlocks?": "number",
    "cachedTokens?": "number",
    "liveQueryTokens?": "number",
    "repairedTokens?": "number",
  },

  GenerateResult: {
    tokenIds: "number[]",
    state: "LlmRuntime",
    "timings?": "GenerateTimings",
  },

  CacheBlockOptions: {
    "depth?": "number",
  },

  GpuWeightFormat: "'f16' | 'f32' | 'wq4'",
  Lfm2LayerKind: "'conv' | 'attention'",

  Lfm2RuntimeConfig: {
    contextLength: "number",
    hiddenSize: "number",
    feedForwardSize: "number",
    attentionHeads: "number",
    kvHeadsByLayer: "number[]",
    headDim: "number",
    ropeTheta: "number",
    vocabSize: "number",
    convCacheLength: "number",
    normEpsilon: "number",
    eosToken: "number",
    blockCount: "number",
    layers: "Lfm2LayerKind[]",
    attentionLayerSlots: "number[]",
  },

  MatmulDispatchArgs: {
    rowCount: "number",
    tokenCount: "number",
    inputDim: "number",
    outputDim: "number",
  },

  GPUBuffer: GPUBufferType,
  GPUDevice: GPUDeviceType,

  GpuTensorPage: {
    buffer: "GPUBuffer",
    rowStart: "number",
    rowCount: "number",
    byteLength: "number",
  },

  GpuTensor: {
    name: "string",
    format: "GpuWeightFormat",
    dimensions: "number[]",
    pages: "GpuTensorPage[]",
    "byteLength?": "number",
  },

  Lfm2RuntimeModel: {
    device: "GPUDevice",
    config: "Lfm2RuntimeConfig",
    tensor: "Function",
  },

  MatmulKernelSpec: {
    entryPoint: "string",
    "wgsl?": "string",
    "workgroups?": "Function",
  },

  Lfm2RuntimeOptions: {
    "contextCapacity?": "number",
    "maxNewTokens?": "number",
    "matmulKernels?": {
      "f16?": "MatmulKernelSpec",
      "f32?": "MatmulKernelSpec",
      "wq4?": "MatmulKernelSpec",
    },
  },
});


export const $ = schema;
