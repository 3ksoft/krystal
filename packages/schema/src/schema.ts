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


export const schema = scope({
  ...wgsl.import(),

  Lfm2Mode: "'prefill' | 'decode' | 'continuation'",

  OpParams: {
    inputOffset: "u32 = 0",
    outputOffset: "u32 = 0",
    auxOffset: "u32 = 0",
    aux2Offset: "u32 = 0",

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

    // One uniform slot. The legacy runtime only consumes the first 64 B;
    // Sandblaster can use the full 256 B layout directly.
    reserved: ["u32[] == 48", "=", () => new Array(48).fill(0)],
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

