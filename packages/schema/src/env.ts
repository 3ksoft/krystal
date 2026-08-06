/// <reference types="@webgpu/types" />

/**
 * Keep native WebGPU host objects opaque when ArkType computes inferred types.
 * This is type-level configuration only; no runtime constructors/polyfills are
 * required for the WebGPU interfaces themselves.
 */
declare global {
  interface ArkEnv {
    prototypes():
      | GPUAdapter
      | GPUDevice
      | GPUQueue
      | GPUBuffer
      | GPUTexture
      | GPUTextureView
      | GPUSampler
      | GPUBindGroup
      | GPUBindGroupLayout
      | GPUPipelineLayout
      | GPUShaderModule
      | GPUComputePipeline
      | GPURenderPipeline
      | GPUCommandEncoder
      | GPUComputePassEncoder
      | GPURenderPassEncoder;
  }
}

export {};
