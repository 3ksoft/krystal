import type {
  GpuConstraintDecoderState,
  GpuConstraintProgram,
  GpuConstraintTokenizer,
} from "../../engine-ts/src/gpu-constraint.ts";
// import { GPU_CONSTRAINT_STATE } from "../../engine-ts/src/gpu-constraint.ts";
// import { shaderSources } from "./shaders.generated";

const MASK_WORKGROUP_SIZE = 64;

function align4(value: number): number {
  return (value + 3) & ~3;
}

function createStorageBuffer(
  device: GPUDevice,
  label: string,
  data: Uint32Array,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: Math.max(4, align4(data.byteLength)),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | usage,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

export interface GpuConstraintMaskResources {
  readonly maskBuffer: GPUBuffer;
  readonly maskWords: number;
  readonly vocabSize: number;
  writeState(state: GpuConstraintDecoderState): void;
  encode(encoder: GPUCommandEncoder): void;
  readMask(): Promise<Uint32Array>;
  destroy(): void;
}

/**
 * Raw-WebGPU execution wrapper for the upload blobs produced by engine-ts.
 *
 * This is intentionally not wired into Lfm2Runtime yet. It proves the exact
 * byte-VM ABI first; the runtime integration can later bind `maskBuffer`
 * directly to masked argmax without ever reading it back to the CPU.
 */
export async function createGpuConstraintMaskResources(
  device: GPUDevice,
  program: GpuConstraintProgram,
  tokenizer: GpuConstraintTokenizer,
  state: GpuConstraintDecoderState,
): Promise<GpuConstraintMaskResources> {
  if (state.byteLength !== GPU_CONSTRAINT_STATE.byteLength) {
    throw new Error(`Constraint decoder state must be ${GPU_CONSTRAINT_STATE.byteLength} bytes`);
  }

  const vocabSize = tokenizer.header[0]!;
  const maskWords = Math.ceil(vocabSize / 32);

  const programBuffer = createStorageBuffer(
    device,
    "constraint.program",
    program.blob,
    0,
  );
  const tokenizerBuffer = createStorageBuffer(
    device,
    "constraint.tokenizer",
    tokenizer.blob,
    0,
  );
  const stateBuffer = createStorageBuffer(
    device,
    "constraint.state",
    state,
    GPUBufferUsage.COPY_SRC,
  );
  const maskBuffer = device.createBuffer({
    label: "constraint.mask",
    size: Math.max(4, maskWords * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const module = device.createShaderModule({
    label: "constraint-mask",
    code: shaderSources.constraint_mask,
  });

  const pipeline = await device.createComputePipelineAsync({
    label: "constraint-mask",
    layout: "auto",
    compute: { module, entryPoint: "constraint_mask" },
  });

  const bindGroup = device.createBindGroup({
    label: "constraint-mask",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: programBuffer } },
      { binding: 1, resource: { buffer: tokenizerBuffer } },
      { binding: 2, resource: { buffer: stateBuffer } },
      { binding: 3, resource: { buffer: maskBuffer } },
    ],
  });

  const writeState = (next: GpuConstraintDecoderState): void => {
    if (next.byteLength !== GPU_CONSTRAINT_STATE.byteLength) {
      throw new Error(`Constraint decoder state must be ${GPU_CONSTRAINT_STATE.byteLength} bytes`);
    }
    device.queue.writeBuffer(stateBuffer, 0, next);
  };

  const encode = (encoder: GPUCommandEncoder): void => {
    const pass = encoder.beginComputePass({ label: "constraint-mask" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(maskWords / MASK_WORKGROUP_SIZE));
    pass.end();
  };

  const readMask = async (): Promise<Uint32Array> => {
    const staging = device.createBuffer({
      label: "constraint.mask.readback",
      size: Math.max(4, maskWords * 4),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = device.createCommandEncoder({ label: "constraint-mask.readback" });
      encode(encoder);
      encoder.copyBufferToBuffer(maskBuffer, 0, staging, 0, maskWords * 4);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      return new Uint32Array(staging.getMappedRange().slice(0));
    } finally {
      if (staging.mapState === "mapped") staging.unmap();
      staging.destroy();
    }
  };

  return {
    maskBuffer,
    maskWords,
    vocabSize,
    writeState,
    encode,
    readMask,
    destroy() {
      programBuffer.destroy();
      tokenizerBuffer.destroy();
      stateBuffer.destroy();
      maskBuffer.destroy();
    },
  };
}
