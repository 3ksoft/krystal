import type {
  GpuConstraintDecoderState,
  GpuConstraintProgram,
  GpuConstraintTokenizer,
} from "../../engine-ts/src/gpu-constraint.ts";
import { GPU_CONSTRAINT_STATE } from "../../engine-ts/src/gpu-constraint.ts";
import {
  CONSTRAINT_MASK_WORDS,
  CONSTRAINT_PROGRAM_WORD_CAPACITY,
  CONSTRAINT_TOKENIZER_WORD_CAPACITY,
} from "./lfm2-layout";
import type { Lfm2Definition } from "./lfm2-artifact";

export interface GpuConstraintUpload {
  readonly vocabSize: number;
  readonly maskWords: number;
}

function assertCapacity(label: string, actual: number, capacity: number): void {
  if (actual > capacity) {
    throw new Error(`${label} requires ${actual} u32 words, capacity is ${capacity}`);
  }
}

/**
 * Upload one compiled constraint plus the model-global tokenizer byte table.
 *
 * All four buffers are Sandblaster resources declared in lfm2-definition.ts.
 * This helper deliberately owns no shader module, pipeline or bind group.
 */
export function uploadGpuConstraint(
  definition: Lfm2Definition,
  program: GpuConstraintProgram,
  tokenizer: GpuConstraintTokenizer,
  state: GpuConstraintDecoderState,
): GpuConstraintUpload {
  if (state.byteLength !== GPU_CONSTRAINT_STATE.byteLength) {
    throw new Error(`Constraint decoder state must be ${GPU_CONSTRAINT_STATE.byteLength} bytes`);
  }

  const vocabSize = tokenizer.header[0]!;
  const maskWords = Math.ceil(vocabSize / 32);

  assertCapacity("Constraint program", program.blob.length, CONSTRAINT_PROGRAM_WORD_CAPACITY);
  assertCapacity("Constraint tokenizer", tokenizer.blob.length, CONSTRAINT_TOKENIZER_WORD_CAPACITY);
  assertCapacity("Constraint mask", maskWords, CONSTRAINT_MASK_WORDS);

  const { engine, resources } = definition;
  const queue = engine.device.queue;
  queue.writeBuffer(resources.constraintProgram.gpu, 0, program.blob);
  queue.writeBuffer(resources.constraintTokenizer.gpu, 0, tokenizer.blob);
  queue.writeBuffer(resources.constraintState.gpu, 0, state);

  return { vocabSize, maskWords };
}

/** Update only the 64-byte transactional VM state between decode steps. */
export function writeGpuConstraintState(
  definition: Lfm2Definition,
  state: GpuConstraintDecoderState,
): void {
  if (state.byteLength !== GPU_CONSTRAINT_STATE.byteLength) {
    throw new Error(`Constraint decoder state must be ${GPU_CONSTRAINT_STATE.byteLength} bytes`);
  }
  definition.engine.device.queue.writeBuffer(definition.resources.constraintState.gpu, 0, state);
}

/**
 * Encode the exact vocabulary mask through the normal Sandblaster program.
 * The linked/serialized artifact owns WGSL, layouts and the compute pipeline.
 */
export function dispatchGpuConstraintMask(definition: Lfm2Definition): void {
  definition.engine.submit((encoder) => {
    encoder.compute({ label: "constraint.mask" }, (pass) => {
      pass.run(
        definition.programs.constraint_mask,
        { workgroups: [definition.constraint.maskWorkgroups, 1, 1] },
      );
    });
  });
}

/** Diagnostic readback only; constrained inference should keep the mask on GPU. */
export async function readGpuConstraintMask(
  definition: Lfm2Definition,
  maskWords = CONSTRAINT_MASK_WORDS,
): Promise<Uint32Array> {
  const device = definition.engine.device;
  const byteLength = maskWords * 4;
  const staging = device.createBuffer({
    label: "constraint.mask.readback",
    size: Math.max(4, byteLength),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  try {
    const encoder = device.createCommandEncoder({ label: "constraint.mask.readback" });
    encoder.copyBufferToBuffer(
      definition.resources.constraintMask.gpu,
      0,
      staging,
      0,
      byteLength,
    );
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ, 0, byteLength);
    return new Uint32Array(staging.getMappedRange(0, byteLength).slice(0));
  } finally {
    if (staging.mapState === "mapped") staging.unmap();
    staging.destroy();
  }
}
/** Diagnostic/state-equivalence readback for the 64-byte decoder state. */
export async function readGpuConstraintState(
  definition: Lfm2Definition,
): Promise<Uint32Array> {
  const device = definition.engine.device;
  const byteLength = GPU_CONSTRAINT_STATE.byteLength;
  const staging = device.createBuffer({
    label: "constraint.state.readback",
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  try {
    const encoder = device.createCommandEncoder({ label: "constraint.state.readback" });
    encoder.copyBufferToBuffer(
      definition.resources.constraintState.gpu,
      0,
      staging,
      0,
      byteLength,
    );
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ, 0, byteLength);
    return new Uint32Array(staging.getMappedRange(0, byteLength).slice(0));
  } finally {
    if (staging.mapState === "mapped") staging.unmap();
    staging.destroy();
  }
}
