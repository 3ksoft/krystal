// Shared GPU harness for the M1 training vertical slice tests.
//
// `KrystalComputePass.run` resolves passes through the module-level `krystal`
// definition, so the harness compiles that singleton engine exactly once per
// process on a single shared device — the same convention as `installDawn`
// (which guards against a second native Dawn bootstrap aborting). All training
// test files share this one harness instance.
//
// Readbacks go through the Sandblaster `readback()` path (the tested path; a
// raw mapAsync probe earlier returned AbortError). The small training-readback
// staging buffer keeps per-readback cost proportional to the region, not the
// whole arena.
import { expect } from "bun:test";
import { createWebGpuDevice } from "../src/device.ts";
import { krystal, type KrystalDefinition } from "../src/krystal.ts";
import { KrystalExecutor } from "../src/pass.ts";
import { TRAINING_READBACK_ELEMENTS } from "../src/krystal-layout.ts";
import { installDawn } from "./dawn.ts";
import type { KrystalOpParams } from "../src/krystal-layout.ts";

export interface TrainingHarness {
  device: GPUDevice;
  executor: KrystalExecutor;
  definition: KrystalDefinition;
}

let instance: Promise<TrainingHarness> | undefined;

export function getTrainingHarness(): Promise<TrainingHarness> {
  instance ??= (async () => {
    await installDawn();
    // The shared arena now exceeds the default 128 MiB storage-buffer binding
    // limit (M3 added the backward regions). Ask for the adapter's maximum so
    // every program can bind the whole arena; fall back to the default when
    // the adapter reports no higher limit.
    const gpu = (globalThis as { navigator?: { gpu?: GPU } }).navigator?.gpu;
    const adapter = await gpu!.requestAdapter({});
    const limit = adapter!.limits.maxStorageBufferBindingSize;
    const { device } = await createWebGpuDevice({
      label: "krystal.training",
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(limit, 2147483644),
      },
    });
    const compiled = await krystal.engine.compile({ device });
    expect(compiled.failed).toBe(0);
    const executor = new KrystalExecutor(krystal);
    return { device, executor, definition: krystal };
  })();
  return instance;
}

/** Write f32 values into the shared arena at an element offset. */
export async function uploadArena(h: TrainingHarness, offset: number, values: Float32Array): Promise<void> {
  h.device.queue.writeBuffer(h.definition.resources.arena.gpu, offset * 4, values);
  await h.device.queue.onSubmittedWorkDone();
}

/** Write token ids (u32) into the token-id buffer. */
export async function uploadTokens(h: TrainingHarness, values: Uint32Array): Promise<void> {
  h.device.queue.writeBuffer(h.definition.resources.tokens.gpu, 0, values);
  await h.device.queue.onSubmittedWorkDone();
}

/** Write target ids (u32) into the targets buffer. */
export async function uploadTargets(h: TrainingHarness, values: Uint32Array): Promise<void> {
  h.device.queue.writeBuffer(h.definition.resources.targets.gpu, 0, values);
  await h.device.queue.onSubmittedWorkDone();
}

/** Create an f32 weight page (STORAGE | COPY_DST) with the given values. */
export function createWeightPage(h: TrainingHarness, values: Float32Array): GPUBuffer {
  const page = h.device.createBuffer({
    label: "training.weight-page",
    size: Math.max(4, values.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  h.device.queue.writeBuffer(page, 0, values);
  return page;
}

/** Run a single named pass with the given OpParams and optional weight page. */
export async function runPassWait(
  h: TrainingHarness,
  name: string,
  op: Readonly<KrystalOpParams>,
  weight?: GPUBuffer,
): Promise<void> {
  h.executor.submit((encoder) => {
    encoder.compute((pass) => pass.run(name as never, op, weight));
  });
  await h.device.queue.onSubmittedWorkDone();
}

/**
 * Copy an arena region into the small staging buffer and read it back through
 * the Sandblaster readback path.
 */
export async function readArenaRegion(h: TrainingHarness, offset: number, elements: number): Promise<Float32Array> {
  expect(elements).toBeLessThanOrEqual(TRAINING_READBACK_ELEMENTS);
  const arena = h.definition.resources.arena;
  const staging = h.definition.resources.trainingReadback;
  const encoder = h.device.createCommandEncoder();
  encoder.copyBufferToBuffer(arena.gpu, offset * 4, staging.gpu, 0, elements * 4);
  h.device.queue.submit([encoder.finish()]);
  await h.device.queue.onSubmittedWorkDone();
  // Sandblaster readback() returns a plain Array for f32 buffers; normalize
  // to a real Float32Array so callers can rely on buffer/byteOffset semantics.
  const raw = (await staging.readback()) as unknown as ArrayLike<number>;
  return Float32Array.from(raw as ArrayLike<number>).slice(0, elements);
}
