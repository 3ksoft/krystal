// Local loadModel adapter — the seam behind tests/checkpoint.test.ts.
//
// Default leg: the REAL engine. When the WQ4 model file exists and the
// `webgpu` Dawn bindings are installed (packages/backend/node_modules/webgpu),
// loadModel() runs the actual Lfm2Forward on the GPU:
//
//   device -> lfm2.engine.compile -> Lfm2GpuModel.open -> Lfm2Forward
//     -> createLfm2WebGpuTransport -> engine-ts Engine
//
// Checkpoints are exact physical KV/conv snapshots, so the checkpoint tests
// verify real semantics (the checkpoint prefix is never re-prefilled).
//
// Fallback leg (no model file / no Dawn / no GPU): spawns the native mock exe
// and speaks the Bridge ABI over stdio, so the same tests still pass on
// GPU-less CI. A warning is printed so it is never silent.
//
// Sandblaster's lfm2 singleton can compile() only once per process, so the
// device/model/forward are created lazily once and shared; each loadModel()
// returns a fresh Engine over its own in-process transport. dispose() closes
// that Engine (releasing its per-engine checkpoint buffers); the shared GPU
// runtime intentionally lives for the whole process.
import { existsSync } from "node:fs";
import { open, type FileHandle } from "node:fs/promises";
import { resolve } from "node:path";
import { BinaryEngineTransport } from "@chomato/engine-ts/binary-transport";
import { SpawnedNativeChannel } from "@chomato/engine-ts/spawn";
import { Engine } from "@chomato/engine-ts/transport";
import type { RandomAccessSource } from "../../quant/src/gguf/source.ts";
import { createLfm2WebGpuTransport } from "../../webgpu/src/engine-transport.ts";
import { createWebGpuDevice } from "../../webgpu/src/device.ts";
import { Lfm2Forward } from "../../webgpu/src/forward.ts";
import { lfm2 } from "../../webgpu/src/lfm2.ts";
import { Lfm2GpuModel } from "../../webgpu/src/model.ts";
import { pickExeCommand } from "./exe/pick-command.ts";

export interface LocalModel {
  readonly engine: Engine;
  /**
   * Real WebGPU engine instance when available (undefined on the mock exe
   * fallback). Exposed for benchmark/timing tests that need forward-level
   * control: per-phase timing, kernel micro-benchmarks and structured
   * generation (which the mock backend does not implement).
   */
  readonly forward?: Lfm2Forward;
  dispose(): Promise<void>;
}

const GIB = 1024 * 1024 * 1024;
const DEFAULT_MODEL = "models/LFM2.5-1.2B-Instruct-WQ4.wq4";

/** Streaming file source for the WQ4 container (no whole-file slurp). */
class NodeFileSource implements RandomAccessSource {
  readonly size: number;

  private constructor(private readonly handle: FileHandle, size: number) {
    this.size = size;
  }

  static async open(path: string): Promise<NodeFileSource> {
    const handle = await open(path, "r");
    try {
      const stat = await handle.stat();
      if (!stat.isFile) throw new Error(`Not a file: ${path}`);
      return new NodeFileSource(handle, stat.size);
    } catch (error) {
      await handle.close();
      throw error;
    }
  }

  async read(offset: number, length: number): Promise<Uint8Array> {
    if (offset < 0 || length < 0 || offset + length > this.size) {
      throw new RangeError(`Read [${offset}, ${offset + length}) outside file size ${this.size}`);
    }
    const out = new Uint8Array(length);
    let cursor = 0;
    while (cursor < length) {
      const { bytesRead } = await this.handle.read(out, cursor, length - cursor, offset + cursor);
      if (bytesRead === 0) throw new Error(`Unexpected EOF at ${offset + cursor}`);
      cursor += bytesRead;
    }
    return out;
  }

  async close(): Promise<void> {
    await this.handle.close();
  }
}

interface SharedRealEngine {
  readonly device: GPUDevice;
  readonly model: Lfm2GpuModel;
  readonly forward: Lfm2Forward;
}

let sharedReal: SharedRealEngine | undefined;

/** Resolve the model path against cwd and the repo root; also try the default. */
function resolveModelPath(modelPath?: string): string | undefined {
  const repoRoot = resolve(import.meta.dir, "..", "..", "..");
  const candidates = [
    modelPath ? resolve(process.cwd(), modelPath) : undefined,
    modelPath ? resolve(repoRoot, modelPath) : undefined,
    resolve(repoRoot, DEFAULT_MODEL),
  ].filter((candidate): candidate is string => Boolean(candidate));
  return candidates.find((candidate) => existsSync(candidate));
}

async function createSharedRealEngine(modelPath: string): Promise<SharedRealEngine> {
  // Dawn bindings: install navigator.gpu exactly like dawn-backend.ts.
  const { create, globals } = await import("webgpu");
  Object.assign(globalThis, globals);
  Object.defineProperty(globalThis, "navigator", { value: { gpu: create([]) }, configurable: true });

  const { adapter, device } = await createWebGpuDevice({
    label: "chomato-local-model",
    requiredLimits: {
      maxBufferSize: GIB,
      maxStorageBufferBindingSize: GIB,
      maxComputeWorkgroupsPerDimension: 65535,
    },
  });
  void adapter;

  const compiled = await lfm2.engine.compile({ device });
  if (compiled.failed) throw new Error(`LFM2 compile failed ${compiled.failed}/${compiled.total}`);

  const model = await Lfm2GpuModel.open(device, await NodeFileSource.open(modelPath), {
    preload: "all",
    drainUploads: true,
  });
  const forward = new Lfm2Forward(model);
  await forward.prepareAll();
  return { device, model, forward };
}

function loadMock(): LocalModel {
  const exe = pickExeCommand();
  const engine = new Engine(
    new BinaryEngineTransport(new SpawnedNativeChannel(exe.command, exe.args)),
  );
  return {
    engine,
    dispose: () => engine.close(),
  };
}

/** Adapter behind `loadModel(...).engine`: real WebGPU engine or mock fallback. */
export async function loadModel(modelPath?: string): Promise<LocalModel> {
  const resolved = resolveModelPath(modelPath);

  if (!resolved) {
    console.warn(
      `[loadModel] WQ4 model not found (${modelPath ?? DEFAULT_MODEL}) — falling back to the mock exe backend.`,
    );
    return loadMock();
  }

  if (!sharedReal) {
    try {
      sharedReal = await createSharedRealEngine(resolved);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.warn(
        `[loadModel] real engine unavailable (${message}) — falling back to the mock exe backend.`,
      );
      return loadMock();
    }
  }

  const engine = new Engine(createLfm2WebGpuTransport(sharedReal.forward));
  return {
    engine,
    forward: sharedReal.forward,
    dispose: async () => {
      await engine.close();
      // The shared device/model/forward intentionally outlive individual
      // loadModel() calls: lfm2.engine.compile() can only run once per process.
    },
  };
}
