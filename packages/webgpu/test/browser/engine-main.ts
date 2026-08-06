import { Engine } from "@chomato/engine-ts/transport";
import { BlobSource } from "../../../quant/src/gguf/source";
import { createWebGpuDevice } from "../../src/device";
import { createLfm2WebGpuTransport } from "../../src/engine-transport";
import { Lfm2Forward } from "../../src/forward";
import { lfm2 } from "../../src/lfm2";
import { Lfm2GpuModel } from "../../src/model";

const status = document.querySelector<HTMLDivElement>("#status")!;
const output = document.querySelector<HTMLPreElement>("#output")!;
const fileInput = document.querySelector<HTMLInputElement>("#model")!;
const runButton = document.querySelector<HTMLButtonElement>("#run")!;

function log(message: string, value?: unknown) {
  const suffix = value === undefined ? "" : ` ${typeof value === "string" ? value : JSON.stringify(value, null, 2)}`;
  output.textContent += `${message}${suffix}\n`;
  console.log(message, value ?? "");
}

function sameNumbers(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

async function collect(iterable: AsyncIterable<number>): Promise<number[]> {
  const result: number[] = [];
  for await (const value of iterable) result.push(value);
  return result;
}

async function getDevice(): Promise<GPUDevice> {
  const GIB = 1024 * 1024 * 1024;
  const { device } = await createWebGpuDevice({
    label: "chomato-engine-bridge-smoke",
    requiredLimits: {
      maxBufferSize: GIB,
      maxStorageBufferBindingSize: GIB,
      maxComputeWorkgroupsPerDimension: 65535,
    },
  });
  const compiled = await lfm2.engine.compile({ device });
  if (compiled.failed) throw new Error(`LFM2 compile failed (${compiled.failed}/${compiled.total})`);
  log(`✓ LFM2 programs compiled by Dawn (${compiled.ok}/${compiled.total})`);
  return device;
}

async function run(): Promise<void> {
  const file = fileInput.files?.[0];
  if (!file) throw new Error("Choose WQ4 v3 model first");
  runButton.disabled = true;
  output.textContent = "";
  status.textContent = "RUNNING";

  let model: Lfm2GpuModel | undefined;
  let engine: Engine | undefined;
  try {
    const device = await getDevice();
    model = await Lfm2GpuModel.open(device, new BlobSource(file), { preload: false, drainUploads: true });
    const forward = new Lfm2Forward(model);
    await forward.prepareAll();

    engine = new Engine(createLfm2WebGpuTransport(forward));

    const stableTokens = Uint32Array.of(model.config.bosToken, 42);
    const branchATokens = Uint32Array.of(43);
    const branchBTokens = Uint32Array.of(44);

    const stable = await engine.putBlock(stableTokens);
    const checkpoint = await engine.checkpoint({ blocks: [stable] });
    const branchA = await engine.putBlock(branchATokens);
    const branchB = await engine.putBlock(branchBTokens);

    log("✓ bridge resources", { stable, checkpoint, branchA, branchB });

    const viaBridgeA = await collect(engine.generate({ checkpoint, blocks: [branchA] }, { maxTokens: 2 }));
    const directA = await forward.generateGreedy(
      Uint32Array.of(...stableTokens, ...branchATokens),
      { maxNewTokens: 2, resetState: true },
    );
    if (!sameNumbers(viaBridgeA, directA.tokens)) {
      throw new Error(`branch A mismatch: bridge=${viaBridgeA} direct=${directA.tokens}`);
    }

    const viaBridgeB = await collect(engine.generate({ checkpoint, blocks: [branchB] }, { maxTokens: 2 }));
    const directB = await forward.generateGreedy(
      Uint32Array.of(...stableTokens, ...branchBTokens),
      { maxNewTokens: 2, resetState: true },
    );
    if (!sameNumbers(viaBridgeB, directB.tokens)) {
      throw new Error(`branch B mismatch: bridge=${viaBridgeB} direct=${directB.tokens}`);
    }

    await engine.dropBlock(branchA);
    await engine.dropBlock(branchB);
    await engine.dropCheckpoint(checkpoint);
    await engine.dropBlock(stable);

    log("✓ engine-ts ↔ WebGPU context bridge", {
      checkpointSemantics: "exact recompute",
      branchA: viaBridgeA,
      branchB: viaBridgeB,
      matchesDirectForward: true,
    });
    status.textContent = "PASS · engine-ts ↔ WebGPU bridge";
    status.className = "ok";
  } finally {
    await engine?.close();
    model?.dispose();
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", () => {
  run().catch((error) => {
    status.textContent = "FAILED";
    status.className = "error";
    log("✗", error instanceof Error ? `${error.name}: ${error.message}` : String(error));
    console.error(error);
  });
});
