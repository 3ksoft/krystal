import { GPU_SCHEMA_SENTINELS } from "../../../schema/src/sparse";
import { createWebGpuDevice } from "../../src/device";
import { Lfm2Forward } from "../../src/forward";
import { LFM2_ARENA, lfm2 } from "../../src/lfm2";
import type { Lfm2GpuModel } from "../../src/model";

const status = document.querySelector<HTMLDivElement>("#status")!;
const output = document.querySelector<HTMLPreElement>("#output")!;

function log(message: string, value?: unknown): void {
  const suffix = value === undefined
    ? ""
    : ` ${typeof value === "string" ? value : JSON.stringify(value, null, 2)}`;
  output.textContent += `${message}${suffix}\n`;
  console.log(message, value ?? "");
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function createDevice(): Promise<GPUDevice> {
  const GIB = 1024 * 1024 * 1024;
  const { adapter, device } = await createWebGpuDevice({
    label: "chomato-guide-smoke",
    requiredLimits: {
      maxBufferSize: GIB,
      maxStorageBufferBindingSize: GIB,
      maxComputeWorkgroupsPerDimension: 65535,
    },
  });

  log("✓ WebGPU device", {
    adapter: adapter.info
      ? `${adapter.info.vendor} · ${adapter.info.architecture}`
      : "unknown",
  });

  const compiled = await lfm2.engine.compile({ device });
  if (compiled.failed) {
    throw new Error(`LFM2 compile failed ${compiled.failed}/${compiled.total}`);
  }
  log(`✓ LFM2 programs compiled by Dawn (${compiled.ok}/${compiled.total})`);
  return device;
}

/**
 * The guide sampler itself needs no model weights. Lfm2Forward only needs the
 * device plus vocab/eos metadata for request bookkeeping on this path.
 */
function createSamplerForward(device: GPUDevice): Lfm2Forward {
  const model = {
    device,
    config: {
      vocabSize: 0x10000,
      eosToken: 64401,
    },
  } as unknown as Lfm2GpuModel;
  return new Lfm2Forward(model);
}

function writeSyntheticLogits(
  device: GPUDevice,
  values: ReadonlyMap<number, number>,
): void {
  const logits = new Float32Array(0x10000);
  logits.fill(-1000);
  for (const [token, value] of values) logits[token] = value;
  device.queue.writeBuffer(
    lfm2.resources.arena.gpu,
    LFM2_ARENA.logits * 4,
    logits,
  );
}

async function selectedToken(device: GPUDevice): Promise<number> {
  await device.queue.onSubmittedWorkDone();
  const readback = await lfm2.resources.tokens.readback({
    dropIfBusy: false,
  }) as any;
  return Number(readback[lfm2.capacities.context]);
}

async function runtimeState(device: GPUDevice): Promise<any> {
  await device.queue.onSubmittedWorkDone();
  return await lfm2.resources.runtime.readback({ dropIfBusy: false }) as any;
}

async function constrainedCandidateSmoke(
  device: GPUDevice,
  forward: Lfm2Forward,
): Promise<void> {
  const EMPTY = GPU_SCHEMA_SENTINELS.emptyToken;
  const allowedA = 101;
  const allowedB = 202;
  const forbiddenWinner = 303;

  // The globally best logit is deliberately NOT in the guide candidate set.
  // EMPTY is even higher to prove it is padding, never a real candidate.
  writeSyntheticLogits(device, new Map([
    [allowedA, 4],
    [allowedB, 7],
    [forbiddenWinner, 10_000],
    [EMPTY, 20_000],
  ]));

  const candidates = Uint32Array.of(allowedA, EMPTY, allowedB, EMPTY);
  forward.initializeRequest(1, 1);
  forward.writeCandidateTokens(candidates);
  forward.executor.submit((encoder) => {
    encoder.compute((pass) => {
      forward.commitArgmaxCandidates(pass, candidates.length, "prefill");
    }, { label: "lfm2.guide-smoke.constrained" });
  });

  const token = await selectedToken(device);
  assert(token === allowedB, `guide selected ${token}, expected ${allowedB}`);

  log("✓ sparse guide constrains argmax", {
    selected: token,
    allowed: [allowedA, allowedB],
    forbiddenGlobalWinner: forbiddenWinner,
    empty: `0x${EMPTY.toString(16)}`,
  });
}

async function ordinaryArgmaxSentinelSmoke(
  device: GPUDevice,
  forward: Lfm2Forward,
): Promise<void> {
  const EMPTY = GPU_SCHEMA_SENTINELS.emptyToken;
  const legalWinner = 303;

  // Reuse synthetic logits from the previous test: EMPTY has the largest
  // value, but ordinary argmax must globally reserve it as a sentinel.
  forward.initializeRequest(1, 1);
  forward.executor.submit((encoder) => {
    encoder.compute((pass) => {
      forward.commitArgmax(pass, "prefill");
    }, { label: "lfm2.guide-smoke.normal-argmax" });
  });

  const token = await selectedToken(device);
  assert(token === legalWinner, `ordinary argmax selected ${token}, expected ${legalWinner}`);
  log("✓ ordinary argmax globally reserves EMPTY_TOKEN", { selected: token });
}

async function emptyGuideSmoke(
  device: GPUDevice,
  forward: Lfm2Forward,
): Promise<void> {
  const EMPTY = GPU_SCHEMA_SENTINELS.emptyToken;
  const candidates = Uint32Array.of(EMPTY, EMPTY, EMPTY, EMPTY);

  forward.initializeRequest(1, 1);
  forward.writeCandidateTokens(candidates);
  forward.executor.submit((encoder) => {
    encoder.compute((pass) => {
      forward.commitArgmaxCandidates(pass, candidates.length, "prefill");
    }, { label: "lfm2.guide-smoke.empty" });
  });

  const runtime = await runtimeState(device);
  assert(String(runtime.status) === "error", `empty guide status=${String(runtime.status)}, expected error`);
  assert(Number(runtime.errorCode) === 0x47554944, `empty guide errorCode=0x${Number(runtime.errorCode).toString(16)}, expected GUID`);

  log("✓ all-EMPTY guide fails closed", {
    status: String(runtime.status),
    errorCode: `0x${Number(runtime.errorCode).toString(16)}`,
  });
}

async function run(): Promise<void> {
  output.textContent = "";
  status.textContent = "initializing Dawn…";

  const device = await createDevice();
  device.addEventListener("uncapturederror", (event: GPUUncapturedErrorEvent) => {
    console.error("WebGPU uncaptured error", event.error);
  });

  const forward = createSamplerForward(device);
  await constrainedCandidateSmoke(device, forward);
  await ordinaryArgmaxSentinelSmoke(device, forward);
  await emptyGuideSmoke(device, forward);

  status.textContent = "PASS";
  status.className = "ok";
  log("✓ guide → candidates → logits → constrained token");
}

run().catch((error) => {
  status.textContent = "FAIL";
  status.className = "error";
  log("✗", error instanceof Error ? `${error.name}: ${error.message}` : String(error));
  console.error(error);
});
