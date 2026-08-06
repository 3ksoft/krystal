const divStatus = document.querySelector<HTMLDivElement>("#divStatus")!;
const divOutput = document.querySelector<HTMLPreElement>("#divOutput")!;

function log(message: string, value?: unknown) {
  const suffix = value === undefined
    ? ""
    : ` ${typeof value === "string" ? value : JSON.stringify(value, null, 2)}`;
  divOutput.textContent += `${message}${suffix}\n`;
  console.log(message, value ?? "");
}

function fail(error: unknown): never {
  divStatus.textContent = "FAILED";
  divStatus.className = "error";
  const message = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  log("✗", message);
  console.error(error);
  throw error;
}

window.addEventListener("unhandledrejection", (event) => {
  divStatus.textContent = "FAILED";
  divStatus.className = "error";
  log("✗ unhandled rejection", String(event.reason));
});

window.addEventListener("error", (event) => {
  divStatus.textContent = "FAILED";
  divStatus.className = "error";
  log("✗ window error", event.message);
});

async function main() {
  log("[smoke] importing LFM2 GPU definition…");

  // Dynamic import lets the page report parser/linker failures instead of
  // dying before the harness itself has initialized.
  const { lfm2, LFM2_SHADER_NAMES } = await import("../../src/lfm2");
  log(`✓ linked ${LFM2_SHADER_NAMES.length} shader programs`);

  if (!navigator.gpu) throw new Error("navigator.gpu is unavailable; open this page in a WebGPU-enabled Chromium build");

  const GIB = 1024 * 1024 * 1024;
  const requiredLimits: Record<string, number> = {
    maxBufferSize: GIB,
    maxStorageBufferBindingSize: GIB,
    maxComputeWorkgroupsPerDimension: 65535,
  };

  log("[smoke] compiling through Chromium/Dawn…");
  const compiled = await lfm2.engine.compile({ requiredLimits } as any);
  const device: GPUDevice = compiled.device;

  device.addEventListener("uncapturederror", (event) => {
    divStatus.textContent = "FAILED";
    divStatus.className = "error";
    log("✗ WebGPU uncaptured error", event.error.message);
    console.error(event.error);
  });

  log("✓ WebGPU device", {
    maxBufferSizeMiB: Math.round(Number(device.limits.maxBufferSize) / 1048576),
    maxStorageBufferBindingSizeMiB: Math.round(Number(device.limits.maxStorageBufferBindingSize) / 1048576),
    maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
    maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
  });

  log("✓ LFM2 programs compiled by Dawn");
  log("✓ pass definitions", Object.keys(lfm2.passes));

  divStatus.textContent = "PASS · parser + linker + Dawn compile";
  divStatus.className = "ok";
}

main().catch(fail);
