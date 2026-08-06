const divStatus = document.querySelector<HTMLDivElement>("#status")!;
const divOutput = document.querySelector<HTMLPreElement>("#output")!;

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

function assertNear(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  epsilon = 1e-4,
): void {
  if (actual.length !== expected.length) {
    throw new Error(`length mismatch: ${actual.length} !== ${expected.length}`);
  }
  for (let i = 0; i < actual.length; i++) {
    const delta = Math.abs(actual[i]! - expected[i]!);
    if (!(delta <= epsilon)) {
      throw new Error(
        `value mismatch at ${i}: got ${actual[i]}, expected ${expected[i]} (delta=${delta})`,
      );
    }
  }
}

async function readArenaF32(
  device: GPUDevice,
  arena: GPUBuffer,
  elementOffset: number,
  count: number,
): Promise<Float32Array> {
  const byteLength = count * 4;
  const staging = device.createBuffer({
    label: "chomato-smoke.arena-readback",
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  try {
    const encoder = device.createCommandEncoder({ label: "chomato-smoke.readback" });
    encoder.copyBufferToBuffer(arena, elementOffset * 4, staging, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ, 0, byteLength);
    const mapped = staging.getMappedRange(0, byteLength);
    return new Float32Array(mapped.slice(0));
  } finally {
    if (staging.mapState === "mapped") staging.unmap();
    staging.destroy();
  }
}

async function main() {
  log("[smoke] importing LFM2 GPU definition…");

  // Dynamic import lets the page report parser/linker failures instead of
  // dying before the harness itself has initialized.
  const [
    { lfm2, LFM2_SHADER_NAMES },
    { Lfm2Executor },
    { createWebGpuDevice },
  ] = await Promise.all([
    import("../../src/lfm2"),
    import("../../src/pass"),
    import("../../src/device"),
  ]);
  log(`✓ linked ${LFM2_SHADER_NAMES.length} shader programs`);

  if (!navigator.gpu) {
    throw new Error(
      "WebGPU is unavailable: open this page in a WebGPU-enabled Chromium build "
      + "(about:gpu should report 'WebGPU' under Graphics Feature Status).",
    );
  }

  const GIB = 1024 * 1024 * 1024;
  const requiredLimits: Record<string, number> = {
    maxBufferSize: GIB,
    maxStorageBufferBindingSize: GIB,
    maxComputeWorkgroupsPerDimension: 65535,
  };

  // Explicit WebGPU bootstrap: acquire an adapter and a device before any GPU
  // work, so limits, features and error reporting are owned by the harness.
  log("[smoke] requesting WebGPU adapter + device…");
  const { adapter, device } = await createWebGpuDevice({
    label: "chomato-smoke",
    requiredLimits,
  });

  // Attach the uncaptured-error handler before compiling, so validation
  // failures during pipeline creation are reported instead of swallowed.
  device.addEventListener("uncapturederror", (event) => {
    divStatus.textContent = "FAILED";
    divStatus.className = "error";
    log("✗ WebGPU uncaptured error", event.error.message);
    console.error(event.error);
  });

  log("✓ WebGPU device", {
    adapter: adapter.info
      ? `${adapter.info.vendor} · ${adapter.info.architecture} · ${adapter.info.description}`
      : adapter,
    maxBufferSizeMiB: Math.round(Number(device.limits.maxBufferSize) / 1048576),
    maxStorageBufferBindingSizeMiB: Math.round(Number(device.limits.maxStorageBufferBindingSize) / 1048576),
    maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
    maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
  });

  log("[smoke] compiling through Chromium/Dawn…");
  const compiled = await lfm2.engine.compile({ device });
  if (compiled.failed > 0) {
    const lines = compiled.programs.flatMap((program) => {
      if (program.status !== "failed") return [];
      return [
        `— ${program.label} [${program.phase}]`,
        ...program.errors.map((error) =>
          `  ${error.lineNum !== undefined ? `[${error.lineNum}:${error.linePos}] ` : ""}${error.message}`),
      ];
    });
    throw new Error(
      `LFM2 compile failed (${compiled.failed}/${compiled.total} programs):\n${lines.join("\n")}`,
    );
  }

  log(`✓ LFM2 programs compiled by Dawn (${compiled.ok}/${compiled.total})`);
  log("✓ pass definitions", Object.keys(lfm2.passes));

  // -------------------------------------------------------------------------
  // Execution smoke
  // -------------------------------------------------------------------------
  // This deliberately uses no model file. It tests the actual migrated host
  // path: three OpParams records, dynamic uniform offsets, inter-dispatch arena
  // dependencies, a runtime weight-page override and one Sandblaster submit.
  log("[smoke] executing residual_add → arena_copy → rms_norm…");

  const inputDim = 8;
  const left = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
  const right = new Float32Array([8, 7, 6, 5, 4, 3, 2, 1]);
  const normWeights = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);

  const leftOffset = lfm2.arena.hiddenA;
  const rightOffset = lfm2.arena.hiddenB;
  const sumOffset = lfm2.arena.tmpH;
  const copiedOffset = lfm2.arena.tmpA;
  const outputOffset = lfm2.arena.tmpB;

  device.queue.writeBuffer(lfm2.resources.arena.gpu, leftOffset * 4, left);
  device.queue.writeBuffer(lfm2.resources.arena.gpu, rightOffset * 4, right);

  const weightPage = device.createBuffer({
    label: "chomato-smoke.rms-weight",
    size: normWeights.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(weightPage, 0, normWeights);

  try {
    const executor = new Lfm2Executor(lfm2);
    executor.submit((encoder) => {
      encoder.compute((pass) => {
        pass.run("residual_add", {
          inputOffset: leftOffset,
          auxOffset: rightOffset,
          outputOffset: sumOffset,
          tokenCount: 1,
          inputDim,
        });

        pass.run("arena_copy", {
          inputOffset: sumOffset,
          outputOffset: copiedOffset,
          tokenCount: 1,
          inputDim,
        });

        pass.run("rms_norm", {
          inputOffset: copiedOffset,
          outputOffset,
          tokenCount: 1,
          inputDim,
          f0: 1e-8,
        }, weightPage);
      }, { label: "chomato-smoke.execute" });
    });

    await device.queue.onSubmittedWorkDone();

    const sum = await readArenaF32(device, lfm2.resources.arena.gpu, sumOffset, inputDim);
    const copied = await readArenaF32(device, lfm2.resources.arena.gpu, copiedOffset, inputDim);
    const normalized = await readArenaF32(device, lfm2.resources.arena.gpu, outputOffset, inputDim);

    const expectedSum = new Float32Array(inputDim).fill(9);
    assertNear(sum, expectedSum);
    assertNear(copied, expectedSum);
    // RMS(9,9,...)=9, so with epsilon≈0 the normalized vector is exactly the
    // supplied weight page. This makes a broken weight override immediately
    // visible without needing a CPU reference implementation of RMSNorm.
    assertNear(normalized, normWeights, 1e-3);

    log("✓ execution smoke", {
      sum: Array.from(sum),
      copied: Array.from(copied),
      normalized: Array.from(normalized),
      expected: Array.from(normWeights),
    });
  } finally {
    weightPage.destroy();
  }

  divStatus.textContent = "PASS · parser + linker + Dawn compile + execution";
  divStatus.className = "ok";
}

main().catch(fail);
