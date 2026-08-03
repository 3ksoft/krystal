import { createWebGpuDevice } from "../packages/webgpu/src/index.ts";

const params = new URLSearchParams(location.search);
const pageMiB = Number(params.get("page") ?? 64);
const targetGiB = Number(params.get("target") ?? 4);
const pageBytes = pageMiB * 1024 * 1024;
const targetBytes = targetGiB * 1024 * 1024 * 1024;

console.log("[probe] acquiring WebGPU device...");
const { device } = await createWebGpuDevice();
console.log("[probe] limits", {
  maxBufferSizeMiB: Math.round(Number(device.limits.maxBufferSize) / 1048576),
  maxStorageBindingMiB: Math.round(Number(device.limits.maxStorageBufferBindingSize) / 1048576),
  maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
});

if (pageBytes > Number(device.limits.maxBufferSize)) {
  throw new Error(`page ${pageMiB} MiB exceeds maxBufferSize`);
}
if (pageBytes > Number(device.limits.maxStorageBufferBindingSize)) {
  throw new Error(`page ${pageMiB} MiB exceeds maxStorageBufferBindingSize`);
}

const buffers: GPUBuffer[] = [];
let resident = 0;
try {
  while (resident + pageBytes <= targetBytes) {
    const index = buffers.length;
    device.pushErrorScope("out-of-memory");
    device.pushErrorScope("validation");
    const buffer = device.createBuffer({
      label: `chomato.probe.${index}`,
      size: pageBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const validation = await device.popErrorScope();
    const oom = await device.popErrorScope();
    if (validation || oom) {
      buffer.destroy();
      throw validation ?? oom ?? new Error("unknown allocation error");
    }

    // Force Dawn/Vulkan to actually back the resource instead of only creating
    // a lazily allocated handle.
    const encoder = device.createCommandEncoder({ label: `chomato.probe.touch.${index}` });
    encoder.clearBuffer(buffer);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    buffers.push(buffer);
    resident += pageBytes;
    console.log(`[probe] ${(resident / 1073741824).toFixed(2)} GiB resident`);
  }
  console.log(`[probe] PASS: touched ${(resident / 1073741824).toFixed(2)} GiB`);
} catch (error) {
  console.error(
    `[probe] FAIL at ${(resident / 1073741824).toFixed(2)} GiB resident, next page ${pageMiB} MiB`,
    error,
  );
  throw error;
} finally {
  for (const buffer of buffers) buffer.destroy();
}
