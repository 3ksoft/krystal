import { createWebGpuDevice } from "../packages/webgpu/src/engine.ts";

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const pageMiB = Number(Deno.args[0] ?? 64);
const targetGiB = Number(Deno.args[1] ?? 4);
const limitGiB = Number(Deno.args[2] ?? 1); // explicit maxBufferSize / maxStorageBufferBindingSize
const pageBytes = pageMiB * MIB;

// Deno/wgpu was failing with "not enough memory left" on the very first
// storage buffer. Requesting the limits explicitly sometimes fixes the
// allocation path, so they are passed through compile({ requiredLimits }).
const requiredLimits: Record<string, number> = {
  maxBufferSize: limitGiB * GIB,
  maxStorageBufferBindingSize: limitGiB * GIB,
  maxComputeWorkgroupsPerDimension: 65535,
};

console.log("Acquiring WebGPU device...");
// bun-types lacks the WebGPU DOM types, so compile()'s descriptor is not fully
// typed here; the cast is required until the project adds a WebGPU type lib.
const { device } = await createWebGpuDevice({ requiredLimits });
console.log("limits", {
  maxBufferSizeMiB: Math.round(Number(device.limits.maxBufferSize) / MIB),
  maxStorageBindingMiB: Math.round(Number(device.limits.maxStorageBufferBindingSize) / MIB),
  maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
});

const buffers: GPUBuffer[] = [];
let bytes = 0;
try {
  while (bytes + pageBytes <= targetGiB * GIB) {
    const label = `probe-${buffers.length}`;
    device.pushErrorScope("out-of-memory");
    device.pushErrorScope("validation");
    const buffer = device.createBuffer({
      label,
      size: pageBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const validation = await device.popErrorScope();
    const oom = await device.popErrorScope();
    if (validation || oom) {
      buffer.destroy();
      console.error(`FAIL at ${(bytes / GIB).toFixed(2)} GiB resident, next page ${pageMiB} MiB`);
      console.error((validation ?? oom)?.message ?? validation ?? oom);
      Deno.exit(1);
    }
    buffers.push(buffer);
    bytes += pageBytes;
    if (buffers.length % Math.max(1, Math.floor(256 / pageMiB)) === 0) {
      console.log(`  ${(bytes / GIB).toFixed(2)} GiB allocated`);
    }
  }
  console.log(`PASS: ${(bytes / GIB).toFixed(2)} GiB allocated as ${pageMiB} MiB storage pages`);
} finally {
  for (const buffer of buffers) buffer.destroy();
}
