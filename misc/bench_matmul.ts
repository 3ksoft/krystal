// bench_matmul.ts
// Uruchomienie: deno run --unstable-webgpu --allow-read bench_matmul.ts

if (!(navigator as any).gpu) {
  console.error("Wymagane WebGPU! Uruchom z flagą --unstable-webgpu");
  Deno.exit(1);
}

const M = 128;   // Prefill batch
const K = 2048;  // Hidden
const N = 8192;  // FFN width

console.log(`[bench] Testowanie MatMul: M=${M}, K=${K}, N=${N}`);
const totalOps = 2 * M * N * K;
const totalBytes = (M * K + K * N + M * N) * 4;
console.log(`[bench] Alokacja pamięci VRAM: ${(totalBytes / (1024 * 1024)).toFixed(2)} MB`);

const adapter = await (navigator as any).gpu.requestAdapter();
const device = await adapter.requestDevice();

// Dane CPU
const inputCpu = new Float32Array(M * K);
const weightCpu = new Float32Array(K * N); // layout: weight[k * N + n]  ← zmieniony!
for (let i = 0; i < inputCpu.length; i++) inputCpu[i] = (Math.random() - 0.5) * 0.1;
for (let i = 0; i < weightCpu.length; i++) weightCpu[i] = (Math.random() - 0.5) * 0.1;

// Oracle (pierwszy element)
let refOutput0 = 0;
for (let k = 0; k < K; k++) {
  refOutput0 += inputCpu[k]! * weightCpu[k * N + 0]!;
}

// Bufory
const inputBuffer = device.createBuffer({
  size: inputCpu.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const weightBuffer = device.createBuffer({
  size: weightCpu.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const outputBuffer = device.createBuffer({
  size: M * N * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});
const stagingBuffer = device.createBuffer({
  size: M * N * 4,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

device.queue.writeBuffer(inputBuffer, 0, inputCpu);
device.queue.writeBuffer(weightBuffer, 0, weightCpu);

// ====================== ZOPTYMALIZOWANY SHADER ======================
const TILE_M = 16;   // ile wierszy outputu liczy workgroup
const TILE_N = 16;   // ile kolumn outputu
const TILE_K = 16;   // rozmiar kafelka po K

const wgslCode = `
struct Params {
  M: u32,
  K: u32,
  N: u32,
  pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;   // [M, K]
@group(0) @binding(2) var<storage, read> weight: array<f32>;  // [K, N]  ← row-major po N
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const TILE_M: u32 = ${TILE_M}u;
const TILE_N: u32 = ${TILE_N}u;
const TILE_K: u32 = ${TILE_K}u;

var<workgroup> As: array<array<f32, TILE_K>, TILE_M>;
var<workgroup> Bs: array<array<f32, TILE_N>, TILE_K>;

@compute @workgroup_size(${TILE_M}, ${TILE_N}, 1)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let row = wid.y * TILE_M + lid.y;   // globalny wiersz outputu
  let col = wid.x * TILE_N + lid.x;   // globalna kolumna outputu

  var sum: f32 = 0.0;

  // Liczba kafelków po K
  let numTiles = (params.K + TILE_K - 1u) / TILE_K;

  for (var t = 0u; t < numTiles; t++) {
    // --- Load A tile (input) ---
    let aRow = row;
    let aCol = t * TILE_K + lid.x;
    if (aRow < params.M && aCol < params.K) {
      As[lid.y][lid.x] = input[aRow * params.K + aCol];
    } else {
      As[lid.y][lid.x] = 0.0;
    }

    // --- Load B tile (weight) ---
    // weight jest [K, N] → weight[k * N + n]
    let bRow = t * TILE_K + lid.y;
    let bCol = col;
    if (bRow < params.K && bCol < params.N) {
      Bs[lid.y][lid.x] = weight[bRow * params.N + bCol];
    } else {
      Bs[lid.y][lid.x] = 0.0;
    }

    workgroupBarrier();

    // --- Multiply-accumulate ---
    for (var k = 0u; k < TILE_K; k++) {
      sum += As[lid.y][k] * Bs[k][lid.x];
    }

    workgroupBarrier();
  }

  if (row < params.M && col < params.N) {
    output[row * params.N + col] = sum;
  }
}
`;

const module = device.createShaderModule({ code: wgslCode });
const pipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module, entryPoint: "main" },
});

const paramsBuf = device.createBuffer({
  size: 16,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([M, K, N, 0]));

const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: paramsBuf } },
    { binding: 1, resource: { buffer: inputBuffer } },
    { binding: 2, resource: { buffer: weightBuffer } },
    { binding: 3, resource: { buffer: outputBuffer } },
  ],
});

// Warmup
{
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(N / TILE_N), Math.ceil(M / TILE_M));
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

// Benchmark
const ITERATIONS = 20;
const start = performance.now();

const commandEncoder = device.createCommandEncoder();
for (let i = 0; i < ITERATIONS; i++) {
  const pass = commandEncoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(N / TILE_N), Math.ceil(M / TILE_M));
  pass.end();
}
commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, M * N * 4);
device.queue.submit([commandEncoder.finish()]);
await device.queue.onSubmittedWorkDone();

const elapsedMs = (performance.now() - start) / ITERATIONS;

// Weryfikacja
await stagingBuffer.mapAsync(GPUMapMode.READ);
const gpuOutput = new Float32Array(stagingBuffer.getMappedRange());
const gpuVal0 = gpuOutput[0]!;
stagingBuffer.unmap();

const diff = Math.abs(gpuVal0 - refOutput0);
const passed = diff < 0.001;

const gflops = (totalOps / (elapsedMs / 1000)) / 1e9;
console.log("==========================================");
console.log(`⏱️ Czas shadera MatMul: ${elapsedMs.toFixed(3)} ms / iterację`);
console.log(`🚀 Wydajność:           ${gflops.toFixed(2)} GFLOPS`);
console.log(`🧪 Test Poprawności:     ${passed ? "✅ PASS" : "❌ FAIL"} (Różnica: ${diff.toFixed(6)})`);
console.log("==========================================");
export {};
