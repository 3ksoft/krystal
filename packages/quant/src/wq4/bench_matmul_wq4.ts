// bench_matmul.ts
// Runtime-faithful F16 vs WQ4 microbenchmark for Chomato/LFM2.
//
// Run:
//   deno run --unstable-webgpu bench_matmul.ts
//   deno run --unstable-webgpu bench_matmul.ts --m 1
//   deno run --unstable-webgpu bench_matmul.ts --m 1,128 --k 2048 --n 8192
//
// The section marked WQ4 CANDIDATE is intentionally self-contained: edit only
// that entry point while iterating on the kernel. F16 and wq4_runtime are kept
// as stable baselines.

if (!(navigator as any).gpu) {
  console.error("Wymagane WebGPU! Uruchom z flagą --unstable-webgpu");
  Deno.exit(1);
}

type Cli = {
  m: number[];
  k: number;
  n: number;
  samples: number;
  repeatsDecode: number;
  repeatsPrefill: number;
};

function parseNumberList(value: string): number[] {
  const values = value.split(",").map((v) => Number(v.trim())).filter((v) => Number.isFinite(v) && v > 0);
  if (values.length === 0) throw new Error(`Invalid number list: ${value}`);
  return values;
}

function parseArgs(args: string[]): Cli {
  const out: Cli = {
    m: [1, 128],
    k: 2048,
    n: 8192,
    samples: 7,
    repeatsDecode: 30,
    repeatsPrefill: 2,
  };
  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;
    const next = () => {
      const value = args[++i];
      if (value === undefined) throw new Error(`Missing value after ${arg}`);
      return value;
    };
    if (arg === "--m") out.m = parseNumberList(next());
    else if (arg === "--k") out.k = Number(next());
    else if (arg === "--n") out.n = Number(next());
    else if (arg === "--samples") out.samples = Number(next());
    else if (arg === "--repeats-decode") out.repeatsDecode = Number(next());
    else if (arg === "--repeats-prefill") out.repeatsPrefill = Number(next());
    else if (arg === "--help" || arg === "-h") {
      console.log(`Usage: deno run --unstable-webgpu bench_matmul.ts [options]\n\n` +
        `  --m 1,128             token counts to benchmark\n` +
        `  --k 2048              input dimension\n` +
        `  --n 8192              output rows\n` +
        `  --samples 7           timing samples per kernel\n` +
        `  --repeats-decode 30   dispatches/sample for M<=4\n` +
        `  --repeats-prefill 2   dispatches/sample for M>4\n`);
      Deno.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (out.k % 32 !== 0) throw new Error(`K must be divisible by 32 for WQ4, got ${out.k}`);
  return out;
}

const cli = parseArgs(Deno.args);
const K = cli.k;
const N = cli.n;
const WEIGHT_COUNT = K * N;
const BLOCK_SIZE = 32;
const WORDS_PER_BLOCK = 5;
const ZERO_WORD = 0x88888888;
const MIN_EXP = -24;
const MAX_EXP = 8;

console.log(`[bench] Runtime layout: weights=[N,K], K=${K}, N=${N}`);
console.log(`[bench] Cases: M=${cli.m.join(", ")}`);
console.log(`[bench] Logical weights: ${(WEIGHT_COUNT * 4 / 2 ** 20).toFixed(1)} MiB F32 source`);

const adapter = await (navigator as any).gpu.requestAdapter();
if (!adapter) throw new Error("No WebGPU adapter");
const device = await adapter.requestDevice();

let adapterDesc = "unknown";
try {
  const info = (adapter as any).info ?? await (adapter as any).requestAdapterInfo?.();
  if (info) {
    adapterDesc = `${info.description ?? ""} ${info.vendor ?? ""} ${info.architecture ?? ""} ${info.device ?? ""}`.trim() || "unknown";
  }
} catch { /* adapter info is best-effort */ }
console.log(`[bench] adapter: ${adapterDesc}`);
console.log(`[bench] adapter/device ready; maxStorageBufferBindingSize=${(device.limits.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MiB`);

// ---------------------------------------------------------------------------
// Deterministic source data + the same WQ4 v2 quantizer used by the converter.
// We keep weights [N,K], i.e. one contiguous input row per output neuron,
// exactly as matmul_f16/matmul_wq4 expect in the runtime.
// ---------------------------------------------------------------------------

let rngState = 0x12345678;
function random01(): number {
  // xorshift32: deterministic and cheap enough for a benchmark fixture.
  let x = rngState | 0;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  rngState = x >>> 0;
  return rngState / 0x100000000;
}

const weightF32 = new Float32Array(WEIGHT_COUNT);
for (let i = 0; i < weightF32.length; i++) weightF32[i] = (random01() - 0.5) * 0.1;

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);
function f32ToF16Bits(value: number): number {
  f32Scratch[0] = value;
  const x = u32Scratch[0]!;
  const sign = (x >>> 16) & 0x8000;
  let mantissa = x & 0x007fffff;
  let exp = (x >>> 23) & 0xff;

  if (exp === 0xff) {
    if (mantissa !== 0) return sign | 0x7e00;
    return sign | 0x7c00;
  }

  exp = exp - 127 + 15;
  if (exp >= 0x1f) return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mantissa |= 0x00800000;
    const shift = 14 - exp;
    let halfMantissa = mantissa >>> shift;
    const roundBit = 1 << (shift - 1);
    if ((mantissa & roundBit) && ((mantissa & (roundBit - 1)) || (halfMantissa & 1))) halfMantissa++;
    return sign | halfMantissa;
  }

  let half = sign | (exp << 10) | (mantissa >>> 13);
  const round = mantissa & 0x1fff;
  if (round > 0x1000 || (round === 0x1000 && (half & 1))) half++;
  return half & 0xffff;
}

console.log("[bench] packing F16...");
const weightF16Raw = new Uint32Array(Math.ceil(WEIGHT_COUNT / 2));
for (let i = 0; i < WEIGHT_COUNT; i += 2) {
  const lo = f32ToF16Bits(weightF32[i]!);
  const hi = i + 1 < WEIGHT_COUNT ? f32ToF16Bits(weightF32[i + 1]!) : 0;
  weightF16Raw[i >> 1] = (lo | (hi << 16)) >>> 0;
}

function clampExp(exp: number): number {
  return Math.max(MIN_EXP, Math.min(MAX_EXP, exp));
}
function quantizedValue(value: number, scale: number): number {
  return Math.max(-8, Math.min(7, Math.round(value / scale)));
}
function quantizeBlockWq4(data: Float32Array, start: number, out: Uint32Array, outOffset: number): void {
  const end = start + BLOCK_SIZE;
  let maxPositive = 0;
  let minNegative = 0;
  for (let i = start; i < end; i++) {
    const value = data[i]!;
    if (value > maxPositive) maxPositive = value;
    if (value < minNegative) minNegative = value;
  }
  if (maxPositive === 0 && minNegative === 0) {
    out[outOffset] = ZERO_WORD;
    out[outOffset + 1] = ZERO_WORD;
    out[outOffset + 2] = ZERO_WORD;
    out[outOffset + 3] = ZERO_WORD;
    out[outOffset + 4] = MIN_EXP >>> 0;
    return;
  }

  const maxAbs = Math.max(maxPositive, -minNegative);
  const legacyExp = clampExp(Math.floor(Math.log2(maxAbs / 7)));
  const noClipScale = Math.max(maxPositive / 7, (-minNegative) / 8, 2 ** MIN_EXP);
  const exactExp = Math.log2(noClipScale);
  const expLo = clampExp(Math.floor(exactExp));
  const expHi = clampExp(Math.ceil(exactExp));

  const candidates = [legacyExp, expLo, expHi];
  let bestExp = candidates[0]!;
  let bestSse = Number.POSITIVE_INFINITY;
  for (const exp of candidates) {
    const scale = 2 ** exp;
    let sse = 0;
    for (let i = start; i < end; i++) {
      const value = data[i]!;
      const err = value - quantizedValue(value, scale) * scale;
      sse += err * err;
    }
    if (sse < bestSse) {
      bestSse = sse;
      bestExp = exp;
    }
  }

  const scale = 2 ** bestExp;
  for (let wordIndex = 0; wordIndex < 4; wordIndex++) {
    let word = 0;
    for (let nibbleIndex = 0; nibbleIndex < 8; nibbleIndex++) {
      const idx = start + wordIndex * 8 + nibbleIndex;
      const q = quantizedValue(data[idx]!, scale);
      word |= ((q + 8) & 0x0f) << (nibbleIndex * 4);
    }
    out[outOffset + wordIndex] = word >>> 0;
  }
  out[outOffset + 4] = bestExp >>> 0;
}

console.log("[bench] quantizing WQ4...");
const blockCount = WEIGHT_COUNT / BLOCK_SIZE;
const weightWq4Raw = new Uint32Array(blockCount * WORDS_PER_BLOCK);
for (let b = 0; b < blockCount; b++) {
  quantizeBlockWq4(weightF32, b * BLOCK_SIZE, weightWq4Raw, b * WORDS_PER_BLOCK);
}

console.log(`[bench] F16 weights: ${(weightF16Raw.byteLength / 2 ** 20).toFixed(2)} MiB`);
console.log(`[bench] WQ4 weights: ${(weightWq4Raw.byteLength / 2 ** 20).toFixed(2)} MiB (${(weightF16Raw.byteLength / weightWq4Raw.byteLength).toFixed(2)}x smaller)`);

const f16WeightBuffer = device.createBuffer({
  size: weightF16Raw.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const wq4WeightBuffer = device.createBuffer({
  size: weightWq4Raw.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(f16WeightBuffer, 0, weightF16Raw);
device.queue.writeBuffer(wq4WeightBuffer, 0, weightWq4Raw);

// 8-bit intermediate weight layout (v6): per block, 8 u32 of packed signed
// bytes (q-8, 4 per word) + 1 u32 scale = 9 u32 / block. 32 bytes of
// weights instead of 16, but per-MAC work drops to load+unpack+cvt+fma.
console.log("[bench] repacking WQ4 into i8 intermediate (v6)...");
const weight8Raw = new Uint32Array(blockCount * 9);
for (let b = 0; b < blockCount; b++) {
  const srcBase = b * WORDS_PER_BLOCK;
  const dstBase = b * 9;
  for (let i = 0; i < 32; i++) {
    const byte = ((weightWq4Raw[srcBase + (i >> 3)]! >> ((i & 7) * 4)) & 0x0f) - 8;
    weight8Raw[dstBase + (i >> 2)] =
      (weight8Raw[dstBase + (i >> 2)]! | ((byte & 0xff) << ((i & 3) * 8))) >>> 0;
  }
  weight8Raw[dstBase + 8] = weightWq4Raw[srcBase + 4]!;
}
const weight8Buffer = device.createBuffer({
  size: weight8Raw.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(weight8Buffer, 0, weight8Raw);
console.log(`[bench] i8 weights: ${(weight8Raw.byteLength / 2 ** 20).toFixed(2)} MiB`);

// ---------------------------------------------------------------------------
// Shader. `matmul_f16` and `matmul_wq4_runtime` mirror runtime.wgsl.
// Edit the `matmul_wq4_candidate*` kernels while optimizing.
// ---------------------------------------------------------------------------

const wgslCode = /* wgsl */ `
struct Params {
  tokenCount: u32,
  inputDim: u32,
  outputDim: u32,
  rowCount: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weightRaw: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> reduceF32: array<f32, 64>;
var<workgroup> reduceVec4: array<vec4<f32>, 32>;

fn load_f16(index: u32) -> f32 {
  let packed = weightRaw[index >> 1u];
  let pair = unpack2x16float(packed);
  return select(pair.x, pair.y, (index & 1u) != 0u);
}

@compute @workgroup_size(64)
fn matmul_f16(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let localRow = wid.x;
  let tokenIndex = wid.y;
  if (localRow >= p.rowCount || tokenIndex >= p.tokenCount) { return; }

  let inputBase = tokenIndex * p.inputDim;
  var sum = 0.0;
  var k = lid.x;
  loop {
    if (k >= p.inputDim) { break; }
    sum += input[inputBase + k] * load_f16(localRow * p.inputDim + k);
    k += 64u;
  }
  reduceF32[lid.x] = sum;
  workgroupBarrier();

  var width = 32u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) { reduceF32[lid.x] += reduceF32[lid.x + width]; }
    workgroupBarrier();
    width >>= 1u;
  }

  if (lid.x == 0u) { output[tokenIndex * p.outputDim + localRow] = reduceF32[0]; };
}

// Current runtime kernel: keep this unchanged as the WQ4 baseline.
@compute @workgroup_size(64)
fn matmul_wq4_runtime(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let localRow = wid.x;
  let tokenIndex = wid.y;
  if (localRow >= p.rowCount || tokenIndex >= p.tokenCount) { return; }

  let inputBase = tokenIndex * p.inputDim;
  let blocksPerRow = p.inputDim / 32u;
  let rowBlockStart = localRow * blocksPerRow;

  var sum: f32 = 0.0;
  var b = lid.x;
  loop {
    if (b >= blocksPerRow) { break; }

    let blockIdx = rowBlockStart + b;
    let baseU32 = blockIdx * 5u;
    let expVal = bitcast<i32>(weightRaw[baseU32 + 4u]);
    let scale = exp2(f32(expVal));
    let kStart = b * 32u;

    for (var w = 0u; w < 4u; w++) {
      let packed = weightRaw[baseU32 + w];
      let kBase = kStart + w * 8u;
      sum += (f32((packed >>  0u) & 0x0Fu) - 8.0) * scale * input[inputBase + kBase + 0u];
      sum += (f32((packed >>  4u) & 0x0Fu) - 8.0) * scale * input[inputBase + kBase + 1u];
      sum += (f32((packed >>  8u) & 0x0Fu) - 8.0) * scale * input[inputBase + kBase + 2u];
      sum += (f32((packed >> 12u) & 0x0Fu) - 8.0) * scale * input[inputBase + kBase + 3u];
      sum += (f32((packed >> 16u) & 0x0Fu) - 8.0) * scale * input[inputBase + kBase + 4u];
      sum += (f32((packed >> 20u) & 0x0Fu) - 8.0) * scale * input[inputBase + kBase + 5u];
      sum += (f32((packed >> 24u) & 0x0Fu) - 8.0) * scale * input[inputBase + kBase + 6u];
      sum += (f32((packed >> 28u) & 0x0Fu) - 8.0) * scale * input[inputBase + kBase + 7u];
    }
    b += 64u;
  }

  reduceF32[lid.x] = sum;
  workgroupBarrier();
  var width = 32u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) { reduceF32[lid.x] += reduceF32[lid.x + width]; }
    workgroupBarrier();
    width >>= 1u;
  }
  if (lid.x == 0u) { output[tokenIndex * p.outputDim + localRow] = reduceF32[0]; };
}

// ==========================================================================
// WQ4 CANDIDATE v3 — decode latency first.
//   workgroup_size(32), 1 output row per workgroup, no shared memory,
//   one 32->1 reduction, fma + manual unroll, scale folded into the FMA.
//   Each thread owns ceil(blocksPerRow/32) blocks of K; the trip count is a
//   TS-interpolated constant (ceil(K/1024)) so the t-loop unrolls fully for
//   any --k while staying correct (bounds check guards partial tails).
// ==========================================================================
fn pow2i(exp: i32) -> f32 {
  return bitcast<f32>(u32(exp + 127) << 23u);
}

@compute @workgroup_size(32)
fn matmul_wq4_candidate(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let localRow = wid.x;
  let tokenIndex = wid.y;
  if (localRow >= p.rowCount || tokenIndex >= p.tokenCount) { return; }

  let inputBase = tokenIndex * p.inputDim;
  let blocksPerRow = p.inputDim / 32u; // = 64 for K=2048
  let rowBlockStart = localRow * blocksPerRow;

  var sum = 0.0;
  for (var t = 0u; t < ${Math.ceil(K / 1024)}u; t++) {
    let b = lid.x + t * 32u;
    if (b >= blocksPerRow) { continue; }

    let blockIdx = rowBlockStart + b;
    let baseU32 = blockIdx * 5u;
    let scale = pow2i(bitcast<i32>(weightRaw[baseU32 + 4u]));
    let kStart = b * 32u;

    // 4 words x 8 nibbles, fully unrolled; (q - 8) * scale folded into fma.
    for (var w = 0u; w < 4u; w++) {
      let packed = weightRaw[baseU32 + w];
      let k = kStart + w * 8u;
      sum = fma((f32((packed      ) & 0x0Fu) - 8.0) * scale, input[inputBase + k + 0u], sum);
      sum = fma((f32((packed >>  4u) & 0x0Fu) - 8.0) * scale, input[inputBase + k + 1u], sum);
      sum = fma((f32((packed >>  8u) & 0x0Fu) - 8.0) * scale, input[inputBase + k + 2u], sum);
      sum = fma((f32((packed >> 12u) & 0x0Fu) - 8.0) * scale, input[inputBase + k + 3u], sum);
      sum = fma((f32((packed >> 16u) & 0x0Fu) - 8.0) * scale, input[inputBase + k + 4u], sum);
      sum = fma((f32((packed >> 20u) & 0x0Fu) - 8.0) * scale, input[inputBase + k + 5u], sum);
      sum = fma((f32((packed >> 24u) & 0x0Fu) - 8.0) * scale, input[inputBase + k + 6u], sum);
      sum = fma((f32((packed >> 28u) & 0x0Fu) - 8.0) * scale, input[inputBase + k + 7u], sum);
    }
  }

  // Reduce 32 -> 1 (single tree, 5 barriers).
  reduceF32[lid.x] = sum;
  workgroupBarrier();
  if (lid.x < 16u) { reduceF32[lid.x] += reduceF32[lid.x + 16u]; }
  workgroupBarrier();
  if (lid.x <  8u) { reduceF32[lid.x] += reduceF32[lid.x +  8u]; }
  workgroupBarrier();
  if (lid.x <  4u) { reduceF32[lid.x] += reduceF32[lid.x +  4u]; }
  workgroupBarrier();
  if (lid.x <  2u) { reduceF32[lid.x] += reduceF32[lid.x +  2u]; }
  workgroupBarrier();
  if (lid.x == 0u) {
    output[tokenIndex * p.outputDim + localRow] = reduceF32[0] + reduceF32[1];
  }
}

// ==========================================================================
// WQ4 CANDIDATE v4 — decode-focused.
//   Same shape as v3 (workgroup_size(32), 2 blocks/thread for K=2048) but:
//   - 4 accumulators interleaved per word for instruction-level parallelism,
//   - block scale applied ONCE per block after the MAC loop (v3 folded it
//     into every fma, costing 31 extra multiplies per block).
//   NOTE: an earlier variant also read the input via a second storage binding
//   (array<vec4<f32>>); double-binding the same buffer returned zeros on this
//   driver, so we keep scalar input reads (they coalesce in L1 anyway).
// ==========================================================================
@compute @workgroup_size(32)
fn matmul_wq4_candidate_v4(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let localRow = wid.x;
  let tokenIndex = wid.y;
  if (localRow >= p.rowCount || tokenIndex >= p.tokenCount) { return; }

  let inputBase = tokenIndex * p.inputDim;
  let blocksPerRow = p.inputDim / 32u;
  let rowBlockStart = localRow * blocksPerRow;

  var total = 0.0;
  for (var t = 0u; t < ${Math.ceil(K / 1024)}u; t++) {
    let b = lid.x + t * 32u;
    if (b >= blocksPerRow) { continue; }

    let baseU32 = (rowBlockStart + b) * 5u;
    let scale = pow2i(bitcast<i32>(weightRaw[baseU32 + 4u]));
    let kStart = b * 32u;

    var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
    for (var w = 0u; w < 4u; w++) {
      let packed = weightRaw[baseU32 + w];
      let k = kStart + w * 8u;
      let q0 = f32(packed & 0x0Fu) - 8.0;
      let q1 = f32((packed >>  4u) & 0x0Fu) - 8.0;
      let q2 = f32((packed >>  8u) & 0x0Fu) - 8.0;
      let q3 = f32((packed >> 12u) & 0x0Fu) - 8.0;
      let q4 = f32((packed >> 16u) & 0x0Fu) - 8.0;
      let q5 = f32((packed >> 20u) & 0x0Fu) - 8.0;
      let q6 = f32((packed >> 24u) & 0x0Fu) - 8.0;
      let q7 = f32((packed >> 28u) & 0x0Fu) - 8.0;
      s0 = fma(q0, input[inputBase + k + 0u], s0);
      s1 = fma(q1, input[inputBase + k + 1u], s1);
      s2 = fma(q2, input[inputBase + k + 2u], s2);
      s3 = fma(q3, input[inputBase + k + 3u], s3);
      s0 = fma(q4, input[inputBase + k + 4u], s0);
      s1 = fma(q5, input[inputBase + k + 5u], s1);
      s2 = fma(q6, input[inputBase + k + 6u], s2);
      s3 = fma(q7, input[inputBase + k + 7u], s3);
    }
    total += (s0 + s1 + s2 + s3) * scale;
  }

  reduceF32[lid.x] = total;
  workgroupBarrier();
  if (lid.x < 16u) { reduceF32[lid.x] += reduceF32[lid.x + 16u]; }
  workgroupBarrier();
  if (lid.x <  8u) { reduceF32[lid.x] += reduceF32[lid.x +  8u]; }
  workgroupBarrier();
  if (lid.x <  4u) { reduceF32[lid.x] += reduceF32[lid.x +  4u]; }
  workgroupBarrier();
  if (lid.x <  2u) { reduceF32[lid.x] += reduceF32[lid.x +  2u]; }
  workgroupBarrier();
  if (lid.x == 0u) {
    output[tokenIndex * p.outputDim + localRow] = reduceF32[0] + reduceF32[1];
  }
}

// ==========================================================================
// WQ4 CANDIDATE v5 — prefill-focused: 8 tokens per workgroup.
//   workgroup_size(64), wid.x = row, wid.y = token group (ceil(M/8) groups).
//   Each thread owns blocks b = lid.x, lid.x+64, ...; the 4 packed words of a
//   block are unpacked ONCE and the nibbles are reused across all 8 tokens, so
//   unpack cost and weight traffic are amortized x8 vs 1 token/WG.
//   Dispatch: dispatchWorkgroups(N, ceil(M/8), 1).
// ==========================================================================
@compute @workgroup_size(64)
fn matmul_wq4_candidate_v5(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let localRow = wid.x;
  let tokenBase = wid.y * 8u;
  if (localRow >= p.rowCount) { return; }

  let blocksPerRow = p.inputDim / 32u;
  let rowBlockStart = localRow * blocksPerRow;

  var sums: array<f32, 8>;
  for (var t = 0u; t < 8u; t++) { sums[t] = 0.0; }

  var b = lid.x;
  loop {
    if (b >= blocksPerRow) { break; }
    let baseU32 = (rowBlockStart + b) * 5u;
    let scale = pow2i(bitcast<i32>(weightRaw[baseU32 + 4u]));
    let kStart = b * 32u;

    var tokSum: array<f32, 8>;
    for (var t = 0u; t < 8u; t++) { tokSum[t] = 0.0; }

    for (var w = 0u; w < 4u; w++) {
      let packed = weightRaw[baseU32 + w];
      let k = kStart + w * 8u;
      var q: array<f32, 8>;
      for (var n = 0u; n < 8u; n++) {
        q[n] = f32((packed >> (n * 4u)) & 0x0Fu) - 8.0;
      }
      for (var t = 0u; t < 8u; t++) {
        let tok = tokenBase + t;
        if (tok >= p.tokenCount) { continue; }
        let ib = tok * p.inputDim + k;
        tokSum[t] += q[0] * input[ib + 0u] + q[1] * input[ib + 1u] + q[2] * input[ib + 2u] + q[3] * input[ib + 3u]
                   + q[4] * input[ib + 4u] + q[5] * input[ib + 5u] + q[6] * input[ib + 6u] + q[7] * input[ib + 7u];
      }
    }

    for (var t = 0u; t < 8u; t++) {
      let tok = tokenBase + t;
      if (tok >= p.tokenCount) { continue; }
      sums[t] += tokSum[t] * scale;
    }
    b += 64u;
  }

  // Reduce 64 -> 1 once per token.
  for (var t = 0u; t < 8u; t++) {
    let tok = tokenBase + t;
    if (tok >= p.tokenCount) { continue; }
    reduceF32[lid.x] = sums[t];
    workgroupBarrier();
    var width = 32u;
    loop {
      if (width == 0u) { break; }
      if (lid.x < width) { reduceF32[lid.x] += reduceF32[lid.x + width]; }
      workgroupBarrier();
      width >>= 1u;
    }
    if (lid.x == 0u) { output[tok * p.outputDim + localRow] = reduceF32[0]; }
    workgroupBarrier();
  }
}

// ==========================================================================
// WQ4 CANDIDATE v6 — 8-bit intermediate weights.
//   Same shape as v3/v4 (workgroup_size(32), 2 blocks/thread for K=2048) but
//   weights are pre-unpacked to signed bytes (q-8), 4 per u32. Per block the
//   kernel does 8 u32 loads + 8 unpack4xI8 + 32 converts + 32 FMAs instead of
//   4 loads + 32*(shift+and+conv+sub) — roughly half the instructions per MAC.
//   Layout per block: 8 u32 of packed i8 + 1 u32 scale (9 u32 / block).
//   Both decode and prefill are instruction-bound, so this helps both.
// ==========================================================================
@compute @workgroup_size(32)
fn matmul_wq4_candidate_v6(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let localRow = wid.x;
  let tokenIndex = wid.y;
  if (localRow >= p.rowCount || tokenIndex >= p.tokenCount) { return; }

  let inputBase = tokenIndex * p.inputDim;
  let blocksPerRow = p.inputDim / 32u;
  let rowBlockStart = localRow * blocksPerRow;

  var total = 0.0;
  for (var t = 0u; t < ${Math.ceil(K / 1024)}u; t++) {
    let b = lid.x + t * 32u;
    if (b >= blocksPerRow) { continue; }

    let baseU32 = (rowBlockStart + b) * 9u;
    let scale = pow2i(bitcast<i32>(weightRaw[baseU32 + 8u]));
    let kStart = b * 32u;

    var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
    for (var w = 0u; w < 8u; w++) {
      let w4 = vec4<f32>(unpack4xI8(weightRaw[baseU32 + w]));
      let k = kStart + w * 4u;
      s0 = fma(w4.x, input[inputBase + k + 0u], s0);
      s1 = fma(w4.y, input[inputBase + k + 1u], s1);
      s2 = fma(w4.z, input[inputBase + k + 2u], s2);
      s3 = fma(w4.w, input[inputBase + k + 3u], s3);
    }
    total += (s0 + s1 + s2 + s3) * scale;
  }

  reduceF32[lid.x] = total;
  workgroupBarrier();
  if (lid.x < 16u) { reduceF32[lid.x] += reduceF32[lid.x + 16u]; }
  workgroupBarrier();
  if (lid.x <  8u) { reduceF32[lid.x] += reduceF32[lid.x +  8u]; }
  workgroupBarrier();
  if (lid.x <  4u) { reduceF32[lid.x] += reduceF32[lid.x +  4u]; }
  workgroupBarrier();
  if (lid.x <  2u) { reduceF32[lid.x] += reduceF32[lid.x +  2u]; }
  workgroupBarrier();
  if (lid.x == 0u) {
    output[tokenIndex * p.outputDim + localRow] = reduceF32[0] + reduceF32[1];
  }
}

// ==========================================================================
// WQ4 CANDIDATE v7 — element-per-thread (coalesced input access).
//   Thread t owns element e = lid.x WITHIN every K-block instead of owning
//   whole blocks. The warp then reads input[base + b*32 + 0..31] (consecutive
//   floats, one cache line per load) instead of 32 separate 128B-strided
//   lines. Weight nibbles are picked by position (shift by constant offset
//   per thread). One reduce 32->1. Uses the original wq4 nibble layout.
// ==========================================================================
@compute @workgroup_size(32)
fn matmul_wq4_candidate_v7(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let localRow = wid.x;
  let tokenIndex = wid.y;
  if (localRow >= p.rowCount || tokenIndex >= p.tokenCount) { return; }

  let inputBase = tokenIndex * p.inputDim;
  let blocksPerRow = p.inputDim / 32u;
  let rowBlockStart = localRow * blocksPerRow;

  let e = lid.x;
  let wordInBlock = e >> 3u;
  let nibbleShift = (e & 7u) * 4u;

  var sum = 0.0;
  for (var b = 0u; b < ${K / 32}u; b++) {
    let blockBase = (rowBlockStart + b) * 5u;
    let scale = pow2i(bitcast<i32>(weightRaw[blockBase + 4u]));
    let q = f32((weightRaw[blockBase + wordInBlock] >> nibbleShift) & 0x0Fu) - 8.0;
    sum = fma(q * scale, input[inputBase + b * 32u + e], sum);
  }

  reduceF32[lid.x] = sum;
  workgroupBarrier();
  if (lid.x < 16u) { reduceF32[lid.x] += reduceF32[lid.x + 16u]; }
  workgroupBarrier();
  if (lid.x <  8u) { reduceF32[lid.x] += reduceF32[lid.x +  8u]; }
  workgroupBarrier();
  if (lid.x <  4u) { reduceF32[lid.x] += reduceF32[lid.x +  4u]; }
  workgroupBarrier();
  if (lid.x <  2u) { reduceF32[lid.x] += reduceF32[lid.x +  2u]; }
  workgroupBarrier();
  if (lid.x == 0u) {
    output[tokenIndex * p.outputDim + localRow] = reduceF32[0] + reduceF32[1];
  }
}

// ==========================================================================
// WQ4 CANDIDATE v9 — 4 rows per workgroup + element-per-thread + vec4 reduce.
//   Combines everything learned: coalesced input access (thread t owns element
//   e across all K blocks), 4 output rows per workgroup (N/4 workgroups), and
//   ONE vec4 tree reduction covering all 4 rows (5 barriers total instead of
//   4x the scalar tree). The input value for element e is loaded once and
//   reused for all 4 rows.
//   Dispatch: dispatchWorkgroups(ceil(N/4), M, 1).
// ==========================================================================
@compute @workgroup_size(32)
fn matmul_wq4_candidate_v9(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let rowBase = wid.x * 4u;
  let tokenIndex = wid.y;
  if (tokenIndex >= p.tokenCount) { return; }

  let inputBase = tokenIndex * p.inputDim;
  let blocksPerRow = p.inputDim / 32u;

  let e = lid.x;
  let wordInBlock = e >> 3u;
  let nibbleShift = (e & 7u) * 4u;

  var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
  for (var b = 0u; b < ${K / 32}u; b++) {
    let x = input[inputBase + b * 32u + e];
    let r0 = rowBase;
    if (r0 < p.rowCount) {
      let bb = (r0 * blocksPerRow + b) * 5u;
      let q = f32((weightRaw[bb + wordInBlock] >> nibbleShift) & 0x0Fu) - 8.0;
      s0 = fma(q * pow2i(bitcast<i32>(weightRaw[bb + 4u])), x, s0);
    }
    let r1 = rowBase + 1u;
    if (r1 < p.rowCount) {
      let bb = (r1 * blocksPerRow + b) * 5u;
      let q = f32((weightRaw[bb + wordInBlock] >> nibbleShift) & 0x0Fu) - 8.0;
      s1 = fma(q * pow2i(bitcast<i32>(weightRaw[bb + 4u])), x, s1);
    }
    let r2 = rowBase + 2u;
    if (r2 < p.rowCount) {
      let bb = (r2 * blocksPerRow + b) * 5u;
      let q = f32((weightRaw[bb + wordInBlock] >> nibbleShift) & 0x0Fu) - 8.0;
      s2 = fma(q * pow2i(bitcast<i32>(weightRaw[bb + 4u])), x, s2);
    }
    let r3 = rowBase + 3u;
    if (r3 < p.rowCount) {
      let bb = (r3 * blocksPerRow + b) * 5u;
      let q = f32((weightRaw[bb + wordInBlock] >> nibbleShift) & 0x0Fu) - 8.0;
      s3 = fma(q * pow2i(bitcast<i32>(weightRaw[bb + 4u])), x, s3);
    }
  }

  reduceVec4[lid.x] = vec4<f32>(s0, s1, s2, s3);
  workgroupBarrier();
  if (lid.x < 16u) { reduceVec4[lid.x] += reduceVec4[lid.x + 16u]; }
  workgroupBarrier();
  if (lid.x <  8u) { reduceVec4[lid.x] += reduceVec4[lid.x +  8u]; }
  workgroupBarrier();
  if (lid.x <  4u) { reduceVec4[lid.x] += reduceVec4[lid.x +  4u]; }
  workgroupBarrier();
  if (lid.x <  2u) { reduceVec4[lid.x] += reduceVec4[lid.x +  2u]; }
  workgroupBarrier();
  if (lid.x == 0u) {
    let v = reduceVec4[0] + reduceVec4[1];
    let ob = tokenIndex * p.outputDim + rowBase;
    if (rowBase < p.rowCount) { output[ob] = v.x; }
    if (rowBase + 1u < p.rowCount) { output[ob + 1u] = v.y; }
    if (rowBase + 2u < p.rowCount) { output[ob + 2u] = v.z; }
    if (rowBase + 3u < p.rowCount) { output[ob + 3u] = v.w; }
  }
}

// ==========================================================================
// WQ4 CANDIDATE v10 — v7 but workgroup_size(64), 2 warps per workgroup.
//   Two threads per element: warp 0 (lid.x<32) covers even K-blocks, warp 1
//   covers odd blocks, both keep the coalesced input access of v7. Mirrors the
//   f16 kernel's 64-thread shape; tests whether the remaining ~11% gap to F16
//   at decode is thread-count related.
// ==========================================================================
@compute @workgroup_size(64)
fn matmul_wq4_candidate_v10(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let localRow = wid.x;
  let tokenIndex = wid.y;
  if (localRow >= p.rowCount || tokenIndex >= p.tokenCount) { return; }

  let inputBase = tokenIndex * p.inputDim;
  let blocksPerRow = p.inputDim / 32u;
  let rowBlockStart = localRow * blocksPerRow;

  let e = lid.x & 31u;
  let blockOffset = lid.x >> 5u;
  let wordInBlock = e >> 3u;
  let nibbleShift = (e & 7u) * 4u;

  var sum = 0.0;
  for (var j = 0u; j < ${Math.ceil(K / 64)}u; j++) {
    let b = blockOffset + j * 2u;
    if (b >= blocksPerRow) { continue; }
    let blockBase = (rowBlockStart + b) * 5u;
    let scale = pow2i(bitcast<i32>(weightRaw[blockBase + 4u]));
    let q = f32((weightRaw[blockBase + wordInBlock] >> nibbleShift) & 0x0Fu) - 8.0;
    sum = fma(q * scale, input[inputBase + b * 32u + e], sum);
  }

  reduceF32[lid.x] = sum;
  workgroupBarrier();
  var width = 32u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) { reduceF32[lid.x] += reduceF32[lid.x + width]; }
    workgroupBarrier();
    width >>= 1u;
  }
  if (lid.x == 0u) { output[tokenIndex * p.outputDim + localRow] = reduceF32[0]; }
}

// ==========================================================================
// WQ4 CANDIDATE v11 — user paste, kept 1:1 (renamed).
//   Still workgroup_size(64), 1 nibble per block per thread (same work
//   distribution as v10) but the reduction is manually unrolled instead of
//   the width loop. Control variant: isolates the reduction form.
// ==========================================================================
@compute @workgroup_size(64)
fn matmul_wq4_candidate_v11(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let localRow = wid.x;
  let tokenIndex = wid.y;
  if (localRow >= p.rowCount || tokenIndex >= p.tokenCount) { return; }

  let inputBase = tokenIndex * p.inputDim;
  let blocksPerRow = p.inputDim / 32u;
  let rowBlockStart = localRow * blocksPerRow;

  let e = lid.x & 31u;
  let warp = lid.x >> 5u;

  var sum = 0.0;
  for (var j = 0u; j < ${Math.ceil(K / 64)}u; j++) {
    let b = warp + j * 2u;
    if (b >= blocksPerRow) { continue; }

    let blockBase = (rowBlockStart + b) * 5u;
    let scale = pow2i(bitcast<i32>(weightRaw[blockBase + 4u]));

    let wordInBlock = e >> 3u;
    let nibbleShift = (e & 7u) * 4u;

    let q = f32((weightRaw[blockBase + wordInBlock] >> nibbleShift) & 0x0Fu) - 8.0;
    sum = fma(q * scale, input[inputBase + b * 32u + e], sum);
  }

  reduceF32[lid.x] = sum;
  workgroupBarrier();
  if (lid.x < 32u) { reduceF32[lid.x] += reduceF32[lid.x + 32u]; }
  workgroupBarrier();
  if (lid.x < 16u) { reduceF32[lid.x] += reduceF32[lid.x + 16u]; }
  workgroupBarrier();
  if (lid.x <  8u) { reduceF32[lid.x] += reduceF32[lid.x +  8u]; }
  workgroupBarrier();
  if (lid.x <  4u) { reduceF32[lid.x] += reduceF32[lid.x +  4u]; }
  workgroupBarrier();
  if (lid.x <  2u) { reduceF32[lid.x] += reduceF32[lid.x +  2u]; }
  workgroupBarrier();
  if (lid.x == 0u) {
    output[tokenIndex * p.outputDim + localRow] = reduceF32[0] + reduceF32[1];
  }
}

// ==========================================================================
// WQ4 CANDIDATE v12 — workgroup_size(32) + 2 consecutive nibbles per thread.
//   Thread t owns elements {2t, 2t+1} of a block (always the same u32 word:
//   adjacent nibbles), so 16 threads cover one 32-element block and the warp
//   covers 2 adjacent blocks per iteration. Input reads are input[base + 64j
//   .. +64j+63] across the warp — perfectly coalesced, 2 cache lines per
//   iteration. 32 iterations x 2 MACs = 64 MACs/thread = 2x v10's per-thread
//   work, half the workgroups, one u32 weight word per 2 MACs.
// ==========================================================================
@compute @workgroup_size(32)
fn matmul_wq4_candidate_v12(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let localRow = wid.x;
  let tokenIndex = wid.y;
  if (localRow >= p.rowCount || tokenIndex >= p.tokenCount) { return; }

  let inputBase = tokenIndex * p.inputDim;
  let blocksPerRow = p.inputDim / 32u;
  let rowBlockStart = localRow * blocksPerRow;

  // e0 = 2t, e1 = 2t+1 -> same word: wordInBlock = (t&15)>>2, shift = (t&3)*8.
  let wordInBlock = (lid.x & 15u) >> 2u;
  let nibbleShift = (lid.x & 3u) * 8u;
  let eBase = (lid.x & 15u) * 2u;

  var sum0 = 0.0;
  var sum1 = 0.0;
  for (var j = 0u; j < ${Math.ceil(K / 64)}u; j++) {
    let b = j * 2u + (lid.x >> 4u);
    if (b >= blocksPerRow) { continue; }

    let blockBase = (rowBlockStart + b) * 5u;
    let scale = pow2i(bitcast<i32>(weightRaw[blockBase + 4u]));
    let packed = weightRaw[blockBase + wordInBlock];
    let q0 = f32((packed >> nibbleShift) & 0x0Fu) - 8.0;
    let q1 = f32((packed >> (nibbleShift + 4u)) & 0x0Fu) - 8.0;
    let xBase = inputBase + b * 32u + eBase;
    sum0 = fma(q0 * scale, input[xBase], sum0);
    sum1 = fma(q1 * scale, input[xBase + 1u], sum1);
  }

  reduceF32[lid.x] = sum0 + sum1;
  workgroupBarrier();
  if (lid.x < 16u) { reduceF32[lid.x] += reduceF32[lid.x + 16u]; }
  workgroupBarrier();
  if (lid.x <  8u) { reduceF32[lid.x] += reduceF32[lid.x +  8u]; }
  workgroupBarrier();
  if (lid.x <  4u) { reduceF32[lid.x] += reduceF32[lid.x +  4u]; }
  workgroupBarrier();
  if (lid.x <  2u) { reduceF32[lid.x] += reduceF32[lid.x +  2u]; }
  workgroupBarrier();
  if (lid.x == 0u) {
    output[tokenIndex * p.outputDim + localRow] = reduceF32[0] + reduceF32[1];
  }
}

// ==========================================================================
// WQ4 CANDIDATE v13a — v11 shape + i8 pre-unpacked weights.
//   Same as v11 (workgroup_size(64), element-per-thread, warp splits even/odd
//   blocks, unrolled reduction) but weights come from the v6 i8 layout
//   (8 u32 of packed signed bytes + 1 u32 scale = 9 u32/block) and the nibble
//   is extracted with unpack4xI8 + a uniform-per-thread select chain instead
//   of shift/and/sub. q values are already pre-subtracted (q-8).
// ==========================================================================
@compute @workgroup_size(64)
fn matmul_wq4_candidate_v13a(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let localRow = wid.x;
  let tokenIndex = wid.y;
  if (localRow >= p.rowCount || tokenIndex >= p.tokenCount) { return; }

  let inputBase = tokenIndex * p.inputDim;
  let blocksPerRow = p.inputDim / 32u;
  let rowBlockStart = localRow * blocksPerRow;

  let e = lid.x & 31u;
  let warp = lid.x >> 5u;
  let wordInBlock = e >> 2u;       // i8: 4 elements per u32 word
  let byteShift = (e & 3u) * 8u;   // element position within the word

  var sum = 0.0;
  for (var j = 0u; j < ${Math.ceil(K / 64)}u; j++) {
    let b = warp + j * 2u;
    if (b >= blocksPerRow) { continue; }

    let blockBase = (rowBlockStart + b) * 9u;
    let scale = pow2i(bitcast<i32>(weightRaw[blockBase + 8u]));
    let w4 = unpack4xI8(weightRaw[blockBase + wordInBlock]);
    // byteShift is uniform per thread -> the select chain folds to a single
    // lane pick; the two selects only depend on loop-invariant byteShift.
    let q = select(
      select(w4.x, w4.y, byteShift == 8u),
      select(w4.z, w4.w, byteShift == 24u),
      byteShift >= 16u);
    sum = fma(f32(q) * scale, input[inputBase + b * 32u + e], sum);
  }

  reduceF32[lid.x] = sum;
  workgroupBarrier();
  if (lid.x < 32u) { reduceF32[lid.x] += reduceF32[lid.x + 32u]; }
  workgroupBarrier();
  if (lid.x < 16u) { reduceF32[lid.x] += reduceF32[lid.x + 16u]; }
  workgroupBarrier();
  if (lid.x <  8u) { reduceF32[lid.x] += reduceF32[lid.x +  8u]; }
  workgroupBarrier();
  if (lid.x <  4u) { reduceF32[lid.x] += reduceF32[lid.x +  4u]; }
  workgroupBarrier();
  if (lid.x <  2u) { reduceF32[lid.x] += reduceF32[lid.x +  2u]; }
  workgroupBarrier();
  if (lid.x == 0u) {
    output[tokenIndex * p.outputDim + localRow] = reduceF32[0] + reduceF32[1];
  }
}

// ==========================================================================
// WQ4 CANDIDATE v13b — vec4-per-thread + i8 weights.
//   Each thread owns ONE u32 word of a block (4 elements, 4 consecutive
//   floats of input) and does 4 FMAs per unpack4xI8: 1 unpack + 4 FMAs per
//   4 MACs vs v11's 4x(shift+and+sub). 64 threads cover 8 blocks per
//   iteration (word = lid.x&7, block = j*8 + (lid.x>>3)), so K=2048 needs
//   ceil(K/256) = 8 iterations. Same unrolled 64->1 reduction as v11.
// ==========================================================================
@compute @workgroup_size(64)
fn matmul_wq4_candidate_v13b(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let localRow = wid.x;
  let tokenIndex = wid.y;
  if (localRow >= p.rowCount || tokenIndex >= p.tokenCount) { return; }

  let inputBase = tokenIndex * p.inputDim;
  let blocksPerRow = p.inputDim / 32u;
  let rowBlockStart = localRow * blocksPerRow;

  let wordInBlock = lid.x & 7u;
  let group = lid.x >> 3u;  // 0..7 -> which block of this iteration's 8

  var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
  for (var j = 0u; j < ${Math.ceil(K / 256)}u; j++) {
    let b = j * 8u + group;
    if (b >= blocksPerRow) { continue; }

    let blockBase = (rowBlockStart + b) * 9u;
    let scale = pow2i(bitcast<i32>(weightRaw[blockBase + 8u]));
    let w4 = vec4<f32>(unpack4xI8(weightRaw[blockBase + wordInBlock]));
    let xb = inputBase + b * 32u + wordInBlock * 4u;
    s0 = fma(w4.x * scale, input[xb + 0u], s0);
    s1 = fma(w4.y * scale, input[xb + 1u], s1);
    s2 = fma(w4.z * scale, input[xb + 2u], s2);
    s3 = fma(w4.w * scale, input[xb + 3u], s3);
  }

  reduceF32[lid.x] = s0 + s1 + s2 + s3;
  workgroupBarrier();
  if (lid.x < 32u) { reduceF32[lid.x] += reduceF32[lid.x + 32u]; }
  workgroupBarrier();
  if (lid.x < 16u) { reduceF32[lid.x] += reduceF32[lid.x + 16u]; }
  workgroupBarrier();
  if (lid.x <  8u) { reduceF32[lid.x] += reduceF32[lid.x +  8u]; }
  workgroupBarrier();
  if (lid.x <  4u) { reduceF32[lid.x] += reduceF32[lid.x +  4u]; }
  workgroupBarrier();
  if (lid.x <  2u) { reduceF32[lid.x] += reduceF32[lid.x +  2u]; }
  workgroupBarrier();
  if (lid.x == 0u) {
    output[tokenIndex * p.outputDim + localRow] = reduceF32[0] + reduceF32[1];
  }
}
`;

const module = device.createShaderModule({ code: wgslCode });
const compilation = await module.getCompilationInfo();
for (const message of compilation.messages) {
  console.log(`[WGSL ${message.type}] ${message.lineNum}:${message.linePos} ${message.message}`);
}
if (compilation.messages.some((m: any) => m.type === "error")) throw new Error("WGSL compilation failed");

const entryPoints = [
  "matmul_f16",
  "matmul_wq4_runtime",
  "matmul_wq4_candidate",
  "matmul_wq4_candidate_v4",
  "matmul_wq4_candidate_v5",
  "matmul_wq4_candidate_v6",
  "matmul_wq4_candidate_v7",
  "matmul_wq4_candidate_v9",
  "matmul_wq4_candidate_v10",
  "matmul_wq4_candidate_v11",
  "matmul_wq4_candidate_v12",
  "matmul_wq4_candidate_v13a",
  "matmul_wq4_candidate_v13b",
] as const;
type EntryPoint = typeof entryPoints[number];
const pipelines = new Map<EntryPoint, GPUComputePipeline>();
for (const entryPoint of entryPoints) {
  pipelines.set(entryPoint, device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint },
  }));
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length & 1 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) * 0.5;
}
function percentile(values: number[], p: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * p)));
  return sorted[idx]!;
}

async function runCase(M: number): Promise<void> {
  const inputCpu = new Float32Array(M * K);
  for (let i = 0; i < inputCpu.length; i++) inputCpu[i] = (random01() - 0.5) * 0.1;

  const inputBuffer = device.createBuffer({
    size: inputCpu.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const outputBytes = M * N * 4;
  const outputs = new Map<EntryPoint, GPUBuffer>();
  for (const entryPoint of entryPoints) {
    outputs.set(entryPoint, device.createBuffer({
      size: outputBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    }));
  }
  const paramsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuffer, 0, inputCpu);
  device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([M, K, N, N]));

  function weightFor(entryPoint: EntryPoint): GPUBuffer {
    if (entryPoint === "matmul_f16") return f16WeightBuffer;
    if (entryPoint === "matmul_wq4_candidate_v6" || entryPoint === "matmul_wq4_candidate_v13a" ||
        entryPoint === "matmul_wq4_candidate_v13b") return weight8Buffer;
    return wq4WeightBuffer;
  }

  const bindGroups = new Map<EntryPoint, GPUBindGroup>();
  for (const entryPoint of entryPoints) {
    const pipeline = pipelines.get(entryPoint)!;
    bindGroups.set(entryPoint, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: inputBuffer } },
        { binding: 2, resource: { buffer: weightFor(entryPoint) } },
        { binding: 3, resource: { buffer: outputs.get(entryPoint)! } },
      ],
    }));
  }

  function commandBuffer(entryPoint: EntryPoint, repeats: number): GPUCommandBuffer {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines.get(entryPoint)!);
    pass.setBindGroup(0, bindGroups.get(entryPoint)!);
    // v5 packs 8 tokens per workgroup (wid.y = token group);
    // v9 packs 4 rows per workgroup (wid.x = row band); all others 1:1.
    const xGroups = entryPoint === "matmul_wq4_candidate_v9" ? Math.ceil(N / 4) : N;
    const yGroups = entryPoint === "matmul_wq4_candidate_v5" ? Math.ceil(M / 8) : M;
    for (let i = 0; i < repeats; i++) pass.dispatchWorkgroups(xGroups, yGroups, 1);
    pass.end();
    return encoder.finish();
  }

  const repeats = M <= 4 ? cli.repeatsDecode : cli.repeatsPrefill;

  // Warm all pipelines before measuring.
  for (const entryPoint of entryPoints) {
    device.queue.submit([commandBuffer(entryPoint, 1)]);
    await device.queue.onSubmittedWorkDone();
  }

  const times = new Map<EntryPoint, number[]>();
  for (const entryPoint of entryPoints) times.set(entryPoint, []);

  // Interleave kernels per sample so boost/temperature effects are less biased.
  for (let sample = 0; sample < cli.samples; sample++) {
    const order = sample & 1 ? [...entryPoints].reverse() : [...entryPoints];
    for (const entryPoint of order) {
      const cmd = commandBuffer(entryPoint, repeats); // encode outside timing
      const start = performance.now();
      device.queue.submit([cmd]);
      await device.queue.onSubmittedWorkDone();
      times.get(entryPoint)!.push((performance.now() - start) / repeats);
    }
  }

  // Read outputs once, outside timing, to catch broken candidate kernels.
  async function readOutput(entryPoint: EntryPoint): Promise<Float32Array> {
    const staging = device.createBuffer({
      size: outputBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(outputs.get(entryPoint)!, 0, staging, 0, outputBytes);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await staging.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return copy;
  }

  const outs = new Map<EntryPoint, Float32Array>();
  for (const ep of entryPoints) outs.set(ep, await readOutput(ep));
  const f16Out = outs.get("matmul_f16")!;
  const runtimeOut = outs.get("matmul_wq4_runtime")!;

  function compare(reference: Float32Array, actual: Float32Array) {
    let sqErr = 0;
    let sqRef = 0;
    let maxAbs = 0;
    for (let i = 0; i < reference.length; i++) {
      const r = reference[i]!;
      const d = actual[i]! - r;
      sqErr += d * d;
      sqRef += r * r;
      maxAbs = Math.max(maxAbs, Math.abs(d));
    }
    return {
      relRmse: Math.sqrt(sqErr / Math.max(sqRef, 1e-30)),
      maxAbs,
    };
  }

  const quantError = compare(f16Out, runtimeOut);
  const wq4Candidates = entryPoints.filter((ep) => ep.startsWith("matmul_wq4_candidate"));
  const drifts = wq4Candidates.map((ep) => ({ ep, ...compare(runtimeOut, outs.get(ep)!) }));
  const ops = 2 * M * N * K;

  console.log("\n============================================================");
  console.log(`[case] M=${M}, K=${K}, N=${N}   repeats=${repeats}, samples=${cli.samples}`);
  console.log("------------------------------------------------------------");

  const medians = new Map<EntryPoint, number>();
  for (const entryPoint of entryPoints) medians.set(entryPoint, median(times.get(entryPoint)!));
  const f16Ms = medians.get("matmul_f16")!;
  for (const entryPoint of entryPoints) {
    const values = times.get(entryPoint)!;
    const ms = medians.get(entryPoint)!;
    const gflops = (ops / (ms / 1000)) / 1e9;
    const ratio = ms / f16Ms;
    console.log(`${entryPoint.padEnd(24)} ${ms.toFixed(3).padStart(9)} ms  ` +
      `${gflops.toFixed(1).padStart(8)} GFLOP/s  ` +
      `${ratio.toFixed(2).padStart(5)}x F16  ` +
      `[p10 ${percentile(values, 0.10).toFixed(3)}, p90 ${percentile(values, 0.90).toFixed(3)}]`);
  }
  console.log("------------------------------------------------------------");
  console.log(`[quality] WQ4 vs F16:       relRMSE=${(quantError.relRmse * 100).toFixed(3)}%  maxAbs=${quantError.maxAbs.toExponential(3)}`);
  for (const d of drifts) {
    console.log(`[quality] ${d.ep.padEnd(24)} drift relRMSE=${(d.relRmse * 100).toFixed(6)}%  maxAbs=${d.maxAbs.toExponential(3)}`);
  }
  console.log("          drift should be ~floating-point noise; large drift = kernel bug.");
  console.log("============================================================");

  inputBuffer.destroy();
  paramsBuffer.destroy();
  for (const buffer of outputs.values()) buffer.destroy();
}

for (const M of cli.m) await runCase(M);

f16WeightBuffer.destroy();
wq4WeightBuffer.destroy();
device.destroy();
