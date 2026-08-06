// TEMPORARY COMPATIBILITY BACKEND. New kernels live as split body-only shaders.
// Keep this layout in lock-step with schema.ts:LlmRuntime.
// schema-pop still owns CPU serialization/deserialization; the WGSL declaration
// is explicit so the runtime does not depend on Sandblaster exposing its LayoutPlan.
struct LlmRuntime {
  contextCapacity: u32,
  maxNewTokens: u32,
  eosToken: u32,
  promptTokenCount: u32,
  position: u32,
  generatedCount: u32,
  currentToken: u32,
  status: u32,
  telemetryRevision: u32,
  lastToken: u32,
  errorCode: u32,
  pad0: u32,
};

struct OpParams {
  inputOffset: u32,
  outputOffset: u32,
  auxOffset: u32,
  aux2Offset: u32,

  tokenCount: u32,
  inputDim: u32,
  outputDim: u32,
  rowStart: u32,

  rowCount: u32,
  layerIndex: u32,
  attentionSlot: u32,
  mode: u32,

  f0: f32,
  f1: f32,
  u0: u32,
  u1: u32,
};

@group(0) @binding(0) var<uniform> op: OpParams;
@group(0) @binding(1) var<storage, read_write> runtime: LlmRuntime;
@group(0) @binding(2) var<storage, read_write> tokens: array<u32>;
@group(0) @binding(3) var<storage, read_write> arena: array<f32>;
@group(0) @binding(4) var<storage, read_write> kvCache: array<f32>;
@group(0) @binding(5) var<storage, read_write> convCache: array<f32>;

// Only one of the two weight bindings is used by a given kernel. Keeping both
// bindings stable makes every weight tensor share one bind-group layout.
@group(1) @binding(0) var<storage, read> weightRaw: array<u32>;
@group(1) @binding(1) var<storage, read> weight32: array<f32>;

const WG: u32 = 64u;
const ARGMAX_WG: u32 = 256u;
const MAX_CONTEXT: u32 = 1024u;
const HEAD_DIM: u32 = 64u;
const KV_HEADS: u32 = 8u;
const QUERY_HEADS: u32 = 32u;
const KV_DIM: u32 = KV_HEADS * HEAD_DIM;

fn load_f16(index: u32) -> f32 {
  let packed = weightRaw[index >> 1u];
  let pair = unpack2x16float(packed);
  return select(pair.x, pair.y, (index & 1u) != 0u);
}

fn load_wq4(index: u32) -> f32 {
  let block = index / 32u;
  let lane = index % 32u;
  let baseU32 = block * 5u;
  let packed = weightRaw[baseU32 + lane / 8u];
  let shift = (lane % 8u) * 4u;
  let expVal = bitcast<i32>(weightRaw[baseU32 + 4u]);
  return (f32((packed >> shift) & 0x0Fu) - 8.0) * exp2(f32(expVal));
}

fn token_position(tokenIndex: u32) -> u32 {
  if (op.mode == 1u) { return runtime.position; }
  if (op.mode == 2u) { return op.u1 + tokenIndex; }
  return tokenIndex;
}

fn arena_index(base: u32, tokenIndex: u32, dim: u32, stride: u32) -> u32 {
  return base + tokenIndex * stride + dim;
}

var<workgroup> reduceF32: array<f32, 256>;
var<workgroup> reduceU32: array<u32, 256>;
var<workgroup> attentionScores: array<f32, 1024>;

@compute @workgroup_size(256)
fn embedding(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = op.tokenCount * op.outputDim;
  let linear = gid.x;
  if (linear >= total) { return; }

  let tokenIndex = linear / op.outputDim;
  let dim = linear % op.outputDim;
  let sourceToken = op.u0 + tokenIndex;
  let tokenId = select(tokens[sourceToken], runtime.currentToken, op.mode != 0u);

  if (tokenId < op.rowStart || tokenId >= op.rowStart + op.rowCount) { return; }
  let localRow = tokenId - op.rowStart;
  let value = load_f16(localRow * op.outputDim + dim);
  arena[arena_index(op.outputOffset, tokenIndex, dim, op.outputDim)] = value;
}

@compute @workgroup_size(256)
fn embedding_wq4(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = op.tokenCount * op.outputDim;
  let linear = gid.x;
  if (linear >= total) { return; }

  let tokenIndex = linear / op.outputDim;
  let dim = linear % op.outputDim;
  let sourceToken = op.u0 + tokenIndex;
  let tokenId = select(tokens[sourceToken], runtime.currentToken, op.mode != 0u);

  if (tokenId < op.rowStart || tokenId >= op.rowStart + op.rowCount) { return; }
  let localRow = tokenId - op.rowStart;
  let value = load_wq4(localRow * op.outputDim + dim);
  arena[arena_index(op.outputOffset, tokenIndex, dim, op.outputDim)] = value;
}

@compute @workgroup_size(64)
fn rms_norm(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tokenIndex = wid.x;
  if (tokenIndex >= op.tokenCount) { return; }

  var sum = 0.0;
  var d = lid.x;
  loop {
    if (d >= op.inputDim) { break; }
    let x = arena[arena_index(op.inputOffset, tokenIndex, d, op.inputDim)];
    sum += x * x;
    d += WG;
  }
  reduceF32[lid.x] = sum;
  workgroupBarrier();

  var width = WG >> 1u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) { reduceF32[lid.x] += reduceF32[lid.x + width]; }
    workgroupBarrier();
    width >>= 1u;
  }

  let invRms = inverseSqrt(reduceF32[0] / f32(op.inputDim) + op.f0);
  d = lid.x;
  loop {
    if (d >= op.inputDim) { break; }
    let x = arena[arena_index(op.inputOffset, tokenIndex, d, op.inputDim)];
    arena[arena_index(op.outputOffset, tokenIndex, d, op.inputDim)] = x * invRms * weight32[d];
    d += WG;
  }
}

@compute @workgroup_size(64)
fn matmul_f16(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let localRow = wid.x;
  let tokenIndex = wid.y;
  if (localRow >= op.rowCount || tokenIndex >= op.tokenCount) { return; }

  var sum = 0.0;
  var k = lid.x;
  loop {
    if (k >= op.inputDim) { break; }
    let x = arena[arena_index(op.inputOffset, tokenIndex, k, op.inputDim)];
    let w = load_f16(localRow * op.inputDim + k);
    sum += x * w;
    k += WG;
  }
  reduceF32[lid.x] = sum;
  workgroupBarrier();

  var width = WG >> 1u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) { reduceF32[lid.x] += reduceF32[lid.x + width]; }
    workgroupBarrier();
    width >>= 1u;
  }

  if (lid.x == 0u) {
    let logicalRow = op.rowStart + localRow;
    arena[op.outputOffset + tokenIndex * op.outputDim + logicalRow] = reduceF32[0];
  }
}

@compute @workgroup_size(64)
fn matmul_f32(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let localRow = wid.x;
  let tokenIndex = wid.y;
  if (localRow >= op.rowCount || tokenIndex >= op.tokenCount) { return; }

  var sum = 0.0;
  var k = lid.x;
  loop {
    if (k >= op.inputDim) { break; }
    let x = arena[arena_index(op.inputOffset, tokenIndex, k, op.inputDim)];
    sum += x * weight32[localRow * op.inputDim + k];
    k += WG;
  }
  reduceF32[lid.x] = sum;
  workgroupBarrier();

  var width = WG >> 1u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) { reduceF32[lid.x] += reduceF32[lid.x + width]; }
    workgroupBarrier();
    width >>= 1u;
  }

  if (lid.x == 0u) {
    let logicalRow = op.rowStart + localRow;
    arena[op.outputOffset + tokenIndex * op.outputDim + logicalRow] = reduceF32[0];
  }
}

@compute @workgroup_size(64)
fn matmul_wq4(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let localRow = wid.x;
  let tokenIndex = wid.y;
  if (localRow >= op.rowCount || tokenIndex >= op.tokenCount) { return; }

  let inputBase = op.inputOffset + tokenIndex * op.inputDim;
  let blocksPerRow = op.inputDim / 32u;
  // Weight pages are row-local. rowStart is only the logical output-row offset.
  let rowBlockStart = localRow * blocksPerRow;

  var sum: f32 = 0.0;
  var b = lid.x;
  loop {
    if (b >= blocksPerRow) { break; }

    let blockIdx = rowBlockStart + b;
    let baseU32 = blockIdx * 5u; // 4 packed words + i32 pow2 exponent = 20 B
    let expVal = bitcast<i32>(weightRaw[baseU32 + 4u]);
    let scale = exp2(f32(expVal));
    let kStart = b * 32u;

    for (var w = 0u; w < 4u; w++) {
      let packed = weightRaw[baseU32 + w];
      let kBase = kStart + w * 8u;

      sum += (f32((packed >>  0u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 0u];
      sum += (f32((packed >>  4u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 1u];
      sum += (f32((packed >>  8u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 2u];
      sum += (f32((packed >> 12u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 3u];
      sum += (f32((packed >> 16u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 4u];
      sum += (f32((packed >> 20u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 5u];
      sum += (f32((packed >> 24u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 6u];
      sum += (f32((packed >> 28u) & 0x0Fu) - 8.0) * scale * arena[inputBase + kBase + 7u];
    }

    b += WG;
  }

  reduceF32[lid.x] = sum;
  workgroupBarrier();

  var width = WG >> 1u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) { reduceF32[lid.x] += reduceF32[lid.x + width]; }
    workgroupBarrier();
    width >>= 1u;
  }

  if (lid.x == 0u) {
    let logicalRow = op.rowStart + localRow;
    arena[op.outputOffset + tokenIndex * op.outputDim + logicalRow] = reduceF32[0];
  }
}

@compute @workgroup_size(256)
fn residual_add(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = op.tokenCount * op.inputDim;
  let i = gid.x;
  if (i >= total) { return; }
  arena[op.outputOffset + i] = arena[op.inputOffset + i] + arena[op.auxOffset + i];
}

@compute @workgroup_size(256)
fn silu_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = op.tokenCount * op.inputDim;
  let i = gid.x;
  if (i >= total) { return; }
  let gate = arena[op.inputOffset + i];
  let up = arena[op.auxOffset + i];
  arena[op.outputOffset + i] = (gate / (1.0 + exp(-gate))) * up;
}

@compute @workgroup_size(256)
fn shortconv_prefill(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = op.tokenCount * op.inputDim;
  let linear = gid.x;
  if (linear >= total) { return; }
  let t = linear / op.inputDim;
  let d = linear % op.inputDim;

  // in_proj output layout is [B | C | x], each hidden-sized.
  let base = op.inputOffset + t * (op.inputDim * 3u);
  let B = arena[base + d];
  let C = arena[base + op.inputDim + d];
  let x = arena[base + op.inputDim * 2u + d];
  let bx0 = B * x;

  var y = bx0 * weight32[d * 3u + 2u];
  if (t >= 1u) {
    let p = op.inputOffset + (t - 1u) * (op.inputDim * 3u);
    y += arena[p + d] * arena[p + op.inputDim * 2u + d] * weight32[d * 3u + 1u];
  }
  if (t >= 2u) {
    let p = op.inputOffset + (t - 2u) * (op.inputDim * 3u);
    y += arena[p + d] * arena[p + op.inputDim * 2u + d] * weight32[d * 3u];
  }

  arena[op.outputOffset + t * op.inputDim + d] = C * y;

  // Persist only the tail needed by the first decode step.
  let keepStart = select(0u, op.tokenCount - 3u, op.tokenCount > 3u);
  if (t >= keepStart) {
    let slot = select(3u - op.tokenCount + t, t - keepStart, op.tokenCount >= 3u);
    let cacheBase = op.layerIndex * op.inputDim * 3u + d * 3u;
    convCache[cacheBase + slot] = bx0;
  }
}

// Continue a cached prefix with a small live query. One workgroup owns one
// hidden dimension and walks the query tokens sequentially, so it can consume
// the three-value conv tail left by the cached context without any cross-
// workgroup synchronization.
@compute @workgroup_size(1)
fn shortconv_continue(@builtin(workgroup_id) wid: vec3<u32>) {
  let d = wid.x;
  if (d >= op.inputDim || op.tokenCount == 0u) { return; }

  let cacheBase = op.layerIndex * op.inputDim * 3u + d * 3u;
  var c0 = convCache[cacheBase];
  var c1 = convCache[cacheBase + 1u];
  var c2 = convCache[cacheBase + 2u];

  for (var t = 0u; t < op.tokenCount; t++) {
    let base = op.inputOffset + t * (op.inputDim * 3u);
    let B = arena[base + d];
    let C = arena[base + op.inputDim + d];
    let x = arena[base + op.inputDim * 2u + d];
    let bx = B * x;

    let y = c1 * weight32[d * 3u] +
            c2 * weight32[d * 3u + 1u] +
            bx * weight32[d * 3u + 2u];
    arena[op.outputOffset + t * op.inputDim + d] = C * y;

    c0 = c1;
    c1 = c2;
    c2 = bx;
  }

  convCache[cacheBase] = c0;
  convCache[cacheBase + 1u] = c1;
  convCache[cacheBase + 2u] = c2;
}

@compute @workgroup_size(256)
fn shortconv_decode(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;
  if (d >= op.inputDim) { return; }
  let base = op.inputOffset;
  let B = arena[base + d];
  let C = arena[base + op.inputDim + d];
  let x = arena[base + op.inputDim * 2u + d];
  let bx = B * x;

  let cacheBase = op.layerIndex * op.inputDim * 3u + d * 3u;
  let a = convCache[cacheBase + 1u];
  let b = convCache[cacheBase + 2u];
  convCache[cacheBase] = a;
  convCache[cacheBase + 1u] = b;
  convCache[cacheBase + 2u] = bx;

  let y = a * weight32[d * 3u] +
          b * weight32[d * 3u + 1u] +
          bx * weight32[d * 3u + 2u];
  arena[op.outputOffset + d] = C * y;
}

fn rope_component(base: u32, tokenIndex: u32, head: u32, d: u32, headCount: u32) -> f32 {
  let dim = headCount * HEAD_DIM;
  return arena[base + tokenIndex * dim + head * HEAD_DIM + d];
}

@compute @workgroup_size(64)
fn qk_norm_rope(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let head = wid.x;
  let tokenIndex = wid.y;
  if (tokenIndex >= op.tokenCount) { return; }
  let isK = op.u0 != 0u;
  let headCount = select(QUERY_HEADS, KV_HEADS, isK);
  if (head >= headCount) { return; }

  let base = select(op.inputOffset, op.auxOffset, isK);
  var x = rope_component(base, tokenIndex, head, lid.x, headCount);
  reduceF32[lid.x] = x * x;
  workgroupBarrier();

  var width = WG >> 1u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) { reduceF32[lid.x] += reduceF32[lid.x + width]; }
    workgroupBarrier();
    width >>= 1u;
  }

  let norm = inverseSqrt(reduceF32[0] / 64.0 + op.f0);
  x *= norm * weight32[lid.x];

  // Liquid's RoPE uses rotate_half over the two 32-wide halves.
  let half = HEAD_DIM / 2u;
  let pairDim = select(lid.x + half, lid.x - half, lid.x >= half);
  var pair = rope_component(base, tokenIndex, head, pairDim, headCount);
  pair *= norm * weight32[pairDim];
  pair = select(-pair, pair, lid.x >= half);
  // Every lane must capture its unrotated partner before any lane overwrites Q/K.
  workgroupBarrier();

  let freqDim = lid.x % half;
  let invFreq = pow(op.f1, -2.0 * f32(freqDim) / f32(HEAD_DIM));
  let angle = f32(token_position(tokenIndex)) * invFreq;
  let rotated = x * cos(angle) + pair * sin(angle);

  let dim = headCount * HEAD_DIM;
  arena[base + tokenIndex * dim + head * HEAD_DIM + lid.x] = rotated;
}

@compute @workgroup_size(256)
fn kv_store(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = op.tokenCount * KV_DIM;
  let linear = gid.x;
  if (linear >= total) { return; }
  let t = linear / KV_DIM;
  let d = linear % KV_DIM;
  let position = token_position(t);
  if (position >= runtime.contextCapacity) { return; }

  let layerStride = runtime.contextCapacity * KV_DIM;
  let baseK = (op.attentionSlot * 2u) * layerStride;
  let baseV = baseK + layerStride;
  kvCache[baseK + position * KV_DIM + d] = arena[op.inputOffset + t * KV_DIM + d];
  kvCache[baseV + position * KV_DIM + d] = arena[op.auxOffset + t * KV_DIM + d];
}

@compute @workgroup_size(64)
fn attention(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let head = wid.x;
  let tokenIndex = wid.y;
  if (head >= QUERY_HEADS || tokenIndex >= op.tokenCount) { return; }

  let qBase = op.inputOffset + tokenIndex * (QUERY_HEADS * HEAD_DIM) + head * HEAD_DIM;
  let kvHead = head / (QUERY_HEADS / KV_HEADS);
  let position = token_position(tokenIndex);
  // Clamp instead of early-returning: storage-buffer loads are non-uniform to
  // WGSL's uniformity analysis, so a return here would put the workgroupBarrier()
  // below in non-uniform control flow (rejected by Dawn). position < MAX_CONTEXT
  // is guaranteed by the runtime (contextCapacity <= MAX_CONTEXT), so clamping
  // is lossless.
  let contextCount = min(position + 1u, MAX_CONTEXT);

  let layerStride = runtime.contextCapacity * KV_DIM;
  let cacheK = (op.attentionSlot * 2u) * layerStride;
  let cacheV = cacheK + layerStride;

  var p = lid.x;
  loop {
    if (p >= contextCount) { break; }
    var dot = 0.0;
    for (var d = 0u; d < HEAD_DIM; d++) {
      let q = arena[qBase + d];
      let k = kvCache[cacheK + p * KV_DIM + kvHead * HEAD_DIM + d];
      dot += q * k;
    }
    attentionScores[p] = dot * 0.125; // 1 / sqrt(64)
    p += WG;
  }
  workgroupBarrier();

  if (lid.x == 0u) {
    var maxScore = -3.402823466e+38;
    for (var i = 0u; i < contextCount; i++) { maxScore = max(maxScore, attentionScores[i]); }
    var sumExp = 0.0;
    for (var i = 0u; i < contextCount; i++) {
      let e = exp(attentionScores[i] - maxScore);
      attentionScores[i] = e;
      sumExp += e;
    }
    let inv = 1.0 / max(sumExp, 1e-20);
    for (var i = 0u; i < contextCount; i++) { attentionScores[i] *= inv; }
  }
  workgroupBarrier();

  let d = lid.x;
  var value = 0.0;
  for (var i = 0u; i < contextCount; i++) {
    value += attentionScores[i] * kvCache[cacheV + i * KV_DIM + kvHead * HEAD_DIM + d];
  }
  let outIndex = tokenIndex * (QUERY_HEADS * HEAD_DIM) + head * HEAD_DIM + d;
  arena[op.outputOffset + outIndex] = value;
}

@compute @workgroup_size(256)
fn arena_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = op.tokenCount * op.inputDim;
  let i = gid.x;
  if (i >= total) { return; }
  arena[op.outputOffset + i] = arena[op.inputOffset + i];
}

@compute @workgroup_size(256)
fn argmax(
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  // No early return before the barriers below: runtime.status is a storage-buffer
  // load (non-uniform to WGSL), which would put workgroupBarrier() in non-uniform
  // control flow. The status gate is applied on the final lane-0 write instead.
  var bestValue = -3.402823466e+38;
  var bestToken = 0u;
  var i = lid.x;
  loop {
    if (i >= op.inputDim) { break; }
    let v = arena[op.inputOffset + i];
    if (v > bestValue || (v == bestValue && i < bestToken)) {
      bestValue = v;
      bestToken = i;
    }
    i += ARGMAX_WG;
  }
  reduceF32[lid.x] = bestValue;
  reduceU32[lid.x] = bestToken;
  workgroupBarrier();

  var width = ARGMAX_WG >> 1u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) {
      let rv = reduceF32[lid.x + width];
      let rt = reduceU32[lid.x + width];
      let lv = reduceF32[lid.x];
      let lt = reduceU32[lid.x];
      if (rv > lv || (rv == lv && rt < lt)) {
        reduceF32[lid.x] = rv;
        reduceU32[lid.x] = rt;
      }
    }
    workgroupBarrier();
    width >>= 1u;
  }

  if (lid.x == 0u) {
    if (runtime.status == 1u) {
      if (op.mode != 0u) { runtime.position += 1u; }
      let outIndex = runtime.contextCapacity + runtime.generatedCount;
      let token = reduceU32[0];
      tokens[outIndex] = token;
      runtime.currentToken = token;
      runtime.lastToken = token;
      runtime.generatedCount += 1u;
      runtime.telemetryRevision += 1u;

      if (token == runtime.eosToken) {
        runtime.status = 2u;
      } else if (runtime.generatedCount >= runtime.maxNewTokens) {
        runtime.status = 3u;
      }
    }
  }
}
