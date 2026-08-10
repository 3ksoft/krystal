// Vision tower: LayerNorm with weight + bias (ADA-0009, M2).
//
// y[i] = (x[i] - mean) / sqrt(var + eps) * w[i] + b[i]  (population variance,
// exactly like the CPU oracle / HF modeling_siglip2.py).
//
// One workgroup per token row; two WG=64 tree reductions (mean, variance).
// Params: inOff/outOff, wOff=weight, bOff=bias, inputDim=hidden, tokenCount.

struct VisionParams {
  tokenCount: u32,
  inputDim: u32,
  outputDim: u32,
  headDim: u32,
  heads: u32,
  gridW: u32,
  gridH: u32,
  inOff: u32,
  outOff: u32,
  wOff: u32,
  bOff: u32,
  auxOff: u32,
  aux2Off: u32,
  factor: u32,
  projectorDim: u32,
  eps: f32,
  scale: f32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

// Params live in a storage buffer (not uniform): the per-dispatch record ring
// exceeds the 64 KiB uniform binding limit once the tower runs all 27 blocks.
@group(0) @binding(0) var<storage, read> params: VisionParams;
@group(0) @binding(1) var<storage, read_write> arena: array<f32>;
@group(1) @binding(0) var<storage, read_write> weights: array<f32>;

var<workgroup> reduce: array<f32, 64>;

@compute @workgroup_size(64)
fn layernorm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let p = wid.x;
  if (p >= params.tokenCount) { return; }
  let dim = params.inputDim;

  var sum = 0.0;
  var i = lid.x;
  loop {
    if (i >= dim) { break; }
    sum += arena[params.inOff + p * dim + i];
    i += 64u;
  }
  reduce[lid.x] = sum;
  workgroupBarrier();

  var width = 32u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) { reduce[lid.x] += reduce[lid.x + width]; }
    workgroupBarrier();
    width >>= 1u;
  }

  let mean = reduce[0] / f32(dim);

  var sq = 0.0;
  i = lid.x;
  loop {
    if (i >= dim) { break; }
    let d = arena[params.inOff + p * dim + i] - mean;
    sq += d * d;
    i += 64u;
  }
  reduce[lid.x] = sq;
  workgroupBarrier();

  width = 32u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) { reduce[lid.x] += reduce[lid.x + width]; }
    workgroupBarrier();
    width >>= 1u;
  }

  let inv = inverseSqrt(reduce[0] / f32(dim) + params.eps);
  i = lid.x;
  loop {
    if (i >= dim) { break; }
    let x = arena[params.inOff + p * dim + i];
    arena[params.outOff + p * dim + i] = (x - mean) * inv * weights[params.wOff + i] + weights[params.bOff + i];
    i += 64u;
  }
}
