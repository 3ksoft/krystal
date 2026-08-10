// Vision tower: generic matmul + bias (ADA-0009, M2).
//
// out[token, o] = bias[o] + sum_k in[token, k] * w[o*inputDim + k]
// (weights row-major [output, input], GGUF convention — same as the oracle's
// linearInto). Used for q/k/v, attention out-projection, MLP up/down.
//
// One workgroup per (output row, token); WG=64 reduction over inputDim.
// Params: inOff/outOff, wOff=weight, bOff=bias (bias always present for the
// tower's linear layers), inputDim, outputDim, tokenCount.

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
fn matmul(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.x;
  let p = wid.y;
  if (row >= params.outputDim || p >= params.tokenCount) { return; }
  let inputDim = params.inputDim;

  var acc = 0.0;
  var k = lid.x;
  loop {
    if (k >= inputDim) { break; }
    acc += arena[params.inOff + p * inputDim + k] * weights[params.wOff + row * inputDim + k];
    k += 64u;
  }
  reduce[lid.x] = acc;
  workgroupBarrier();

  var width = 32u;
  loop {
    if (width == 0u) { break; }
    if (lid.x < width) { reduce[lid.x] += reduce[lid.x + width]; }
    workgroupBarrier();
    width >>= 1u;
  }

  if (lid.x == 0u) {
    arena[params.outOff + p * params.outputDim + row] = reduce[0] + weights[params.bOff + row];
  }
}
