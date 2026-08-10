// Vision tower: out[i] += in[i] (residual streams after attention and MLP,
// ADA-0009, M2). One invocation per element.
// Params: inOff=addend (h), outOff=accumulator (hidden), tokenCount * dim = count.

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

@compute @workgroup_size(256)
fn residual_add(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.tokenCount * params.inputDim;
  let i = gid.x;
  if (i >= total) { return; }
  arena[params.outOff + i] += arena[params.inOff + i];
}
