// Vision tower: in-place tanh-approximated GELU (`gelu_pytorch_tanh`, the
// tower MLP activation — ADA-0009, M2). Same formula as the CPU oracle.
//
// g(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// One invocation per element; dispatch ceil(tokenCount * outputDim / 256).
// Params: inOff==outOff (in-place), tokenCount * outputDim = element count.

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
fn gelu_tanh(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.tokenCount * params.outputDim;
  let i = gid.x;
  if (i >= total) { return; }
  let x = arena[params.inOff + i];
  arena[params.inOff + i] = 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}
