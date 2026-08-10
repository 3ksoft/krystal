// Vision tower: patch embedding + position embeddings (ADA-0009, M2).
//
// One workgroup per (output row, token); WG=64 invocations each accumulate a
// strided partial dot product, then a workgroup tree-reduce produces the full
// 3*patch^2-term sum (same shape as the matmul kernel — the earlier per-token
// workgroups summed only every 64th input, which is why the head diverged).
// Weights are the conv kernel flattened as a [hidden, 3*patch^2] matmul
// (input order (c,h,w), w fastest — same flattening as patchify()).
// The patch embedding is an nn.Linear with bias (HF Siglip2VisionEmbeddings,
// `v.patch_embd.bias` F32 in the mmproj) — the bias is added before the
// position embeddings, exactly like the CPU oracle.
//
// Params: inOff=patches, outOff=hidden, wOff=patchEmb, auxOff=patchEmbBias,
//         bOff=posEmb, inputDim=3*patch^2, outputDim=hiddenSize,
//         tokenCount=patchCount.
// Bindings: group0 { 0: params(storage read), 1: arena(storage read_write) },
//           group1 { 0: misc weights(storage read_write) }.

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
fn patch_embed(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
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
    arena[params.outOff + p * params.outputDim + row] =
      reduce[0] + weights[params.auxOff + row] + arena[params.bOff + p * params.outputDim + row];
  }
}
