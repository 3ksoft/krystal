// Vision tower tail: pixel-unshuffle + projector Linear -> GELU (exact erf) ->
// Linear, producing the image embeddings [tokens, projectorDim] that get
// injected into the LM sequence (ADA-0009, M2).
//
// One workgroup per output token; WG=256 invocations split both linear rows.
// Unshuffle channel order follows the mmproj/llama.cpp convention (verified
// against llama.cpp ground truth): u = c + dim*i + dim*factor*j with i the
// vertical sub-pixel and j the horizontal one — the sub-pixel bits are BLOCKED
// at the top of the channel space, NOT interleaved like torch PixelUnshuffle
// (c*factor^2 + i*factor + j). The GGUF mm.1 columns are stored in this order;
// the torch order yields embeddings that differ per image but never match the
// content. Gathered from the post-LN hidden buffer at
// ((oy*factor+i)*gridW + (ox*factor+j))*dim + c — same indexing as the CPU oracle.
//
// Params: tokenCount=T, inputDim=hidden, outputDim=projectorDim,
//         inOff=post-LN hidden, outOff=embeddings, wOff=mm1, bOff=mm1Bias,
//         auxOff=mm2, aux2Off=mm2Bias, gridW, gridH, factor.
// Bindings: group0 { 0: params(uniform), 1: arena(storage) },
//           group1 { 0: misc weights(storage) }.

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

var<workgroup> mid: array<f32, 2048>;

// Abramowitz–Stegun 7.1.26 erf approximation (max abs error ~1.5e-7), the same
// coefficients as the CPU oracle. WGSL core has no erf().
fn erf(x: f32) -> f32 {
  let sign = select(1.0, -1.0, x < 0.0);
  let ax = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
  return sign * y;
}

@compute @workgroup_size(256)
fn unshuffle_project(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  if (t >= params.tokenCount) { return; }
  let dim = params.inputDim;
  let factor = params.factor;
  let outW = params.gridW / factor;
  let oy = t / outW;
  let ox = t % outW;
  let unshDim = dim * factor * factor;
  let projDim = params.outputDim;

  // mm.1: gather unshuffled (c, i, j), GELU-erf
  var o = lid.x;
  loop {
    if (o >= projDim) { break; }
    var acc = weights[params.bOff + o];
    for (var c = 0u; c < dim; c++) {
      for (var ii = 0u; ii < factor; ii++) {
        for (var jj = 0u; jj < factor; jj++) {
          let src = ((oy * factor + ii) * params.gridW + (ox * factor + jj)) * dim + c;
          let u = c + dim * ii + dim * factor * jj;
          acc += arena[params.inOff + src] * weights[params.wOff + o * unshDim + u];
        }
      }
    }
    let g = 0.5 * acc * (1.0 + erf(acc * 0.7071067811865476));
    mid[o] = g;
    o += 256u;
  }
  workgroupBarrier();

  // mm.2: projector hidden -> projector hidden
  var o2 = lid.x;
  loop {
    if (o2 >= projDim) { break; }
    var acc = weights[params.aux2Off + o2];
    for (var m = 0u; m < projDim; m++) {
      acc += mid[m] * weights[params.auxOff + o2 * projDim + m];
    }
    arena[params.outOff + t * projDim + o2] = acc;
    o2 += 256u;
  }
}
