// Vision tower: bidirectional multi-head attention with a padding mask
// (ADA-0009, M2). No RoPE, no causal mask: every query attends to every key
// (scale = headDim^-0.5), mirroring the CPU oracle / modeling_siglip2.py.
//
// One workgroup per (head, query token); WG=256. The qkv buffer packs per-token
// rows [q | k | v]. Scores for all keys live in workgroup memory (max 1024
// keys, 4 KiB). The kernel is split into three phases:
//
//   1. parallel scores: each invocation computes the dot product for its own
//      key slice (b = lid.x, lid.x+256, ...) into scores[] — no cross-talk,
//      no barrier needed inside this phase.
//   2. SERIAL softmax by invocation 0: max-subtraction, exp, and the sum are
//      computed by one invocation after a single workgroup barrier. This is
//      deliberate: the parallel exp + tree-reduced max/sum variant showed
//      nondeterministic whole-row corruption on the Dawn/NVIDIA (Ampere)
//      stack (a few workgroups per run, values off by ~5e-2, no NaN, stale
//      score entries when the V phase read them). Serializing the softmax is
//      provably deterministic and costs O(n) per workgroup — negligible next
//      to the O(n * headDim) V accumulation for n <= 1024.
//   3. parallel V accumulation: the first headDim invocations each own one
//      output dim and scan all keys (no cross-invocation reduction needed).
//
// Mask semantics (identical to the oracle): a masked QUERY leaves its attn row
// zero; a masked KEY contributes score -1e30 (exp -> 0); a query whose keys are
// all masked leaves its row zero (no NaN).
//
// Params: inOff=qkv, outOff=attn, bOff=padMask, tokenCount=patchCount,
//         inputDim=hidden, headDim, heads, scale.
// Bindings: group0 { 0: params(storage read), 1: arena(storage read_write) },
//           group1 { 0: layer weights(storage) } (unused here).

const MAX_N: u32 = 1024u;

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

var<workgroup> scores: array<f32, MAX_N>;
var<workgroup> wgSum: array<f32, 256>;

@compute @workgroup_size(256)
fn attention(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let head = wid.x;
  let a = wid.y;
  if (head >= params.heads || a >= params.tokenCount) { return; }
  let n = params.tokenCount;
  let dim = params.inputDim;
  let headDim = params.headDim;
  let hOff = head * headDim;
  let qBase = params.inOff + a * dim + hOff;

  let queryMasked = arena[params.bOff + a] == 0.0;

  // phase 1: scores[b] = q . k[b] * scale  (masked keys -> -1e30 sentinel,
  // so exp -> 0). Each invocation owns its own key slice.
  var b = lid.x;
  loop {
    if (b >= n) { break; }
    var dot = 0.0;
    for (var d = 0u; d < headDim; d++) {
      dot += arena[qBase + d] * arena[params.inOff + (n + b) * dim + hOff + d];
    }
    let masked = arena[params.bOff + b] == 0.0;
    scores[b] = select(dot * params.scale, -1.0e30, masked);
    b += 256u;
  }
  workgroupBarrier();

  // phase 2: serial max-subtraction softmax + sum by invocation 0. This is
  // the deterministic structure — see the module comment for why the parallel
  // variant was rejected.
  if (lid.x == 0u) {
    var maxScore = -1.0e30;
    for (var i = 0u; i < n; i++) {
      maxScore = max(maxScore, scores[i]);
    }
    var sum = 0.0;
    for (var i = 0u; i < n; i++) {
      let e = exp(scores[i] - maxScore);
      scores[i] = e;
      sum += e;
    }
    wgSum[0] = maxScore;
    wgSum[1] = sum;
  }
  workgroupBarrier();

  // masked query (or all-masked keys) -> attn row stays zero, like the oracle
  // (the oracle skips queries whose keys are all -Inf; -1e30 sentinel is the
  // f32 stand-in, so maxScore staying at the sentinel means every key masked)
  if (queryMasked || wgSum[0] < -1.0e29) {
    if (lid.x < headDim) {
      arena[params.outOff + a * dim + hOff + lid.x] = 0.0;
    }
    return;
  }

  let inv = 1.0 / wgSum[1];
  // phase 3: first headDim invocations own one output dim each and scan all
  // keys (weights already in scores[] from phase 2). Requires headDim <= 256
  // (workgroup size); a larger head would leave dims >= 256 unwritten (stale
  // garbage) — the LFM2.5-VL tower has headDim = 72.
  if (lid.x < headDim) {
    let d = lid.x;
    var acc = 0.0;
    for (var kk = 0u; kk < n; kk++) {
      acc += scores[kk] * arena[params.inOff + (2u * n + kk) * dim + hOff + d];
    }
    arena[params.outOff + a * dim + hOff + d] = acc * inv;
  }
}
