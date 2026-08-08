// Seeded top-k / temperature sampling support for the argmax kernel.
//
// Sampling is Gumbel-max: adding an independent Gumbel(0,1) variate to every
// scaled logit and taking the argmax draws exactly from softmax(logits / T).
// That is the whole reason this fits the existing kernel — no softmax, no
// running sum, no prefix scan, no second dispatch. The reduction that already
// finds the maximum is the sampler.
//
// The randomness is stateless: the variate for one candidate is a pure hash of
// (seed, decode step, token id). There is no RNG state buffer, so checkpoints,
// restores and cancelled generations need no extra snapshot, and resuming from
// a checkpoint reproduces the same stream. Same seed + same context + same
// options => same tokens, on any device.

// Must match SAMPLE_TOP_K_MAX in lfm2-layout.ts, which sizes the arena scratch.
const SAMPLE_TOP_K_MAX: u32 = 64u;

// Winning candidates of the tournament, in descending logit order.
var<workgroup> sampleTopLogit: array<f32, SAMPLE_TOP_K_MAX>;
var<workgroup> sampleTopToken: array<u32, SAMPLE_TOP_K_MAX>;

// murmur3-style 32-bit finalizer. Cheap, and good enough that neighbouring
// (step, token) pairs decorrelate — a weaker mixer shows up as visible token
// affinity across decode steps.
fn sample_hash(value: u32) -> u32 {
  var h = value;
  h ^= h >> 16u;
  h *= 0x7feb352du;
  h ^= h >> 15u;
  h *= 0x846ca68bu;
  h ^= h >> 16u;
  return h;
}

/**
 * Uniform variate in the OPEN interval (0, 1).
 *
 * Both ends must be excluded: log(0) is -inf and log(1) is 0, either of which
 * turns the Gumbel variate below into inf/NaN and poisons the comparison. 24
 * bits plus the half-step offset keeps every result exactly representable.
 */
fn sample_unit(seed: u32, step: u32, token: u32) -> f32 {
  let mixed = sample_hash(seed ^ sample_hash(step ^ sample_hash(token * 0x9e3779b9u)));
  return (f32(mixed >> 8u) + 0.5) * (1.0 / 16777216.0);
}

fn sample_gumbel(seed: u32, step: u32, token: u32) -> f32 {
  return -log(-log(sample_unit(seed, step, token)));
}

// Per-lane candidate lists in the arena, laid out slot-major so that the 256
// lanes of one slot are contiguous and their loads coalesce.
fn sample_slot_index(base: u32, slot: u32, lane: u32) -> u32 {
  return base + slot * ARGMAX_WG + lane;
}

fn sample_token_index(base: u32, slot: u32, lane: u32) -> u32 {
  return base + (SAMPLE_TOP_K_MAX + slot) * ARGMAX_WG + lane;
}

/**
 * Total order on candidates: higher logit first, lower token id breaks ties.
 * `emptyToken` marks an unfilled slot and loses to everything, which is what
 * makes an empty list sort correctly without a separate length counter.
 */
fn sample_outranks(
  logit: f32,
  token: u32,
  otherLogit: f32,
  otherToken: u32,
  emptyToken: u32,
) -> bool {
  if (otherToken == emptyToken) { return true; }
  if (token == emptyToken) { return false; }
  return logit > otherLogit || (logit == otherLogit && token < otherToken);
}
