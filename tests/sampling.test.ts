// Seeded sampling has to be reproducible, and greedy has to stay greedy.
//
// The sampler shares one kernel with argmax, so both properties are load-bearing
// and neither shows up as an error when it breaks: a seed that does not fully
// determine the draw degrades into "sometimes different output", and a greedy
// path that quietly picks up noise looks like a worse model, not like a bug.
//
// The invariants below pin the RNG contract without asserting a distribution:
//   - a seed reproduces its token stream exactly,
//   - different seeds do diverge (otherwise the seed is being ignored),
//   - topK = 1 is greedy at any temperature, because a one-candidate Gumbel-max
//     has nothing to choose between,
//   - temperature = 0 is greedy at any topK.
import { expect, test } from "bun:test";
import { loadModel } from "../src";
import { SAMPLE_TOP_K_MAX } from "../packages/webgpu/src/lfm2.ts";

const MODEL = "./models/LFM2.5-1.2B-Instruct-WQ4.wq4";
const PROMPT = new Uint32Array([1, 2, 3, 4]);
const MAX_NEW_TOKENS = 24;

test("seeded top-k sampling is reproducible and greedy stays greedy", async () => {
  const model = await loadModel(MODEL);
  const forward = model.forward!;
  try {
    await forward.prepareAll();
    const generate = async (sampling?: { temperature: number; topK: number; seed: number }) =>
      (await forward.generateGreedy(PROMPT, {
        maxNewTokens: MAX_NEW_TOKENS,
        resetState: true,
        sampling,
      })).tokens;

    const greedy = await generate();
    expect(greedy.length).toBeGreaterThan(1);

    // Same seed, twice: identical. This is the whole point of hashing
    // (seed, step, token) instead of carrying RNG state across dispatches.
    const first = await generate({ temperature: 0.9, topK: 40, seed: 12345 });
    const second = await generate({ temperature: 0.9, topK: 40, seed: 12345 });
    expect(second).toEqual(first);

    // Different seeds must actually move the draw. One seed could coincide with
    // another for a short continuation, so this fails only if none of them do.
    // Sequentially: the runtime owns one set of readback buffers, so overlapping
    // generations collide on them rather than running in parallel.
    const others: number[][] = [];
    for (const seed of [7, 99, 4242]) {
      others.push(await generate({ temperature: 0.9, topK: 40, seed }));
    }
    expect(others.some((tokens) => JSON.stringify(tokens) !== JSON.stringify(first))).toBe(true);

    // A near-zero temperature collapses onto the peak, which is the one check
    // here that the k candidates are ordered and scaled correctly rather than
    // merely stable: a scrambled list or an inverted comparison still gives
    // reproducible tokens, just not these. Flipping this needs two logits
    // within ~0.02 nats, which the tie-break on token id already decides.
    expect(await generate({ temperature: 0.02, topK: 40, seed: 7 })).toEqual(greedy);
    // topK = 1: exactly one candidate, so the Gumbel variate cannot change it.
    expect(await generate({ temperature: 2.5, topK: 1, seed: 999 })).toEqual(greedy);
    // temperature = 0: the sampler is switched off, whatever topK says.
    expect(await generate({ temperature: 0, topK: SAMPLE_TOP_K_MAX, seed: 999 })).toEqual(greedy);
  } finally {
    await model.dispose();
  }
}, 180_000);

test("sampling from a checkpoint reproduces the same tokens", async () => {
  const model = await loadModel(MODEL);
  const forward = model.forward!;
  try {
    await forward.prepareAll();
    const checkpoint = await forward.createCheckpoint(PROMPT);
    try {
      const sampling = { temperature: 0.9, topK: 32, seed: 2026 };
      // The RNG is keyed on the decode step, which restarts at 0 for every
      // generation — so a restored checkpoint replays its stream exactly, and
      // branching off one checkpoint twice is reproducible per branch.
      const first = await forward.generateGreedyFromCheckpoint(checkpoint, [], {
        maxNewTokens: MAX_NEW_TOKENS,
        sampling,
      });
      const second = await forward.generateGreedyFromCheckpoint(checkpoint, [], {
        maxNewTokens: MAX_NEW_TOKENS,
        sampling,
      });
      expect(second.tokens).toEqual(first.tokens);
    } finally {
      checkpoint.destroy();
    }
  } finally {
    await model.dispose();
  }
}, 180_000);

test("out-of-range sampling options are rejected before reaching the GPU", async () => {
  const model = await loadModel(MODEL);
  const forward = model.forward!;
  try {
    const reject = (sampling: { temperature: number; topK: number; seed: number }) =>
      expect(forward.generateGreedy(PROMPT, { maxNewTokens: 2, sampling })).rejects.toThrow(RangeError);

    await reject({ temperature: 1, topK: 0, seed: 0 });
    await reject({ temperature: 1, topK: SAMPLE_TOP_K_MAX + 1, seed: 0 });
    await reject({ temperature: 1, topK: 8, seed: -1 });
    await reject({ temperature: 1, topK: 8, seed: 2 ** 32 });
    await reject({ temperature: -0.5, topK: 8, seed: 0 });
    await reject({ temperature: Number.NaN, topK: 8, seed: 0 });
  } finally {
    await model.dispose();
  }
}, 120_000);
