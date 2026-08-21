import { describe, expect, test } from "bun:test";
import { BrainSession, type HostRecord } from "./index.ts";
import { BRAIN_FORWARD_CONFIG } from "../forward/model.ts";

const rows = (count: number, offset = 0): Uint32Array =>
  Uint32Array.from({ length: count }, (_, id) => (id + offset) % 64);

const world = (): HostRecord[] => [
  { schemaId: 1, band: 3, tokens: [10, 11] },
  { schemaId: 1, band: 3, tokens: [20, 21] },
  { schemaId: 9, query: true, tokens: [40, 41] },
];

/** A small profile: this exercises the same layout for a fraction of the bytes. */
const small = {
  ...BRAIN_FORWARD_CONFIG,
  hiddenSize: 32,
  headCount: 1,
  headDim: 32,
  ffnSize: 64,
  encoderBlocks: 1,
  mixerBlocks: 1,
};

describe("writing a brain down", () => {
  test("a restored brain answers exactly as the one that was written", () => {
    const trained = new BrainSession({ tokenRows: rows(4096), seed: 7, config: small });
    // Move it off its seed, so the test is about the checkpoint and not about
    // two sessions built from the same number.
    trained.learn([{ records: world(), chosen: [0], reward: 0.5 }]);
    const before = trained.think(world()).selections[0]!;

    const cold = new BrainSession({ tokenRows: rows(4096), seed: 99, config: small });
    expect(cold.restore(trained.snapshot())).toBeNull();
    const after = cold.think(world()).selections[0]!;
    expect(after.record).toBe(before.record);
    expect([...after.distribution]).toEqual([...before.distribution]);
  });

  test("a checkpoint of another vocabulary is refused, not coerced", () => {
    const trained = new BrainSession({ tokenRows: rows(4096), seed: 7, config: small });
    const other = new BrainSession({ tokenRows: rows(4096, 1), seed: 7, config: small });
    const untouched = Float32Array.from(other.weights.selector.wq);
    // The same weights under a different mapping denote something else, with
    // nothing at runtime to signal it. That is the failure worth refusing.
    expect(other.restore(trained.snapshot())).toBe("a different vocabulary");
    expect(other.weights.selector.wq).toEqual(untouched);
  });

  test("a checkpoint of another geometry is refused", () => {
    const trained = new BrainSession({ tokenRows: rows(4096), seed: 7, config: small });
    const wider = new BrainSession({
      tokenRows: rows(4096),
      seed: 7,
      config: { ...small, hiddenSize: 64, headDim: 64 },
    });
    expect(wider.restore(trained.snapshot())).toBe("a different geometry");
  });

  test("something that is not a checkpoint at all says so", () => {
    const session = new BrainSession({ tokenRows: rows(4096), seed: 7, config: small });
    expect(session.restore(new Uint8Array(64))).toBe("not a krystal checkpoint");
    expect(session.restore(new Uint8Array(3))).toBe("truncated");
  });

  test("a checkpoint read back from an unaligned buffer still loads", () => {
    const trained = new BrainSession({ tokenRows: rows(4096), seed: 7, config: small });
    const snapshot = trained.snapshot();
    // What a file read or a network buffer hands over: the bytes are there, but
    // they do not start on a word boundary.
    const shifted = new Uint8Array(snapshot.length + 1);
    shifted.set(snapshot, 1);
    const cold = new BrainSession({ tokenRows: rows(4096), seed: 99, config: small });
    expect(cold.restore(shifted.subarray(1))).toBeNull();
  });
});
