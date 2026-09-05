import { describe, expect, test } from "bun:test";
import { sampleRow } from "./sampling.ts";

describe("drawing one entry from a row", () => {
  test("lands where the cumulative mass passes the draw", () => {
    const p = Float32Array.from([0.2, 0.5, 0.3]);
    expect(sampleRow(p, 0, 3, 0.1)).toBe(0);
    expect(sampleRow(p, 0, 3, 0.6)).toBe(1);
    expect(sampleRow(p, 0, 3, 0.95)).toBe(2);
  });

  test("a row that adds up to slightly less than the draw falls back to its last live entry", () => {
    const p = Float32Array.from([0.5, 0.49, 0]);
    expect(sampleRow(p, 0, 3, 0.999)).toBe(1);
  });

  test("a row of numbers that are not numbers chooses nothing, not its last entry", () => {
    // Measured in a world: NaN weights made every frame answer with its last
    // record, which was the one act nothing may do — and it looked decided.
    const p = Float32Array.from([Number.NaN, Number.NaN, Number.NaN]);
    expect(sampleRow(p, 0, 3, 0.5)).toBe(-1);
    const mixed = Float32Array.from([Number.NaN, 0.4, Number.NaN]);
    expect(sampleRow(mixed, 0, 3, 0.9)).toBe(1);
  });
});
