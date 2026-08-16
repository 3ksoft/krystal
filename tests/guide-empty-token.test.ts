// The GPU guide's empty-state sentinel must never collide with a real token.
//
// This used to read the LFM2 tokenizer to prove 0xffff was unoccupied. The
// Krystal ABI replaces the LFM2 tokenizer with a fixed 12-bit token space
// (KRYSTAL_TOKEN_RANGES ends at 0xfff), so the invariant is now structural:
// every legal token id stays below 0xffff, and the schema reserves 0xffff as
// EMPTY_TOKEN / NONE_NODE for fixed-size GPU tables.
import { expect, test } from "bun:test";
import { KRYSTAL_TOKEN_RANGES } from "../packages/schema/src/krystal-engine-schema.ts";
import { GPU_SCHEMA_LIMITS, GPU_SCHEMA_SENTINELS } from "../packages/schema/src/sparse";

test("0xffff is reserved for GPU guide empty state", () => {
  const empty = GPU_SCHEMA_SENTINELS.emptyToken;
  expect(empty).toBe(0xffff);

  // The entire Krystal token space is 12-bit (0x000..0xfff), so no token id can
  // ever reach 0xffff and collide with the empty-state sentinel.
  const maxTokenId = Math.max(...Object.values(KRYSTAL_TOKEN_RANGES).map(([, hi]) => hi));
  expect(maxTokenId).toBe(0xfff);
  expect(maxTokenId).toBeLessThan(empty);

  // maxNodes is a count: valid indices stop at 0xfffe, keeping 0xffff free.
  expect(GPU_SCHEMA_LIMITS.maxNodes - 1).toBe(GPU_SCHEMA_SENTINELS.noneNode - 1);
});
