// The GPU guide's empty-state sentinel must never collide with a real token.
//
// This used to read the LFM2 tokenizer to prove 0xffff was unoccupied, and
// then rested on the Krystal token space being 12-bit. At 16 bits the two
// spaces meet, so the invariant is no longer free: the reference half stops at
// 0xfffe and KRYSTAL_ABI.reservedEmptyToken holds 0xffff back deliberately.
import { expect, test } from "bun:test";
import { KRYSTAL_ABI, KRYSTAL_TOKEN_RANGES } from "../../packages/schema/src/krystal-engine-schema.ts";
import { GPU_SCHEMA_LIMITS, GPU_SCHEMA_SENTINELS } from "../../packages/schema/src/sparse.ts";

test("0xffff is reserved for GPU guide empty state", () => {
  const empty = GPU_SCHEMA_SENTINELS.emptyToken;
  expect(empty).toBe(0xffff);

  // The Krystal token space is 16-bit and now reaches up to the sentinel, so
  // the invariant is no longer free: the reference half stops one short at
  // 0xfffe (KRYSTAL_ABI.reservedEmptyToken) precisely so no token id can
  // collide with the empty state. Assert the invariant, not the old ceiling.
  const maxTokenId = Math.max(...Object.values(KRYSTAL_TOKEN_RANGES).map(([, hi]) => hi));
  expect(maxTokenId).toBeLessThan(empty);
  expect(maxTokenId).toBe(KRYSTAL_ABI.reservedEmptyToken - 1);
  expect(maxTokenId).toBeLessThan(empty);

  // maxNodes is a count: valid indices stop at 0xfffe, keeping 0xffff free.
  expect(GPU_SCHEMA_LIMITS.maxNodes - 1).toBe(GPU_SCHEMA_SENTINELS.noneNode - 1);
});
