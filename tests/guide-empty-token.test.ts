import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { GPU_SCHEMA_LIMITS, GPU_SCHEMA_SENTINELS } from "../packages/schema/src/sparse";

test("0xffff is reserved for GPU guide empty state", () => {
  const tokenizer = JSON.parse(
    readFileSync(new URL("../packages/finetune/tokenizer.json", import.meta.url), "utf8"),
  ) as {
    model?: { vocab?: Record<string, number> };
    added_tokens?: Array<{ id?: number; content?: string }>;
  };

  const occupied = new Map<number, string>();
  for (const [token, id] of Object.entries(tokenizer.model?.vocab ?? {})) occupied.set(id, token);
  for (const token of tokenizer.added_tokens ?? []) {
    if (token.id !== undefined) occupied.set(token.id, token.content ?? "<added-token>");
  }

  const empty = GPU_SCHEMA_SENTINELS.emptyToken;
  expect(empty).toBe(0xffff);
  expect(occupied.get(empty)).toBeUndefined();

  // maxNodes is a count: valid indices stop at 0xfffe, keeping 0xffff free.
  expect(GPU_SCHEMA_LIMITS.maxNodes - 1).toBe(GPU_SCHEMA_SENTINELS.noneNode - 1);
});
