import { expect, test } from "bun:test";
import {
  aliasFor,
  buildReservedTable,
  collapseReserved,
  expandReserved,
} from "../src/engine/reserved.ts";

// Mirrors the real LFM2.5 layout: reserved numbering starts above 1, named
// tokens interrupt the id run, and — the trap this fixture exists for — some
// entries spelled `<|reserved_N|>` are ordinary NORMAL tokens the tokenizer
// will not resolve, so they must not be addressable.
const VOCAB = [
  "<|pad|>",            // 0  special
  "<|startoftext|>",    // 1  special
  "<|reserved_4|>",     // 2  NOT special — spelled like one, tokenizes as text
  "<|reserved_5|>",     // 3  NOT special
  "<|reserved_6|>",     // 4  special -> [-token-1-]
  "<|audio_start|>",    // 5  special, named (gap in the reserved run)
  "<|reserved_7|>",     // 6  special -> [-token-2-]
  "hello",              // 7  ordinary
];
const NOT_SPECIAL = new Set([2, 3, 7]);
const isSpecial = (id: number) => !NOT_SPECIAL.has(id);

const table = buildReservedTable(VOCAB, isSpecial);

test("only tokenizer-resolvable reserved entries are addressable", () => {
  expect(table.literals).toEqual(["<|reserved_6|>", "<|reserved_7|>"]);
  expect(table.ids).toEqual([4, 6]);
});

test("reserved-looking NORMAL tokens are excluded", () => {
  expect(table.literals).not.toContain("<|reserved_4|>");
  expect(table.ids).not.toContain(2);
});

test("aliases expand to their literal", () => {
  const out = expandReserved("a [-token-1-] b [-token-2-]", table);
  expect(out.text).toBe("a <|reserved_6|> b <|reserved_7|>");
  expect(out.used).toEqual([1, 2]);
  expect(out.unknown).toEqual([]);
});

test("out-of-range aliases are reported and left verbatim", () => {
  const out = expandReserved("x [-token-99-] y", table);
  expect(out.text).toBe("x [-token-99-] y");
  expect(out.unknown).toEqual([99]);
  expect(out.used).toEqual([]);
});

test("index 0 is not a valid alias", () => {
  const out = expandReserved("[-token-0-]", table);
  expect(out.text).toBe("[-token-0-]");
  expect(out.unknown).toEqual([0]);
});

test("text with no aliases is untouched", () => {
  const out = expandReserved("plain text <|im_start|>", table);
  expect(out.text).toBe("plain text <|im_start|>");
  expect(out.used).toEqual([]);
});

test("collapse is the inverse of expand", () => {
  const source = "a [-token-1-] b [-token-2-]";
  expect(collapseReserved(expandReserved(source, table).text, table)).toBe(source);
});

test("collapse leaves reserved literals the model does not expose", () => {
  expect(collapseReserved("<|reserved_999|>", table)).toBe("<|reserved_999|>");
});

test("aliasFor maps a token id back to its alias", () => {
  expect(aliasFor(table, 6)).toBe("[-token-2-]");
  // Ordinary and named-special ids have no alias.
  expect(aliasFor(table, 7)).toBeNull();
  expect(aliasFor(table, 5)).toBeNull();
});

test("an empty vocabulary yields an empty table", () => {
  const empty = buildReservedTable([], () => true);
  expect(empty.literals).toEqual([]);
  expect(expandReserved("[-token-1-]", empty).unknown).toEqual([1]);
});
