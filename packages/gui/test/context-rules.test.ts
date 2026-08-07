// Locks in the rule behind a real regression: two blocks each carrying BOS
// compose into a context the model silently ignores.
import { expect, test } from "bun:test";
import { contextIssues, type ContextPart } from "../src/engine/context-rules.ts";

const withBos = (what: string): ContextPart => ({ what, bosLeading: true, bosInterior: 0 });
const noBos = (what: string): ContextPart => ({ what, bosLeading: false, bosInterior: 0 });

test("a well-formed context reports nothing", () => {
  expect(contextIssues({ parts: [withBos("b#0"), noBos("b#1")], tokens: 32 })).toEqual([]);
});

test("BOS on a non-first block is flagged", () => {
  const issues = contextIssues({ parts: [withBos("b#0"), withBos("b#1")], tokens: 32 });
  expect(issues).toHaveLength(1);
  expect(issues[0]).toContain("b#1");
  expect(issues[0]).toContain("resets the context");
});

test("a context that never opens with BOS is flagged", () => {
  const issues = contextIssues({ parts: [noBos("b#0"), noBos("b#1")], tokens: 32 });
  expect(issues).toHaveLength(1);
  expect(issues[0]).toContain("does not start with BOS");
});

test("BOS buried inside a block is flagged", () => {
  const issues = contextIssues({
    parts: [{ what: "b#0", bosLeading: true, bosInterior: 2 }],
    tokens: 32,
  });
  expect(issues).toHaveLength(1);
  expect(issues[0]).toContain("2x");
});

test("checkpoints participate in the same rule", () => {
  const issues = contextIssues({
    parts: [{ what: "ckpt#1", bosLeading: true, bosInterior: 0 }, withBos("b#4")],
    tokens: 64,
  });
  expect(issues).toHaveLength(1);
  expect(issues[0]).toContain("b#4");
});

test("over-capacity is reported alongside shape problems", () => {
  const issues = contextIssues({ parts: [withBos("b#0")], tokens: 2048, capacity: 1024 });
  expect(issues).toHaveLength(1);
  expect(issues[0]).toContain("capacity is 1024");
});

test("an empty context is not an error", () => {
  expect(contextIssues({ parts: [], tokens: 0, capacity: 1024 })).toEqual([]);
});
