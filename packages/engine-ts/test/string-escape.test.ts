/**
 * The string VM no longer accepts `\uXXXX` (NO_UNICODE_ESCAPE in structured.ts).
 *
 * Two things are under test and they pull in opposite directions: the escape
 * has to be genuinely rejected, and nothing that was expressible may have
 * become unexpressible. The second is the reason the change is cheap — the body
 * phase takes any byte >= 0x20, so every character that `\uXXXX` could name is
 * still reachable as raw UTF-8.
 */
import { expect, test } from "bun:test";
import { type } from "arktype";
import {
  compileStructuredGeneration,
  createGpuConstraintDecoderState,
  feedGpuConstraintBytes,
  linkGpuConstraintProgram,
} from "../src";
import { compileJsonSchemaProgram } from "../src/json-schema-constraint.ts";

const utf8 = new TextEncoder();

function vm(schema: Parameters<typeof compileStructuredGeneration>[0]) {
  const program = linkGpuConstraintProgram(compileJsonSchemaProgram(schema.toJsonSchema()));
  return { program, state: createGpuConstraintDecoderState(program) };
}

/** Feed a payload byte by byte; returns how many bytes were accepted. */
function accepted(schema: Parameters<typeof compileStructuredGeneration>[0], text: string): number {
  const { program, state } = vm(schema);
  const bytes = utf8.encode(text);
  let count = 0;
  for (const byte of bytes) {
    if (!feedGpuConstraintBytes(program, state, new Uint8Array([byte]))) break;
    count++;
  }
  return count;
}

const STRING = type("string < 16");

test("a unicode escape is rejected at the u", () => {
  // `"` and `\` are consumed, then `u` has no continuation.
  expect(accepted(STRING, `"\\u0041"`)).toBe(2);
});

test("short escapes still work", () => {
  expect(accepted(STRING, `"a\\nb"`)).toBe(6);
  expect(accepted(STRING, `"say \\"hi\\""`)).toBe(12);
});

test("every character a unicode escape could name is still reachable raw", () => {
  // é is gone; é is not. Same string, spelling the VM already accepted.
  expect(accepted(STRING, `"é"`)).toBe(4); // 2 UTF-8 bytes + both quotes
  expect(accepted(STRING, `"日本"`)).toBe(8); // 3 bytes each
});

test("the decode budget drops to 2 bytes per length unit", () => {
  // string < 16 means maxLength 15: two quotes + 15 * 2.
  expect(compileStructuredGeneration(STRING).maxJsonBytes).toBe(2 + 15 * 2);
});

/**
 * What a string field costs inside a real schema.
 *
 * Asserted as the *difference* one field makes, not as a total: the total also
 * carries the number bound (maxNumberChars), which is tuned independently, and
 * pinning it here would make this test fail for a reason it does not own.
 */
test("a string field costs 2 bytes per length unit wherever it appears", () => {
  const rows = (extra: boolean) =>
    compileStructuredGeneration(
      (extra
        ? type({ id: "number", name: "string < 24" })
        : type({ id: "number" })).array().atLeastLength(1).atMostLength(10) as never,
    ).maxJsonBytes;

  // 10 rows, each gaining `"name":` plus the string itself. The string's own
  // share is 2 quotes + 23 * 2; the rest is the key literal and a separator.
  const perRow = (rows(true) - rows(false)) / 10;
  const stringShare = 2 + 23 * 2;
  expect(perRow).toBeGreaterThanOrEqual(stringShare);
  expect(perRow - stringShare).toBeLessThanOrEqual(10); // key + comma overhead
});
