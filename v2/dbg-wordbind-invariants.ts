// Front-loaded conformance for the same-word attention bias
// (docs/word_attention_bias.md, "Required parity and regression coverage").
// Pure CPU on the mask compiler — these must hold before any training runs,
// because every downstream profile is uninterpretable if they do not.
import { compileRecordMask, INVALID_WORD_ID } from "../packages/krystal/src/forward/masks.ts";

const W = 8; // recordWidth
// Two records, four tokens each: frame-token indices 0..3 and 8..11.
const activeTokens = Uint32Array.from([0, 1, 2, 3, 8, 9, 10, 11]);

// Record 0: tokens {0,1} are word A, {2,3} are word B. Record 1 reuses the
// SAME local ids, which must not bind across the record boundary.
const wordIds: Record<number, number> = {
  0: 7, 1: 7, 2: 9, 3: 9,
  8: 7, 9: 7, 10: 9, 11: 9,
};
const ALPHA = 4.0;

const eq = (a: Float32Array, b: Float32Array) =>
  a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
const results: [string, boolean][] = [];

// 1. alpha = 0 is bit-identical to the unbiased mask.
const plain = compileRecordMask(activeTokens).mask;
results.push(["alpha=0 identyczna bit w bit z maską bez biasu",
  eq(plain, compileRecordMask(activeTokens, { wordIds, alpha: 0 }).mask)]);

// 2. The bias actually fires, and only inside a record.
const biased = compileRecordMask(activeTokens, { wordIds, alpha: ALPHA }).mask;
const at = (i: number, j: number) => biased[i * activeTokens.length + j]!;
results.push(["ten sam rekord + to samo słowo → +alpha", at(0, 1) === ALPHA && at(2, 3) === ALPHA]);
results.push(["ten sam rekord + inne słowo → 0", at(0, 2) === 0 && at(1, 3) === 0]);
// f32 storage rounds the -1e30 sentinel, so compare by magnitude, not equality.
const blocked = (i: number, j: number) => at(i, j) < -1e29;
results.push(["inny rekord → zablokowany (bez zmian)", blocked(0, 4) && blocked(3, 7)]);

// 3. Equal local ids in DIFFERENT records must not bind (cross-record leak).
results.push(["te same lokalne id w innych rekordach nie wiążą",
  blocked(0, 4) && blocked(1, 5) && blocked(2, 6)]);

// 4. INVALID word id receives no bias (KEY/VALUE/query control slots).
const withInvalid = { ...wordIds, 1: INVALID_WORD_ID };
const inv = compileRecordMask(activeTokens, { wordIds: withInvalid, alpha: ALPHA }).mask;
results.push(["wordId=INVALID nie dostaje biasu",
  inv[0 * 8 + 1] === 0 && inv[1 * 8 + 0] === 0]);

// 5. P3: an arbitrary bijection of local word ids leaves the mask IDENTICAL.
// Only equality is consulted, so renumbering cannot reach the model at all.
const biject: Record<number, number> = {};
for (const [tok, id] of Object.entries(wordIds)) biject[Number(tok)] = id === 7 ? 31 : 4;
results.push(["P3: przenumerowanie word ID nie zmienia maski",
  eq(biased, compileRecordMask(activeTokens, { wordIds: biject, alpha: ALPHA }).mask)]);

// 6. Permuting which physical positions carry a word must change the mask —
// otherwise the bias would be indistinguishable from plain adjacency.
const swapped = { ...wordIds, 1: 9, 2: 7 };
results.push(["inne przypisanie tokenów do słów daje inną maskę",
  !eq(biased, compileRecordMask(activeTokens, { wordIds: swapped, alpha: ALPHA }).mask)]);

let failed = 0;
for (const [name, ok] of results) {
  if (!ok) failed++;
  console.log(`${ok ? "OK  " : "FAIL"}  ${name}`);
}
console.log(failed === 0 ? "\nwszystkie niezmienniki spełnione" : `\n${failed} niespełnione`);
process.exit(failed === 0 ? 0 : 1);
