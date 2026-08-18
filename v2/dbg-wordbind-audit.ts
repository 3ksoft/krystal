// Anti-shortcut audit (docs/word_attention_bias.md). The pair construction
// makes this an exact proof rather than a statistical bound: both variants of
// a semantic pair are byte-identical in every channel the model reads except
// the word-membership sidecar, so no function of the token tape can separate
// the labels.
import { wordBindPair, wordBindPairs } from "./tests/wordbind-fixture.ts";
import { packBrainFrame } from "../packages/krystal/src/frame/packer.ts";

const seeds = Array.from({ length: 512 }, (_, i) => i);
let identical = 0;
let wordDiffers = 0;
const labels = { EAT: 0, LOOK: 0 };

for (const seed of seeds) {
  const [a, b] = wordBindPair(seed);
  labels[a.gold.action]++;
  labels[b.gold.action]++;
  const fa = packBrainFrame(a.frame).frame;
  const fb = packBrainFrame(b.frame).frame;
  const same =
    fa.tokenIds.join(",") === fb.tokenIds.join(",") &&
    fa.fieldRoles.join(",") === fb.fieldRoles.join(",") &&
    fa.schemaIds.join(",") === fb.schemaIds.join(",") &&
    fa.bandIds.join(",") === fb.bandIds.join(",") &&
    fa.recordFlags.join(",") === fb.recordFlags.join(",") &&
    fa.runtimeRefs.join(",") === fb.runtimeRefs.join(",") &&
    fa.activeRecordIndices.join(",") === fb.activeRecordIndices.join(",");
  if (same) identical++;
  if (JSON.stringify(a.wordIds) !== JSON.stringify(b.wordIds)) wordDiffers++;
  if (a.gold.action === b.gold.action) {
    console.log(`FAIL seed=${seed}: para nie jest kontrfaktyczna (${a.gold.action})`);
    process.exit(1);
  }
}

const pairs = seeds.length;
const ok1 = identical === pairs;
const ok2 = wordDiffers === pairs;
const ok3 = labels.EAT === pairs && labels.LOOK === pairs;

console.log(`par: ${pairs}`);
console.log(`${ok1 ? "OK  " : "FAIL"}  ramki w parze identyczne we WSZYSTKICH kanałach GPU  ${identical}/${pairs}`);
console.log(`${ok2 ? "OK  " : "FAIL"}  sidecar przynależności słów się różni              ${wordDiffers}/${pairs}`);
console.log(`${ok3 ? "OK  " : "FAIL"}  etykiety zbalansowane                              EAT=${labels.EAT} LOOK=${labels.LOOK}`);

// Word-id VALUES must not carry the label either: check the id attached to RED
// is not predictive (it is drawn per pair, shared by both variants).
const flat = wordBindPairs(seeds);
const byId = new Map<number, { eat: number; n: number }>();
for (const e of flat) {
  const id = Math.min(...Object.values(e.wordIds));
  const b = byId.get(id) ?? { eat: 0, n: 0 };
  b.n++; if (e.gold.action === "EAT") b.eat++;
  byId.set(id, b);
}
let worst = 0;
for (const [, b] of byId) worst = Math.max(worst, Math.abs(b.eat / b.n - 0.5));
const ok4 = worst === 0;
console.log(`${ok4 ? "OK  " : "FAIL"}  wartości word ID nieprzewidujące                  maxDev=${worst.toFixed(4)}`);

const failed = [ok1, ok2, ok3, ok4].filter((v) => !v).length;
console.log(failed === 0
  ? "\nAudyt zdany konstrukcyjnie: tasma tokenów jest identyczna, więc żaden płaski model nie może przekroczyć 50%."
  : `\n${failed} niezdanych`);
process.exit(failed === 0 ? 0 : 1);
