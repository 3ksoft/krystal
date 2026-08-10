// Compare our base pos-emb grid against llama.cpp's 16x16 pos_embed
// (within-dim differences cancel the constant patch term).
import { NodeFileSource } from "../packages/quant/src/gguf/source-node.ts";
import { GgufReader } from "../packages/quant/src/gguf/reader.ts";
import { parseVisionConfig } from "../packages/quant/src/vision/config.ts";
import { loadVisionWeights, visionTensorSource } from "../packages/quant/src/vision/weights.ts";
import { readFileSync } from "node:fs";

const gt = JSON.parse(readFileSync("/tmp/mtmd-pos256.json", "utf8")).pos_embed256;
const source = await NodeFileSource.open("models/mmproj-LFM2.5-VL-1.6b-F16.gguf");
const reader = await GgufReader.open(source);
const config = parseVisionConfig((key) => reader.metadata(key));
const weights = await loadVisionWeights(visionTensorSource(reader), config);
const base = weights.posEmb; // raw GGUF layout [position, dim] (no transpose)
const dims = [0, 1, 2, 1151];
const tokens = [0, 1, 2, 253, 254, 255];
console.log("gt pos_embed dim0 t0..2 diff vs t0:", gt.rows[0].slice(1, 3).map((v: number, i: number) => (v - gt.rows[0][0]).toFixed(4)).join(", "));
// Also test the raw flat layout: [d*256 + p] read as if it were [t, d]
const raw = await (async () => {
  const src = await NodeFileSource.open("models/mmproj-LFM2.5-VL-1.6b-F16.gguf");
  const r = await GgufReader.open(src);
  const info = r.tensor("v.position_embd.weight");
  const bytes = await r.readTensor(info);
  const f32 = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  await src.close();
  return f32;
})();

const variants: [string, (t: number, d: number) => number][] = [
  ["transposed [t,d]", (t, d) => base[t * 1152 + d]!],
  ["raw flat as [t,d]", (t, d) => raw[t * 1152 + d]!],
  ["raw as [d,t] index-inverted", (t, d) => raw[d * 256 + t]!],
];
for (const [label, get] of variants) {
  console.log(`--- ${label} ---`);
  for (const d of dims) {
    const gRow = gt.rows[dims.indexOf(d)]!;
    const gDiffs = tokens.map((t) => gRow[tokens.indexOf(t)]! - gRow[0]!);
    const oDiffs = tokens.map((t) => get(t, d) - get(tokens[0]!, d));
    const maxDiff = Math.max(...gDiffs.map((g, i) => Math.abs(g - oDiffs[i]!)));
    console.log(`dim${d}: maxDiff ${maxDiff.toExponential(3)}  gt: [${gDiffs.map((v) => v.toFixed(3)).join(", ")}]  ours: [${oDiffs.map((v) => v.toFixed(3)).join(", ")}]`);
  }
}
process.exit(0);
