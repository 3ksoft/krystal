#!/usr/bin/env bun
// Verify the v.position_embd.weight GGUF layout + resize against llama.cpp
// ground truth (mtmd-pos256.json = base grid pos_embed node for red 256 image;
// mtmd-nodes.json pos_embed = pos_embed node for red 512 image).
//
// The mtmd-debug print format: rows are TOKENS (i1), columns are dims (i0),
// first 3 + last 3 of each. pos_embed node = patch_embedding + position
// embedding; for a uniform red image the patch term is constant per token,
// so token diffs vs token 0 isolate the position embedding.
//
//   bun run misc/vl-poslayout-check.ts
import { readFileSync } from "node:fs";
import { NodeFileSource } from "../packages/quant/src/gguf/source-node.ts";
import { GgufReader } from "../packages/quant/src/gguf/reader.ts";
import { parseVisionConfig } from "../packages/quant/src/vision/config.ts";
import { loadVisionWeights, visionTensorSource } from "../packages/quant/src/vision/weights.ts";
import { resizePositionEmbedding } from "../packages/webgpu/src/vision/processor.ts";

const pos256Gt = JSON.parse(readFileSync("/tmp/mtmd-pos256.json", "utf8")).pos_embed256;
const nodes = JSON.parse(readFileSync("/tmp/mtmd-nodes.json", "utf8"));

const source = await NodeFileSource.open("models/mmproj-LFM2.5-VL-1.6b-F16.gguf");
const reader = await GgufReader.open(source);
const config = parseVisionConfig((key) => reader.metadata(key));
const weights = await loadVisionWeights(visionTensorSource(reader), config);

// Raw tensor dims + flat data (bypasses the loader's layout interpretation).
const info = reader.tensor("v.position_embd.weight")!;
const bytes = await reader.readTensor(info);
const raw = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
console.log(`v.position_embd.weight GGUF dims: [${info.dimensions.join(", ")}]  (${raw.length} f32)`);

const H = config.hiddenSize; // 1152
const N = 16 * 16; // 256

// Candidate layouts: (position, dim) row-major, dim fastest
const basePD = new Float32Array(N * H); // data[p*H + d]
const baseDP = new Float32Array(N * H); // data[d*N + p]  (what weights.ts emits)
for (let p = 0; p < N; p++) {
  for (let d = 0; d < H; d++) {
    basePD[p * H + d] = raw[p * H + d]!;
    baseDP[p * H + d] = raw[d * N + p]!;
  }
}
const dims = [0, 1, 2, H - 3, H - 2, H - 1];
const toks256 = [0, 1, 2, 253, 254, 255];
const toks1024 = [0, 1, 2, 1021, 1022, 1023];

// ---- 1) base grid vs 256-grid GT (diffs vs token 0, per dim) ----
function diffs(base: Float32Array, tokens: number[], dim: number): number[] {
  const t0 = base[tokens[0]! * H + dim]!;
  return tokens.map((t) => base[t * H + dim]! - t0);
}
function gtDiffs(rows: number[][], dimIdx: number): number[] {
  // rows are tokens; row columns are dims [0,1,2,last-2,last-1,last]
  const t0 = rows[0]![dimIdx]!;
  return rows.map((r) => r[dimIdx]! - t0);
}
console.log("\n== base grid vs 256-grid GT (pos_embed node, diffs vs token0) ==");
for (const [label, base] of [["base[p*H+d] (GGUF row-major)", basePD], ["base[d*N+p] (weights.ts)", baseDP]] as const) {
  let worst = 0;
  for (const d of dims) {
    const gt = gtDiffs(pos256Gt.rows, dims.indexOf(d));
    const ours = diffs(base, toks256, d);
    const md = Math.max(...gt.map((g, i) => Math.abs(g - ours[i]!)));
    worst = Math.max(worst, md);
  }
  console.log(`${label}: maxDiff vs GT = ${worst.toExponential(3)}`);
}

// ---- 2) resized 32x32 grid vs 1024-grid GT (diffs vs token 0, per dim) ----
console.log("\n== resized 32x32 grid vs 1024-grid GT (pos_embed node, diffs vs token0) ==");
for (const [label, base] of [["base[p*H+d]", basePD], ["base[d*N+p]", baseDP]] as const) {
  const resized = resizePositionEmbedding(base, 16, 32, 32, H);
  let worst = 0;
  for (const d of dims) {
    const gt = gtDiffs(nodes.pos_embed.rows, dims.indexOf(d));
    const ours = diffs(resized, toks1024, d);
    const md = Math.max(...gt.map((g, i) => Math.abs(g - ours[i]!)));
    worst = Math.max(worst, md);
    if (d <= 2) {
      console.log(
        `  ${label} dim${d}: maxDiff ${md.toExponential(3)}  gt: [${gt.map((v) => v.toFixed(4)).join(", ")}]  ours: [${ours.map((v) => v.toFixed(4)).join(", ")}]`,
      );
    }
  }
  console.log(`${label}: worst dim maxDiff = ${worst.toExponential(3)}`);
}

// ---- 3) what the loader emits today ----
console.log("\n== weights.ts loader output base grid dim0 diffs ==");
const loaderBase = weights.posEmb; // [p*H+d]
console.log("loader base[p*H+d] t1..2 diffs:", (loaderBase[1 * H]! - loaderBase[0]!).toFixed(4), (loaderBase[2 * H]! - loaderBase[0]!).toFixed(4));
console.log("GT base t1..2 diffs:          ", (pos256Gt.rows[1][0]! - pos256Gt.rows[0][0]!).toFixed(4), (pos256Gt.rows[2][0]! - pos256Gt.rows[0][0]!).toFixed(4));
source.close();
process.exit(0);
