#!/usr/bin/env bun
// End-to-end tower check against llama.cpp ground truth (mtmd-debug encode
// path: RAW unnormalized [1,0,0] red input, no mean/std applied — the debug
// tool feeds floats straight to the graph, unlike llama-cli which normalizes).
//
//   bun run misc/vl-fullcheck.ts
//
// Compares (a) the full 256-token output vs /tmp/node869.json (rows = tokens
// 0,1,2,253,254,255, cols = dims 0,1,2,2045,2046,2047 + global sum) and
// (b) stage-0 pos_embed vs /tmp/mtmd-nodes.json with the fixed pos-emb grid.
import { readFileSync } from "node:fs";
import { NodeFileSource } from "../packages/quant/src/gguf/source-node.ts";
import { GgufReader } from "../packages/quant/src/gguf/reader.ts";
import { parseVisionConfig } from "../packages/quant/src/vision/config.ts";
import { loadVisionWeights, visionTensorSource } from "../packages/quant/src/vision/weights.ts";
import { resizePositionEmbedding } from "../packages/webgpu/src/vision/processor.ts";
import { VisionTower } from "../packages/webgpu/src/vision/tower.ts";
import { installDawn } from "../tests/dawn.ts";
import { createWebGpuDevice } from "../packages/webgpu/src/device.ts";

const gtFinal = JSON.parse(readFileSync("/tmp/node869.json", "utf8"));
const nodes = JSON.parse(readFileSync("/tmp/mtmd-nodes.json", "utf8"));

const source = await NodeFileSource.open("models/mmproj-LFM2.5-VL-1.6b-F16.gguf");
const reader = await GgufReader.open(source);
const config = parseVisionConfig((key) => reader.metadata(key));
const weights = await loadVisionWeights(visionTensorSource(reader), config);

// Raw red 512x512 patch grid (32x32 = 1024 patches), (c,h,w) w-fastest:
// R=1.0, G=0, B=0 per pixel — exactly the mtmd-debug "red" input.
const gridH = 32;
const gridW = 32;
const P = gridH * gridW;
const patchDim = 3 * config.patchSize * config.patchSize;
const patches = new Float32Array(config.maxPatches * patchDim);
for (let p = 0; p < P; p++) {
  for (let i = 0; i < 256; i++) patches[p * patchDim + i] = 1.0; // R channel
}
const posEmb = new Float32Array(config.maxPatches * config.hiddenSize);
posEmb.set(resizePositionEmbedding(weights.posEmb, config.positionEmbeddingGrid, gridH, gridW, config.hiddenSize));
const paddingMask = new Uint8Array(config.maxPatches).fill(1, 0, P);
const image = { patches, posEmb, paddingMask, patchCount: P, gridH, gridW, imageW: 512, imageH: 512 };

await installDawn();
const { device } = await createWebGpuDevice({ label: "vl-fullcheck" });
const tower = await VisionTower.create({ device, config, weights });

// ---- stage 0: pos_embed node (patches + pos emb) ----
const h0 = await tower.run(image, { stopAtLayer: 0 });
const dim = config.hiddenSize;
const gtRows = nodes.pos_embed.rows as number[][];
const gtDims = [0, 1, 2, dim - 3, dim - 2, dim - 1];
const gtToks = [0, 1, 2, P - 3, P - 2, P - 1];
let max0 = 0;
for (let t = 0; t < 6; t++) {
  for (let d = 0; d < 6; d++) {
    const g = gtRows[t]![d]!;
    const o = h0[gtToks[t]! * dim + gtDims[d]!]!;
    max0 = Math.max(max0, Math.abs(g - o));
  }
}
console.log(`[stage0 pos_embed] maxAbsDiff = ${max0.toExponential(3)}  (llama.cpp GT vs our raw-input stage 0)`);

// ---- bisect hidden states vs llama.cpp layer_out-N nodes ----
for (const n of [0, 5, 13, 26]) {
  const layer = await tower.run(image, { stopAtLayer: n + 1 });
  const gtn = nodes[`layer_out-${n}`] as { rows: number[][]; dims: number[] };
  let md = 0;
  for (let t = 0; t < 6; t++) {
    for (let d = 0; d < 6; d++) {
      const g = gtn.rows[t]![d]!;
      const o = layer[gtToks[t]! * dim + gtDims[d]!]!;
      md = Math.max(md, Math.abs(g - o));
    }
  }
  console.log(`[layer_out-${n}] maxAbsDiff = ${md.toExponential(3)}`);
}

// ---- full tower output vs node869 (256 tokens x 2048) ----
const emb = await tower.run(image);
const proj = config.projectorHiddenSize;
const rows = gtFinal.rows as number[][];
const fdims = [0, 1, 2, proj - 3, proj - 2, proj - 1];
const ftoks = [0, 1, 2, 253, 254, 255];
let maxF = 0;
for (let t = 0; t < 6; t++) {
  for (let d = 0; d < 6; d++) {
    const g = rows[t]![d]!;
    const o = emb[ftoks[t]! * proj + fdims[d]!]!;
    maxF = Math.max(maxF, Math.abs(g - o));
  }
}
let sum = 0;
for (const v of emb) sum += v;
console.log(`[final embeddings] maxAbsDiff on 36 GT points = ${maxF.toExponential(3)}`);
console.log(`[final embeddings] our sum = ${sum.toFixed(4)}  gt sum = ${gtFinal.sum.toFixed(4)}`);
console.log(`[final embeddings] our token0 dims 0..2: ${emb[0]!.toFixed(4)}, ${emb[1]!.toFixed(4)}, ${emb[2]!.toFixed(4)}  gt: ${rows[0]!.slice(0, 3).map((v) => v.toFixed(4)).join(", ")}`);

tower.destroy();
source.close();
process.exit(0);
