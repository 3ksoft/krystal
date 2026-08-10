#!/usr/bin/env bun
// Bisect the tower tail against fresh llama.cpp ground truth dumped from
// llama-mtmd-debug (encode mode, raw red 512): norm_b (post-LN), pixel_shuffle,
// ffn_up (mm1+bias pre-GELU) and ffn_out (post-GELU mm2 output).
//
//   bun run misc/vl-tail-check.ts
import { readFileSync } from "node:fs";
import { NodeFileSource } from "../packages/quant/src/gguf/source-node.ts";
import { GgufReader } from "../packages/quant/src/gguf/reader.ts";
import { parseVisionConfig } from "../packages/quant/src/vision/config.ts";
import { loadVisionWeights, visionTensorSource } from "../packages/quant/src/vision/weights.ts";
import { resizePositionEmbedding } from "../packages/webgpu/src/vision/processor.ts";
import { layerNormInto, geluErf } from "../packages/quant/src/vision/reference.ts";
import { VisionTower } from "../packages/webgpu/src/vision/tower.ts";
import { installDawn } from "../tests/dawn.ts";
import { createWebGpuDevice } from "../packages/webgpu/src/device.ts";

/** Parse a dumped node block: rows are tokens (0,1,2,last-2..), cols are dims. */
function parseNode(name: string): { rows: number[][]; sum: number } {
  const log = readFileSync("/tmp/mtmd-full.log", "utf8");
  const lines = log.split("\n");
  const start = lines.findIndex((l) => l.includes(`${name} =`) && l.includes("="));
  if (start < 0) throw new Error(`node ${name} not found`);
  // find the first "[" line after start; the dump nests [ (i3) then [ (i2)
  // before the i1 rows, so rows begin two lines below the first bracket.
  let i = start;
  while (i < lines.length && !lines[i]!.trim().startsWith("[")) i++;
  const rows: number[][] = [];
  // The dump nests 7 lines between the bracket open and close: three real
  // token rows, the "..." elision, then three more token rows.
  for (let r = 0; r < 7; r++) {
    const line = lines[i + 2 + r]!;
    const match = line.match(/-?\d+(?:\.\d+)?/g);
    rows.push(match ? match.map(Number).filter((v) => Number.isFinite(v)) : []);
  }
  const sumLine = lines.slice(start, i + 12).find((l) => l.includes("sum ="));
  const sum = sumLine ? Number(sumLine.split("sum =")[1]) : NaN;
  return { rows, sum };
}

const gt = {
  norm_b: parseNode("norm_b"),
  pixel_shuffle: parseNode("pixel_shuffle"),
  ffn_up_b: parseNode("ffn_up_b"),
};

const source = await NodeFileSource.open("models/mmproj-LFM2.5-VL-1.6b-F16.gguf");
const reader = await GgufReader.open(source);
const config = parseVisionConfig((key) => reader.metadata(key));
const weights = await loadVisionWeights(visionTensorSource(reader), config);

const gridH = 32;
const gridW = 32;
const P = gridH * gridW;
const patchDim = 3 * config.patchSize * config.patchSize;
const patches = new Float32Array(config.maxPatches * patchDim);
for (let p = 0; p < P; p++) for (let i = 0; i < 256; i++) patches[p * patchDim + i] = 1.0;
const posEmb = new Float32Array(config.maxPatches * config.hiddenSize);
posEmb.set(resizePositionEmbedding(weights.posEmb, config.positionEmbeddingGrid, gridH, gridW, config.hiddenSize));
const paddingMask = new Uint8Array(config.maxPatches).fill(1, 0, P);
const image = { patches, posEmb, paddingMask, patchCount: P, gridH, gridW, imageW: 512, imageH: 512 };

await installDawn();
const { device } = await createWebGpuDevice({ label: "vl-tail" });
const tower = await VisionTower.create({ device, config, weights });
const hidden = await tower.run(image, { stopAtLayer: config.blockCount });
const dim = config.hiddenSize;

function compare(g: { rows: number[][]; sum: number }, ours: Float32Array, stride: number, toks: number[], dims: number[], label: string): void {
  let md = 0;
  // Dump rows: [t0, t1, t2, "...", tN-3, tN-2, tN-1] — skip the elision row.
  const gtRow = (t: number) => g.rows[t < 3 ? t : t + 1]!;
  for (let t = 0; t < 6; t++) {
    for (let d = 0; d < 6; d++) {
      md = Math.max(md, Math.abs(gtRow(t)[d]! - ours[toks[t]! * stride + dims[d]!]!));
    }
  }
  console.log(`${label}: maxAbsDiff = ${md.toExponential(3)}`);
  if (label.includes("pixel_shuffle") || label.includes("norm_b vs our post")) {
    for (let t = 0; t < 3; t++) {
      const g = gtRow(t);
      const o = [0, 1, 2].map((d) => ours[toks[t]! * stride + dims[d]!]!.toFixed(4));
      console.log(`  t${toks[t]} gt: [${g.slice(0, 3).map((v) => v.toFixed(4)).join(", ")}] ours: [${o.join(", ")}]`);
    }
  }
}

const hToks = [0, 1, 2, P - 3, P - 2, P - 1];
const hDims = [0, 1, 2, dim - 3, dim - 2, dim - 1];
compare(gt.norm_b, hidden, dim, hToks, hDims, "norm_b vs our hidden (pre post-LN)");

// our post-LN (CPU)
const post = new Float32Array(P * dim);
for (let p = 0; p < P; p++) {
  layerNormInto(hidden, p * dim, weights.postLn.weight, weights.postLn.bias, config.layerNormEpsilon, dim, post, p * dim);
}
compare(gt.norm_b, post, dim, hToks, hDims, "norm_b vs our post-LN");

// llama.cpp-order unshuffle
const factor = config.projectorScaleFactor;
const T = (gridH / factor) * (gridW / factor);
const unshDim = dim * factor * factor;
const unsh = new Float32Array(T * unshDim);
for (let y = 0; y < gridH; y++) {
  for (let x = 0; x < gridW; x++) {
    const v = (x >> 2) + 8 * y;
    const tok = (v % 16) * (gridW / factor) + (v >> 4);
    for (let c = 0; c < dim; c++) {
      unsh[tok * unshDim + c + 1152 * (x & 1) + 2304 * ((x >> 1) & 1)] = post[(y * gridW + x) * dim + c]!;
    }
  }
}
compare(gt.pixel_shuffle, unsh, unshDim, [0, 1, 2, 253, 254, 255], [0, 1, 2, unshDim - 3, unshDim - 2, unshDim - 1], "pixel_shuffle vs our llama-order unshuffle");

// projector: mm1 + bias
const proj = config.projectorHiddenSize;
const up = new Float32Array(T * proj);
for (let t = 0; t < T; t++) {
  const srcBase = t * unshDim;
  for (let o = 0; o < proj; o++) {
    let acc = weights.projector.mm1Bias[o]!;
    const row = o * unshDim;
    for (let i = 0; i < unshDim; i++) acc += unsh[srcBase + i]! * weights.projector.mm1[row + i]!;
    up[t * proj + o] = acc;
  }
}
compare(gt.ffn_up_b, up, proj, [0, 1, 2, 253, 254, 255], [0, 1, 2, proj - 3, proj - 2, proj - 1], "ffn_up_b (mm1+bias) vs our");

tower.destroy();
source.close();
process.exit(0);
