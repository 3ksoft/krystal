#!/usr/bin/env bun
// Infer llama.cpp's pixel-unshuffle mapping empirically: for each sampled
// pixel_shuffle entry (token, dim) find which (source token, source dim) of the
// post-LN tensor it copies, by argmin over our full post-LN (which matches
// llama.cpp's norm_b to ~4e-2). Then print the inferred (srcToken, srcDim,
// srcX, srcY, srcC) tuples to spot the pattern.
//
//   bun run misc/vl-unshuffle-map.ts
import { readFileSync } from "node:fs";
import { NodeFileSource } from "../packages/quant/src/gguf/source-node.ts";
import { GgufReader } from "../packages/quant/src/gguf/reader.ts";
import { parseVisionConfig } from "../packages/quant/src/vision/config.ts";
import { loadVisionWeights, visionTensorSource } from "../packages/quant/src/vision/weights.ts";
import { resizePositionEmbedding } from "../packages/webgpu/src/vision/processor.ts";
import { layerNormInto } from "../packages/quant/src/vision/reference.ts";
import { VisionTower } from "../packages/webgpu/src/vision/tower.ts";
import { installDawn } from "../tests/dawn.ts";
import { createWebGpuDevice } from "../packages/webgpu/src/device.ts";

function parseNode(name: string): { rows: number[][] } {
  const lines = readFileSync("/tmp/mtmd-full.log", "utf8").split("\n");
  const start = lines.findIndex((l) => l.includes(`${name} =`) && l.includes("="));
  let i = start;
  while (i < lines.length && !lines[i]!.trim().startsWith("[")) i++;
  const rows: number[][] = [];
  for (let r = 0; r < 7; r++) {
    const line = lines[i + 2 + r]!;
    const m = line.match(/-?\d+(?:\.\d+)?/g);
    rows.push(m ? m.map(Number).filter((v) => Number.isFinite(v)) : []);
  }
  return { rows };
}

const ps = parseNode("pixel_shuffle").rows;
const gtRow = (t: number) => ps[t < 3 ? t : t + 1]!;

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
const { device } = await createWebGpuDevice({ label: "vl-unsh-map" });
const tower = await VisionTower.create({ device, config, weights });
const hidden = await tower.run(image, { stopAtLayer: config.blockCount });
const dim = config.hiddenSize;

const post = new Float32Array(P * dim);
for (let p = 0; p < P; p++) {
  layerNormInto(hidden, p * dim, weights.postLn.weight, weights.postLn.bias, config.layerNormEpsilon, dim, post, p * dim);
}

const toks = [0, 1, 2, 253, 254, 255];

// Find the source (x, y) whose dims [0,1,2] (or [dim-3..dim-1]) best match the
// three sampled values of a pixel_shuffle token — channel offset 0 / 3456.
function bestSrc3(values: number[], dims3: number[]): { x: number; y: number; err: number } {
  let best = { x: -1, y: -1, err: Infinity };
  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW; x++) {
      let err = 0;
      for (let d = 0; d < 3; d++) {
        const e = values[d]! - post[(y * gridW + x) * dim + dims3[d]!]!;
        err += e * e;
      }
      if (err < best.err) best = { x, y, err };
    }
  }
  return best;
}

for (let t = 0; t < 6; t++) {
  const token = toks[t]!;
  const row = gtRow(t);
  const a = bestSrc3(row.slice(0, 3), [0, 1, 2]);
  const b = bestSrc3(row.slice(3, 6), [dim - 3, dim - 2, dim - 1]);
  console.log(
    `ps token ${String(token).padStart(3)}: ch0 <- (x=${a.x}, y=${a.y}) err=${a.err.toExponential(2)}  | ch3456 <- (x=${b.x}, y=${b.y}) err=${b.err.toExponential(2)}`,
  );
}
tower.destroy();
source.close();
process.exit(0);
