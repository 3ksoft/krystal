#!/usr/bin/env bun
// Which pixel-unshuffle order does the mmproj's mm.1 expect? llama.cpp's
// build_patch_merge_permute produces a different channel/token permutation
// than torch PixelUnshuffle; the node869 ground truth (raw red 512) decides.
//
//   bun run misc/vl-unshuffle-check.ts
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

const gtFinal = JSON.parse(readFileSync("/tmp/node869.json", "utf8"));

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
const { device } = await createWebGpuDevice({ label: "vl-unshuffle" });
const tower = await VisionTower.create({ device, config, weights });

// hidden after all 27 blocks (before post-LN)
const hidden = await tower.run(image, { stopAtLayer: config.blockCount });
const dim = config.hiddenSize;

// post-LN (CPU)
const post = new Float32Array(P * dim);
for (let p = 0; p < P; p++) {
  layerNormInto(hidden, p * dim, weights.postLn.weight, weights.postLn.bias, config.layerNormEpsilon, dim, post, p * dim);
}

const factor = config.projectorScaleFactor;
const outH = gridH / factor;
const outW = gridW / factor;
const T = outH * outW;
const unshDim = dim * factor * factor;

// ---- candidate unshuffle orders ----
// Each returns unshuffled[token*unshDim + channel] for input patch (x, y).
const orders: [string, (u: Float32Array, x: number, y: number) => void][] = [
  [
    "torch PixelUnshuffle (current)",
    (u, x, y) => {
      const tok = (y >> 1) * outW + (x >> 1);
      for (let c = 0; c < dim; c++) {
        const ch = c * 4 + (y & 1) * 2 + (x & 1);
        u[tok * unshDim + ch] = post[(y * gridW + x) * dim + c]!;
      }
    },
  ],
  [
    "llama.cpp build_patch_merge_permute",
    (u, x, y) => {
      const v = (x >> 2) + 8 * y;
      const tok = ((v % 16) * outW) + (v >> 4); // y'' = v mod 16, x'' = v div 16
      for (let c = 0; c < dim; c++) {
        const ch = c + 1152 * (x & 1) + 2304 * ((x >> 1) & 1);
        u[tok * unshDim + ch] = post[(y * gridW + x) * dim + c]!;
      }
    },
  ],
  [
    "llama.cpp order, torch channel (token perm only)",
    (u, x, y) => {
      const v = (x >> 2) + 8 * y;
      const tok = ((v % 16) * outW) + (v >> 4);
      for (let c = 0; c < dim; c++) {
        const ch = c * 4 + (y & 1) * 2 + (x & 1);
        u[tok * unshDim + ch] = post[(y * gridW + x) * dim + c]!;
      }
    },
  ],
  [
    "torch token, llama channel",
    (u, x, y) => {
      const tok = (y >> 1) * outW + (x >> 1);
      for (let c = 0; c < dim; c++) {
        const ch = c + 1152 * (x & 1) + 2304 * ((x >> 1) & 1);
        u[tok * unshDim + ch] = post[(y * gridW + x) * dim + c]!;
      }
    },
  ],
  [
    "EMPIRICAL llama (tok=(x>>1)+16*(y>>1), ch=d+1152*(y&1)+2304*(x&1))",
    (u, x, y) => {
      const tok = (x >> 1) + outW * (y >> 1);
      for (let c = 0; c < dim; c++) {
        const ch = c + 1152 * (y & 1) + 2304 * (x & 1);
        u[tok * unshDim + ch] = post[(y * gridW + x) * dim + c]!;
      }
    },
  ],
];

const proj = config.projectorHiddenSize;
const rows = gtFinal.rows as number[][];
const fdims = [0, 1, 2, proj - 3, proj - 2, proj - 1];
const ftoks = [0, 1, 2, 253, 254, 255];

for (const [label, fill] of orders) {
  const unsh = new Float32Array(T * unshDim);
  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW; x++) fill(unsh, x, y);
  }
  // mm.1 -> GELU -> mm.2 (CPU)
  const out = new Float32Array(T * proj);
  const mid = new Float32Array(proj);
  for (let t = 0; t < T; t++) {
    const srcBase = t * unshDim;
    for (let o = 0; o < proj; o++) {
      let acc = weights.projector.mm1Bias[o]!;
      const row = o * unshDim;
      for (let i = 0; i < unshDim; i++) acc += unsh[srcBase + i]! * weights.projector.mm1[row + i]!;
      mid[o] = geluErf(acc);
    }
    for (let o = 0; o < proj; o++) {
      let acc = weights.projector.mm2Bias[o]!;
      const row = o * proj;
      for (let i = 0; i < proj; i++) acc += mid[i]! * weights.projector.mm2[row + i]!;
      out[t * proj + o] = acc;
    }
  }
  let maxDiff = 0;
  for (let t = 0; t < 6; t++) {
    for (let d = 0; d < 6; d++) {
      maxDiff = Math.max(maxDiff, Math.abs(rows[t]![d]! - out[ftoks[t]! * proj + fdims[d]!]!));
    }
  }
  let sum = 0;
  for (const v of out) sum += v;
  console.log(
    `${label}: maxAbsDiff = ${maxDiff.toExponential(3)}  sum = ${sum.toFixed(4)} (gt ${gtFinal.sum.toFixed(4)})`,
  );
}

tower.destroy();
source.close();
process.exit(0);
