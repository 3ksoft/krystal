// CPU reference vision tower benchmark at the v0 512×512 envelope.
//
//   bun run misc/vl-cpu-embed.ts [mmproj.gguf] [imageWidth] [imageHeight]
//
// Runs the M0 processor + exact reference tower over a synthetic image and
// prints timing + output-token stats. Useful to sanity-check the real
// resolution path before the WGSL implementation lands.
import { GgufReader } from "../packages/quant/src/gguf/reader.ts";
import { NodeFileSource } from "../packages/quant/src/gguf/source-node.ts";
import { parseVisionConfig } from "../packages/quant/src/vision/config.ts";
import { forwardVision } from "../packages/quant/src/vision/reference.ts";
import { loadVisionWeights, visionTensorSource } from "../packages/quant/src/vision/weights.ts";
import { encodeImage } from "../packages/webgpu/src/vision/processor.ts";

const args = process.argv.slice(2);
// The JS reference is O(patches * layers * MLP). `targetSize` is pinned to
// the input dims so the grid stays small (the v0 real-envelope 512×512 grid
// of 1024 patches is ~64× this cost and only makes sense on the GPU).
// 64×64 → 4×4 grid → ~35 s. Default 64×64 (a smoke run); pass dims for
// bigger runs.
const path = args[0] ?? "models/mmproj-LFM2.5-VL-1.6b-F16.gguf";
const width = Number(args[1] ?? 64);
const height = Number(args[2] ?? 64);

function syntheticRgba(w: number, h: number): Uint8Array {
  const rgba = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const base = (y * w + x) * 4;
      rgba[base + 0] = (x * 37) % 256;
      rgba[base + 1] = (y * 53) % 256;
      rgba[base + 2] = (x * 17 + y * 29) % 256;
      rgba[base + 3] = 255;
    }
  }
  return rgba;
}

const source = await NodeFileSource.open(path);
try {
  const reader = await GgufReader.open(source);
  const config = parseVisionConfig((key) => reader.metadata(key));
  console.log(`vision config: hidden=${config.hiddenSize} blocks=${config.blockCount} heads=${config.attentionHeads} patch=${config.patchSize}`);

  const t0 = performance.now();
  const weights = await loadVisionWeights(visionTensorSource(reader), config);
  const t1 = performance.now();
  console.log(`weights loaded in ${(t1 - t0).toFixed(0)} ms`);

  const rgba = syntheticRgba(width, height);
  const t2 = performance.now();
  const processed = encodeImage(rgba, width, height, weights.posEmb, {
    hiddenSize: config.hiddenSize,
    patchSize: config.patchSize,
    imageMean: config.imageMean,
    imageStd: config.imageStd,
    positionEmbeddingGrid: config.positionEmbeddingGrid,
    maxPatches: config.maxPatches,
  }, { targetSize: width });
  const t3 = performance.now();
  console.log(`image ${width}x${height} -> grid ${processed.gridH}x${processed.gridW}, ${processed.patchCount} patches in ${(t3 - t2).toFixed(1)} ms`);

  const result = forwardVision(weights, config, {
    patches: processed.patches,
    posEmb: processed.posEmb,
    patchCount: processed.patchCount,
    gridH: processed.gridH,
    gridW: processed.gridW,
  });
  const t4 = performance.now();
  console.log(`forward in ${((t4 - t3) / 1000).toFixed(1)} s -> ${result.tokens} image tokens x ${config.projectorHiddenSize}`);

  let norm = 0;
  for (const value of result.embeddings) norm += value * value;
  norm = Math.sqrt(norm / result.embeddings.length);
  console.log(`embedding RMS: ${norm.toFixed(4)}`);
  console.log(`first token head: ${Array.from(result.embeddings.slice(0, 8)).map((v) => v.toFixed(3)).join(", ")}`);
} finally {
  await source.close();
}
