// Vision tower GPU differential tests (ADA-0009, M2).
//
// Runs the REAL WGSL tower (Dawn via the `webgpu` npm bindings, like
// tests/checkpoint.test.ts) against the exact CPU reference oracle
// (packages/quant/src/vision/reference.ts) on the same F16 GGUF weights and
// the same inputs. This is the M2 gate: it isolates pure kernel error from
// quantization/semantic error.
//
//   bun test tests/vision-gpu.test.ts
//
// Skips (with a warning) when ./models/mmproj-LFM2.5-VL-1.6b-F16.gguf is
// missing; GPU/device failures surface as test errors.
import { describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { GgufReader } from "../packages/quant/src/gguf/reader.ts";
import { NodeFileSource } from "../packages/quant/src/gguf/source-node.ts";
import { parseVisionConfig, type VisionConfig } from "../packages/quant/src/vision/config.ts";
import { forwardVision, type VisionReferenceWeights } from "../packages/quant/src/vision/reference.ts";
import { loadVisionWeights, visionTensorSource } from "../packages/quant/src/vision/weights.ts";
import {
  encodeImage,
  resizePositionEmbedding,
  type ProcessedImage,
} from "../packages/webgpu/src/vision/processor.ts";
import { VisionTower } from "../packages/webgpu/src/vision/tower.ts";
import { installDawn } from "./dawn.ts";
import { createWebGpuDevice } from "../packages/webgpu/src/device.ts";

const MMPROJ_GGUF = "models/mmproj-LFM2.5-VL-1.6b-F16.gguf";
const hasModel = existsSync(MMPROJ_GGUF);

if (!hasModel) {
  console.warn("[vision-gpu] models/mmproj-LFM2.5-VL-1.6b-F16.gguf missing — GPU differential tests skipped");
}

function syntheticRgba(width: number, height: number, seed = 1): Uint8Array {
  const rgba = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const base = (y * width + x) * 4;
      rgba[base + 0] = (x * 37 + seed * 11) % 256;
      rgba[base + 1] = (y * 53 + seed * 7) % 256;
      rgba[base + 2] = (x * 17 + y * 29 + seed * 3) % 256;
      rgba[base + 3] = 255;
    }
  }
  return rgba;
}

interface Boot {
  config: VisionConfig;
  weights: VisionReferenceWeights;
  tower: VisionTower;
}

let bootPromise: Promise<Boot> | undefined;

async function boot(): Promise<Boot> {
  bootPromise ??= (async () => {
    await installDawn();
    const { device } = await createWebGpuDevice({ label: "vision-gpu-test" });
    const source = await NodeFileSource.open(MMPROJ_GGUF);
    try {
      const reader = await GgufReader.open(source);
      const config = parseVisionConfig((key) => reader.metadata(key));
      const weights = await loadVisionWeights(visionTensorSource(reader), config);
      const tower = await VisionTower.create({ device, config, weights });
      return { config, weights, tower };
    } finally {
      await source.close();
    }
  })();
  return bootPromise;
}

/** Encode a synthetic image, optionally masking the tail patches. */
async function encode(
  weights: VisionReferenceWeights,
  config: VisionConfig,
  seed: number,
  validCount?: number,
): Promise<ProcessedImage> {
  const rgba = syntheticRgba(64, 64, seed);
  const processed = encodeImage(rgba, 64, 64, weights.posEmb, {
    hiddenSize: config.hiddenSize,
    patchSize: config.patchSize,
    imageMean: config.imageMean,
    imageStd: config.imageStd,
    positionEmbeddingGrid: config.positionEmbeddingGrid,
    maxPatches: config.maxPatches,
  }, { targetSize: 64 });
  if (validCount !== undefined && validCount < processed.patchCount) {
    processed.paddingMask.fill(0, validCount, processed.patchCount);
  }
  return processed;
}

/** Oracle embeddings for the same ProcessedImage the tower consumes. */
function oracleEmbeddings(
  weights: VisionReferenceWeights,
  config: VisionConfig,
  image: ProcessedImage,
): Float32Array {
  const result = forwardVision(weights, config, {
    patches: image.patches,
    posEmb: image.posEmb,
    patchCount: image.gridH * image.gridW,
    gridH: image.gridH,
    gridW: image.gridW,
    paddingMask: image.paddingMask.subarray(0, image.gridH * image.gridW),
  });
  return result.embeddings;
}

function maxAbsDiff(a: Float32Array, b: Float32Array): { max: number; scale: number } {
  let max = 0;
  let scale = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    max = Math.max(max, Math.abs(a[i]! - b[i]!));
    scale = Math.max(scale, Math.abs(b[i]!));
  }
  return { max, scale };
}

describe("vision tower GPU (M2 differential)", () => {
  test.skipIf(!hasModel)(
    "GPU matches the CPU oracle on a 4x4 grid (16 patches, all valid)",
    async () => {
      const { config, weights, tower } = await boot();
      const image = await encode(weights, config, 1);
      const expected = oracleEmbeddings(weights, config, image);
      const actual = await tower.run(image);

      expect(actual.length).toBe(expected.length);
      const { max, scale } = maxAbsDiff(actual, expected);
      // f32 kernel accumulation across 27 blocks vs f64-ish oracle math:
      // observed ~1.5e-6; the bound is ~60x above that so a 100x regression
      // still fails, while cross-driver f32 variation stays green.
      const tolerance = Math.max(1e-4, 2e-4 * scale);
      console.log(`[vision-gpu] all-valid: maxAbsDiff=${max.toExponential(2)} scale=${scale.toExponential(2)}`);
      expect(max).toBeLessThanOrEqual(tolerance);
    },
    300_000,
  );

  test.skipIf(!hasModel)(
    "GPU matches the CPU oracle with a padding mask (12 valid / 16 grid)",
    async () => {
      const { config, weights, tower } = await boot();
      const image = await encode(weights, config, 2, 12);
      const expected = oracleEmbeddings(weights, config, image);
      const actual = await tower.run(image);

      expect(actual.length).toBe(expected.length);
      const { max, scale } = maxAbsDiff(actual, expected);
      const tolerance = Math.max(1e-4, 2e-4 * scale);
      console.log(`[vision-gpu] masked: maxAbsDiff=${max.toExponential(2)} scale=${scale.toExponential(2)}`);
      expect(max).toBeLessThanOrEqual(tolerance);
    },
    300_000,
  );

  test.skipIf(!hasModel)(
    "GPU matches the CPU oracle across the 256-patch attention chunk boundary (18x16 grid)",
    async () => {
      const { config, weights, tower } = await boot();
      // 288 patches > 256: the attention shader's per-invocation multi-key
      // accumulation (b += 256u chunks) is only exercised past this boundary;
      // the 4x4-grid tests never hit it. The CPU oracle costs ~0.5 s/patch,
      // so this one is intentionally slow.
      const gridH = 18;
      const gridW = 16;
      const P = gridH * gridW;
      const patchDim = 3 * config.patchSize * config.patchSize;
      const dim = config.hiddenSize;
      const patches = new Float32Array(config.maxPatches * patchDim);
      const posEmb = new Float32Array(config.maxPatches * dim);
      let state = 0x2f6e2b1 >>> 0;
      const rand = () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 4294967296;
      };
      for (let i = 0; i < P * patchDim; i++) patches[i] = rand() * 2 - 1;
      posEmb.set(resizePositionEmbedding(weights.posEmb, config.positionEmbeddingGrid, gridH, gridW, dim));
      const paddingMask = new Uint8Array(config.maxPatches);
      paddingMask.fill(1, 0, P);
      const image: ProcessedImage = {
        patches, posEmb, paddingMask, patchCount: P, gridH, gridW,
        imageW: gridW * config.patchSize, imageH: gridH * config.patchSize,
      };

      const expected = oracleEmbeddings(weights, config, image);
      const actual = await tower.run(image);
      expect(actual.length).toBe(expected.length);
      const { max, scale } = maxAbsDiff(actual, expected);
      const tolerance = Math.max(1e-4, 2e-4 * scale);
      console.log(`[vision-gpu] chunked(288): maxAbsDiff=${max.toExponential(2)} scale=${scale.toExponential(2)}`);
      expect(max).toBeLessThanOrEqual(tolerance);

      // Determinism guard: the attention race this file once caught (whole
      // (head, token) workgroup rows corrupted, ~5e-2, probabilistic from
      // n>=65 on the Dawn/NVIDIA Ampere stack) would slip through a single
      // differential pass. Two identical runs must agree to f32 noise.
      const rerun = await tower.run(image);
      const { max: rt } = maxAbsDiff(actual, rerun);
      console.log(`[vision-gpu] chunked(288): run-to-run maxAbsDiff=${rt.toExponential(2)}`);
      expect(rt).toBeLessThanOrEqual(1e-4);
    },
    600_000,
  );

  test.skipIf(!hasModel)(
    "GPU tower output is input-sensitive",
    async () => {
      const { config, weights, tower } = await boot();
      const imageA = await encode(weights, config, 1);
      const imageB = await encode(weights, config, 7);
      const a = await tower.run(imageA);
      const b = await tower.run(imageB);
      const { max } = maxAbsDiff(a, b);
      // two synthetic images differ well above f32 noise (observed ~0.2)
      expect(max).toBeGreaterThan(0.05);
    },
    300_000,
  );
});
