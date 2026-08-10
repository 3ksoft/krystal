// Vision pipeline tests (bun:test).
//
// Covers (ADA-0009 M0 + oracle):
//   - vision config parsing from the mmproj WQ4 sidecar metadata,
//   - <image> token id derivation from the VL text WQ4 tokenizer,
//   - M0 processor math (fit/resize/normalize/patchify/pos-emb resize) with
//     hand-computed expectations,
//   - the exact CPU reference vision tower on a small synthetic image.
//
// Model-dependent tests skip with a warning when ./models files are missing;
// everything else is pure math and always runs.
import { describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { GgufReader } from "../packages/quant/src/gguf/reader.ts";
import { NodeFileSource } from "../packages/quant/src/gguf/source-node.ts";
import { Wq4Reader } from "../packages/quant/src/wq4/reader.ts";
import { findImageTokenId, parseVisionConfig } from "../packages/quant/src/vision/config.ts";
import { forwardVision } from "../packages/quant/src/vision/reference.ts";
import { dequantizeWq4Tensor, loadVisionWeights, visionTensorSource } from "../packages/quant/src/vision/weights.ts";
import {
  encodeImage,
  fitSize,
  patchify,
  resizeBilinearToRgb,
  resizePositionEmbedding,
} from "../packages/webgpu/src/vision/processor.ts";

const MMPROJ_GGUF = "models/mmproj-LFM2.5-VL-1.6b-F16.gguf";
const MMPROJ_WQ4 = "models/LFM2.5-VL-mmproj-WQ4.wq4";
const VL_WQ4 = "models/LFM2.5-VL-1.6B-WQ4.wq4";

const hasMmprojGguf = existsSync(MMPROJ_GGUF);
const hasMmprojWq4 = existsSync(MMPROJ_WQ4);
const hasTextWq4 = existsSync(VL_WQ4);

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

const TEST_CONFIG = {
  hiddenSize: 1152,
  patchSize: 16,
  imageMean: [0.5, 0.5, 0.5],
  imageStd: [0.5, 0.5, 0.5],
  positionEmbeddingGrid: 16,
  maxPatches: 1024,
} as const;

if (!hasMmprojGguf || !hasMmprojWq4 || !hasTextWq4) {
  console.warn(
    "[vision] one or more ./models files missing — model-dependent tests will be skipped " +
    "(need mmproj GGUF, mmproj WQ4, VL text WQ4)",
  );
}

describe("vision WQ4 host dequant", () => {
  test("decodes a crafted block with signed nibbles and a power-of-two scale", () => {
    // One block = 32 elements = 5 u32 words (4 packed + 1 exponent).
    // word 0 nibbles [0..7] -> values (n-8) * 2^3 = -64..-8; other words use
    // nibble 8 (encoded zero) -> decode 0; exponent 3 -> scale 8.
    const bytes = new Uint8Array(2 * 20);
    const view = new DataView(bytes.buffer);
    const put = (block: number, word: number, value: number): void => {
      view.setUint32(block * 20 + word * 4, value, true);
    };
    put(0, 0, 0x76543210); // nibbles 0,1,2,3,4,5,6,7
    put(0, 1, 0x88888888);
    put(0, 2, 0x88888888);
    put(0, 3, 0x88888888);
    put(0, 4, 3); // scale 2^3 = 8
    // zero block: all-zero nibbles, exponent MIN_EXP (-24) bit pattern
    put(1, 0, 0x88888888);
    put(1, 1, 0x88888888);
    put(1, 2, 0x88888888);
    put(1, 3, 0x88888888);
    put(1, 4, (-24 >>> 0));

    const out = dequantizeWq4Tensor(bytes, 64);
    expect(out.length).toBe(64);
    expect([...out.slice(0, 8)]).toEqual([-64, -56, -48, -40, -32, -24, -16, -8]);
    expect([...out.slice(8, 32)].every((v) => v === 0)).toBe(true);
    expect([...out.slice(32)].every((v) => v === 0)).toBe(true);
  });

  test("rejects malformed block counts and byte lengths", () => {
    const bytes = new Uint8Array(20);
    expect(() => dequantizeWq4Tensor(bytes, 33)).toThrow(/multiple of 32/);
    expect(() => dequantizeWq4Tensor(new Uint8Array(21), 32)).toThrow(/20-byte blocks/);
    expect(() => dequantizeWq4Tensor(new Uint8Array(20), 0)).toThrow(/multiple of 32/);
    expect(() => dequantizeWq4Tensor(new Uint8Array(20), 64)).toThrow(/words/);
  });
});

describe("vision config", () => {
  test.skipIf(!hasMmprojWq4)("parses clip.* metadata from the mmproj sidecar", async () => {
    const source = await NodeFileSource.open(MMPROJ_WQ4);
    try {
      const reader = await Wq4Reader.open(source);
      const config = parseVisionConfig((key) => reader.metadataValue(key), { imageTokenId: 396 });
      expect(config.hiddenSize).toBe(1152);
      expect(config.blockCount).toBe(27);
      expect(config.attentionHeads).toBe(16);
      expect(config.headDim).toBe(72);
      expect(config.patchSize).toBe(16);
      expect(config.feedForwardSize).toBe(4304);
      expect(config.layerNormEpsilon).toBeCloseTo(1e-6, 12);
      expect(config.projectorScaleFactor).toBe(2);
      expect(config.projectorHiddenSize).toBe(2048);
      expect(config.imageMean).toEqual([0.5, 0.5, 0.5]);
      expect(config.imageStd).toEqual([0.5, 0.5, 0.5]);
      expect(config.useGelu).toBe(true);
      expect(config.imageTokenId).toBe(396);
    } finally {
      await source.close();
    }
  });

  test.skipIf(!hasTextWq4)("derives the <image> placeholder id from the VL tokenizer", async () => {
    const source = await NodeFileSource.open(VL_WQ4);
    try {
      const reader = await Wq4Reader.open(source);
      const tokens = reader.metadataValue<string[]>("tokenizer.ggml.tokens");
      expect(tokens[396]).toBe("<image>");
      expect(findImageTokenId(tokens)).toBe(396);
    } finally {
      await source.close();
    }
  });

  test.skipIf(!hasMmprojGguf)(
    "pos-emb loader emits the raw [position, dim] GGUF layout (no transpose)",
    async () => {
      // Regression: v.position_embd.weight is stored row-major [grid*grid,
      // hidden] (dims [1152, 256], element (p, d) at p*1152 + d). A former
      // spurious transpose read data[d*256 + p], scrambling the grid — the
      // tower stayed CPU==GPU consistent while the LM could not localize any
      // patch content ("sees" images differently but never correctly).
      const source = await NodeFileSource.open(MMPROJ_GGUF);
      try {
        const reader = await GgufReader.open(source);
        const config = parseVisionConfig((key) => reader.metadata(key));
        const weights = await loadVisionWeights(visionTensorSource(reader), config);
        const info = reader.tensor("v.position_embd.weight")!;
        const bytes = await reader.readTensor(info);
        const raw = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
        const h = config.hiddenSize;
        const n = weights.posEmb.length / h;
        expect(n).toBe(16 * 16);
        // Element-wise equality with the raw GGUF data over a spread of indices.
        for (let i = 1; i < raw.length; i += 997) {
          expect(weights.posEmb[i]).toBe(raw[i]!);
        }
        // The old misread would map pos 1 to hidden dim 256: element (p=0, d=1)
        // must read raw[1], not raw[256].
        expect(weights.posEmb[1]!).toBe(raw[1]!);
        expect(weights.posEmb[1]!).not.toBe(raw[256]!);
      } finally {
        await source.close();
      }
    },
    120_000,
  );

  test("reports missing required keys", () => {
    expect(() => parseVisionConfig(() => undefined)).toThrow(/clip\.vision\.embedding_length/);
  });
});

describe("vision processor (M0)", () => {
  test("fitSize keeps aspect ratio and rounds to patch multiples", () => {
    expect(fitSize(1000, 800, 512, 16)).toEqual({ w: 512, h: 416 });
    expect(fitSize(512, 512, 512, 16)).toEqual({ w: 512, h: 512 });
    // Small images upscale into the target (consistent with min_image_tokens).
    expect(fitSize(64, 64, 512, 16)).toEqual({ w: 512, h: 512 });
    expect(fitSize(100, 50, 512, 16)).toEqual({ w: 512, h: 256 });
  });

  test("resize + normalize a constant image is exact", () => {
    const rgba = new Uint8Array(2 * 2 * 4).fill(255);
    const out = new Float32Array(4 * 4 * 3);
    resizeBilinearToRgb(rgba, 2, 2, 4, 4, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], out);
    for (let i = 0; i < out.length; i++) expect(out[i]).toBe(1);
  });

  test("resize bilinear matches hand-computed interpolation", () => {
    // 2x1 RGBA row: left=0, right=255 (white) -> 3 wide output.
    const rgba = new Uint8Array([0, 0, 0, 255, 255, 255, 255, 255]);
    const out = new Float32Array(3 * 1 * 3);
    resizeBilinearToRgb(rgba, 2, 1, 3, 1, [0, 0, 0], [1, 1, 1], out);
    // align_corners=False: dst x=0 -> src (0.5*2/3-0.5) = -0.1667 -> clamp 0
    expect(out[0]).toBe(0);
    // dst x=1 -> src (1.5*2/3-0.5) = 0.5 -> 0.5
    expect(out[3]).toBeCloseTo(0.5, 6);
    // dst x=2 -> src (2.5*2/3-0.5) = 1.1667 -> clamp 1
    expect(out[6]).toBe(1);
  });

  test("patchify lays patches out in (c,h,w) order, w fastest", () => {
    const size = 32;
    const rgb = new Float32Array(size * size * 3);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const base = (y * size + x) * 3;
        rgb[base + 0] = y;
        rgb[base + 1] = x;
        rgb[base + 2] = y + x;
      }
    }
    const patches = patchify(rgb, 2, 2, 16);
    expect(patches.length).toBe(4 * 768);
    // patch (0,0), channel 0, local (h,w) must equal pixel (h,w) channel 0.
    for (let i = 0; i < 16; i++) {
      for (let j = 0; j < 16; j++) {
        expect(patches[0 * 768 + 0 * 256 + i * 16 + j]).toBe(i);
        expect(patches[0 * 768 + 1 * 256 + i * 16 + j]).toBe(j);
        expect(patches[0 * 768 + 2 * 256 + i * 16 + j]).toBe(i + j);
      }
    }
    // patch (1,1) starts at pixel (16,16).
    expect(patches[3 * 768 + 0 * 256 + 0 * 16 + 0]).toBe(16);
    expect(patches[3 * 768 + 1 * 256 + 15 * 16 + 15]).toBe(31);
  });

  test("position embedding resize is bilinear align_corners=False", () => {
    const baseSide = 2;
    const hidden = 1;
    // base grid values 0..3, dims 1
    const base = new Float32Array([0, 1, 2, 3]);
    const out = resizePositionEmbedding(base, baseSide, 4, 4, hidden);
    expect(out.length).toBe(16);
    // dst (0,0): src (-0.25,-0.25) -> clamp (0,0) -> 0
    expect(out[0]).toBe(0);
    // dst (3,3): src (1.25,1.25) -> clamp (1,1) -> 3
    expect(out[3 * 4 + 3]).toBe(3);
    // dst (1,1): src (0.25,0.25) -> top row lerp(0,1,.25)=.25, bottom row lerp(2,3,.25)=2.25,
    // then vertical lerp(.25,2.25,.25) = .25 + 2*.25 = .75
    expect(out[1 * 4 + 1]).toBeCloseTo(0.75, 6);
    // dst (2,1): src x 0.25, y 0.75 -> .25 + 2*.75 = 1.75
    expect(out[2 * 4 + 1]).toBeCloseTo(1.75, 6);
  });

  test("encodeImage pads to maxPatches with a valid mask", () => {
    const rgba = syntheticRgba(64, 64);
    const posEmbBase = new Float32Array(16 * 16 * 1152).fill(0.01);
    const processed = encodeImage(rgba, 64, 64, posEmbBase, TEST_CONFIG, { targetSize: 64 });
    expect(processed.gridH).toBe(4);
    expect(processed.gridW).toBe(4);
    expect(processed.patchCount).toBe(16);
    expect(processed.patches.length).toBe(1024 * 768);
    expect(processed.posEmb.length).toBe(1024 * 1152);
    expect(processed.paddingMask.length).toBe(1024);
    expect([...processed.paddingMask.slice(0, 16)].every((v) => v === 1)).toBe(true);
    expect([...processed.paddingMask.slice(16)].every((v) => v === 0)).toBe(true);
    expect(Number.isFinite(processed.patches[0])).toBe(true);
  });
});

describe("vision WQ4 sidecar weights (M1 runtime path)", () => {
  // The sidecar stores matmul matrices as WQ4 blocks and everything else raw
  // (F16 `v.patch_embd`/`ffn_down` are not WQ4-safe: widths 16/4304 are not
  // multiples of 32; norms/biases/pos-emb are exact tensors). The loader must
  // dequantize the former and pass the latter through byte-for-byte, so the
  // decoded layout is source-agnostic and the tower/oracle never see WQ4.
  test.skipIf(!hasMmprojGguf || !hasMmprojWq4)(
    "dequantizes WQ4 matrices exactly and passes raw tensors through unchanged",
    async () => {
      const ggufSource = await NodeFileSource.open(MMPROJ_GGUF);
      const wq4Source = await NodeFileSource.open(MMPROJ_WQ4);
      try {
        const ggufReader = await GgufReader.open(ggufSource);
        const wq4Reader = await Wq4Reader.open(wq4Source);
        const config = parseVisionConfig((key) => ggufReader.metadata(key));
        const [ggufWeights, wq4Weights] = await Promise.all([
          loadVisionWeights(visionTensorSource(ggufReader), config),
          loadVisionWeights(visionTensorSource(wq4Reader), config),
        ]);

        // Raw tensors: byte-identical round trip (F16/F32 source bytes).
        // Element-wise compare — these arrays are up to ~4.9M elements.
        const expectEqual = (a: Float32Array, b: Float32Array, label: string): void => {
          expect(a.length).toBe(b.length);
          for (let i = 0; i < a.length; i++) expect(a[i], `${label}[${i}]`).toBe(b[i]);
        };
        expectEqual(wq4Weights.patchEmb, ggufWeights.patchEmb, "patchEmb");
        expectEqual(wq4Weights.posEmb, ggufWeights.posEmb, "posEmb");
        expectEqual(wq4Weights.patchEmbBias, ggufWeights.patchEmbBias, "patchEmbBias");
        expectEqual(wq4Weights.blocks[0]!.ln1.weight, ggufWeights.blocks[0]!.ln1.weight, "blk0.ln1");
        expectEqual(wq4Weights.blocks[0]!.down, ggufWeights.blocks[0]!.down, "blk0.down"); // ffn_down raw F16

        // WQ4 matrices: within block-quantization error of the F16 source.
        const checkClose = (a: Float32Array, b: Float32Array, label: string): void => {
          let maxAbs = 0;
          let maxDiff = 0;
          for (let i = 0; i < a.length; i++) {
            maxAbs = Math.max(maxAbs, Math.abs(a[i]!));
            maxDiff = Math.max(maxDiff, Math.abs(a[i]! - b[i]!));
          }
          // Per-block step is at most maxAbs/7 for balanced blocks, but the
          // converter's int clamp (-8..+7) can cost more on unbalanced blocks:
          // measured worst maxDiff/maxAbs across all tower matrices is 0.277
          // (mm.1). 0.35 keeps the gate honest while still catching any decode
          // bug (wrong nibble order/scale -> maxDiff/maxAbs ~1+).
          expect(maxDiff, `${label}: maxAbsDiff ${maxDiff} vs bound ${0.35 * maxAbs}`).toBeLessThan(0.35 * maxAbs);
        };
        checkClose(ggufWeights.blocks[0]!.q, wq4Weights.blocks[0]!.q, "blk0.q");
        checkClose(ggufWeights.blocks[0]!.up, wq4Weights.blocks[0]!.up, "blk0.up");
        checkClose(ggufWeights.blocks[0]!.o, wq4Weights.blocks[0]!.o, "blk0.o");
        checkClose(ggufWeights.projector.mm1, wq4Weights.projector.mm1, "mm.1");
        checkClose(ggufWeights.projector.mm2, wq4Weights.projector.mm2, "mm.2");
      } finally {
        await wq4Source.close();
        await ggufSource.close();
      }
    },
    240_000,
  );

  test.skipIf(!hasMmprojGguf || !hasMmprojWq4)(
    "tower oracle forward with WQ4 weights tracks the F16 reference",
    async () => {
      const ggufSource = await NodeFileSource.open(MMPROJ_GGUF);
      const wq4Source = await NodeFileSource.open(MMPROJ_WQ4);
      try {
        const ggufReader = await GgufReader.open(ggufSource);
        const wq4Reader = await Wq4Reader.open(wq4Source);
        const config = parseVisionConfig((key) => ggufReader.metadata(key));
        const [ggufWeights, wq4Weights] = await Promise.all([
          loadVisionWeights(visionTensorSource(ggufReader), config),
          loadVisionWeights(visionTensorSource(wq4Reader), config),
        ]);

        const rgba = syntheticRgba(64, 64, 3);
        const run = (weights: typeof ggufWeights) => {
          const processed = encodeImage(rgba, 64, 64, weights.posEmb, {
            hiddenSize: config.hiddenSize,
            patchSize: config.patchSize,
            imageMean: config.imageMean,
            imageStd: config.imageStd,
            positionEmbeddingGrid: config.positionEmbeddingGrid,
            maxPatches: config.maxPatches,
          }, { targetSize: 64 });
          return forwardVision(weights, config, {
            patches: processed.patches,
            posEmb: processed.posEmb,
            patchCount: processed.patchCount,
            gridH: processed.gridH,
            gridW: processed.gridW,
          });
        };
        const ref = run(ggufWeights);
        const wq4 = run(wq4Weights);
        expect(ref.tokens).toBe(4);
        expect(wq4.tokens).toBe(4);

        // 27 layers of 4-bit weight noise measurably rotate the embeddings
        // (measured cosine ~0.90, mean abs drift ~0.11 of the reference max on
        // a 64x64 grid) but must not destroy them: a dequant decode bug would
        // collapse the cosine to ~0 and blow the mean drift past the reference
        // magnitude. Semantic quality is the WQ4 cat gate's job (vision-vl).
        let dot = 0;
        let normA = 0;
        let normB = 0;
        let meanAbsDiff = 0;
        let refMax = 0;
        for (let i = 0; i < ref.embeddings.length; i++) {
          dot += ref.embeddings[i]! * wq4.embeddings[i]!;
          normA += ref.embeddings[i]! ** 2;
          normB += wq4.embeddings[i]! ** 2;
          meanAbsDiff += Math.abs(ref.embeddings[i]! - wq4.embeddings[i]!);
          refMax = Math.max(refMax, Math.abs(ref.embeddings[i]!));
        }
        meanAbsDiff /= ref.embeddings.length;
        const cosine = dot / Math.sqrt(normA * normB);
        expect(cosine).toBeGreaterThan(0.8);
        expect(meanAbsDiff, `meanAbsDiff ${meanAbsDiff} vs ref max ${refMax}`).toBeLessThan(0.3 * refMax);
      } finally {
        await wq4Source.close();
        await ggufSource.close();
      }
    },
    300_000,
  );
});

describe("vision reference tower", () => {
  test.skipIf(!hasMmprojGguf)("forward produces finite, deterministic, input-sensitive embeddings", async () => {
    const source = await NodeFileSource.open(MMPROJ_GGUF);
    try {
      const reader = await GgufReader.open(source);
      const config = parseVisionConfig((key) => reader.metadata(key));
      const weights = await loadVisionWeights(visionTensorSource(reader), config);

      const rgba = syntheticRgba(64, 64, 1);
      const processed = encodeImage(rgba, 64, 64, weights.posEmb, {
        hiddenSize: config.hiddenSize,
        patchSize: config.patchSize,
        imageMean: config.imageMean,
        imageStd: config.imageStd,
        positionEmbeddingGrid: config.positionEmbeddingGrid,
        maxPatches: config.maxPatches,
      }, { targetSize: 64 });

      const result = forwardVision(weights, config, {
        patches: processed.patches,
        posEmb: processed.posEmb,
        patchCount: processed.patchCount,
        gridH: processed.gridH,
        gridW: processed.gridW,
      });

      expect(result.tokens).toBe(4);
      expect(result.embeddings.length).toBe(4 * 2048);
      for (const value of result.embeddings) expect(Number.isFinite(value)).toBe(true);

      // deterministic
      const again = forwardVision(weights, config, {
        patches: processed.patches,
        posEmb: processed.posEmb,
        patchCount: processed.patchCount,
        gridH: processed.gridH,
        gridW: processed.gridW,
      });
      expect(Array.from(again.embeddings)).toEqual(Array.from(result.embeddings));

      // input-sensitive: a different image must produce different embeddings
      const rgba2 = syntheticRgba(64, 64, 7);
      const processed2 = encodeImage(rgba2, 64, 64, weights.posEmb, {
        hiddenSize: config.hiddenSize,
        patchSize: config.patchSize,
        imageMean: config.imageMean,
        imageStd: config.imageStd,
        positionEmbeddingGrid: config.positionEmbeddingGrid,
        maxPatches: config.maxPatches,
      }, { targetSize: 64 });
      const result2 = forwardVision(weights, config, {
        patches: processed2.patches,
        posEmb: processed2.posEmb,
        patchCount: processed2.patchCount,
        gridH: processed2.gridH,
        gridW: processed2.gridW,
      });
      let maxDiff = 0;
      for (let i = 0; i < result.embeddings.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(result.embeddings[i]! - result2.embeddings[i]!));
      }
      expect(maxDiff).toBeGreaterThan(0.01);
    } finally {
      await source.close();
    }
  }, 120_000);
});
