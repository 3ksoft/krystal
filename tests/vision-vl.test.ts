// Vision-language integration tests (ADA-0009, M3).
//
// Runs the REAL text backbone (WQ4, via Dawn) + the REAL WGSL vision tower
// (F16 mmproj) + the host processor end to end through
// VisionLfm2Session.chat. The M3 gate: an image of a cat → the model says it
// sees a cat (tests/fixtures/cat.jpg, a well-known Commons photograph).
//
//   bun test tests/vision-vl.test.ts
//
// Skips (with a warning) when ./models/LFM2.5-VL-1.6B-WQ4.wq4 or the mmproj
// F16 GGUF is missing.
import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { decode as decodeJpeg } from "jpeg-js";
import { NodeFileSource } from "../packages/quant/src/gguf/source-node.ts";
import { VisionLfm2Session } from "../packages/webgpu/src/vision/integration.ts";
import { installDawn } from "./dawn.ts";
import { createWebGpuDevice } from "../packages/webgpu/src/device.ts";

const TEXT_WQ4 = "models/LFM2.5-VL-1.6B-WQ4.wq4";
const MMPROJ_GGUF = "models/mmproj-LFM2.5-VL-1.6b-F16.gguf";
const MMPROJ_WQ4 = "models/LFM2.5-VL-mmproj-WQ4.wq4";
const CAT_FIXTURE = "tests/fixtures/cat.jpg";
const hasModels = existsSync(TEXT_WQ4) && existsSync(MMPROJ_GGUF);

if (!hasModels) {
  console.warn("[vision-vl] VL models missing — M3 integration tests skipped");
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

const sessions = new Map<string, Promise<VisionLfm2Session>>();

let devicePromise: Promise<GPUDevice> | undefined;

// The Sandblaster LFM2 definition compiles once per process on the FIRST
// device; every session (and Lfm2Forward) must share exactly that device.
async function getDevice(): Promise<GPUDevice> {
  devicePromise ??= (async () => {
    await installDawn();
    const { device } = await createWebGpuDevice({ label: "vision-vl-test" });
    return device;
  })();
  return devicePromise;
}

async function boot(targetSize?: number, visionPath: string = MMPROJ_GGUF): Promise<VisionLfm2Session> {
  const key = `${targetSize ?? ""}:${visionPath}`;
  let promise = sessions.get(key);
  if (!promise) {
    promise = (async () => {
      return VisionLfm2Session.create({
        device: await getDevice(),
        textSource: await NodeFileSource.open(TEXT_WQ4),
        visionSource: await NodeFileSource.open(visionPath),
        ...(targetSize !== undefined ? { targetSize } : {}),
      });
    })();
    sessions.set(key, promise);
  }
  return promise;
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let max = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    max = Math.max(max, Math.abs(a[i]! - b[i]!));
  }
  return max;
}

describe("vision-language integration (M3)", () => {
  test.skipIf(!hasModels)(
    "the <image> placeholder is special token 396 and parses from text",
    async () => {
      const session = await boot();
      expect(session.tokenizer.idToToken[396]).toBe("<image>");
      expect(session.tokenizer.isSpecialToken(396)).toBe(true);
      const ids = session.tokenizer.encodeUserPrompt("What is in the <image>?");
      expect(ids).toContain(396);
      expect(ids[0]).toBe(session.tokenizer.bos);
      expect(session.tokenizer.decode([396], { skipSpecial: true })).toBe("");
    },
    120_000,
  );

  test.skipIf(!hasModels)(
    "tower embeddings match the processed image grid (64x64 @ target 64 -> 4 image tokens)",
    async () => {
      // A target of 64 keeps the grid small (4x4); the default 512 target would
      // upscale a 64x64 image 8x to a 32x32 grid (256 tokens).
      const session = await boot(64);
      const rgba = syntheticRgba(64, 64, 1);
      const { image, embeddings, imageTokens } = await session.embedImage(rgba, 64, 64);
      expect(imageTokens).toBe(4);
      expect(image.gridH).toBe(4);
      expect(image.gridW).toBe(4);
      expect(embeddings.length).toBe(4 * session.textModel.config.hiddenSize);
      let energy = 0;
      for (let i = 0; i < embeddings.length; i++) energy += embeddings[i]! * embeddings[i]!;
      expect(energy).toBeGreaterThan(0); // non-zero: real tower output, not placeholder zeros
    },
    300_000,
  );

  test.skipIf(!hasModels)(
    "different images produce different embeddings (tower input sensitivity)",
    async () => {
      const session = await boot();
      const a = await session.embedImage(syntheticRgba(64, 64, 1), 64, 64);
      const b = await session.embedImage(syntheticRgba(64, 64, 7), 64, 64);
      expect(maxAbsDiff(a.embeddings, b.embeddings)).toBeGreaterThan(1e-2);
    },
    300_000,
  );

  test.skipIf(!hasModels)(
    "injected image changes the LM output vs the placeholder-only baseline",
    async () => {
      const session = await boot();
      const rgba = syntheticRgba(64, 64, 1);
      const { embeddings, imageTokens } = await session.embedImage(rgba, 64, 64);
      // Same token sequence both runs; only the hiddenA rows differ.
      const raw = session.tokenizer.encodeUserPrompt("What is in the <image>?");
      const placeholderId = session.tokenizer.tokenToId.get("<image>")!;
      const expanded: number[] = [];
      let imageStart = -1;
      for (const id of raw) {
        if (id === placeholderId) {
          imageStart = expanded.length;
          for (let i = 0; i < imageTokens; i++) expanded.push(placeholderId);
        } else {
          expanded.push(id);
        }
      }
      const withImage = await session.forward.generateWithImageEmbeddings(expanded, embeddings, {
        imageStart,
        maxNewTokens: 1,
      });
      // Baseline: same sequence, placeholder embeddings left in place (no image).
      const withoutImage = await session.forward.generateGreedy(expanded, { maxNewTokens: 1 });
      expect(withImage.tokens.length).toBe(1);
      expect(withoutImage.tokens.length).toBe(1);
      expect(withImage.tokens[0]).not.toBe(withoutImage.tokens[0]);
      console.log(
        `[vision-vl] injection: image-first=${withImage.tokens[0]} baseline-first=${withoutImage.tokens[0]}`,
      );
    },
    300_000,
  );

  test.skipIf(!hasModels)(
    "chat is deterministic for identical image + prompt (greedy)",
    async () => {
      const session = await boot();
      const rgba = syntheticRgba(64, 64, 1);
      const request = { rgba, width: 64, height: 64, prompt: "Describe this image briefly.", maxNewTokens: 8 };
      const first = await session.chat(request);
      const second = await session.chat(request);
      expect(first.tokens).toEqual(second.tokens);
      expect(first.text.length).toBeGreaterThan(0);
      console.log(`[vision-vl] deterministic text: ${JSON.stringify(first.text)}`);
    },
    300_000,
  );

  test.skipIf(!hasModels || !existsSync(CAT_FIXTURE))(
    "M3 gate: an image of a cat -> the model says it sees a cat",
    async () => {
      const session = await boot();
      const bytes = readFileSync(CAT_FIXTURE);
      const { width, height, data } = decodeJpeg(bytes, { useTArray: true });
      const result = await session.chat({
        rgba: data,
        width,
        height,
        prompt: "What animal is shown in this image? Answer with a single word.",
        maxNewTokens: 24,
      });
      console.log(`[vision-vl] cat gate: ${JSON.stringify(result.text)}`);
      const lower = result.text.toLowerCase();
      expect(lower).toContain("cat");
    },
    600_000,
  );

  test.skipIf(!hasModels || !existsSync(MMPROJ_WQ4) || !existsSync(CAT_FIXTURE))(
    "M3 gate over the WQ4 sidecar: the tower loader dequant path sees a cat",
    async () => {
      // Same gate as above but the vision tower loads from the WQ4 sidecar
      // (models/LFM2.5-VL-mmproj-WQ4.wq4) via the auto-detected host-dequant
      // path — the whole runtime pair is then WQ4, like the deployed setup.
      const session = await boot(undefined, MMPROJ_WQ4);
      const bytes = readFileSync(CAT_FIXTURE);
      const { width, height, data } = decodeJpeg(bytes, { useTArray: true });
      const result = await session.chat({
        rgba: data,
        width,
        height,
        prompt: "What animal is shown in this image? Answer with a single word.",
        maxNewTokens: 24,
      });
      console.log(`[vision-vl] WQ4 cat gate: ${JSON.stringify(result.text)}`);
      const lower = result.text.toLowerCase();
      expect(lower).toContain("cat");
    },
    600_000,
  );
});
