#!/usr/bin/env bun
// Chat with LFM2.5-VL about one image (ADA-0009, M3 gate runner).
//
//   bun run misc/vl-chat.ts --image tests/fixtures/cat.jpg \
//     --prompt "What animal is this?" [--system "..."] [--max-tokens 64] \
//     [--vision models/LFM2.5-VL-mmproj-WQ4.wq4]
//
// The vision tower defaults to the WQ4 sidecar (models/LFM2.5-VL-mmproj-WQ4.wq4,
// host-dequantized by the shared vision loader); pass --vision with the F16 GGUF
// path to run the exact differential reference instead. Format is auto-detected.
//
// JPEG input is decoded with jpeg-js; PNG with pngjs. RGBA input is composited
// onto a WHITE background before the tower: transparent pixels carry no
// meaningful RGB, and product renders like the nobg asset set are cutouts that
// should read as objects on white (llama.cpp's stb loader would instead drop
// alpha and feed the raw, often black, under-alpha RGB).
import { readFileSync } from "node:fs";
import { decode as decodeJpeg } from "jpeg-js";
// @ts-ignore - pngjs has no bundled types; runtime-only tool.
import { PNG } from "pngjs";

/** Composite RGBA onto white; returns RGB as RGBA with alpha 255. */
function compositeWhite(rgba: Uint8Array): Uint8Array {
  const out = new Uint8Array(rgba.length);
  for (let i = 0; i < rgba.length; i += 4) {
    const a = rgba[i + 3]! / 255;
    for (let c = 0; c < 3; c++) out[i + c] = Math.round(rgba[i + c]! * a + 255 * (1 - a));
    out[i + 3] = 255;
  }
  return out;
}
import { NodeFileSource } from "../packages/quant/src/gguf/source-node.ts";
import { installDawn } from "../tests/dawn.ts";
import { createWebGpuDevice } from "../packages/webgpu/src/device.ts";
import { VisionLfm2Session } from "../packages/webgpu/src/vision/integration.ts";

function arg(name: string): string | undefined {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

const imagePath = arg("image");
const prompt = arg("prompt") ?? "Describe this image.";
const system = arg("system");
const maxNewTokens = Number(arg("max-tokens") ?? 64);
const visionPath = arg("vision") ?? "models/LFM2.5-VL-mmproj-WQ4.wq4";

if (!imagePath) {
  console.error(
    "usage: bun run misc/vl-chat.ts --image <file> [--prompt <text>] [--system <text>] [--max-tokens N] [--vision <mmproj.wq4|.gguf>]",
  );
  process.exit(1);
}
if (!process.env.CHOMATO_MODELS_DIR) {
  // Models live in ./models by default; allow override for other layouts.
}

const t0 = performance.now();
const bytes = readFileSync(imagePath);
let width: number;
let height: number;
let data: Uint8Array;
if (imagePath.toLowerCase().endsWith(".png")) {
  const png = PNG.sync.read(bytes);
  width = png.width;
  height = png.height;
  data = new Uint8Array(png.data.buffer, png.data.byteOffset, png.data.byteLength);
} else {
  const jpg = decodeJpeg(bytes, { useTArray: true });
  width = jpg.width;
  height = jpg.height;
  data = new Uint8Array(jpg.data.buffer, jpg.data.byteOffset, jpg.data.byteLength);
}
data = compositeWhite(data);
console.log(`[vl-chat] decoded ${imagePath}: ${width}x${height} (RGBA composited on white)`);

await installDawn();
const { device } = await createWebGpuDevice({ label: "vl-chat" });
const session = await VisionLfm2Session.create({
  device,
  textSource: await NodeFileSource.open("models/LFM2.5-VL-1.6B-WQ4.wq4"),
  visionSource: await NodeFileSource.open(visionPath),
});
console.log(`[vl-chat] vision tower source: ${visionPath}`);
console.log(`[vl-chat] session ready in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

const t1 = performance.now();
const result = await session.chat({
  rgba: data,
  width,
  height,
  prompt,
  system,
  maxNewTokens,
});
console.log(`[vl-chat] image tokens: ${result.imageTokens} (grid ${result.grid.w}x${result.grid.h}), ` +
  `start @ token ${result.imageStart}, ${result.tokens.length} generated in ` +
  `${((performance.now() - t1) / 1000).toFixed(1)}s`);
console.log(`[vl-chat] tokens: [${result.tokens.join(", ")}]`);
console.log(`[vl-chat] response: ${result.text}`);
session.destroy();
process.exit(0);
