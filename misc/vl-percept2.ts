// Perception probe (post patch-bias fix): solid colors, a white digit, and the
// cat at two target sizes, so we can isolate tower bugs from resolution/tiling.
//   bun run misc/vl-percept2.ts
import { readFileSync } from "node:fs";
// @ts-ignore - pngjs has no bundled types; runtime-only probe.
import { PNG } from "pngjs";
import { NodeFileSource } from "../packages/quant/src/gguf/source-node.ts";
import { installDawn } from "../tests/dawn.ts";
import { createWebGpuDevice } from "../packages/webgpu/src/device.ts";
import { VisionLfm2Session } from "../packages/webgpu/src/vision/integration.ts";

async function sessionFor(targetSize: number | undefined): Promise<VisionLfm2Session> {
  const { device } = await createWebGpuDevice({ label: "vl-percept" });
  return VisionLfm2Session.create({
    device,
    textSource: await NodeFileSource.open("models/LFM2.5-VL-1.6B-WQ4.wq4"),
    visionSource: await NodeFileSource.open("models/mmproj-LFM2.5-VL-1.6b-F16.gguf"),
    ...(targetSize ? { targetSize } : {}),
  });
}

function loadPng(path: string): { rgba: Uint8Array; width: number; height: number } {
  const png = PNG.sync.read(readFileSync(path));
  return { rgba: new Uint8Array(png.data.buffer, png.data.byteOffset, png.data.byteLength), width: png.width, height: png.height };
}

const colors: [string, string][] = [
  ["red", "What color is this image? Answer with one word."],
  ["blue", "What color is this image? Answer with one word."],
  ["white", "What color is this image? Answer with one word."],
  ["black", "What color is this image? Answer with one word."],
  ["green", "What color is this image? Answer with one word."],
];

await installDawn();
const session = await sessionFor(512);

for (const [name, prompt] of colors) {
  const img = loadPng(`/tmp/vlpx/${name}.png`);
  const r = await session.chat({ rgba: img.rgba, width: img.width, height: img.height, prompt, maxNewTokens: 8 });
  console.log(`[color ${name}] -> "${r.text}"`);
}

const digit = loadPng("/tmp/vlpx/digit3.png");
const rd = await session.chat({ rgba: digit.rgba, width: digit.width, height: digit.height, prompt: "What digit is shown in this image? Answer with a single digit.", maxNewTokens: 8 });
console.log(`[digit3] -> "${rd.text}"`);

const cat = loadPng("tests/fixtures/cat.jpg".replace(/\.jpg$/, ".png")); // not a png; use jpeg decode below instead
// cat is a jpeg — reuse jpeg-js
const { decode: decodeJpeg } = await import("jpeg-js");
const jpg = decodeJpeg(readFileSync("tests/fixtures/cat.jpg"), { useTArray: true });
const catRgba = new Uint8Array(jpg.data.buffer, jpg.data.byteOffset, jpg.data.byteLength);

const rc = await session.chat({ rgba: catRgba, width: jpg.width, height: jpg.height, prompt: "Describe this image in detail.", maxNewTokens: 48 });
console.log(`[cat @512] -> "${rc.text}"`);
const rc1 = await session.chat({ rgba: catRgba, width: jpg.width, height: jpg.height, prompt: "Is there a cat in this image? Answer with one word.", maxNewTokens: 8 });
console.log(`[cat @512 gate] -> "${rc1.text}"`);

session.destroy();

// high-resolution cat: does detail matter?
const session2 = await sessionFor(1024);
const r2 = await session2.chat({ rgba: catRgba, width: jpg.width, height: jpg.height, prompt: "Describe this image in detail.", maxNewTokens: 48 });
console.log(`[cat @1024] -> "${r2.text}"`);
const r2g = await session2.chat({ rgba: catRgba, width: jpg.width, height: jpg.height, prompt: "Is there a cat in this image? Answer with one word.", maxNewTokens: 8 });
console.log(`[cat @1024 gate] -> "${r2g.text}"`);
session2.destroy();
process.exit(0);
