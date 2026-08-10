// Compare our vision tower against llama.cpp's ground-truth embeddings
// (node_869 parsed from mtmd-debug output for a solid red 512x512 image).
// The GPU tower is the exact-f32 mirror of the CPU oracle (M2 differential),
// so it IS the oracle at 1024 patches (the CPU JS path is too slow there).
//   bun run misc/vl-diff869.ts
import { readFileSync } from "node:fs";
import { NodeFileSource } from "../packages/quant/src/gguf/source-node.ts";
import { GgufReader } from "../packages/quant/src/gguf/reader.ts";
import { parseVisionConfig } from "../packages/quant/src/vision/config.ts";
import { loadVisionWeights, visionTensorSource } from "../packages/quant/src/vision/weights.ts";
import { encodeImage } from "../packages/webgpu/src/vision/processor.ts";
import { VisionTower } from "../packages/webgpu/src/vision/tower.ts";
import { installDawn } from "../tests/dawn.ts";
import { createWebGpuDevice } from "../packages/webgpu/src/device.ts";

// llama.cpp ground truth: node_869 rows (dim, tokens) — first 3 + last 3 rows, each first 3 + last 3 tokens.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const gt = JSON.parse(readFileSync("/tmp/node869.json", "utf8"));
// eslint-disable-next-line @typescript-eslint/no-var-requires
const nodes = JSON.parse(readFileSync("/tmp/mtmd-nodes.json", "utf8")) as Record<
  string,
  { dims: number[]; rows: number[][] }
>;

const source = await NodeFileSource.open("models/mmproj-LFM2.5-VL-1.6b-F16.gguf");
const reader = await GgufReader.open(source);
const config = parseVisionConfig((key) => reader.metadata(key));
const weights = await loadVisionWeights(visionTensorSource(reader), config);

await installDawn();
const { device } = await createWebGpuDevice({ label: "vl-diff" });
const tower = await VisionTower.create({ device, config, weights });

// uniform red 512x512, RGBA
const size = 512;
const rgba = new Uint8Array(size * size * 4);
for (let i = 0; i < size * size; i++) {
  rgba[i * 4] = 255; rgba[i * 4 + 1] = 0; rgba[i * 4 + 2] = 0; rgba[i * 4 + 3] = 255;
}

// our standard preprocessing (normalize 0.5/0.5), targetSize 512
const img = encodeImage(rgba, size, size, weights.posEmb, config, { targetSize: 512 });
const embeddings = await tower.run(img);
console.log(`[tower] grid ${img.gridH}x${img.gridW}, tokens ${embeddings.length / config.projectorHiddenSize}`);

function rowOf(emb: Float32Array, dim: number, token: number, stride: number): number {
  return emb[token * stride + dim]!;
}

const gtRows = gt.rows as [number, number, number, number, number, number][];
const gtSum = (gt as { sum: number }).sum;

function compare(
  emb: Float32Array,
  label: string,
  stride: number,
  gtNode: { dims: number[]; rows: number[][] } | null = null,
): void {
  const dimCount = gtNode ? gtNode.dims[0]! : config.projectorHiddenSize;
  const tokenCount = gtNode ? gtNode.dims[1]! : 256;
  const rows = gtNode ? gtNode.rows : (gtRows as unknown as number[][]);
  const sum = gtNode ? undefined : gtSum;
  const dims = [0, 1, 2, dimCount - 3, dimCount - 2, dimCount - 1];
  const tokens = [0, 1, 2, tokenCount - 3, tokenCount - 2, tokenCount - 1];
  let maxDiff = 0;
  for (let d = 0; d < dims.length; d++) {
    for (let t = 0; t < tokens.length; t++) {
      const g = rows[d]![t]!;
      const o = rowOf(emb, dims[d]!, tokens[t]!, stride);
      maxDiff = Math.max(maxDiff, Math.abs(g - o));
    }
  }
  console.log(`[${label}] maxDiff on 36 ground-truth points: ${maxDiff.toExponential(3)}`);
  console.log(`[${label}] our dim0 tokens 0..2: ${[0, 1, 2].map((t) => rowOf(emb, 0, t, stride).toFixed(4)).join(", ")}  gt: ${rows[0]!.slice(0, 3).map((v) => v.toFixed(4)).join(", ")}`);
  if (sum !== undefined) {
    let s = 0;
    for (const v of emb) s += v;
    console.log(`[${label}] our sum: ${s.toFixed(4)}  gt sum: ${sum.toFixed(4)}`);
  }
}

const dim = config.hiddenSize;

// stage 0: patch embedding + pos emb (llama.cpp pos_embed)
const h0 = await tower.run(img, { stopAtLayer: 0 });
compare(h0, "pos_embed", dim, nodes["pos_embed"]!);

// stage 1: after block 0
const h1 = await tower.run(img, { stopAtLayer: 1 });
compare(h1, "layer_out-0", dim, nodes["layer_out-0"]!);

tower.destroy();
process.exit(0);
