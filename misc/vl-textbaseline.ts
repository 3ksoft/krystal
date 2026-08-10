// Text-only baseline: the color prompt with NO image (priors check).
import { NodeFileSource } from "../packages/quant/src/gguf/source-node.ts";
import { installDawn } from "../tests/dawn.ts";
import { createWebGpuDevice } from "../packages/webgpu/src/device.ts";
import { Lfm2GpuModel } from "../packages/webgpu/src/model.ts";
import { Lfm2Forward } from "../packages/webgpu/src/forward.ts";
import { lfm2 } from "../packages/webgpu/src/lfm2.ts";
import { Lfm2Tokenizer } from "../packages/lfm2/src/tokenizer.ts";

await installDawn();
const { device } = await createWebGpuDevice({ label: "vl-text-base" });
await lfm2.engine.compile({ device });
const model = await Lfm2GpuModel.open(device, await NodeFileSource.open("models/LFM2.5-VL-1.6B-WQ4.wq4"), { preload: false });
const forward = new Lfm2Forward(model);
const tok = new Lfm2Tokenizer({ metadata: (k) => { try { return model.metadata(k); } catch { return undefined; } } } as any);
const prompts = [
  "What color is this image? Answer with one word.",
  "What color is a red apple? Answer with one word.",
  "What digit is shown in this image? Answer with a single digit.",
];
for (const p of prompts) {
  const res = await forward.generateGreedy(tok.encodeUserPrompt(p), { maxNewTokens: 8 });
  console.log(`[text-only] ${JSON.stringify(p)} -> "${tok.decode(res.tokens, { skipSpecial: true })}"`);
}
process.exit(0);
