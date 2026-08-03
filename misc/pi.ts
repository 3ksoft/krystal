import { Sandblaster } from "./src/sandblaster.ts";
import { Lfm2Model } from "./src/lfm2/model.ts";
import { Lfm2Runtime } from "./src/lfm2/runtime.ts";
import { $ } from "./src/lfm2/schema.ts";
import { Lfm2Tokenizer } from "./src/lfm2/tokenizer.ts";

function usage(): never {
  console.error(`Usage:
  deno run --unstable-webgpu --allow-read chomato/pi.ts <model-f16.gguf> [prompt]

Example:
  deno run --unstable-webgpu --allow-read chomato/pi.ts \\
    ./models/LFM2.5-1.2B-Instruct-F16.gguf \\
    "What is the capital of France?"`);
  Deno.exit(2);
}

const args = [...Deno.args];
const modelPath = args.shift() ?? usage();
let wq4Path: string | undefined;
const wq4Index = args.indexOf("--wq4");
if (wq4Index >= 0) {
  wq4Path = args[wq4Index + 1] ?? usage();
  args.splice(wq4Index, 2);
}
const prompt = args.join(" ") || "What is the capital of France?";

console.log("Acquiring WebGPU device...");
const GIB = 1024 * 1024 * 1024;
const requiredLimits: Record<string, number> = {
  maxBufferSize: GIB,
  maxStorageBufferBindingSize: GIB,
  maxComputeWorkgroupsPerDimension: 65535,
};
const engine = await Sandblaster.create($).compile({ requiredLimits } as any);

let lastPercent = -1;
console.log(`Loading ${modelPath}${wq4Path ? ` + ${wq4Path}` : ""}...`);
console.log("WebGPU limits:", {
  maxBufferSizeMiB: Math.round(Number(engine.device.limits.maxBufferSize) / 1048576),
  maxStorageBindingMiB: Math.round(Number(engine.device.limits.maxStorageBufferBindingSize) / 1048576),
  maxComputeWorkgroupsPerDimension: engine.device.limits.maxComputeWorkgroupsPerDimension,
});
const model = await Lfm2Model.loadPath(engine.device, modelPath, {
  wq4Path,
  maxPageBytes: 64 * 1024 * 1024,
  drainUploads: true,
  onProgress(progress) {
    const percent = Math.floor((progress.uploadedBytes / progress.totalBytes) * 100);
    if (percent !== lastPercent && (percent % 5 === 0 || percent === 100)) {
      lastPercent = percent;
      console.log(`  GGUF ${percent}%  allocated ${(progress.allocatedBytes / 1073741824).toFixed(2)} GiB  ${progress.tensor}`);
    }
  },
});

console.log(
  `LFM2: ${model.config.blockCount} layers, ${model.config.hiddenSize} hidden, ` +
  `${model.config.feedForwardSize} FF, ${model.config.vocabSize} vocab`,
);
console.log(`Layer plan: ${model.config.layers.map((x) => x === "attention" ? "A" : "C").join("")}`);

const tokenizer = new Lfm2Tokenizer(model.reader);
const promptTokens = tokenizer.encodeUserPrompt(prompt);
console.log(`Prompt tokens: ${promptTokens.length}`);
if (promptTokens.length > 512) {
  throw new Error("Bring-up runtime currently caps context at 512 tokens. Use a shorter prompt.");
}

const runtime = await Lfm2Runtime.create(engine, model, {
  contextCapacity: 512,
  maxNewTokens: 32,
});

console.log(`Prompt: ${prompt}`);
const started = performance.now();
const result = await runtime.generateTokens(promptTokens, { maxNewTokens: 32 });
const elapsed = performance.now() - started;

console.log("\n--- GPU output ---");
console.log(tokenizer.decode(result.tokenIds));
console.log("------------------");
console.log({
  generated: result.state.generatedCount,
  lastToken: result.state.lastToken,
  status: result.state.status,
  telemetryRevision: result.state.telemetryRevision,
  elapsedMs: Number(elapsed.toFixed(1)),
});

runtime.destroy();
model.destroy();
