// Reproduces the GUI's strict tokenizer path (forward.tokenizer) against the
// VL WQ4 to confirm the optional add_eos_token read no longer throws.
import { NodeFileSource } from "../packages/quant/src/gguf/source-node.ts";
import { installDawn } from "../tests/dawn.ts";
import { createWebGpuDevice } from "../packages/webgpu/src/device.ts";
import { Lfm2GpuModel } from "../packages/webgpu/src/model.ts";
import { Lfm2Forward } from "../packages/webgpu/src/forward.ts";
import { lfm2 } from "../packages/webgpu/src/lfm2.ts";

await installDawn();
const { device } = await createWebGpuDevice({ label: "vl-strict-tok" });
await lfm2.engine.compile({ device });
const model = await Lfm2GpuModel.open(device, await NodeFileSource.open("models/LFM2.5-VL-1.6B-WQ4.wq4"), { preload: false });
const forward = new Lfm2Forward(model);
const tokenizer = forward.tokenizer; // the exact strict path useEngine.boot takes
console.log("strict tokenizer OK:",
  "addBos =", tokenizer.addBosByDefault,
  "| addEos =", tokenizer.addEosByDefault,
  "| <image> id =", tokenizer.tokenToId.get("<image>"),
  "| <|im_start|> id =", tokenizer.tokenToId.get("<|im_start|>"));
model.destroy();
process.exit(0);
