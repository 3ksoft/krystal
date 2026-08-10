// Which tokenizer.* metadata keys exist in the WQ4 containers?
import { Wq4Reader } from "../packages/quant/src/wq4/reader.ts";
import { NodeFileSource } from "../packages/quant/src/gguf/source-node.ts";

for (const path of [
  "models/LFM2.5-1.2B-Instruct-WQ4.wq4",
  "models/LFM2.5-VL-1.6B-WQ4.wq4",
]) {
  const source = await NodeFileSource.open(path);
  const reader = await Wq4Reader.open(source);
  const keys = Object.keys((reader as unknown as { metadataMap: Record<string, unknown> }).metadataMap)
    .filter((k) => k.startsWith("tokenizer.ggml."))
    .sort();
  console.log(`== ${path}`);
  console.log("  tokenizer keys:", keys.join(", ") || "(none)");
  source.close();
}
process.exit(0);
