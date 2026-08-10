// Verify the <image> placeholder token in the VL text WQ4 tokenizer metadata.
//
// bun run misc/vl-tokenizer-check.ts models/LFM2.5-VL-1.6B-WQ4.wq4

import { NodeFileSource } from "../packages/quant/src/gguf/source-node.ts";
import { Wq4Reader } from "../packages/quant/src/wq4/reader.ts";

const path = Deno.args[0] ?? "models/LFM2.5-VL-1.6B-WQ4.wq4";
const source = await NodeFileSource.open(path);
try {
  const reader = await Wq4Reader.open(source);
  const tokens = reader.metadataValue<string[]>("tokenizer.ggml.tokens");
  const types = reader.metadataValue<number[]>("tokenizer.ggml.token_type");
  console.log(`vocab size: ${tokens.length}`);
  console.log(`token[396] = ${JSON.stringify(tokens[396])}`);
  console.log(`token_type[396] = ${types[396]}  (3=control, 4=user-defined)`);
  const imgIdx = tokens.findIndex((t) => t === "<image>");
  console.log(`first "<image>" index = ${imgIdx}`);
  const imageTokens = tokens
    .map((t, i) => ({ t, i }))
    .filter(({ t }) => t.startsWith("<") && t.includes("image"));
  console.log(`special tokens containing "image": ${imageTokens.map(({ t, i }) => `${t}@${i}`).join(", ")}`);
  console.log(`bos=${reader.metadataValue("tokenizer.ggml.bos_token_id")} eos=${reader.metadataValue("tokenizer.ggml.eos_token_id")}`);
} finally {
  source.close();
}
