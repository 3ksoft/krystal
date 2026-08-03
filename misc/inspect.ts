import { DenoFileSource } from "../packages/gguf/src/source.ts";
import { GgufReader } from "../packages/gguf/src/reader.ts";
import { GgmlType } from "../packages/gguf/src/types.ts";
import { Lfm2Tokenizer } from "../packages/lfm2/src/tokenizer.ts";

const path = Deno.args[0];
if (!path) {
  console.error("Usage: deno run --allow-read chomato/inspect.ts <model.gguf>");
  Deno.exit(2);
}

const source = await DenoFileSource.open(path);
try {
  const reader = await GgufReader.open(source);
  const get = (key: string) => reader.info.metadata.get(key);
  console.log({
    version: reader.info.version,
    alignment: reader.info.alignment,
    tensors: reader.info.tensors.size,
    architecture: get("general.architecture"),
    blockCount: get("lfm2.block_count"),
    contextLength: get("lfm2.context_length"),
    hiddenSize: get("lfm2.embedding_length"),
    feedForwardSize: get("lfm2.feed_forward_length"),
    attentionHeads: get("lfm2.attention.head_count"),
    kvHeadsByLayer: get("lfm2.attention.head_count_kv"),
    ropeTheta: get("lfm2.rope.freq_base"),
    vocabSize: get("lfm2.vocab_size"),
    convCache: get("lfm2.shortconv.l_cache"),
  });

  const byType = new Map<number, { count: number; bytes: number }>();
  for (const tensor of reader.info.tensors.values()) {
    const item = byType.get(tensor.type) ?? { count: 0, bytes: 0 };
    item.count++;
    item.bytes += tensor.byteLength;
    byType.set(tensor.type, item);
  }
  console.log("Tensor types:");
  for (const [type, stats] of [...byType].sort(([a], [b]) => a - b)) {
    const name = GgmlType[type] ?? String(type);
    console.log(`  ${name.padEnd(8)} ${String(stats.count).padStart(3)} tensors  ${(stats.bytes / 1048576).toFixed(1)} MiB`);
  }

  const tokenizer = new Lfm2Tokenizer(reader);
  const probe = tokenizer.encodeUserPrompt("What is the capital of France?");
  console.log("Tokenizer probe:", probe);
  console.log("Decoded probe:", JSON.stringify(tokenizer.decode(probe, { skipSpecial: false })));
} finally {
  source.close();
}
