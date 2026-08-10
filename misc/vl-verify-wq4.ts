// Verify a converted WQ4 file: tensor/encoding summary + selected metadata keys.
//
// bun run misc/vl-verify-wq4.ts models/LFM2.5-VL-1.6B-WQ4.wq4 [tensor-name-filter]

import { NodeFileSource } from "../packages/quant/src/gguf/source-node.ts";
import { GgmlType } from "../packages/quant/src/gguf/types.ts";
import { Wq4Reader } from "../packages/quant/src/wq4/reader.ts";

const path = Deno.args[0];
const filter = Deno.args[1];
if (!path) {
  console.error("Usage: deno run --allow-read misc/vl-verify-wq4.ts <model.wq4> [tensor-name-filter]");
  Deno.exit(2);
}

const source = await NodeFileSource.open(path);
try {
  const reader = await Wq4Reader.open(source);
  console.log(`WQ4 v${reader.version}, tensors: ${reader.tensors.size}, self-contained: ${reader.selfContained}`);

  const probeKeys = [
    "general.architecture",
    "clip.vision.embedding_length",
    "clip.vision.block_count",
    "clip.vision.patch_size",
    "clip.vision.feed_forward_length",
    "clip.vision.projector.scale_factor",
    "clip.projector_type",
    "clip.use_gelu",
    "lfm2.block_count",
    "lfm2.embedding_length",
    "lfm2.feed_forward_length",
    "lfm2.vocab_size",
  ];
  console.log("metadata probes:");
  for (const key of probeKeys) {
    if (reader.hasMetadata(key)) {
      console.log(`  ${key} = ${JSON.stringify(reader.metadataValue(key))}`);
    }
  }

  let wq4Bytes = 0;
  let rawBytes = 0;
  let wq4Count = 0;
  let rawCount = 0;
  const typeSummary = new Map<string, { count: number; bytes: number }>();
  for (const tensor of reader.tensors.values()) {
    const sourceName = tensor.sourceType !== undefined ? GgmlType[tensor.sourceType] ?? String(tensor.sourceType) : "?";
    const kind = `${tensor.encoding}/${sourceName}`;
    const item = typeSummary.get(kind) ?? { count: 0, bytes: 0 };
    item.count++;
    item.bytes += tensor.size;
    typeSummary.set(kind, item);
    if (tensor.encoding === "wq4") {
      wq4Bytes += tensor.size;
      wq4Count++;
    } else {
      rawBytes += tensor.size;
      rawCount++;
    }
    if (!filter || tensor.name.includes(filter)) {
      const dims = tensor.dimensions.join("x");
      const ratio = tensor.encoding === "wq4" ? ` (${(tensor.sourceBytes / tensor.size).toFixed(2)}x)` : "";
      console.log(
        `  ${tensor.name.padEnd(52)} [${dims.padEnd(16)}] ${tensor.encoding.padEnd(4)} ` +
        `${(tensor.size / 1048576).toFixed(2)} MiB${ratio}`,
      );
    }
  }
  console.log("summary:");
  for (const [kind, stats] of typeSummary) {
    console.log(`  ${kind.padEnd(10)} ${String(stats.count).padStart(3)} tensors  ${(stats.bytes / 1048576).toFixed(1)} MiB`);
  }
  console.log(`total: ${wq4Count} WQ4 (${(wq4Bytes / 1048576).toFixed(1)} MiB) + ${rawCount} raw (${(rawBytes / 1048576).toFixed(1)} MiB)`);
} finally {
  source.close();
}
