// Inspect the LFM2.5-VL GGUF pair: text backbone + mmproj vision tower.
//
// bun run misc/vl-inspect.ts \
//   models/LFM2.5-VL-1.6B-F16.gguf models/mmproj-LFM2.5-VL-1.6b-F16.gguf

import { NodeFileSource } from "../packages/quant/src/gguf/source-node.ts";
import { GgufReader } from "../packages/quant/src/gguf/reader.ts";
import { GgmlType } from "../packages/quant/src/gguf/types.ts";

async function inspect(path: string): Promise<void> {
  console.log(`\n======== ${path} ========`);
  const source = await NodeFileSource.open(path);
  try {
    const reader = await GgufReader.open(source);
    console.log(`GGUF v${reader.info.version}, alignment ${reader.info.alignment}, tensors: ${reader.info.tensors.size}`);
    console.log(`metadata keys (${reader.info.metadata.size}):`);
    for (const [key, value] of reader.info.metadata) {
      const rendered = Array.isArray(value) ? `[array len ${value.length}]` : JSON.stringify(value);
      console.log(`  ${key} = ${rendered}`);
    }
    console.log("tensors:");
    for (const [name, tensor] of reader.info.tensors) {
      console.log(
        `  ${name.padEnd(56)} dims [${tensor.dimensions.join("x")}]  ${(GgmlType[tensor.type] ?? "?").padEnd(6)}  ` +
        `${(tensor.byteLength / 1048576).toFixed(2)} MiB`,
      );
    }
  } finally {
    source.close();
  }
}

for (const path of Deno.args) {
  await inspect(path);
}
