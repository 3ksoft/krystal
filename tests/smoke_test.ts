import { BlobSource } from "../packages/gguf/src/source.ts";
import { GgufReader } from "../packages/gguf/src/reader.ts";
import { GgmlType, GgufValueType } from "../packages/gguf/src/types.ts";
import { pretokenizeLfm2 } from "../packages/lfm2/src/tokenizer.ts";

function syntheticGguf(): Blob {
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  const encoder = new TextEncoder();
  // Copy into a fresh ArrayBuffer-backed view: TextEncoder produces
  // Uint8Array<ArrayBufferLike>, which TS 5.7+ rejects as a BlobPart.
  const push = (bytes: Uint8Array) => chunks.push(new Uint8Array(bytes));
  const u32 = (value: number) => {
    const bytes = new Uint8Array(4);
    new DataView(bytes.buffer).setUint32(0, value, true);
    push(bytes);
  };
  const u64 = (value: number) => {
    const bytes = new Uint8Array(8);
    new DataView(bytes.buffer).setBigUint64(0, BigInt(value), true);
    push(bytes);
  };
  const string = (value: string) => {
    const bytes = encoder.encode(value);
    u64(bytes.byteLength);
    push(bytes);
  };

  push(encoder.encode("GGUF"));
  u32(3);
  u64(1); // tensor count
  u64(1); // metadata count
  string("general.architecture");
  u32(GgufValueType.String);
  string("lfm2");

  string("probe");
  u32(2);
  u64(2);
  u64(2);
  u32(GgmlType.F16);
  u64(0);

  const headerBytes = chunks.reduce((sum, x) => sum + x.byteLength, 0);
  const padding = (32 - (headerBytes % 32)) % 32;
  push(new Uint8Array(padding));
  push(Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8));
  return new Blob(chunks);
}

Deno.test("GGUF v3 directory + tensor range", async () => {
  const reader = await GgufReader.open(new BlobSource(syntheticGguf()));
  if (reader.metadata("general.architecture") !== "lfm2") throw new Error("metadata mismatch");
  const tensor = reader.tensor("probe");
  if (tensor.byteLength !== 8 || tensor.dimensions.join(",") !== "2,2") throw new Error("tensor mismatch");
  const bytes = await reader.readTensor(tensor);
  if (bytes.join(",") !== "1,2,3,4,5,6,7,8") throw new Error("payload mismatch");
});

Deno.test("LFM2 pretokenizer preserves source text", () => {
  const text = "Hello, world! 1234\nNext";
  const pieces = pretokenizeLfm2(text);
  if (pieces.join("") !== text) throw new Error(`lost text: ${JSON.stringify(pieces)}`);
});
