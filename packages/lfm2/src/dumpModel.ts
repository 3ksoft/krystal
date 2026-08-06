import { GgmlType, type GgufValue } from "../../quant/src/gguf/types";
import { GgufReader } from "../../quant/src/gguf/reader";
import { DenoFileSource } from "../../quant/src/gguf/source";

function asNumber(value: GgufValue, key: string): number {
  if (typeof value === "number") return value;
  if (typeof value === "bigint") {
    const out = Number(value);
    if (!Number.isSafeInteger(out)) throw new Error(`${key} exceeds JS safe integer range`);
    return out;
  }
  throw new Error(`${key} must be numeric`);
}

function asNumberArray(value: GgufValue, key: string): number[] {
  if (!Array.isArray(value)) throw new Error(`${key} must be an array`);
  return value.map((item, index) => asNumber(item, `${key}[${index}]`));
}

function asString(value: GgufValue, key: string): string {
  if (typeof value !== "string") throw new Error(`${key} must be a string`);
  return value;
}

function asStringArray(value: GgufValue, key: string): string[] {
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) {
    throw new Error(`${key} must be a string array`);
  }
  return value as string[];
}

function dirname(path: string): string {
  const normalized = path.replaceAll("\\", "/");
  const slash = normalized.lastIndexOf("/");
  return slash < 0 ? "." : normalized.slice(0, slash) || "/";
}

function emit(value: unknown, indent = 2): string {
  return JSON.stringify(value, null, indent)
    // Keep the generated file valid when embedded in HTML/script tooling too.
    .replaceAll("\u2028", "\\u2028")
    .replaceAll("\u2029", "\\u2029");
}

type LayerKind = "conv" | "attention";

const ggufPath = Deno.args[0];
const outputPath = Deno.args[1] ?? "./lfm2-init.ts";

if (!ggufPath) {
  console.error(
    "Usage: deno run --allow-read --allow-write misc/dump_lfm2_init.ts <model.gguf> [output.ts]",
  );
  Deno.exit(2);
}

const source = await DenoFileSource.open(ggufPath);
try {
  const reader = await GgufReader.open(source);

  const architecture = asString(reader.metadata("general.architecture"), "general.architecture");
  if (architecture !== "lfm2") {
    throw new Error(`Expected lfm2 GGUF, got '${architecture}'`);
  }

  const blockCount = asNumber(reader.metadata("lfm2.block_count"), "lfm2.block_count");
  const hiddenSize = asNumber(reader.metadata("lfm2.embedding_length"), "lfm2.embedding_length");
  const attentionHeads = asNumber(
    reader.metadata("lfm2.attention.head_count"),
    "lfm2.attention.head_count",
  );
  const kvHeadsByLayer = asNumberArray(
    reader.metadata("lfm2.attention.head_count_kv"),
    "lfm2.attention.head_count_kv",
  );
  if (kvHeadsByLayer.length !== blockCount) {
    throw new Error(
      `Expected ${blockCount} entries in lfm2.attention.head_count_kv, got ${kvHeadsByLayer.length}`,
    );
  }

  const layers: LayerKind[] = kvHeadsByLayer.map((heads) => heads === 0 ? "conv" : "attention");
  const attentionLayerSlots = new Array<number>(blockCount).fill(-1);
  let attentionSlot = 0;
  for (let layer = 0; layer < blockCount; layer++) {
    if (layers[layer] === "attention") attentionLayerSlots[layer] = attentionSlot++;
  }

  const vocab = asStringArray(reader.metadata("tokenizer.ggml.tokens"), "tokenizer.ggml.tokens");
  const vocabSizeFromMetadata = asNumber(reader.metadata("lfm2.vocab_size"), "lfm2.vocab_size");
  if (vocab.length !== vocabSizeFromMetadata) {
    throw new Error(
      `Tokenizer has ${vocab.length} tokens but lfm2.vocab_size=${vocabSizeFromMetadata}`,
    );
  }

  const tokenizerModel = asString(reader.metadata("tokenizer.ggml.model"), "tokenizer.ggml.model");
  const tokenizerPre = asString(reader.metadata("tokenizer.ggml.pre"), "tokenizer.ggml.pre");
  const merges = asStringArray(reader.metadata("tokenizer.ggml.merges"), "tokenizer.ggml.merges");
  const tokenTypes = asNumberArray(
    reader.metadata("tokenizer.ggml.token_type"),
    "tokenizer.ggml.token_type",
  );
  if (tokenTypes.length !== vocab.length) {
    throw new Error(
      `Tokenizer has ${vocab.length} tokens but ${tokenTypes.length} token_type entries`,
    );
  }

  const bosToken = asNumber(reader.metadata("tokenizer.ggml.bos_token_id"), "tokenizer.ggml.bos_token_id");
  const eosToken = asNumber(reader.metadata("tokenizer.ggml.eos_token_id"), "tokenizer.ggml.eos_token_id");
  const addBosToken = Boolean(reader.metadata("tokenizer.ggml.add_bos_token"));
  const addEosToken = Boolean(reader.metadata("tokenizer.ggml.add_eos_token"));

  const config = {
    architecture: "lfm2" as const,
    blockCount,
    contextLength: asNumber(reader.metadata("lfm2.context_length"), "lfm2.context_length"),
    hiddenSize,
    feedForwardSize: asNumber(
      reader.metadata("lfm2.feed_forward_length"),
      "lfm2.feed_forward_length",
    ),
    attentionHeads,
    kvHeadsByLayer,
    headDim: hiddenSize / attentionHeads,
    ropeTheta: asNumber(reader.metadata("lfm2.rope.freq_base"), "lfm2.rope.freq_base"),
    vocabSize: vocab.length,
    convCacheLength: asNumber(reader.metadata("lfm2.shortconv.l_cache"), "lfm2.shortconv.l_cache"),
    normEpsilon: asNumber(
      reader.metadata("lfm2.attention.layer_norm_rms_epsilon"),
      "lfm2.attention.layer_norm_rms_epsilon",
    ),
    bosToken,
    eosToken,
    addBosToken,
    addEosToken,
    layers,
    attentionLayerSlots,
    attentionLayerCount: attentionSlot,
  };

  const requiredTensors = ["token_embd.weight", "token_embd_norm.weight"];
  for (let layer = 0; layer < blockCount; layer++) {
    requiredTensors.push(
      `blk.${layer}.attn_norm.weight`,
      `blk.${layer}.ffn_norm.weight`,
      `blk.${layer}.ffn_gate.weight`,
      `blk.${layer}.ffn_up.weight`,
      `blk.${layer}.ffn_down.weight`,
    );
    if (layers[layer] === "conv") {
      requiredTensors.push(
        `blk.${layer}.shortconv.in_proj.weight`,
        `blk.${layer}.shortconv.conv.weight`,
        `blk.${layer}.shortconv.out_proj.weight`,
      );
    } else {
      requiredTensors.push(
        `blk.${layer}.attn_q.weight`,
        `blk.${layer}.attn_k.weight`,
        `blk.${layer}.attn_v.weight`,
        `blk.${layer}.attn_q_norm.weight`,
        `blk.${layer}.attn_k_norm.weight`,
        `blk.${layer}.attn_output.weight`,
      );
    }
  }

  for (const name of requiredTensors) reader.tensor(name);

  const tensors = Object.fromEntries(
    [...reader.info.tensors.values()].map((tensor) => [
      tensor.name,
      {
        dimensions: tensor.dimensions,
        ggmlType: GgmlType[tensor.type] ?? String(tensor.type),
        ggmlTypeId: tensor.type,
        offset: tensor.offset,
        fileOffset: tensor.fileOffset,
        byteLength: tensor.byteLength,
      },
    ]),
  );

  const tokenizer = {
    model: tokenizerModel,
    pre: tokenizerPre,
    bosToken,
    eosToken,
    addBosToken,
    addEosToken,
    tokens: vocab,
    merges,
    tokenTypes,
  };

  const gguf = {
    version: reader.info.version,
    alignment: reader.info.alignment,
    tensorDataOffset: reader.info.tensorDataOffset,
    fileSize: source.size,
  };

  const generated = `// THIS FILE IS AUTO-GENERATED - DO NOT CHANGE\n` +
`// Source: ${JSON.stringify(ggufPath)}\n\n` +
`export type Lfm2LayerKind = "conv" | "attention";\n\n` +
`export interface Lfm2InitConfig {\n` +
`  architecture: "lfm2";\n` +
`  blockCount: number;\n` +
`  contextLength: number;\n` +
`  hiddenSize: number;\n` +
`  feedForwardSize: number;\n` +
`  attentionHeads: number;\n` +
`  kvHeadsByLayer: number[];\n` +
`  headDim: number;\n` +
`  ropeTheta: number;\n` +
`  vocabSize: number;\n` +
`  convCacheLength: number;\n` +
`  normEpsilon: number;\n` +
`  bosToken: number;\n` +
`  eosToken: number;\n` +
`  addBosToken: boolean;\n` +
`  addEosToken: boolean;\n` +
`  layers: Lfm2LayerKind[];\n` +
`  attentionLayerSlots: number[];\n` +
`  attentionLayerCount: number;\n` +
`}\n\n` +
`export interface Lfm2TokenizerDump {\n` +
`  model: string;\n` +
`  pre: string;\n` +
`  bosToken: number;\n` +
`  eosToken: number;\n` +
`  addBosToken: boolean;\n` +
`  addEosToken: boolean;\n` +
`  tokens: string[];\n` +
`  merges: string[];\n` +
`  tokenTypes: number[];\n` +
`}\n\n` +
`export interface Lfm2TensorDump {\n` +
`  dimensions: number[];\n` +
`  ggmlType: string;\n` +
`  ggmlTypeId: number;\n` +
`  offset: number;\n` +
`  fileOffset: number;\n` +
`  byteLength: number;\n` +
`}\n\n` +
`export const LFM2_GGUF = ${emit(gguf)};\n\n` +
`export const LFM2_CONFIG: Lfm2InitConfig = ${emit(config)};\n\n` +
`export const LFM2_REQUIRED_TENSORS: string[] = ${emit(requiredTensors)};\n\n` +
`export const LFM2_TENSORS: Record<string, Lfm2TensorDump> = ${emit(tensors)};\n\n` +
`export const LFM2_TOKENIZER: Lfm2TokenizerDump = ${emit(tokenizer)};\n\n` +
`/** All vocabulary token ids, ready for a u32[] Sandblaster buffer. */\n` +
`export const LFM2_TOKEN_IDS: number[] = Array.from(\n` +
`  { length: LFM2_CONFIG.vocabSize },\n` +
`  (_, tokenId) => tokenId,\n` +
`);\n\n` +
`export const lfm2 = {\n` +
`  gguf: LFM2_GGUF,\n` +
`  config: LFM2_CONFIG,\n` +
`  tokenizer: LFM2_TOKENIZER,\n` +
`  tensors: LFM2_TENSORS,\n` +
`  requiredTensors: LFM2_REQUIRED_TENSORS,\n` +
`  getTokens(): number[] {\n` +
`    return LFM2_TOKEN_IDS;\n` +
`  },\n` +
`  getVocabulary(): string[] {\n` +
`    return LFM2_TOKENIZER.tokens;\n` +
`  },\n` +
`};\n`;

  await Deno.mkdir(dirname(outputPath), { recursive: true });
  await Deno.writeTextFile(outputPath, generated);

  console.log(`[dump-lfm2] ${ggufPath}`);
  console.log(`[dump-lfm2] -> ${outputPath}`);
  console.log(`[dump-lfm2] layers=${blockCount} attention=${attentionSlot} vocab=${vocab.length}`);
  console.log(`[dump-lfm2] tensors=${reader.info.tensors.size} tokenizer-merges=${merges.length}`);
} finally {
  source.close();
}
