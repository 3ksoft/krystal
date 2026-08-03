// WQ4 v2 prototype sidecar converter.
//
// deno run --allow-read --allow-write convert_gguf_to_wq4.ts input.gguf output.wq4
//
// The GGUF remains the source of model metadata/tokenizer and of small tensors
// that are intentionally kept in F32/F16. This file contains only matrices
// consumed by Chomato's WQ4 embedding/matmul kernels.

import { DenoFileSource } from "./src/gguf/source.ts";
import { GgufReader } from "./src/gguf/reader.ts";
import { GgmlType, type GgufTensorInfo } from "./src/gguf/types.ts";
import {
  WQ4_BLOCK_SIZE,
  WQ4_BYTES_PER_BLOCK,
  WQ4_HEADER_BYTES,
  WQ4_MAGIC,
  WQ4_VERSION,
  type Wq4TensorInfo,
} from "./src/wq4/reader.ts";

const WORDS_PER_BLOCK = WQ4_BYTES_PER_BLOCK / 4; // 4 × packed u32 + 1 × i32 exponent
const MIN_EXP = -24;
const MAX_EXP = 8;
const ZERO_WORD = 0x88888888; // nibble 8 decodes to signed quantized value 0

function clampExp(exp: number): number {
  return Math.max(MIN_EXP, Math.min(MAX_EXP, exp));
}

function decodeF16(u16: number): number {
  const exponent = (u16 >> 10) & 0x1f;
  const fraction = u16 & 0x03ff;
  const sign = u16 & 0x8000 ? -1 : 1;

  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

function quantizedValue(value: number, scale: number): number {
  return Math.max(-8, Math.min(7, Math.round(value / scale)));
}

/**
 * Quantize 32 weights as 5 × u32:
 *   [0..3] packed 4-bit signed values, encoded -8..+7 -> 0..15
 *   [4]    signed power-of-two exponent, scale = 2^exp
 *
 * The exponent is selected offline from the legacy scale plus the two nearest
 * power-of-two scales around the asymmetric no-clipping scale. We choose the
 * lowest-MSE candidate for this block, so quality can only improve relative to
 * the prototype while runtime cost stays identical.
 */
function quantizeBlockWq4(
  f32Data: Float32Array,
  startIdx: number,
  outWords: Uint32Array,
  outWordOffset: number,
): void {
  const endIdx = Math.min(startIdx + WQ4_BLOCK_SIZE, f32Data.length);
  let maxPositive = 0;
  let minNegative = 0;

  for (let i = startIdx; i < endIdx; i++) {
    const value = f32Data[i]!;
    if (!Number.isFinite(value)) throw new Error(`Non-finite weight at element ${i}: ${value}`);
    if (value > maxPositive) maxPositive = value;
    if (value < minNegative) minNegative = value;
  }

  if (maxPositive === 0 && minNegative === 0) {
    outWords[outWordOffset + 0] = ZERO_WORD;
    outWords[outWordOffset + 1] = ZERO_WORD;
    outWords[outWordOffset + 2] = ZERO_WORD;
    outWords[outWordOffset + 3] = ZERO_WORD;
    outWords[outWordOffset + 4] = MIN_EXP >>> 0;
    return;
  }

  // Positive q has +7 max while negative q has -8 min.
  const maxAbs = Math.max(maxPositive, -minNegative);
  const legacyExp = clampExp(Math.floor(Math.log2(maxAbs / 7)));
  const noClipScale = Math.max(maxPositive / 7, (-minNegative) / 8, 2 ** MIN_EXP);
  const exactExp = Math.log2(noClipScale);
  const expLo = clampExp(Math.floor(exactExp));
  const expHi = clampExp(Math.ceil(exactExp));
  const scaleLegacy = 2 ** legacyExp;
  const scaleLo = 2 ** expLo;
  const scaleHi = 2 ** expHi;

  let sseLegacy = 0;
  let sseLo = 0;
  let sseHi = 0;
  for (let i = startIdx; i < endIdx; i++) {
    const value = f32Data[i]!;
    const qLegacy = quantizedValue(value, scaleLegacy);
    const qLo = quantizedValue(value, scaleLo);
    const qHi = quantizedValue(value, scaleHi);
    const eLegacy = value - qLegacy * scaleLegacy;
    const eLo = value - qLo * scaleLo;
    const eHi = value - qHi * scaleHi;
    sseLegacy += eLegacy * eLegacy;
    sseLo += eLo * eLo;
    sseHi += eHi * eHi;
  }

  let exp = legacyExp;
  let bestSse = sseLegacy;
  if (sseLo < bestSse) { exp = expLo; bestSse = sseLo; }
  if (sseHi < bestSse) { exp = expHi; }
  const scale = 2 ** exp;

  for (let wordIndex = 0; wordIndex < 4; wordIndex++) {
    let word = 0;
    for (let nibbleIndex = 0; nibbleIndex < 8; nibbleIndex++) {
      const idx = startIdx + wordIndex * 8 + nibbleIndex;
      const value = idx < endIdx ? f32Data[idx]! : 0;
      const q = quantizedValue(value, scale);
      word |= ((q + 8) & 0x0f) << (nibbleIndex * 4);
    }
    outWords[outWordOffset + wordIndex] = word >>> 0;
  }
  outWords[outWordOffset + 4] = exp >>> 0;
}

function shouldQuantize(info: GgufTensorInfo): { ok: true } | { ok: false; reason: string } {
  if (info.dimensions.length < 2) return { ok: false, reason: "not a matrix" };
  if (info.type !== GgmlType.F16 && info.type !== GgmlType.F32) {
    return { ok: false, reason: `source type ${GgmlType[info.type]}` };
  }
  // This tensor is consumed directly by shortconv_* as scalar F32 weights,
  // not through matrix()/embedding(). Keep it in the GGUF.
  if (info.name.endsWith(".shortconv.conv.weight")) {
    return { ok: false, reason: "direct shortconv kernel weight" };
  }
  const rowElements = info.dimensions[0] ?? 1;
  if (rowElements % WQ4_BLOCK_SIZE !== 0) {
    return { ok: false, reason: `row width ${rowElements} is not divisible by ${WQ4_BLOCK_SIZE}` };
  }
  return { ok: true };
}

async function writeAll(file: Deno.FsFile, bytes: Uint8Array): Promise<void> {
  let offset = 0;
  while (offset < bytes.byteLength) {
    const written = await file.write(bytes.subarray(offset));
    if (written <= 0) throw new Error("Short write while creating WQ4 file");
    offset += written;
  }
}

const inputPath = Deno.args[0] ?? "./models/LFM2.5-1.2B-Instruct-F16.gguf";
const outputPath = Deno.args[1] ?? "./models/LFM2.5-1.2B-Instruct-WQ4.wq4";

console.log(`[WQ4 v${WQ4_VERSION}] Otwieranie: ${inputPath}`);
const source = await DenoFileSource.open(inputPath);

try {
  const reader = await GgufReader.open(source);
  console.log(`[WQ4] GGUF v${reader.info.version}, tensorów: ${reader.info.tensors.size}`);

  const outFile = await Deno.open(outputPath, { create: true, write: true, truncate: true });
  try {
    // Header is patched after writing the data + JSON index.
    await writeAll(outFile, new Uint8Array(WQ4_HEADER_BYTES));

    let currentOffset = WQ4_HEADER_BYTES;
    let totalOriginal = 0;
    let totalQuantized = 0;
    const tensorInfos: Wq4TensorInfo[] = [];

    for (const [name, tensor] of reader.info.tensors.entries()) {
      const decision = shouldQuantize(tensor);
      if ("reason" in decision) {
        console.log(`  [Keep] ${name.padEnd(42)} ${decision.reason}`);
        continue;
      }

      const rawBytes = await reader.readTensor(tensor);
      const rawCopy = new Uint8Array(rawBytes);
      let f32Data: Float32Array;

      if (tensor.type === GgmlType.F16) {
        const u16 = new Uint16Array(rawCopy.buffer, rawCopy.byteOffset, rawCopy.byteLength / 2);
        f32Data = new Float32Array(u16.length);
        for (let i = 0; i < u16.length; i++) f32Data[i] = decodeF16(u16[i]!);
      } else {
        // rawCopy owns an aligned ArrayBuffer, so this view is safe regardless of
        // the GGUF tensor's original file alignment.
        f32Data = new Float32Array(rawCopy.buffer, rawCopy.byteOffset, rawCopy.byteLength / 4);
      }

      const blockCount = f32Data.length / WQ4_BLOCK_SIZE;
      if (!Number.isInteger(blockCount)) throw new Error(`${name}: internal WQ4 block alignment error`);
      const quantBuffer = new Uint32Array(blockCount * WORDS_PER_BLOCK);

      for (let b = 0; b < blockCount; b++) {
        quantizeBlockWq4(f32Data, b * WQ4_BLOCK_SIZE, quantBuffer, b * WORDS_PER_BLOCK);
      }

      const quantBytes = new Uint8Array(quantBuffer.buffer);
      await writeAll(outFile, quantBytes);

      tensorInfos.push({
        name,
        dimensions: [...tensor.dimensions],
        offset: currentOffset,
        size: quantBytes.byteLength,
        sourceBytes: rawBytes.byteLength,
      });
      currentOffset += quantBytes.byteLength;
      totalOriginal += rawBytes.byteLength;
      totalQuantized += quantBytes.byteLength;

      const ratio = rawBytes.byteLength / quantBytes.byteLength;
      console.log(
        `  [WQ4] ${name.padEnd(42)} ${(rawBytes.byteLength / 1e6).toFixed(1)} MB -> ` +
        `${(quantBytes.byteLength / 1e6).toFixed(1)} MB (${ratio.toFixed(2)}x)`,
      );
    }

    const indexOffset = currentOffset;
    const indexBytes = new TextEncoder().encode(JSON.stringify({ tensors: tensorInfos }));
    await writeAll(outFile, indexBytes);

    const header = new ArrayBuffer(WQ4_HEADER_BYTES);
    const view = new DataView(header);
    view.setUint32(0, WQ4_MAGIC, true);
    view.setUint32(4, WQ4_VERSION, true);
    view.setUint32(8, tensorInfos.length, true);
    view.setUint32(12, WQ4_BLOCK_SIZE, true);
    view.setBigUint64(16, BigInt(indexOffset), true);
    view.setBigUint64(24, BigInt(indexBytes.byteLength), true);
    await outFile.seek(0, Deno.SeekMode.Start);
    await writeAll(outFile, new Uint8Array(header));

    console.log("=================================================");
    console.log(`Gotowe. Skwantyzowano ${tensorInfos.length} tensorów.`);
    console.log(`Źródło objęte WQ4: ${(totalOriginal / 1e9).toFixed(2)} GB`);
    console.log(`WQ4:              ${(totalQuantized / 1e9).toFixed(2)} GB`);
    console.log(`Kompresja:         ${(totalOriginal / totalQuantized).toFixed(2)}x`);
    console.log(`Index:              ${(indexBytes.byteLength / 1024).toFixed(1)} KiB`);
    console.log("=================================================");
  } finally {
    outFile.close();
  }
} finally {
  source.close();
}
