import type { RandomAccessSource } from "../gguf/source.ts";

export const WQ4_MAGIC = 0x00345157; // "WQ4\0" little-endian
export const WQ4_VERSION = 2;
export const WQ4_BLOCK_SIZE = 32;
export const WQ4_BYTES_PER_BLOCK = 20;
export const WQ4_HEADER_BYTES = 32;

export interface Wq4TensorInfo {
  name: string;
  dimensions: number[];
  offset: number;
  size: number;
  sourceBytes: number;
}

export interface Wq4SourceTensorInfo {
  name: string;
  type: string;
  dimensions: number[];
  quantizedToWq4: boolean;
  reasonIfNotQuantized?: string;
}

interface Wq4IndexJson {
  version?: number;
  metadata?: Record<string, unknown>;
  tensors: Wq4TensorInfo[];
  ggufTensors?: Wq4SourceTensorInfo[];
}

function asSafeNumber(value: bigint, label: string): number {
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`${label} exceeds JS safe integer range: ${value}`);
  }
  return Number(value);
}

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((value, i) => value === b[i]);
}

export class Wq4Reader {
  readonly tensors = new Map<string, Wq4TensorInfo>();
  readonly sourceTensors = new Map<string, Wq4SourceTensorInfo>();

  private constructor(
    readonly source: RandomAccessSource,
    readonly tensorCount: number,
    readonly metadata: Readonly<Record<string, unknown>>,
  ) {}

  static async open(source: RandomAccessSource): Promise<Wq4Reader> {
    if (source.size < WQ4_HEADER_BYTES) throw new Error(`WQ4 file is too small: ${source.size} bytes`);

    const header = await source.read(0, WQ4_HEADER_BYTES);
    const view = new DataView(header.buffer, header.byteOffset, header.byteLength);
    const magic = view.getUint32(0, true);
    const version = view.getUint32(4, true);
    const tensorCount = view.getUint32(8, true);
    const blockSize = view.getUint32(12, true);
    const indexOffset = asSafeNumber(view.getBigUint64(16, true), "WQ4 index offset");
    const indexSize = asSafeNumber(view.getBigUint64(24, true), "WQ4 index size");

    if (magic !== WQ4_MAGIC) {
      throw new Error(`Not a WQ4 v2 file (magic 0x${magic.toString(16).padStart(8, "0")})`);
    }
    if (version !== WQ4_VERSION) throw new Error(`Unsupported WQ4 version ${version}; expected ${WQ4_VERSION}`);
    if (blockSize !== WQ4_BLOCK_SIZE) throw new Error(`Unsupported WQ4 block size ${blockSize}; expected ${WQ4_BLOCK_SIZE}`);
    if (indexOffset < WQ4_HEADER_BYTES || indexSize <= 0 || indexOffset + indexSize > source.size) {
      throw new Error(`Invalid WQ4 index range [${indexOffset}, ${indexOffset + indexSize}) for ${source.size}-byte file`);
    }

    const indexBytes = await source.read(indexOffset, indexSize);
    let parsed: Wq4IndexJson;
    try {
      parsed = JSON.parse(new TextDecoder().decode(indexBytes)) as Wq4IndexJson;
    } catch (error) {
      throw new Error(`Invalid WQ4 index JSON: ${error instanceof Error ? error.message : String(error)}`);
    }
    if (!parsed || !Array.isArray(parsed.tensors)) throw new Error("Invalid WQ4 index: missing tensors array");
    if (parsed.version !== undefined && parsed.version !== version) {
      throw new Error(`WQ4 index version ${parsed.version} does not match header version ${version}`);
    }
    if (parsed.metadata !== undefined && (parsed.metadata === null || Array.isArray(parsed.metadata) || typeof parsed.metadata !== "object")) {
      throw new Error("Invalid WQ4 index: metadata must be an object");
    }
    if (parsed.ggufTensors !== undefined && !Array.isArray(parsed.ggufTensors)) {
      throw new Error("Invalid WQ4 index: ggufTensors must be an array");
    }
    if (parsed.tensors.length !== tensorCount) {
      throw new Error(`WQ4 header says ${tensorCount} tensors, index contains ${parsed.tensors.length}`);
    }

    const reader = new Wq4Reader(source, tensorCount, Object.freeze({ ...(parsed.metadata ?? {}) }));
    for (const entry of parsed.tensors) {
      if (!entry || typeof entry.name !== "string" || !Array.isArray(entry.dimensions)) {
        throw new Error("Invalid WQ4 tensor index entry");
      }
      if (!Number.isSafeInteger(entry.offset) || !Number.isSafeInteger(entry.size) || entry.offset < WQ4_HEADER_BYTES || entry.size <= 0) {
        throw new Error(`${entry.name}: invalid WQ4 byte range`);
      }
      if (entry.offset + entry.size > indexOffset) {
        throw new Error(`${entry.name}: WQ4 tensor overlaps the index or exceeds the data section`);
      }
      if (reader.tensors.has(entry.name)) throw new Error(`Duplicate WQ4 tensor '${entry.name}'`);
      reader.tensors.set(entry.name, {
        name: entry.name,
        dimensions: entry.dimensions.map(Number),
        offset: entry.offset,
        size: entry.size,
        sourceBytes: Number(entry.sourceBytes ?? 0),
      });
    }

    for (const entry of parsed.ggufTensors ?? []) {
      if (
        !entry ||
        typeof entry.name !== "string" ||
        typeof entry.type !== "string" ||
        !Array.isArray(entry.dimensions) ||
        typeof entry.quantizedToWq4 !== "boolean"
      ) {
        throw new Error("Invalid WQ4 source tensor index entry");
      }
      if (reader.sourceTensors.has(entry.name)) throw new Error(`Duplicate WQ4 source tensor '${entry.name}'`);
      reader.sourceTensors.set(entry.name, {
        name: entry.name,
        type: entry.type,
        dimensions: entry.dimensions.map(Number),
        quantizedToWq4: entry.quantizedToWq4,
        ...(entry.reasonIfNotQuantized !== undefined
          ? { reasonIfNotQuantized: String(entry.reasonIfNotQuantized) }
          : {}),
      });
    }

    for (const [name, sourceTensor] of reader.sourceTensors) {
      const hasWq4Tensor = reader.tensors.has(name);
      if (sourceTensor.quantizedToWq4 !== hasWq4Tensor) {
        throw new Error(
          `${name}: ggufTensors says quantizedToWq4=${sourceTensor.quantizedToWq4}, ` +
          `but WQ4 tensor entry ${hasWq4Tensor ? "exists" : "is missing"}`,
        );
      }
    }

    return reader;
  }

  hasMetadata(key: string): boolean {
    return Object.hasOwn(this.metadata, key);
  }

  metadataValue<T = unknown>(key: string): T {
    if (!this.hasMetadata(key)) throw new Error(`WQ4 metadata not found: ${key}`);
    return this.metadata[key] as T;
  }

  sourceTensor(name: string): Wq4SourceTensorInfo | undefined {
    return this.sourceTensors.get(name);
  }

  tensor(name: string): Wq4TensorInfo | undefined {
    return this.tensors.get(name);
  }

  requireTensor(name: string, dimensions?: readonly number[]): Wq4TensorInfo {
    const info = this.tensors.get(name);
    if (!info) throw new Error(`WQ4 tensor not found: ${name}`);
    if (dimensions && !sameShape(info.dimensions, dimensions)) {
      throw new Error(`${name}: WQ4 shape [${info.dimensions.join(", ")}] != GGUF shape [${dimensions.join(", ")}]`);
    }
    return info;
  }

  readTensor(tensor: string | Wq4TensorInfo, offset = 0, length?: number): Promise<Uint8Array> {
    const info = typeof tensor === "string" ? this.requireTensor(tensor) : tensor;
    const size = length ?? (info.size - offset);
    if (offset < 0 || size < 0 || offset + size > info.size) {
      throw new RangeError(`${info.name}: WQ4 read [${offset}, ${offset + size}) outside tensor size ${info.size}`);
    }
    return this.source.read(info.offset + offset, size);
  }
}
