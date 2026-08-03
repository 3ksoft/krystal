import { GgmlType, GgufValueType, type GgufFileInfo, type GgufTensorInfo, type GgufValue } from "./types.ts";
import type { RandomAccessSource } from "./source.ts";

const DEFAULT_ALIGNMENT = 32;
const CURSOR_CHUNK = 1 << 20;

function asSafeNumber(value: bigint, label: string): number {
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`${label} exceeds JS safe integer range: ${value}`);
  }
  return Number(value);
}

function align(value: number, alignment: number): number {
  return value + ((alignment - (value % alignment)) % alignment);
}

function product(values: readonly number[]): number {
  let n = 1;
  for (const v of values) n *= v;
  return n;
}

export function exactTensorByteLength(type: GgmlType, dimensions: readonly number[]): number | null {
  const elements = product(dimensions);
  switch (type) {
    case GgmlType.F32: return elements * 4;
    case GgmlType.F16:
    case GgmlType.BF16: return elements * 2;
    case GgmlType.I8: return elements;
    case GgmlType.I16: return elements * 2;
    case GgmlType.I32: return elements * 4;
    case GgmlType.I64:
    case GgmlType.F64: return elements * 8;
    default: return null;
  }
}

class BufferedCursor {
  position = 0;
  private buffer: Uint8Array = new Uint8Array();
  private bufferOffset = 0;
  private decoder = new TextDecoder();

  constructor(private readonly source: RandomAccessSource) {}

  private async ensure(length: number): Promise<void> {
    const local = this.position - this.bufferOffset;
    if (local >= 0 && local + length <= this.buffer.length) return;
    const wanted = Math.max(CURSOR_CHUNK, length);
    const available = Math.min(wanted, this.source.size - this.position);
    this.buffer = await this.source.read(this.position, available);
    this.bufferOffset = this.position;
  }

  private async bytes(length: number): Promise<Uint8Array> {
    await this.ensure(length);
    const local = this.position - this.bufferOffset;
    const out = this.buffer.subarray(local, local + length);
    this.position += length;
    return out;
  }

  private async view(length: number): Promise<DataView> {
    const bytes = await this.bytes(length);
    return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }

  async u8(): Promise<number> { return (await this.view(1)).getUint8(0); }
  async i8(): Promise<number> { return (await this.view(1)).getInt8(0); }
  async u16(): Promise<number> { return (await this.view(2)).getUint16(0, true); }
  async i16(): Promise<number> { return (await this.view(2)).getInt16(0, true); }
  async u32(): Promise<number> { return (await this.view(4)).getUint32(0, true); }
  async i32(): Promise<number> { return (await this.view(4)).getInt32(0, true); }
  async f32(): Promise<number> { return (await this.view(4)).getFloat32(0, true); }
  async u64(): Promise<bigint> { return (await this.view(8)).getBigUint64(0, true); }
  async i64(): Promise<bigint> { return (await this.view(8)).getBigInt64(0, true); }
  async f64(): Promise<number> { return (await this.view(8)).getFloat64(0, true); }

  async string(): Promise<string> {
    const length = asSafeNumber(await this.u64(), "GGUF string length");
    return this.decoder.decode(await this.bytes(length));
  }

  async value(type: GgufValueType): Promise<GgufValue> {
    switch (type) {
      case GgufValueType.Uint8: return this.u8();
      case GgufValueType.Int8: return this.i8();
      case GgufValueType.Uint16: return this.u16();
      case GgufValueType.Int16: return this.i16();
      case GgufValueType.Uint32: return this.u32();
      case GgufValueType.Int32: return this.i32();
      case GgufValueType.Float32: return this.f32();
      case GgufValueType.Bool: {
        const value = await this.u8();
        if (value !== 0 && value !== 1) throw new Error(`Invalid GGUF bool value ${value}`);
        return value === 1;
      }
      case GgufValueType.String: return this.string();
      case GgufValueType.Array: {
        const elementType = await this.u32() as GgufValueType;
        const length = asSafeNumber(await this.u64(), "GGUF array length");
        const out: GgufValue[] = new Array(length);
        for (let i = 0; i < length; i++) out[i] = await this.value(elementType);
        return out;
      }
      case GgufValueType.Uint64: return this.u64();
      case GgufValueType.Int64: return this.i64();
      case GgufValueType.Float64: return this.f64();
      default: throw new Error(`Unsupported GGUF metadata value type ${type}`);
    }
  }
}

export class GgufReader {
  private constructor(
    readonly source: RandomAccessSource,
    readonly info: GgufFileInfo,
  ) {}

  static async open(source: RandomAccessSource): Promise<GgufReader> {
    const cursor = new BufferedCursor(source);
    const magic = await source.read(0, 4);
    if (String.fromCharCode(...magic) !== "GGUF") throw new Error("Not a GGUF file");
    cursor.position = 4;

    const version = await cursor.u32();
    if (version !== 3) throw new Error(`Only GGUF v3 is supported for now, got v${version}`);

    const tensorCount = asSafeNumber(await cursor.u64(), "tensor count");
    const metadataCount = asSafeNumber(await cursor.u64(), "metadata count");
    const metadata = new Map<string, GgufValue>();

    for (let i = 0; i < metadataCount; i++) {
      const key = await cursor.string();
      const type = await cursor.u32() as GgufValueType;
      metadata.set(key, await cursor.value(type));
    }

    const rawTensors: Array<Omit<GgufTensorInfo, "fileOffset" | "byteLength">> = [];
    for (let i = 0; i < tensorCount; i++) {
      const name = await cursor.string();
      const nDimensions = await cursor.u32();
      const dimensions: number[] = [];
      for (let d = 0; d < nDimensions; d++) {
        dimensions.push(asSafeNumber(await cursor.u64(), `${name} dimension`));
      }
      const type = await cursor.u32() as GgmlType;
      const offset = asSafeNumber(await cursor.u64(), `${name} offset`);
      rawTensors.push({ name, dimensions, type, offset });
    }

    const rawAlignment = metadata.get("general.alignment");
    const alignment = typeof rawAlignment === "number" ? rawAlignment : DEFAULT_ALIGNMENT;
    const tensorDataOffset = align(cursor.position, alignment);

    const sorted = [...rawTensors].sort((a, b) => a.offset - b.offset);
    const extents = new Map<string, number>();
    for (let i = 0; i < sorted.length; i++) {
      const current = sorted[i]!;
      const nextOffset = sorted[i + 1]?.offset ?? (source.size - tensorDataOffset);
      extents.set(current.name, nextOffset - current.offset);
    }

    const tensors = new Map<string, GgufTensorInfo>();
    for (const tensor of rawTensors) {
      const exact = exactTensorByteLength(tensor.type, tensor.dimensions);
      const extent = extents.get(tensor.name)!;
      const byteLength = exact ?? extent;
      if (byteLength > extent) {
        throw new Error(`${tensor.name}: calculated ${byteLength} bytes exceeds GGUF extent ${extent}`);
      }
      tensors.set(tensor.name, {
        ...tensor,
        fileOffset: tensorDataOffset + tensor.offset,
        byteLength,
      });
    }

    return new GgufReader(source, {
      version,
      alignment,
      tensorDataOffset,
      metadata,
      tensors,
    });
  }

  tensor(name: string): GgufTensorInfo {
    const tensor = this.info.tensors.get(name);
    if (!tensor) throw new Error(`GGUF tensor not found: ${name}`);
    return tensor;
  }

  metadata<T extends GgufValue = GgufValue>(key: string): T {
    if (!this.info.metadata.has(key)) throw new Error(`GGUF metadata not found: ${key}`);
    return this.info.metadata.get(key)! as T;
  }

  readTensor(tensor: string | GgufTensorInfo, offset = 0, length?: number): Promise<Uint8Array> {
    const info = typeof tensor === "string" ? this.tensor(tensor) : tensor;
    const size = length ?? (info.byteLength - offset);
    return this.source.read(info.fileOffset + offset, size);
  }
}
