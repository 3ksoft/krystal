export enum GgufValueType {
  Uint8 = 0,
  Int8 = 1,
  Uint16 = 2,
  Int16 = 3,
  Uint32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  Uint64 = 10,
  Int64 = 11,
  Float64 = 12,
}

export enum GgmlType {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  I8 = 24,
  I16 = 25,
  I32 = 26,
  I64 = 27,
  F64 = 28,
  BF16 = 30,
}

export type GgufScalar = number | bigint | boolean | string;
export type GgufValue = GgufScalar | GgufValue[];

export interface GgufTensorInfo {
  name: string;
  dimensions: number[];
  type: GgmlType;
  /** Offset relative to tensorDataOffset. */
  offset: number;
  /** Absolute file offset. */
  fileOffset: number;
  /** Exact byte length when known for this tensor type. */
  byteLength: number;
}

export interface GgufFileInfo {
  version: number;
  alignment: number;
  tensorDataOffset: number;
  metadata: Map<string, GgufValue>;
  tensors: Map<string, GgufTensorInfo>;
}
