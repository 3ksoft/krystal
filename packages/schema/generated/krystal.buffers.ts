// THIS FILE IS AUTO-GENERATED - DO NOT CHANGE

/**
 * SoA buffer table for `BrainFrameGpu`, derived from the analyzed schema.
 *
 * Do not hand-edit and do not keep a parallel copy: add, remove or resize a
 * buffer by changing `BrainFrameGpu` in the schema and rebuilding.
 */
export interface SoaBufferDescriptor {
  readonly bufferId: number;
  readonly name: string;
  readonly elementCount: number;
  readonly byteSize: number;
}

export const BRAIN_FRAME_GPU_BUFFERS: readonly SoaBufferDescriptor[] = [
  { bufferId: 0, name: "tokenIds", elementCount: 2432, byteSize: 9728 },
  { bufferId: 1, name: "fieldRoles", elementCount: 2432, byteSize: 9728 },
  { bufferId: 2, name: "attentionMask", elementCount: 2432, byteSize: 9728 },
  { bufferId: 3, name: "schemaIds", elementCount: 304, byteSize: 1216 },
  { bufferId: 4, name: "bandIds", elementCount: 304, byteSize: 1216 },
  { bufferId: 5, name: "runtimeRefs", elementCount: 2432, byteSize: 9728 },
  { bufferId: 6, name: "recordFlags", elementCount: 304, byteSize: 1216 },
  { bufferId: 7, name: "activeRecordIndices", elementCount: 304, byteSize: 1216 },
];

export const BINARY_LAYOUT_BUFFER_IDS = {
  tokenIds: 0,
  fieldRoles: 1,
  attentionMask: 2,
  schemaIds: 3,
  bandIds: 4,
  runtimeRefs: 5,
  recordFlags: 6,
  activeRecordIndices: 7,
} as const;

export const BINARY_LAYOUT_BUFFER_COUNT = 8;
