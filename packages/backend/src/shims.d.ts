// Ambient type shims for the scriptc experiment.
//
// scriptc's type environment (es2025 + node types) lacks these DOM globals
// used by packages/quant/src/gguf/source.ts. The @sandblaster/core and
// @schema-pop/* types come from the real linked packages on disk.

interface Blob {
  readonly size: number;
  slice(start?: number, end?: number): Blob;
  arrayBuffer(): Promise<ArrayBuffer>;
}

declare var Blob: {
  prototype: Blob;
  new (parts?: unknown[], options?: unknown): Blob;
};

declare type HeadersInit = readonly (readonly [string, string])[] | Record<string, string>;
