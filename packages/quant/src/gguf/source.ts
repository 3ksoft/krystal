/**
 * The random-access contract every GGUF/WQ4 reader consumes.
 *
 * Runtime-free on purpose. The concrete sources live in sibling modules
 * (source-deno.ts, source-web.ts) because each one drags in globals its
 * runtime does not share: a native build has no `Blob`, `HeadersInit` or
 * `fetch`, and a browser build has no `Deno`. Keeping them together made the
 * whole reader stack untypeable outside a runtime that has all of them.
 *
 * Import this module for the interface; import a concrete source only from
 * code that already commits to that runtime. The `./index.ts` barrel still
 * re-exports everything for callers that do not care.
 */
export interface RandomAccessSource {
  readonly size: number;
  read(offset: number, length: number): Promise<Uint8Array>;
  close?(): void | Promise<void>;
}
