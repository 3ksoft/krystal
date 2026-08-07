/**
 * Public surface of @chomato/quant.
 *
 * Library modules only. The CLI entry points under wq4/ (dumpModel,
 * convert_gguf_to_wq4, bench_matmul_wq4) run work at module scope and must be
 * invoked directly with `deno run` / `bun run`, never re-exported from here.
 */
export * from "./gguf/index.ts";
export * from "./wq4/reader.ts";
