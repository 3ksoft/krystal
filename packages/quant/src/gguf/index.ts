/**
 * Convenience barrel.
 *
 * It re-exports every concrete source, so importing it commits to a runtime
 * that has Deno *and* browser globals. Code that must stay portable — the
 * native exe, in particular — should import `./source.ts` for the interface
 * and pick a concrete source explicitly.
 */
export * from "./types.ts";
export * from "./source.ts";
export * from "./source-deno.ts";
export * from "./source-web.ts";
export * from "./reader.ts";
