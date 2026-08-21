/**
 * Public surface of @krystal/schema.
 *
 * `build.ts` is the code generator: it writes into sibling packages at module
 * scope, so it stays out of the barrel and is run via `bun run build`.
 */

// Type-level ArkEnv configuration that keeps WebGPU host objects opaque.
// Ambient-only, but consumers need it applied before they infer schema types.
// Extensionless within this package: it emits declarations, so TypeScript
// forbids allowImportingTsExtensions here.
import "./env";

export * from "./krystal-engine-schema";
export * from "./world";
