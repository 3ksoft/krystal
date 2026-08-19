import fs from "node:fs";
import path from "node:path";
import { fromModule, SchemaAnalyzer } from "@schema-pop/core";
import { exportPlan } from "@schema-pop/exporter";
import { schema } from "./schema";
import { schema as krystalSchema } from "./krystal-engine-schema";

const HEADER = `// THIS FILE IS AUTO-GENERATED - DO NOT CHANGE\n\n`;

const save = (destPath: string, content: any, noHeader: boolean = false) => {
  const dir = path.dirname(destPath);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(destPath, (noHeader ? "" : HEADER) + String(content));
  console.log("🐑 wrote", destPath);
};

console.log("🐑 Building schema ...");

const analyzer = new SchemaAnalyzer();

const cfg = {
  schemaName: "gpu",
  layout: "std430",
  autoSort: false,
  autoPack: false,
} as const;

const assertSchema = (schema: any) => {
  if (schema.errors?.length) {
    console.error("Schema errors:", schema.errors);
    process.exit(1);
  }
};

/**
 * Build one analyzed schema-pop plan into its generated artifacts.
 *
 * Returns the names of the types the analyzer kept, so callers can build
 * host-import headers that reference the right arktype scope.
 */
function buildTarget(
  name: string,
  scopeModule: any,
  options: {
    /** "ts:exports" import line referencing the arktype scope used by host code. */
    tsExportsImport: string;
    /** Where the arktype-backed host exports land (e.g. webgpu/src/krystal-types.ts). */
    hostTypesPath?: string | undefined;
    /** Where the WGSL struct reference lands. */
    wgslPath?: string | undefined;
    /** Where the runtime-free plain TS interfaces land. */
    plainTypesPath: string;
    /** Where the DataView codec lands. */
    codecPath: string;
    /**
     * Optional SoA buffer table: the struct whose array fields ARE the GPU
     * buffer set, and where to write the derived table.
     */
    soaBuffers?: { structName: string; destPath: string } | undefined;
  },
): void {
  const popSchema = fromModule(scopeModule.export()).schema;
  const result = analyzer.analyze(popSchema, cfg);
  assertSchema(result);
  const ir = result.plan;

  if (options.hostTypesPath) {
    save(options.hostTypesPath, options.tsExportsImport + exportPlan(ir, "ts:exports"));
  }
  // Runtime-free ABI: plain interfaces plus a DataView codec, generated instead
  // of derived at startup. No imports and no code generation, so a statically
  // compiled host can serialize the ABI without arktype/@schema-pop at runtime.
  save(options.plainTypesPath, exportPlan(ir, "ts"));
  const codecAliases = ir.types
    .map((entry: { name: string }) => `type ${entry.name} = v1_0_0.${entry.name};`)
    .join("\n");
  save(
    options.codecPath,
    `import type { v1_0_0 } from "./${path.basename(options.plainTypesPath)}";\n\n${codecAliases}\n\n` +
      exportPlan(ir, "ts:codec"),
  );
  if (options.soaBuffers) {
    save(options.soaBuffers.destPath, exportSoaBufferTable(ir, options.soaBuffers.structName));
  }
  if (options.wgslPath) {
    save(options.wgslPath, "// THIS FILE IS FOR REFERENCE ONLY!! DO NOT INCLUDE IT DIRECTLY!!\n" + exportPlan(ir, "wgsl"));
  }
  console.log(`🐏 ${name} plan complete (${ir.types.length} types).`);
}

/**
 * Derive the SoA buffer table from the analyzed plan.
 *
 * The struct's array fields ARE the buffer set, in declaration order, and the
 * analyzer already knows each one's exact length and byte size. Emitting the
 * table from the IR is what keeps it honest: a buffer added to, removed from or
 * resized in the schema shows up here automatically, so there is no hand-kept
 * parallel list that can quietly disagree with the struct it describes.
 *
 * Non-array fields (the plan header) are not buffers and are skipped.
 */
function exportSoaBufferTable(ir: any, structName: string): string {
  const struct = ir.types.find((entry: any) => entry.name === structName);
  if (!struct) {
    console.error(`SoA buffer table: struct ${structName} not found in plan`);
    process.exit(1);
  }
  const buffers = struct.fields
    .filter((field: any) => field.type?.kind === "array")
    .map((field: any, bufferId: number) => {
      const elementCount = field.type.exactLength;
      if (typeof elementCount !== "number") {
        console.error(`SoA buffer table: ${structName}.${field.name} has no exact length`);
        process.exit(1);
      }
      return { bufferId, name: field.name, elementCount, byteSize: field.size };
    });

  const ids = buffers.map((b: any) => `  ${b.name}: ${b.bufferId},`).join("\n");
  const rows = buffers
    .map(
      (b: any) =>
        `  { bufferId: ${b.bufferId}, name: "${b.name}", elementCount: ${b.elementCount}, byteSize: ${b.byteSize} },`,
    )
    .join("\n");
  const constName = structName.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toUpperCase();

  return [
    "/**",
    ` * SoA buffer table for \`${structName}\`, derived from the analyzed schema.`,
    " *",
    " * Do not hand-edit and do not keep a parallel copy: add, remove or resize a",
    ` * buffer by changing \`${structName}\` in the schema and rebuilding.`,
    " */",
    "export interface SoaBufferDescriptor {",
    "  readonly bufferId: number;",
    "  readonly name: string;",
    "  readonly elementCount: number;",
    "  readonly byteSize: number;",
    "}",
    "",
    `export const ${constName}_BUFFERS: readonly SoaBufferDescriptor[] = [`,
    rows,
    "];",
    "",
    "export const BINARY_LAYOUT_BUFFER_IDS = {",
    ids,
    "} as const;",
    "",
    `export const BINARY_LAYOUT_BUFFER_COUNT = ${buffers.length};`,
    "",
  ].join("\n");
}

// ---------------------------------------------------------------------------
// Krystal brain ABI (forward/backward contracts) — the only remaining target.
// ---------------------------------------------------------------------------

buildTarget("krystal", krystalSchema, {
  tsExportsImport: `import { schema as $ } from "../../schema/src/krystal-engine-schema";\n\n`,
  hostTypesPath: "./../webgpu/src/krystal-types.ts",
  plainTypesPath: "./generated/krystal.types.ts",
  codecPath: "./generated/krystal.codec.ts",
  soaBuffers: { structName: "BrainFrameGpu", destPath: "./generated/krystal.buffers.ts" },
});

console.log("🐏 Schema build complete.");
