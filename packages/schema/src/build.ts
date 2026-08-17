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
  if (options.wgslPath) {
    save(options.wgslPath, "// THIS FILE IS FOR REFERENCE ONLY!! DO NOT INCLUDE IT DIRECTLY!!\n" + exportPlan(ir, "wgsl"));
  }
  console.log(`🐏 ${name} plan complete (${ir.types.length} types).`);
}

// ---------------------------------------------------------------------------
// Krystal brain ABI (forward/backward contracts) — the only remaining target.
// ---------------------------------------------------------------------------

buildTarget("krystal", krystalSchema, {
  tsExportsImport: `import { schema as $ } from "../../schema/src/krystal-engine-schema";\n\n`,
  hostTypesPath: "./../webgpu/src/krystal-types.ts",
  plainTypesPath: "./generated/krystal.types.ts",
  codecPath: "./generated/krystal.codec.ts",
});

console.log("🐏 Schema build complete.");
