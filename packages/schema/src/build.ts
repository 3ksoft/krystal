import fs from "node:fs";
import path from "node:path";
import { fromModule, SchemaAnalyzer } from "@schema-pop/core";
import { exportPlan } from "@schema-pop/exporter";
import { schema } from "./schema";

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

const popSchema = fromModule(schema.export()).schema;
const result = analyzer.analyze(popSchema, cfg);
assertSchema(result);
const schemaIR = result.plan;

const paths = {
  types: "./../webgpu/src/types.ts",
};

const ts_import = `import { $ } from "../../schema/src/schema";\n\n`;
const exports = exportPlan(schemaIR, "ts:exports");

save(paths.types, ts_import + exports);

// Runtime-free ABI: plain interfaces plus a DataView codec, generated instead
// of derived at startup.
//
// The arktype-backed `types.ts` above stays the source of truth for host code,
// but it needs arktype present to infer anything, and the matching codec is
// otherwise built at runtime by @schema-pop (the "jit" mode compiles it with
// `new Function`). Neither survives a static compile. These two files carry the
// same layout with no imports and no code generation, so a native build — the
// scriptc exe in packages/backend — can serialize the ABI without arktype,
// @schema-pop or Sandblaster in the process.
//
// Kept in sync automatically: both come from the same analyzed plan as the
// C++/WGSL exports below, so a schema change cannot leave them disagreeing.
const generatedDir = "./generated";

save(`${generatedDir}/schema.types.ts`, exportPlan(schemaIR, "ts"));

// `ts:codec` emits bare type references while `ts` namespaces its declarations,
// so the codec is prefixed with type-only aliases bridging the two. They erase
// at runtime, keeping the codec module import-free where it matters.
const codecAliases = schemaIR.types
  .map((entry: { name: string }) => `type ${entry.name} = v1_0_0.${entry.name};`)
  .join("\n");
save(
  `${generatedDir}/schema.codec.ts`,
  `import type { v1_0_0 } from "./schema.types.ts";\n\n${codecAliases}\n\n` +
    exportPlan(schemaIR, "ts:codec"),
);

const cpp = exportPlan(schemaIR, "cpp");
save("../backend/src/abi.cpp", cpp)

const webgl = exportPlan(schemaIR, "wgsl");
save("../webgpu/src/shaders/schema.wgsl", "// THIS FILE IS FOR REFERENCE ONLY!! DO NOT INCCLUDE IT DIRECTLY!!\n" + webgl)

console.log("🐏 Schema build complete.");
