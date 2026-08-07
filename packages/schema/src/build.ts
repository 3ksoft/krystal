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

const cpp = exportPlan(schemaIR, "cpp");
save("../backend/src/abi.cpp", cpp)

const webgl = exportPlan(schemaIR, "wgsl");
save("../webgpu/src/shaders/schema.wgsl", "// THIS FILE IS FOR REFERENCE ONLY!! DO NOT INCCLUDE IT DIRECTLY!!\n" + webgl)

console.log("🐏 Schema build complete.");
