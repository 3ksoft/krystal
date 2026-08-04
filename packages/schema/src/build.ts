import fs from "node:fs";
import path from "node:path";
import { fromModule, SchemaAnalyzer } from "@schema-pop/core";
import { exportPlan, tsCodec, tsExports, wgsl } from "@schema-pop/exporter";
import { telemetry } from "./telemetry";

const HEADER = `// THIS FILE IS AUTO-GENERATED - DO NOT CHANGE`;

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
const popSchema = fromModule(telemetry.export()).schema;
const result = analyzer.analyze(popSchema, cfg);
assertSchema(result);
const schemaIR = result.plan;

const paths = {
  codec: "./dist/util/codec.ts",
  buffers: "./dist/util/buffers.ts",
  metadata: "./dist/util/metadata.ts",
  wgslCodec: "./dist/shaders/includes/codec.wgsl",
  wgslSchema: "./dist/shaders/includes/schema.wgsl",
  wgslBindings: "./dist/shaders/includes/bindings.wgsl",
  html: "schema.html",
};
const schemaContent =
  
  tsExports({}).generate(schemaIR);


save(
  paths.codec,
  schemaContent + tsCodec({ importPath: "./schema", generatePatches: true }).generate(schemaIR),
);

save(paths.wgslSchema, wgsl({ outputStyle: "types" }).generate(schemaIR));
save(paths.wgslCodec, wgsl({ outputStyle: "helpers" }).generate(schemaIR));

console.log("🐏 Schema build complete.");
