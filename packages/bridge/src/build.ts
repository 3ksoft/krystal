import fs from "node:fs";
import path from "node:path";
import { fromModule, SchemaAnalyzer } from "@schema-pop/core";
import { exportPlan } from "@schema-pop/exporter";
import { bridge } from "./schema";

const outDir = path.resolve(import.meta.dir, "../generated");
fs.mkdirSync(outDir, { recursive: true });

// Keep extraction diagnostics available, but analyze the extracted PopSchema.
// The bridge itself intentionally contains only fixed-size binary types.
const extracted = fromModule(bridge);
if (extracted.errors.length) {
  throw new Error(`Bridge extraction errors:
${extracted.errors.join("\n")}`);
}
const analyzer = new SchemaAnalyzer();
const result = analyzer.analyze(extracted.schema, {
  schemaName: "chomato_bridge",
  version: "1.0.0",
  endian: "le",
  wordSize: "32",
  layout: "zero-padding",
  mode: "binary",
  autoSort: false,
  autoPack: false,
});

if (result.errors.length) {
  throw new Error(`Bridge schema errors:\n${result.errors.join("\n")}`);
}

const layoutSource = result.plan.types
  .map((entry) => `export const SIZEOF_${entry.name} = ${entry.paddedSize};`)
  .join("\n") + "\n";

const outputs = {
  "bridge.hpp": exportPlan(result.plan, "cpp"),
  "bridge.types.ts": exportPlan(result.plan, "ts"),
  "bridge.layout.ts": layoutSource,
  "bridge.plan.json": JSON.stringify(result.plan, null, 2),
};

for (const [name, source] of Object.entries(outputs)) {
  fs.writeFileSync(path.join(outDir, name), source.replace(/[ \t]+$/gm, ""));
  console.log(`[bridge] wrote generated/${name}`);
}
