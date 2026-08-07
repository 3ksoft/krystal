// report_gpu_constraints.ts
//
// bun run src/report_gpu_constraints.ts [path-to-jsonl]
//
// Compiles every raw JSON schema directly into the CPU constraint graph, then
// links the deterministic GPU byte VM and reports upload sizes. Binary LayoutPlan
// is intentionally bypassed here because it cannot preserve bounded dynamic
// JSON constructs such as maxItems.

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import {
  compileJsonSchemaProgram,
  linkGpuConstraintProgram,
  type GpuConstraintProgramSummary,
} from "../../engine-ts/src/index.ts";

const INPUT = process.argv[2] ?? join(import.meta.dir, "../out/schema-dataset.jsonl");

type DatasetRecord = {
  seed: number;
  id: number;
  task: string;
  rawSchema?: unknown;
  schema?: unknown;
};

type Success = {
  index: number;
  record: DatasetRecord;
  cpuNodes: number;
  gpu: GpuConstraintProgramSummary;
};

type Failure = {
  index: number;
  record: DatasetRecord | null;
  stage: "json" | "schema" | "analyzer" | "cpu" | "gpu";
  message: string;
};

function formatError(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  try {
    return JSON.stringify(error);
  } catch {
    return String(error);
  }
}

function kib(bytes: number): string {
  return `${(bytes / 1024).toFixed(2)} KiB`;
}

const text = await readFile(INPUT, "utf8");
const lines = text.split("\n").filter((line) => line.trim().length > 0);
const successes: Success[] = [];
const failures: Failure[] = [];

for (const [index, line] of lines.entries()) {
  let record: DatasetRecord;
  try {
    record = JSON.parse(line) as DatasetRecord;
  } catch (error) {
    failures.push({ index, record: null, stage: "json", message: formatError(error) });
    continue;
  }

  const rawSchema = record.rawSchema ?? record.schema;
  if (rawSchema === undefined) {
    failures.push({ index, record, stage: "schema", message: "missing rawSchema/schema" });
    continue;
  }

  let cpu;
  try {
    cpu = compileJsonSchemaProgram(rawSchema);
  } catch (error) {
    failures.push({ index, record, stage: "cpu", message: formatError(error) });
    continue;
  }

  try {
    const gpu = linkGpuConstraintProgram(cpu);
    successes.push({ index, record, cpuNodes: cpu.nodes.length, gpu: gpu.summary });
  } catch (error) {
    failures.push({ index, record, stage: "gpu", message: formatError(error) });
  }
}

console.log("\n=== GPU constraint program report ===");
console.log(`input:      ${INPUT}`);
console.log(`records:    ${lines.length}`);
console.log(`linked:     ${successes.length}`);
console.log(`rejected:   ${failures.length}`);

if (successes.length > 0) {
  console.log("\n--- linked records ---");
  console.log(" idx   id  seed   cpu   gpu  jump  switch  edges  literals  pool   blob");
  for (const row of successes) {
    const g = row.gpu;
    console.log(
      `${String(row.index).padStart(4)} ${String(row.record.id).padStart(4)} ${String(row.record.seed).padStart(5)}` +
      ` ${String(row.cpuNodes).padStart(5)} ${String(g.nodes).padStart(5)}` +
      ` ${String(g.jumpNodes).padStart(5)} ${String(g.switchNodes).padStart(7)} ${String(g.edges).padStart(6)}` +
      ` ${String(g.literalNodes).padStart(9)} ${String(g.byteLength).padStart(5)}` +
      ` ${kib(g.blobBytes).padStart(10)}`,
    );
  }

  const total = (pick: (row: Success) => number) => successes.reduce((sum, row) => sum + pick(row), 0);
  const max = (pick: (row: Success) => number) => Math.max(...successes.map(pick));
  const totalBytes = total((row) => row.gpu.blobBytes);
  console.log("\n--- aggregate ---");
  console.log(`total blob: ${kib(totalBytes)}`);
  console.log(`avg blob:   ${kib(totalBytes / successes.length)}`);
  console.log(`max blob:   ${kib(max((row) => row.gpu.blobBytes))}`);
  console.log(`max nodes:  ${max((row) => row.gpu.nodes)}`);
  console.log(`max edges:  ${max((row) => row.gpu.edges)}`);
  console.log(`max pool:   ${max((row) => row.gpu.byteLength)} bytes`);
}

if (failures.length > 0) {
  console.log("\n--- rejected records ---");
  for (const failure of failures) {
    const label = failure.record
      ? `[${failure.index}] id=${failure.record.id} seed=${failure.record.seed}`
      : `[${failure.index}]`;
    console.log(`✗ ${label} [${failure.stage}] ${failure.message}`);
  }
}

console.log("");
