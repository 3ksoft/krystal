// Local test runner for the checkpoint suite.
//
// Runs the engine tests in tests/ against the real engine when a model file
// and the `webgpu` Dawn bindings are available:
//
//   Engine -> InProcessTransport -> Lfm2WebGpuEngineBackend -> Lfm2Forward
//
// Without a model file / GPU it falls back to the mock exe:
//
//   Engine -> BinaryEngineTransport -> SpawnedNativeChannel -> mock exe
//
//   bun run packages/backend/src/run-local-tests.ts
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { pickExeCommand } from "./exe/pick-command.ts";

const repoRoot = resolve(import.meta.dir, "..", "..", "..");
const testFiles = [
  "tests/checkpoint.test.ts",
  "tests/structured-benchmark.test.ts",
];

const exe = pickExeCommand();
console.log("chomato local test runner · real engine (Dawn) / mock fallback");
console.log(`exe: ${exe.command} ${exe.args.join(" ")}${exe.ffi ? " [scriptc FFI]" : " [bun mock]"}`);
console.log("");

let passed = 0;
let failed = 0;
const reports: Array<{ file: string; ok: boolean }> = [];

for (const file of testFiles) {
  const result = spawnSync("bun", ["test", resolve(repoRoot, file)], {
    cwd: repoRoot,
    stdio: "inherit",
  });
  const ok = result.status === 0;
  reports.push({ file, ok });
  console.log("");
  console.log(`${ok ? "✓" : "✗"} ${file} ${ok ? "PASSED" : "FAILED"}`);
  ok ? passed++ : failed++;
}

console.log("");
console.log("── report ──────────────────────────────────────────────");
for (const report of reports) {
  console.log(`${report.ok ? "PASS" : "FAIL"}  ${report.file}`);
}
console.log(`summary: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
