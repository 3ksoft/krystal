// Deno backend validation of the LFM2 WGSL build.
//
// Mirror of cloudCompute's packages/engine/src/util/validateShaders.mjs, adapted
// to the Sandblaster pipeline and to Deno's native WebGPU (navigator.gpu,
// backed by wgpu/naga) — no browser, no Dawn npm package, no libvulkan setup.
//
// The flow is identical to scripts/build-lfm2-artifact.ts:
//   1. read the shader bodies + includes from disk (same name lists),
//   2. defineLfm2({ sources, includes }),
//   3. engine.link() — gives every program its final WGSL source
//      (Sandblaster wraps each body with includes, bindings and the compute
//      entry point; that exact source ships to the browser),
//   4. compile each linked source through a WebGPU device and report
//      per-program diagnostics.
//
// Run:
//   deno run --allow-read --sloppy-imports scripts/validate-lfm2-shaders.ts [--full]
//
// --full additionally runs lfm2.engine.compile({ device }): full pipeline
// creation (bind group layouts, pipelines) — the strongest backend signal.
//
// Exit code: 1 when any shader fails, 0 otherwise. If no WebGPU adapter is
// available the script prints SKIP and exits 0 (same contract as
// validateShaders.mjs).

import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  defineLfm2,
  LFM2_INCLUDE_NAMES,
  LFM2_SHADER_NAMES,
  type Lfm2IncludeName,
  type Lfm2ShaderName,
} from "../src/lfm2-definition";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const shaderDir = resolve(root, "src/shaders");
const includeDir = resolve(shaderDir, "includes");

async function readNamed<K extends string>(
  dir: string,
  names: readonly K[],
): Promise<Record<K, string>> {
  return Object.fromEntries(
    await Promise.all(
      names.map(async (name) => [
        name,
        await readFile(resolve(dir, `${name}.wgsl`), "utf8"),
      ] as const),
    ),
  ) as Record<K, string>;
}

interface NamedSource {
  label: string;
  code: string;
}

interface LinkedLfm2 {
  definition: ReturnType<typeof defineLfm2>;
  programs: NamedSource[];
}

/** Same input set and link flow as scripts/build-lfm2-artifact.ts. */
async function linkCurrentSources(): Promise<LinkedLfm2> {
  const sources = await readNamed<Lfm2ShaderName>(shaderDir, LFM2_SHADER_NAMES);
  const includes = await readNamed<Lfm2IncludeName>(includeDir, LFM2_INCLUDE_NAMES);
  const definition = defineLfm2({ sources, includes });
  definition.engine.link();
  const programs = LFM2_SHADER_NAMES.map((name) => ({
    label: name,
    code: definition.programs[name].source,
  }));
  return { definition, programs };
}

async function requestDevice(): Promise<GPUDevice> {
  if (typeof navigator === "undefined" || !navigator.gpu) {
    throw new Error("navigator.gpu unavailable — run this script under Deno.");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("SKIP: no WebGPU adapter (backend without a GPU driver?)");
  }
  return adapter.requestDevice({ label: "lfm2-shader-validation" });
}

// --- Validation -------------------------------------------------------------

let failed = 0;

function reportError(stage: string, label: string, message: string) {
  failed++;
  console.error(`✗ [${stage}] ${label}`);
  for (const line of message.split("\n").filter((l) => l.trim() !== "")) {
    console.error(`    ${label}: ${line}`);
  }
}

async function validateShaders(stage: string, programs: NamedSource[], device: GPUDevice) {
  for (const { label, code } of programs) {
    // Best-effort error scope: wgpu surfaces the same errors through
    // getCompilationInfo(); pushErrorScope may be unsupported on some backends.
    const hasScope = typeof device.pushErrorScope === "function";
    if (hasScope) device.pushErrorScope("validation");

    const module = device.createShaderModule({ code, label });
    const info = await module.getCompilationInfo();
    const scopeError = hasScope ? await device.popErrorScope() : undefined;

    const errors = info.messages.filter((m) => m.type === "error");
    if (errors.length > 0) {
      const at = (m: { lineNum?: number; linePos?: number }) =>
        m.lineNum !== undefined ? `${label} (${m.lineNum}:${m.linePos ?? 0})` : label;
      reportError(stage, label, errors.map((m) => `${at(m)}: ${m.message}`).join("\n"));
    } else if (scopeError) {
      reportError(stage, label, `${label}: ${scopeError.message}`);
    } else {
      console.log(`✓ [${stage}] ${label}`);
    }
  }
}

// --- Main -------------------------------------------------------------------

const full = Deno.args.includes("--full");

console.log("[validate] linking current shader sources…");
let linked: LinkedLfm2;
try {
  linked = await linkCurrentSources();
} catch (error) {
  console.error(`[validate] link failed: ${error instanceof Error ? error.message : error}`);
  Deno.exit(1);
}

console.log("[validate] requesting WebGPU device…");
let device: GPUDevice;
try {
  device = await requestDevice();
} catch (error) {
  console.warn(`[validate] ${error instanceof Error ? error.message : error}`);
  Deno.exit(0); // SKIP, same contract as validateShaders.mjs
}

await validateShaders("link", linked.programs, device);

if (full) {
  console.log("[validate] engine.compile({ device }) — full pipeline creation…");
  const compiled = await linked.definition.engine.compile({ device });
  for (const program of compiled.programs) {
    if (program.status === "failed") {
      reportError(
        "compile",
        program.label,
        program.errors
          .map((e: { message: string }) => `${program.label}: ${e.message}`)
          .join("\n"),
      );
    } else {
      console.log(`✓ [compile] ${program.label}`);
    }
  }
  console.log(`[compile] ${compiled.ok}/${compiled.total} programs OK in ${compiled.elapsedMs} ms`);
}

console.log(
  failed
    ? `\n${failed} shader(s) with errors`
    : `\n${LFM2_SHADER_NAMES.length}/${LFM2_SHADER_NAMES.length} shaders OK`,
);
Deno.exit(failed ? 1 : 0);
