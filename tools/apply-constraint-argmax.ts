import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

async function install(path: string, payloadPath: string): Promise<void> {
  const payload = await readFile(payloadPath, "utf8");
  let before: string | undefined;
  try { before = await readFile(path, "utf8"); } catch { before = undefined; }
  if (before === payload) { console.log(`= ${path}`); return; }
  await mkdir(dirname(path), { recursive: true });
  if (before !== undefined) await writeFile(`${path}.constraint-argmax.bak`, before);
  await writeFile(path, payload);
  console.log(`✓ ${path}${before === undefined ? " (new)" : ""}`);
}

async function edit(path: string, transform: (source: string) => string): Promise<void> {
  const before = await readFile(path, "utf8");
  const after = transform(before);
  if (after === before) { console.log(`= ${path}`); return; }
  await writeFile(`${path}.constraint-argmax.bak`, before);
  await writeFile(path, after);
  console.log(`✓ ${path}`);
}

async function editIfExists(path: string, transform: (source: string) => string): Promise<void> {
  try { await access(path); } catch { console.log(`- ${path} (not present)`); return; }
  await edit(path, transform);
}

function replaceOnce(source: string, needle: string, replacement: string, label: string): string {
  const at = source.indexOf(needle);
  if (at < 0) throw new Error(`Could not find ${label}`);
  if (source.indexOf(needle, at + needle.length) >= 0) throw new Error(`Ambiguous ${label}`);
  return source.slice(0, at) + replacement + source.slice(at + needle.length);
}

function methodEnd(source: string, methodName: string): number {
  const signature = source.indexOf(`${methodName}(`);
  if (signature < 0) throw new Error(`Could not find ${methodName} method`);
  const open = source.indexOf("{", signature);
  if (open < 0) throw new Error(`Could not find ${methodName} body`);

  let depth = 0;
  let quote = "";
  let escaped = false;
  let lineComment = false;
  let blockComment = false;
  for (let i = open; i < source.length; i++) {
    const ch = source[i]!;
    const next = source[i + 1] ?? "";
    if (lineComment) { if (ch === "\n") lineComment = false; continue; }
    if (blockComment) { if (ch === "*" && next === "/") { blockComment = false; i++; } continue; }
    if (quote) {
      if (escaped) { escaped = false; continue; }
      if (ch === "\\") { escaped = true; continue; }
      if (ch === quote) quote = "";
      continue;
    }
    if (ch === "/" && next === "/") { lineComment = true; i++; continue; }
    if (ch === "/" && next === "*") { blockComment = true; i++; continue; }
    if (ch === '"' || ch === "'" || ch === "`") { quote = ch; continue; }
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) return i + 1;
    }
  }
  throw new Error(`Unterminated ${methodName} method`);
}

function insertAfterMethod(source: string, methodName: string, addition: string): string {
  const end = methodEnd(source, methodName);
  return source.slice(0, end) + addition + source.slice(end);
}

const payloadRoot = "tools/constraint-argmax-payload";
await install(
  "packages/webgpu/src/shaders/constraint_argmax.wgsl",
  `${payloadRoot}/packages/webgpu/src/shaders/constraint_argmax.wgsl`,
);
await install(
  "packages/webgpu/src/shaders/includes/constraint-commit.wgsl",
  `${payloadRoot}/packages/webgpu/src/shaders/includes/constraint-commit.wgsl`,
);

await edit("packages/webgpu/src/lfm2-definition.ts", (input) => {
  let source = input;

  if (!source.includes('  "constraint_argmax",')) {
    if (source.includes('  "constraint_mask",')) {
      source = replaceOnce(
        source,
        '  "constraint_mask",',
        '  "constraint_mask",\n  "constraint_argmax",',
        "constraint shader list",
      );
    } else {
      throw new Error("constraint_mask is not integrated; apply the constraint AOT overlay first");
    }
  }

  if (!source.includes('  "constraint-commit",')) {
    source = replaceOnce(
      source,
      '  "constraint-vm",',
      '  "constraint-vm",\n  "constraint-commit",',
      "constraint include list",
    );
  }

  if (!source.includes("constraint_argmax: engine.compute")) {
    source = replaceOnce(
      source,
      '  } satisfies Record<Lfm2ShaderName, AnyComputeHandle>;',
      `    constraint_argmax: engine.compute({\n      label: "constraint_argmax",\n      resources: {\n        op: r.op,\n        runtime: r.runtime,\n        tokens: r.tokens,\n        arena: r.arena,\n        decodeTelemetry: r.decodeTelemetry,\n        constraintProgram: r.constraintProgram,\n        constraintTokenizer: r.constraintTokenizer,\n        constraintState: nativeWrite(constraintState),\n        constraintMask: nativeRead(constraintMask),\n      },\n      codecs: [engine.type("DecodeTelemetryEntry")],\n      includes: [\n        include("common"),\n        include("telemetry"),\n        include("reduce-f32"),\n        include("reduce-u32"),\n        include("constraint-vm"),\n        include("constraint-commit"),\n      ],\n      compute: {\n        entryPoint: "constraint_argmax",\n        params: lid,\n        workgroupSize: 256,\n        code: sources.constraint_argmax,\n      },\n    }),\n  } satisfies Record<Lfm2ShaderName, AnyComputeHandle>;`,
      "program map terminator",
    );
  }

  if (!source.includes('constraint_argmax: definePass(programs.constraint_argmax')) {
    const anchor = '  argmax: definePass(programs.argmax, "none", () => [1, 1, 1]),';
    source = replaceOnce(
      source,
      anchor,
      `${anchor}\n  constraint_argmax: definePass(programs.constraint_argmax, "none", () => [1, 1, 1]),`,
      "argmax pass",
    );
  }

  return source;
});

await edit("packages/webgpu/src/pass.ts", (input) => {
  if (input.includes('runStatic(name: "constraint_mask"')) return input;
  const hook = input.includes("this.onRun?.(name)") ? "    this.onRun?.(name);\n" : "";
  const method = `\n\n  /** Dispatch a static AOT program that has no OpParams binding. */\n  runStatic(name: "constraint_mask", workgroups: Lfm2Workgroups): void {\n${hook}    this.pass.run(lfm2.programs[name], { workgroups });\n  }`;
  return replaceOnce(
    input,
    "\n}\n\nexport interface Lfm2CommandEncoder",
    `${method}\n}\n\nexport interface Lfm2CommandEncoder`,
    "Lfm2ComputePass terminator",
  );
});

await editIfExists("packages/webgpu/src/forward.ts", (input) => {
  let source = input;

  if (!source.includes("GPU_SCHEMA_SENTINELS")) {
    source = `import { GPU_SCHEMA_SENTINELS } from "../../schema/src/sparse";\n${source}`;
  }

  if (!source.includes("commitConstraintArgmax(")) {
    source = insertAfterMethod(
      source,
      "commitArgmax",
      `\n\n  /** Exact GPU mask -> masked argmax -> transactional VM commit. */\n  commitConstraintArgmax(pass: Lfm2ComputePass, mode: Lfm2Mode = "prefill"): void {\n    pass.runStatic("constraint_mask", [lfm2.constraint.maskWorkgroups, 1, 1]);\n    pass.run("constraint_argmax", {\n      inputOffset: LFM2_ARENA.logits,\n      inputDim: this.model.config.vocabSize,\n      u0: GPU_SCHEMA_SENTINELS.emptyToken,\n      mode,\n    });\n  }`,
    );
  }

  if (!source.includes("forwardAndSampleConstrained(")) {
    source = insertAfterMethod(
      source,
      "forwardAndSample",
      `\n\n  /** Full forward followed by exact structured sampling entirely on GPU. */\n  forwardAndSampleConstrained(\n    pass: Lfm2ComputePass,\n    tokenCount: number,\n    mode: Lfm2Mode = "prefill",\n    work: Lfm2WorkLayout = LFM2_ARENA,\n    tokenOffset = 0,\n  ): void {\n    this.forwardToLogits(pass, tokenCount, mode, work, tokenOffset);\n    this.commitConstraintArgmax(pass, mode);\n  }`,
    );
  }
  return source;
});

await edit("packages/webgpu/src/constraint.ts", (input) => {
  if (input.includes("export async function readGpuConstraintState")) return input;
  const addition = `\n\n/** Diagnostic/state-equivalence readback for the 64-byte decoder state. */\nexport async function readGpuConstraintState(\n  definition: Lfm2Definition,\n): Promise<Uint32Array> {\n  const device = definition.engine.device;\n  const byteLength = GPU_CONSTRAINT_STATE.byteLength;\n  const staging = device.createBuffer({\n    label: "constraint.state.readback",\n    size: byteLength,\n    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,\n  });\n\n  try {\n    const encoder = device.createCommandEncoder({ label: "constraint.state.readback" });\n    encoder.copyBufferToBuffer(\n      definition.resources.constraintState.gpu,\n      0,\n      staging,\n      0,\n      byteLength,\n    );\n    device.queue.submit([encoder.finish()]);\n    await staging.mapAsync(GPUMapMode.READ, 0, byteLength);\n    return new Uint32Array(staging.getMappedRange(0, byteLength).slice(0));\n  } finally {\n    if (staging.mapState === "mapped") staging.unmap();\n    staging.destroy();\n  }\n}\n`;
  return input.endsWith("\n") ? input + addition.trimStart() : input + addition;
});

// Integrate the real Dawn argmax/commit test into the existing local suite.
const testPath = "tests/checkpoint.test.ts";
try {
  await access(testPath);
  const snippet = (await readFile("tools/argmax-test-snippet.txt", "utf8")).trimEnd();
  await edit(testPath, (input) => {
    if (input.includes('test("Dawn constrained argmax respects the exact mask and commits VM state"')) return input;
    let source = input;
    source = source.replace(
      "\tdispatchGpuConstraintMask,\n\treadGpuConstraintMask,",
      "\tdispatchGpuConstraintMask,\n\treadGpuConstraintMask,\n\treadGpuConstraintState,",
    );
    source = source.replace(
      'import { defineLfm2 } from "../packages/webgpu/src/lfm2-definition.ts";',
      'import { defineLfm2, LFM2_ARENA } from "../packages/webgpu/src/lfm2-definition.ts";',
    );
    const anchor = '\n});\n\ndescribe("context checkpoints"';
    source = replaceOnce(
      source,
      anchor,
      `\n${snippet}\n});\n\ndescribe("context checkpoints"`,
      "gpu constraint describe terminator",
    );
    return source;
  });
} catch {
  console.log(`- ${testPath} (not present)`);
}

await editIfExists("packages/webgpu/test/browser/full-main.ts", (input) => {
  let source = input;
  // Structured programs are compiled in the artifact but are not exercised by
  // the ordinary forward-coverage smoke. Keep that smoke's accounting honest.
  source = source.replace(
    "const missingAll = LFM2_PASS_NAMES.filter((name) => !allCoverage.includes(name));",
    'const missingAll = LFM2_PASS_NAMES.filter((name) => name !== "constraint_argmax" && !allCoverage.includes(name));',
  );
  source = source.replace(
    "const missingAll = LFM2_SHADER_NAMES.filter((name) => !allCoverage.includes(name));",
    'const missingAll = LFM2_SHADER_NAMES.filter((name) => name !== "constraint_mask" && name !== "constraint_argmax" && !allCoverage.includes(name));',
  );
  return source;
});


console.log("\nConstraint argmax integration complete.");
console.log("Next: bun run build:webgpu && bun test tests/checkpoint.test.ts -t 'gpu constraint'");
