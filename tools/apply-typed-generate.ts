import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

const BACKUP = ".typed-generate.bak";

async function exists(path: string): Promise<boolean> {
  try { await access(path); return true; } catch { return false; }
}

async function edit(path: string, transform: (source: string) => string): Promise<void> {
  const before = await readFile(path, "utf8");
  const after = transform(before);
  if (after === before) {
    console.log(`= ${path} (already integrated)`);
    return;
  }
  if (!(await exists(`${path}${BACKUP}`))) await writeFile(`${path}${BACKUP}`, before);
  await writeFile(path, after);
  console.log(`✓ ${path}`);
}

async function install(path: string, payloadPath: string): Promise<void> {
  const payload = await readFile(payloadPath, "utf8");
  const before = await exists(path) ? await readFile(path, "utf8") : undefined;
  if (before === payload) {
    console.log(`= ${path} (already installed)`);
    return;
  }
  await mkdir(dirname(path), { recursive: true });
  if (before !== undefined && !(await exists(`${path}${BACKUP}`))) await writeFile(`${path}${BACKUP}`, before);
  await writeFile(path, payload);
  console.log(`✓ ${path}${before === undefined ? " (new)" : ""}`);
}

function replaceOnce(source: string, needle: string, replacement: string, label: string): string {
  const at = source.indexOf(needle);
  if (at < 0) throw new Error(`Could not find ${label}`);
  if (source.indexOf(needle, at + needle.length) >= 0) throw new Error(`Ambiguous ${label}`);
  return source.slice(0, at) + replacement + source.slice(at + needle.length);
}

function findBlockEnd(source: string, open: number): number {
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
  throw new Error("Unterminated block");
}

function namedBlock(source: string, marker: string): { start: number; open: number; end: number; text: string } {
  const start = source.indexOf(marker);
  if (start < 0) throw new Error(`Could not find ${marker}`);
  const open = source.indexOf("{", start);
  if (open < 0) throw new Error(`Could not find body for ${marker}`);
  const end = findBlockEnd(source, open);
  return { start, open, end, text: source.slice(start, end) };
}

function replaceNamedBlock(source: string, marker: string, transform: (block: string) => string): string {
  const block = namedBlock(source, marker);
  const after = transform(block.text);
  return source.slice(0, block.start) + after + source.slice(block.end);
}

function addNamedImport(source: string, moduleName: string, name: string): string {
  const escaped = moduleName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`import\\s*\\{([^}]*)\\}\\s*from\\s*[\"']${escaped}[\"'];?`);
  const match = source.match(re);
  if (match) {
    if (new RegExp(`\\b${name}\\b`).test(match[1]!)) return source;
    const replacement = match[0]!.replace("{", `{\n  ${name},`);
    return source.replace(match[0]!, replacement);
  }
  return `import { ${name} } from "${moduleName}";\n${source}`;
}

function appendExport(source: string, line: string): string {
  if (source.includes(line)) return source;
  return `${source}${source.endsWith("\n") ? "" : "\n"}${line}\n`;
}

const payloadRoot = "tools/typed-generate-payload";
await install(
  "packages/engine-ts/src/structured.ts",
  `${payloadRoot}/packages/engine-ts/src/structured.ts`,
);
await install(
  "tests/structured-generation.test.ts",
  `${payloadRoot}/tests/structured-generation.test.ts`,
);

await edit("packages/bridge/src/constants.ts", (input) => {
  if (input.includes("CONTEXT_FLAG_STRUCTURED")) return input;
  const anchor = "export const NO_CHECKPOINT = 0;";
  return replaceOnce(
    input,
    anchor,
    `${anchor}\n/** ContextRef.reserved bit: Generate payload appends a GPU constraint program. */\nexport const CONTEXT_FLAG_STRUCTURED = 1 << 0;`,
    "NO_CHECKPOINT constant",
  );
});

await edit("packages/engine-ts/src/index.ts", (input) =>
  appendExport(input, 'export * from "./structured.ts";')
);

await edit("packages/engine-ts/src/transport.ts", (input) => {
  let source = input;
  source = addNamedImport(source, "@chomato/bridge/constants", "CONTEXT_FLAG_STRUCTURED");
  if (!source.includes('from "./structured"') && !source.includes('from "./structured.ts"')) {
    source = `import {\n  compileStructuredGeneration,\n  isGeneratableSchema,\n  type GeneratableSchema,\n  type InferGeneratable,\n} from "./structured.ts";\n${source}`;
  }

  if (!source.includes("function concatStructuredPayload(")) {
    const anchor = "/** Encode a u32 vector exactly as the bridge bulk payload. */";
    const helper = `function concatStructuredPayload(blocks: Uint8Array | undefined, constraint: Uint8Array): Uint8Array {\n  const blockBytes = blocks?.byteLength ?? 0;\n  const result = new Uint8Array(blockBytes + constraint.byteLength);\n  if (blocks) result.set(blocks, 0);\n  result.set(constraint, blockBytes);\n  return result;\n}\n\n`;
    source = replaceOnce(source, anchor, helper + anchor, "u32 payload helper comment");
  }

  if (!source.includes("function encodeContext(context: Context, flags = 0)")) {
    source = replaceOnce(
      source,
      "function encodeContext(context: Context)",
      "function encodeContext(context: Context, flags = 0)",
      "encodeContext signature",
    );
    if (!/reserved:\s*0/.test(source)) throw new Error("Could not find encodeContext reserved field");
    source = source.replace(/reserved:\s*0/, "reserved: flags");
  }

  if (!source.includes("private readonly structured = new Map")) {
    const cls = namedBlock(source, "export class Engine ");
    const insertAt = cls.open + 1;
    source = source.slice(0, insertAt)
      + "\n  private readonly structured = new Map<OperationId, Deferred<Uint8Array>>();"
      + source.slice(insertAt);
  }

  if (!source.includes("structured.resolve(frame.payload")) {
    source = replaceOnce(
      source,
      '    if (event.kind === "Completed") {\n      const generation = this.generations.get(event.operation);',
      `    if (event.kind === "Completed") {\n      const structured = this.structured.get(event.operation);\n      if (structured) {\n        this.structured.delete(event.operation);\n        structured.resolve(frame.payload?.slice() ?? new Uint8Array());\n        return;\n      }\n      const generation = this.generations.get(event.operation);`,
      "Completed generation handler",
    );
  }

  if (!source.includes("structured.reject(error)")) {
    source = replaceOnce(
      source,
      "    const error = decodeFailure(event, frame.payload);\n    const generation = this.generations.get(event.operation);",
      `    const error = decodeFailure(event, frame.payload);\n    const structured = this.structured.get(event.operation);\n    if (structured) {\n      this.structured.delete(event.operation);\n      structured.reject(error);\n      return;\n    }\n    const generation = this.generations.get(event.operation);`,
      "Failed generation handler",
    );
  }

  if (!source.includes("private generateTokens(context: Context")) {
    source = source.replace(
      /\n  generate\(context: Context, options: GenerateOptions\): Generation \{/,
      "\n  private generateTokens(context: Context, options: GenerateOptions): Generation {",
    );
    if (!source.includes("private generateTokens(context: Context")) {
      throw new Error("Could not rename raw Engine.generate to generateTokens");
    }
  }

  if (!source.includes("private async generateStructured<")) {
    const marker = "  private generateTokens(context: Context, options: GenerateOptions): Generation {";
    const at = source.indexOf(marker);
    if (at < 0) throw new Error("Could not find generateTokens insertion point");
    const publicMethods = `  generate(context: Context, options: GenerateOptions): Generation;\n  generate<S extends GeneratableSchema>(schema: S, context: Context): Promise<InferGeneratable<S>>;\n  generate(\n    schemaOrContext: GeneratableSchema | Context,\n    contextOrOptions: Context | GenerateOptions,\n  ): Generation | Promise<unknown> {\n    if (isGeneratableSchema(schemaOrContext)) {\n      return this.generateStructured(schemaOrContext, contextOrOptions as Context);\n    }\n    return this.generateTokens(schemaOrContext as Context, contextOrOptions as GenerateOptions);\n  }\n\n  private async generateStructured<S extends GeneratableSchema>(\n    schema: S,\n    context: Context,\n  ): Promise<InferGeneratable<S>> {\n    this.assertOpen();\n    const compiled = compileStructuredGeneration(schema);\n    const operation = this.allocateOperation();\n    const encoded = encodeContext(context, CONTEXT_FLAG_STRUCTURED);\n    const payload = concatStructuredPayload(encoded.payload, encodeU32Payload(compiled.program.blob));\n    const pending = new Deferred<Uint8Array>();\n\n    this.stats.generations++;\n    this.stats.commands++;\n    this.structured.set(operation, pending);\n\n    try {\n      await Promise.resolve(this.transport.send({\n        message: {\n          kind: \"Generate\",\n          operation,\n          context: encoded.ref,\n          maxTokens: compiled.maxTokens,\n        },\n        payload,\n      }));\n      const bytes = await pending.promise;\n      const json = new TextDecoder(\"utf-8\", { fatal: true }).decode(bytes);\n      return JSON.parse(json) as InferGeneratable<S>;\n    } catch (error) {\n      this.structured.delete(operation);\n      throw error;\n    }\n  }\n\n`;
    source = source.slice(0, at) + publicMethods + source.slice(at);
  }

  if (!source.includes("for (const structured of this.structured.values()) structured.reject(error);")) {
    const closeMarker = "    for (const generation of this.generations.values()) generation.fail(error);";
    source = replaceOnce(
      source,
      closeMarker,
      `${closeMarker}\n    for (const structured of this.structured.values()) structured.reject(error);`,
      "Engine.close generation rejection",
    );
    source = replaceOnce(
      source,
      "    this.generations.clear();\n    await this.transport.close?.();",
      "    this.generations.clear();\n    this.structured.clear();\n    await this.transport.close?.();",
      "Engine.close generation clear",
    );
  }

  if (!source.includes("export function completedWithPayload(")) {
    const helper = `\nexport function completedWithPayload(\n  operation: OperationId,\n  payload: Uint8Array,\n): TransportFrame<ABI.EngineEvent> {\n  return { message: { kind: \"Completed\", operation }, payload };\n}\n`;
    source += source.endsWith("\n") ? helper : "\n" + helper;
  }
  return source;
});

await edit("packages/webgpu/src/engine-transport.ts", (input) => {
  let source = input;
  source = addNamedImport(source, "@chomato/bridge/constants", "CONTEXT_FLAG_STRUCTURED");
  source = addNamedImport(source, "@chomato/engine-ts/transport", "completedWithPayload");

  if (!source.includes("function decodeConstraintWords(")) {
    const anchor = "function concatTokens(";
    const at = source.indexOf(anchor);
    if (at < 0) throw new Error("Could not find concatTokens helper");
    const helper = `function decodeConstraintWords(payload: Uint8Array | undefined, byteOffset: number): Uint32Array {\n  const total = payload?.byteLength ?? 0;\n  if (byteOffset < 0 || byteOffset > total) throw new Error(\`Invalid structured payload offset \${byteOffset}/\${total}\`);\n  const bytes = total - byteOffset;\n  if (bytes <= 0 || (bytes & 3) !== 0) throw new Error(\`Invalid constraint payload size \${bytes}\`);\n  const result = new Uint32Array(bytes >>> 2);\n  const view = new DataView(payload!.buffer, payload!.byteOffset + byteOffset, bytes);\n  for (let i = 0; i < result.length; i++) result[i] = view.getUint32(i * 4, true);\n  return result;\n}\n\n`;
    source = source.slice(0, at) + helper + source.slice(at);
  }

  if (!source.includes("generateStructured(")) {
    source = replaceNamedBlock(source, "export interface Lfm2GenerationRuntime", (block) => {
      const close = block.lastIndexOf("}");
      if (close < 0) throw new Error("Malformed Lfm2GenerationRuntime interface");
      const methods = `  generateStructured(\n    promptTokens: Uint32Array | readonly number[],\n    constraintBlob: Uint32Array,\n    options: { readonly maxNewTokens: number },\n  ): Promise<Lfm2RuntimeGenerationResult & { readonly text: string }>;\n  generateStructuredFromCheckpoint(\n    checkpoint: Lfm2RuntimeCheckpoint,\n    tailTokens: Uint32Array | readonly number[],\n    constraintBlob: Uint32Array,\n    options: { readonly maxNewTokens: number },\n  ): Promise<Lfm2RuntimeGenerationResult & { readonly text: string }>;\n`;
      return block.slice(0, close) + methods + block.slice(close);
    });
  }

  if (!source.includes("CONTEXT_FLAG_STRUCTURED) !== 0")) {
    const marker = '      case "Generate": {';
    const at = source.indexOf(marker);
    if (at < 0) throw new Error("Could not find Generate backend case");
    const insertAt = at + marker.length;
    const branch = `\n        if ((command.context.reserved & CONTEXT_FLAG_STRUCTURED) !== 0) {\n          const blockBytes = command.context.blockCount * 4;\n          const totalBytes = frame.payload?.byteLength ?? 0;\n          if (totalBytes < blockBytes) {\n            throw new Error(\`Structured Generate payload has \${totalBytes} bytes; block ids require \${blockBytes}\`);\n          }\n          const blockPayload = blockBytes > 0\n            ? frame.payload!.subarray(0, blockBytes)\n            : new Uint8Array(0);\n          const constraintBlob = decodeConstraintWords(frame.payload, blockBytes);\n          const context = this.resolveContext(command.context, blockPayload);\n          if (context.full.length === 0) throw new Error("Generation context is empty");\n          this.cancelled.delete(command.operation);\n\n          const result = context.checkpoint\n            ? await this.forward.generateStructuredFromCheckpoint(\n                context.checkpoint.state,\n                context.appended,\n                constraintBlob,\n                { maxNewTokens: command.maxTokens },\n              )\n            : await this.forward.generateStructured(\n                context.appended,\n                constraintBlob,\n                { maxNewTokens: command.maxTokens },\n              );\n\n          if (this.cancelled.has(command.operation)) {\n            this.cancelled.delete(command.operation);\n            emit(failed(command.operation, "Cancelled", "Generation cancelled"));\n            return;\n          }\n\n          emit(executionStats(command.operation, {\n            prefillTokens: result.execution.prefillTokens,\n            checkpointHits: context.checkpoint ? 1 : 0,\n            checkpointMisses: 0,\n            restoredBytes: result.execution.restoredCheckpointBytes,\n          }));\n          emit(completedWithPayload(command.operation, new TextEncoder().encode(result.text)));\n          return;\n        }\n`;
    source = source.slice(0, insertAt) + branch + source.slice(insertAt);
  }
  return source;
});

await edit("packages/webgpu/src/forward.ts", (input) => {
  let source = input;
  if (!source.includes('from "../../lfm2/src/tokenizer.ts"') && !source.includes('from "../../lfm2/src/tokenizer"')) {
    source = `import { Lfm2Tokenizer } from "../../lfm2/src/tokenizer.ts";\n${source}`;
  }
  if (!source.includes("gpuConstraintProgramFromBlob")) {
    source = `import {\n  createGpuConstraintDecoderState,\n  linkGpuConstraintTokenizer,\n  type GpuConstraintTokenizer,\n} from "../../engine-ts/src/gpu-constraint.ts";\nimport { gpuConstraintProgramFromBlob } from "../../engine-ts/src/structured.ts";\nimport { uploadGpuConstraint } from "./constraint.ts";\n${source}`;
  }

  if (!source.includes("private structuredTokenizer?: Lfm2Tokenizer")) {
    const cls = namedBlock(source, "export class Lfm2Forward ");
    const insertAt = cls.open + 1;
    source = source.slice(0, insertAt)
      + "\n  private structuredTokenizer?: Lfm2Tokenizer;\n  private structuredConstraintTokenizer?: GpuConstraintTokenizer;"
      + source.slice(insertAt);
  }

  if (!source.includes("private prepareStructuredConstraint(")) {
    const marker = "  private async readGenerationResult(";
    let at = source.indexOf(marker);
    if (at < 0) {
      // Older physical-checkpoint revision: insert immediately before generateGreedy.
      at = source.indexOf("  async generateGreedy(");
    }
    if (at < 0) throw new Error("Could not find structured generation insertion point in Lfm2Forward");
    const methods = `  private getStructuredTokenizer(): Lfm2Tokenizer {\n    if (!this.structuredTokenizer) this.structuredTokenizer = new Lfm2Tokenizer(this.model as any);\n    return this.structuredTokenizer;\n  }\n\n  private getStructuredConstraintTokenizer(): GpuConstraintTokenizer {\n    if (this.structuredConstraintTokenizer) return this.structuredConstraintTokenizer;\n    const tokenizer = this.getStructuredTokenizer();\n    const entries = Array.from({ length: this.model.config.vocabSize }, (_, id) => {\n      const bytes = tokenizer.tokenBytes(id);\n      return {\n        id,\n        bytes,\n        // No-byte ordinary tokens would allow zero-progress decode steps and\n        // invalidate the schema-derived maxTokens bound. Treat them like other\n        // control/special vocabulary entries; EOS remains explicitly handled.\n        special: tokenizer.isSpecialToken(id) || bytes === null || bytes.length === 0,\n      };\n    });\n    this.structuredConstraintTokenizer = linkGpuConstraintTokenizer(\n      entries,\n      this.model.config.eosToken,\n    );\n    return this.structuredConstraintTokenizer;\n  }\n\n  private prepareStructuredConstraint(constraintBlob: Uint32Array): void {\n    const program = gpuConstraintProgramFromBlob(constraintBlob);\n    const state = createGpuConstraintDecoderState(program);\n    uploadGpuConstraint(\n      lfm2,\n      program,\n      this.getStructuredConstraintTokenizer(),\n      state,\n    );\n  }\n\n  async generateStructured(\n    promptTokens: Uint32Array | readonly number[],\n    constraintBlob: Uint32Array,\n    options: { readonly maxNewTokens: number },\n  ) {\n    const prompt = promptTokens instanceof Uint32Array ? promptTokens : Uint32Array.from(promptTokens);\n    const maxNewTokens = options.maxNewTokens;\n    if (maxNewTokens < 1 || maxNewTokens > lfm2.capacities.maxNewTokens) {\n      throw new Error(\`Structured schema requires decode budget \${maxNewTokens}, runtime capacity is \${lfm2.capacities.maxNewTokens}\`);\n    }\n    if (prompt.length < 1) throw new Error("generateStructured requires at least one context token");\n    if (prompt.length > lfm2.capacities.context) {\n      throw new Error(\`Prompt has \${prompt.length} tokens, context capacity is \${lfm2.capacities.context}\`);\n    }\n    if (prompt.length + maxNewTokens - 1 > lfm2.capacities.context) {\n      throw new Error(\`Prompt + structured decode positions (\${prompt.length + maxNewTokens - 1}) exceed context \${lfm2.capacities.context}\`);\n    }\n\n    await this.prepareAll();\n    this.prepareStructuredConstraint(constraintBlob);\n    this.writeRuntime(prompt.length, maxNewTokens);\n    this.writeTokens(prompt, 0);\n\n    this.executor.submit((encoder) => {\n      this.clearState(encoder);\n      encoder.compute((pass) => {\n        this.forwardAndSampleConstrained(pass, prompt.length, "prefill", LFM2_ARENA, 0);\n        for (let step = 1; step < maxNewTokens; step++) {\n          this.forwardAndSampleConstrained(pass, 1, "decode", LFM2_ARENA, 0);\n        }\n      }, { label: "lfm2.generate.structured" });\n    });\n\n    const result = await this.readGenerationResult({\n      prefillTokens: prompt.length,\n      restoredCheckpointBytes: 0,\n      checkpointRestoreUs: 0,\n    });\n    return {\n      ...result,\n      text: this.getStructuredTokenizer().decode(result.tokens, { skipSpecial: true }),\n    };\n  }\n\n  async generateStructuredFromCheckpoint(\n    checkpoint: Lfm2CheckpointState,\n    tailTokens: Uint32Array | readonly number[],\n    constraintBlob: Uint32Array,\n    options: { readonly maxNewTokens: number },\n  ) {\n    const tail = tailTokens instanceof Uint32Array ? tailTokens : Uint32Array.from(tailTokens);\n    const maxNewTokens = options.maxNewTokens;\n    if (maxNewTokens < 1 || maxNewTokens > lfm2.capacities.maxNewTokens) {\n      throw new Error(\`Structured schema requires decode budget \${maxNewTokens}, runtime capacity is \${lfm2.capacities.maxNewTokens}\`);\n    }\n    const promptTokenCount = checkpoint.position + tail.length;\n    if (promptTokenCount + maxNewTokens - 1 > lfm2.capacities.context) {\n      throw new Error(\`Checkpoint + tail + structured decode positions (\${promptTokenCount + maxNewTokens - 1}) exceed context \${lfm2.capacities.context}\`);\n    }\n\n    await this.prepareAll();\n    this.prepareStructuredConstraint(constraintBlob);\n    this.writeRuntime(promptTokenCount, maxNewTokens);\n    if (tail.length) this.writeTokens(tail, 0);\n\n    const restoredCheckpointBytes =\n      checkpoint.kvBytes + checkpoint.convBytes + (tail.length === 0 ? checkpoint.hiddenBytes : 0);\n\n    this.executor.submit((encoder) => {\n      this.restoreCheckpoint(encoder, checkpoint);\n      if (!tail.length) {\n        encoder.gpu.copyBufferToBuffer(\n          checkpoint.lastHidden,\n          0,\n          lfm2.resources.arena.gpu,\n          LFM2_ARENA.hiddenA * 4,\n          this.model.config.hiddenSize * 4,\n        );\n      }\n\n      encoder.compute((pass) => {\n        if (tail.length) {\n          this.embed(pass, tail.length, "prefill", LFM2_ARENA, 0);\n          this.layers(pass, 0, this.model.config.blockCount, tail.length, {\n            mode: "continuation",\n            work: LFM2_ARENA,\n            positionBase: checkpoint.position,\n          });\n          this.projectLogits(pass, tail.length, "prefill", LFM2_ARENA);\n        } else {\n          this.projectLogits(pass, 1, "prefill", LFM2_ARENA);\n        }\n        this.commitConstraintArgmax(pass, "prefill");\n        for (let step = 1; step < maxNewTokens; step++) {\n          this.forwardAndSampleConstrained(pass, 1, "decode", LFM2_ARENA, 0);\n        }\n      }, { label: "lfm2.generate.structured-checkpoint" });\n    });\n\n    const result = await this.readGenerationResult({\n      prefillTokens: tail.length,\n      restoredCheckpointBytes,\n      checkpointRestoreUs: 0,\n    });\n    return {\n      ...result,\n      text: this.getStructuredTokenizer().decode(result.tokens, { skipSpecial: true }),\n    };\n  }\n\n`;
    source = source.slice(0, at) + methods + source.slice(at);
  }
  return source;
});

console.log("\nTyped structured generation integration complete.");
console.log("Next: bun run build:webgpu && bun test tests/structured-generation.test.ts");
