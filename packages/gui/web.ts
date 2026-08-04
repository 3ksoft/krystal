import { HttpRangeSource } from "../quant/src/gguf/source.ts";
import { Lfm2Model } from "../lfm2/src/model.ts";
import { type GenerateResult, type Lfm2CachedBlock, Lfm2Runtime } from "../lfm2/src/runtime.ts";
import { $, LLM_STATUS } from "../schema/src/schema.ts";
import { Lfm2Tokenizer } from "../lfm2/src/tokenizer.ts";
import {
  compileLayoutPlanProgram,
  createLayoutPlanJsonConstraint,
  createLiteralEnumJsonConstraint,
  type ConstraintTraceStep,
  type LayoutConstraintProgram,
  type TokenByteTableEntry,
} from "../engine-ts/src/index.ts";
import { Sandblaster } from  "@sandblaster/core"
import { SchemaAnalyzer } from "@schema-pop/core";
import { jsonSchemaToType } from "@ark/json-schema";
import { scope } from "arktype";

const params = new URLSearchParams(location.search);
const modelUrl = params.get("model") ?? "/models/LFM2.5-1.2B-Instruct-F16.gguf";
const wq4Url = params.get("wq4");
const contextCapacity = Number(params.get("context") ?? 1024);
const runtimeMaxNewTokens = Number(params.get("tokens") ?? 128);
const initialPrompt = params.get("prompt");

const byId = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing #${id}`);
  return element as T;
};

const statusEl = byId<HTMLSpanElement>("status");
const outputEl = byId<HTMLPreElement>("output");
const form = byId<HTMLFormElement>("prompt-form");
const promptEl = byId<HTMLTextAreaElement>("prompt");
const maxTokensEl = byId<HTMLInputElement>("max-tokens");
const profileEl = byId<HTMLInputElement>("profile");
const runEl = byId<HTMLButtonElement>("run");
const budgetEl = byId<HTMLSpanElement>("budget");

const structuredPromptEl = byId<HTMLTextAreaElement>("structured-prompt");
const structuredDatasetEl = byId<HTMLSelectElement>("structured-dataset");
const structuredPrevEl = byId<HTMLButtonElement>("structured-prev");
const structuredRandomEl = byId<HTMLButtonElement>("structured-random");
const structuredNextEl = byId<HTMLButtonElement>("structured-next");
const structuredMaxTokensEl = byId<HTMLInputElement>("structured-max-tokens");
const runStructuredEl = byId<HTMLButtonElement>("run-structured");
const runStructuredAllEl = byId<HTMLButtonElement>("run-structured-all");
const stopStructuredAllEl = byId<HTMLButtonElement>("stop-structured-all");
const structuredBatchEl = byId<HTMLPreElement>("structured-batch");
const structuredOutputEl = byId<HTMLPreElement>("structured-output");
const structuredPlanEl = byId<HTMLPreElement>("structured-plan");
const structuredTraceEl = byId<HTMLPreElement>("structured-trace");
const structuredStatusEl = byId<HTMLSpanElement>("structured-status");

const blockInputEl = byId<HTMLTextAreaElement>("block-input");
const blockRoleEl = byId<HTMLSelectElement>("block-role");
const blockDepthEl = byId<HTMLSelectElement>("block-depth");
const addBlockEl = byId<HTMLButtonElement>("add-block");
const clearBlocksEl = byId<HTMLButtonElement>("clear-blocks");
const blockBudgetEl = byId<HTMLSpanElement>("block-budget");
const blockStatusEl = byId<HTMLSpanElement>("block-status");
const blockListEl = byId<HTMLDivElement>("block-list");
const blockOrderEl = byId<HTMLInputElement>("block-order");
const blockPromptEl = byId<HTMLTextAreaElement>("block-prompt");
const runBlocksEl = byId<HTMLButtonElement>("run-blocks");
const prependBosEl = byId<HTMLInputElement>("prepend-bos");

const statModel = byId<HTMLElement>("stat-model");
const statLoad = byId<HTMLElement>("stat-load");
const statCompile = byId<HTMLElement>("stat-compile");
const statContext = byId<HTMLElement>("stat-context");
const statVram = byId<HTMLElement>("stat-vram");
const statPrompt = byId<HTMLElement>("stat-prompt");
const statOutput = byId<HTMLElement>("stat-output");
const statPrefill = byId<HTMLElement>("stat-prefill");
const statDecode = byId<HTMLElement>("stat-decode");
const statTotal = byId<HTMLElement>("stat-total");
const statRequestStatus = byId<HTMLElement>("stat-request-status");
const statCacheBlocks = byId<HTMLElement>("stat-cache-blocks");
const statCacheBuild = byId<HTMLElement>("stat-cache-build");
const statCacheVram = byId<HTMLElement>("stat-cache-vram");
const statCacheRepair = byId<HTMLElement>("stat-cache-repair");

if (initialPrompt !== null) promptEl.value = initialPrompt;
maxTokensEl.value = String(runtimeMaxNewTokens);
maxTokensEl.max = String(runtimeMaxNewTokens);

const ms = (value: number) => value < 1000 ? `${value.toFixed(1)} ms` : `${(value / 1000).toFixed(2)} s`;
const mib = (bytes: number) => `${(bytes / 1048576).toFixed(1)} MiB`;
const gib = (bytes: number) => `${(bytes / 1073741824).toFixed(2)} GiB`;
const rate = (tokens: number, milliseconds: number) => milliseconds > 0
  ? `${(tokens * 1000 / milliseconds).toFixed(1)} tok/s`
  : "—";

const statusName = (status: number): string => {
  switch (status) {
    case LLM_STATUS.IDLE: return "IDLE";
    case LLM_STATUS.RUNNING: return "RUNNING";
    case LLM_STATUS.EOS: return "EOS";
    case LLM_STATUS.DONE: return "DONE / token limit";
    case LLM_STATUS.ERROR: return "ERROR";
    default: return `UNKNOWN(${status})`;
  }
};

function setStatus(text: string, kind: "normal" | "ok" | "error" = "normal") {
  statusEl.textContent = text;
  statusEl.classList.toggle("ok", kind === "ok");
  statusEl.classList.toggle("error", kind === "error");
}

function setBlockStatus(text: string, kind: "normal" | "ok" | "error" = "normal") {
  blockStatusEl.textContent = text;
  blockStatusEl.classList.toggle("ok", kind === "ok");
  blockStatusEl.classList.toggle("error", kind === "error");
}

console.log("[chomato] Chromium/Dawn bring-up");
console.log("[chomato] model", modelUrl);
if (wq4Url) console.log("[chomato] WQ4 sidecar", wq4Url);

if (!navigator.gpu) throw new Error("navigator.gpu is unavailable");

setStatus("acquiring WebGPU…");
const GIB = 1024 * 1024 * 1024;
const requiredLimits: Record<string, number> = {
  maxBufferSize: GIB,
  maxStorageBufferBindingSize: GIB,
  maxComputeWorkgroupsPerDimension: 65535,
};
const engine = await Sandblaster.create($).compile({ requiredLimits } as any);
const device = engine.device;

console.log("[chomato] WebGPU limits", {
  maxBufferSizeMiB: Math.round(Number(device.limits.maxBufferSize) / 1048576),
  maxStorageBindingMiB: Math.round(Number(device.limits.maxStorageBufferBindingSize) / 1048576),
  maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
  maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
});

setStatus("opening GGUF…");
const source = await HttpRangeSource.open(modelUrl);
console.log(`[chomato] GGUF ${gib(source.size)}, HTTP Range OK`);
const wq4Source = wq4Url ? await HttpRangeSource.open(wq4Url) : undefined;
if (wq4Source) console.log(`[chomato] WQ4 ${gib(wq4Source.size)}, HTTP Range OK`);

let lastPercent = -1;
let allocatedModelBytes = 0;
const loadStarted = performance.now();
const model = await Lfm2Model.load(device, source, {
  wq4Source,
  maxPageBytes: 64 * 1024 * 1024,
  drainUploads: true,
  onProgress(progress) {
    allocatedModelBytes = progress.allocatedBytes;
    const percent = Math.floor((progress.uploadedBytes / progress.totalBytes) * 100);
    setStatus(`loading model ${percent}%…`);
    statVram.textContent = gib(progress.allocatedBytes);
    if (percent !== lastPercent && (percent % 5 === 0 || percent === 100)) {
      lastPercent = percent;
      console.log(
        `[chomato] GGUF ${percent}%  allocated ${gib(progress.allocatedBytes)}  ${progress.tensor}`,
      );
    }
  },
});
const modelLoadMs = performance.now() - loadStarted;

console.log(
  `[chomato] LFM2 ${model.config.blockCount} layers, ${model.config.hiddenSize} hidden, ` +
  `${model.config.feedForwardSize} FF, ${model.config.vocabSize} vocab`,
);
console.log(`[chomato] layer plan ${model.config.layers.map((x) => x === "attention" ? "A" : "C").join("")}`);

const tokenizer = new Lfm2Tokenizer(model.reader);

setStatus("compiling runtime…");
const compileStarted = performance.now();
const runtime = await Lfm2Runtime.create(engine, model, {
  contextCapacity,
  maxNewTokens: runtimeMaxNewTokens,
});
const runtimeCompileMs = performance.now() - compileStarted;

statModel.textContent = `${model.config.blockCount}L / ${model.config.hiddenSize}H / ${model.config.vocabSize}V`;
statLoad.textContent = ms(modelLoadMs);
statCompile.textContent = ms(runtimeCompileMs);
statContext.textContent = `${contextCapacity} tokens`;
statVram.textContent = `${gib(allocatedModelBytes)} model`;
outputEl.textContent = "Ready.";
setStatus("ready", "ok");
runEl.disabled = false;
addBlockEl.disabled = false;
for (const depth of runtime.blockCacheDepths) {
  const option = document.createElement("option");
  option.value = String(depth);
  option.textContent = depth === 2 ? `${depth} (exact)` : String(depth);
  option.selected = depth === runtime.blockCacheDepth;
  blockDepthEl.append(option);
}

let constraintTokenTable: TokenByteTableEntry[] | null = null;

function getConstraintTokenTable(): TokenByteTableEntry[] {
  if (constraintTokenTable) return constraintTokenTable;
  constraintTokenTable = tokenizer.idToToken.map((_, id) => ({
    id,
    bytes: tokenizer.tokenBytes(id),
    special: tokenizer.isSpecialToken(id),
  }));
  return constraintTokenTable;
}

interface SchemaDatasetRow {
  seed: number;
  id: number;
  task: string;
  /** New compact dataset shape. */
  rawSchema?: unknown;
  /** Backward compatibility with the current dataset. */
  schema?: unknown;
  value: unknown;
  sampleText: string;
  /** Backward compatibility only; fresh analysis is preferred when a schema is present. */
  analysis?: {
    plan?: Parameters<typeof compileLayoutPlanProgram>[0];
    warnings?: unknown[];
    errors?: unknown[];
  };
}

type ArkValidator = {
  assert(value: unknown): unknown;
};

interface CompiledDatasetRow {
  row: SchemaDatasetRow;
  plan: Parameters<typeof compileLayoutPlanProgram>[0] | null;
  program: LayoutConstraintProgram | null;
  arkType: ArkValidator | null;
  sampleError: string | null;
  error: string | null;
  lastResult?: "pass" | "validation-fail" | "incomplete" | "decode-fail";
}

let structuredDataset: CompiledDatasetRow[] = [];
let structuredBatchRunning = false;
let structuredBatchStopRequested = false;

function rowSchema(row: SchemaDatasetRow): unknown | undefined {
  return row.rawSchema ?? row.schema;
}

function compileDatasetRow(row: SchemaDatasetRow, index: number): CompiledDatasetRow {
  try {
    const rawSchema = rowSchema(row);
    let plan: Parameters<typeof compileLayoutPlanProgram>[0] | undefined;
    let arkType: ArkValidator | null = null;

    if (rawSchema !== undefined) {
      const converted = jsonSchemaToType(rawSchema as any);
      if (converted === undefined) throw new Error("jsonSchemaToType returned undefined");
      arkType = converted as unknown as ArkValidator;
      const module = scope({ value: converted });
      const analysis = new SchemaAnalyzer().analyze(module, { mode: "binary" });
      if (!analysis.plan || analysis.errors.length > 0) {
        throw new Error(`SchemaAnalyzer failed: ${JSON.stringify(analysis.errors)}`);
      }
      plan = analysis.plan as Parameters<typeof compileLayoutPlanProgram>[0];
    } else if (row.analysis?.plan) {
      // Old fixture fallback: usable for constraint debugging, but cannot validate
      // the generated value without the original schema.
      plan = row.analysis.plan;
    } else {
      throw new Error("dataset row has neither rawSchema/schema nor analysis.plan");
    }

    let sampleError: string | null = null;
    if (arkType) {
      try {
        arkType.assert(row.value);
      } catch (error) {
        sampleError = error instanceof Error ? error.message : String(error);
      }
    }

    return {
      row,
      plan,
      program: compileLayoutPlanProgram(plan),
      arkType,
      sampleError,
      error: null,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.debug(`[chomato] dataset row ${index} unsupported`, message);
    return { row, plan: null, program: null, arkType: null, sampleError: null, error: message };
  }
}

async function loadStructuredDataset(): Promise<CompiledDatasetRow[]> {
  const url = new URL("./schema-dataset.jsonl", import.meta.url);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load schema dataset: HTTP ${response.status}`);
  const text = await response.text();
  return text.split(/\r?\n/).filter(Boolean).map((line, index) =>
    compileDatasetRow(JSON.parse(line) as SchemaDatasetRow, index)
  );
}

function selectedDatasetIndex(): number {
  const index = Number(structuredDatasetEl.value);
  if (!Number.isInteger(index) || index < 0 || index >= structuredDataset.length) {
    throw new Error("No schema dataset row selected");
  }
  return index;
}

function selectedDatasetRow(): CompiledDatasetRow {
  return structuredDataset[selectedDatasetIndex()]!;
}

function datasetPrompt(row: SchemaDatasetRow): string {
  const description = row.sampleText?.trim() || row.task;
  return `Convert the following description into a compact JSON object for ${row.task}. Output only the JSON object.\n\n${description}`;
}

function renderProgramSummary(item: CompiledDatasetRow): string {
  const { row, program, error } = item;
  const lines = [
    `row #${row.id} · seed ${row.seed}`,
    `task: ${row.task}`,
  ];
  if (program) {
    const s = program.summary;
    lines.push(
      `SUPPORTED · root ${s.rootType} · ${s.segments} segments`,
      `fields ${s.fields} · enum ${s.enums} · string ${s.strings} · number ${s.numbers} · bool ${s.booleans} · fixed arrays ${s.arrays}`,
      `optional included ${s.optionalIncluded} · skipped unsupported optional ${s.optionalSkipped}`,
      `validator ${item.arkType ? "ArkType ready" : "unavailable (legacy row without schema)"}`,
      `sample value ${item.arkType ? item.sampleError ? `ArkType FAIL · ${item.sampleError}` : "ArkType PASS" : "not checked"}`,
    );
  } else {
    lines.push(`UNSUPPORTED · ${error}`);
  }
  lines.push(
    "",
    "schema:",
    JSON.stringify(row.rawSchema ?? row.schema ?? null, null, 2),
    "",
    "sample value:",
    JSON.stringify(row.value, null, 2),
    "",
    "LayoutPlan:",
    JSON.stringify(item.plan, null, 2),
  );
  return lines.join("\n");
}

function updateStructuredSelection(index = selectedDatasetIndex()): void {
  if (structuredDataset.length === 0) return;
  const clamped = Math.max(0, Math.min(structuredDataset.length - 1, index));
  structuredDatasetEl.value = String(clamped);
  const item = structuredDataset[clamped]!;
  structuredPromptEl.value = datasetPrompt(item.row);
  structuredPlanEl.textContent = renderProgramSummary(item);
  structuredTraceEl.textContent = "No constrained run yet.";
  structuredOutputEl.textContent = "Ready.";
  if (item.program) {
    const s = item.program.summary;
    structuredStatusEl.textContent = `supported · ${s.fields} fields · ${s.segments} segments`;
    structuredStatusEl.className = "ok";
  } else {
    structuredStatusEl.textContent = item.error ?? "unsupported";
    structuredStatusEl.className = "error";
  }
  runStructuredEl.disabled = structuredBatchRunning || !item.program;
  runStructuredAllEl.disabled = structuredBatchRunning || !structuredDataset.some((entry) => entry.program && entry.arkType);
  stopStructuredAllEl.disabled = !structuredBatchRunning;
}

function moveStructuredSelection(direction: -1 | 1): void {
  if (structuredDataset.length === 0) return;
  const start = selectedDatasetIndex();
  for (let n = 1; n <= structuredDataset.length; n++) {
    const index = (start + direction * n + structuredDataset.length) % structuredDataset.length;
    if (structuredDataset[index]!.program) {
      updateStructuredSelection(index);
      return;
    }
  }
}

function randomSupportedDatasetRow(): void {
  const supported = structuredDataset.map((item, index) => item.program && item.program.summary.fields > 0 ? index : -1).filter((index) => index >= 0);
  if (supported.length === 0) return;
  updateStructuredSelection(supported[Math.floor(Math.random() * supported.length)]!);
}

async function initializeStructuredDataset(): Promise<void> {
  structuredStatusEl.textContent = "loading schema-dataset.jsonl…";
  structuredDataset = await loadStructuredDataset();
  structuredDatasetEl.replaceChildren();
  let supported = 0;
  for (let index = 0; index < structuredDataset.length; index++) {
    const item = structuredDataset[index]!;
    const option = document.createElement("option");
    option.value = String(index);
    const useful = item.program && item.program.summary.fields > 0;
    option.textContent = `${useful ? "✓" : item.program ? "·" : "×"} #${item.row.id} · ${item.row.task}`;
    if (useful) supported++;
    structuredDatasetEl.append(option);
  }
  structuredDatasetEl.disabled = false;
  structuredPrevEl.disabled = false;
  structuredRandomEl.disabled = supported === 0;
  structuredNextEl.disabled = false;
  runStructuredAllEl.disabled = !structuredDataset.some((item) => item.program && item.arkType);
  const firstSupported = structuredDataset.findIndex((item) => item.program !== null && item.program.summary.fields > 0);
  updateStructuredSelection(firstSupported >= 0 ? firstSupported : 0);
  console.log(`[chomato] structured dataset ${supported}/${structuredDataset.length} non-trivial rows supported by current LayoutPlan subset`);
}

await initializeStructuredDataset().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  structuredStatusEl.textContent = message;
  structuredStatusEl.className = "error";
  structuredPlanEl.textContent = message;
  structuredOutputEl.textContent = "Dataset load failed.";
  console.error(error);
});

function renderConstraintTrace(trace: readonly ConstraintTraceStep[]): string {
  if (trace.length === 0) return "No constrained steps yet.";
  const lines: string[] = [];
  for (const item of trace) {
    lines.push(
      `step ${item.step} · ${item.decode ? "decode" : "prefill"} · allowed ${item.allowedCount}`,
      `  prefix: ${JSON.stringify(item.prefixBefore)} -> ${JSON.stringify(item.prefixAfter)}`,
      `  state:  ${item.stateBefore}`,
      `       -> ${item.stateAfter}`,
      `  chosen: #${item.selectedToken} ${JSON.stringify(item.selectedBytes)}${item.complete ? " · COMPLETE" : ""}`,
    );
    if (item.survivingCandidates) {
      lines.push(`  survivors: ${item.survivingCandidates.map((candidate) => JSON.stringify(candidate)).join(" | ") || "<none>"}`);
    }
    if (item.topBefore.length > 0) {
      lines.push("  top logits before mask:");
      for (const top of item.topBefore) {
        lines.push(`    ${top.allowed ? "ALLOW " : "reject"} #${top.token.toString().padEnd(5)} ${top.logit.toFixed(3).padStart(9)}  ${JSON.stringify(top.bytes)}`);
      }
      lines.push("  top allowed:");
      for (const top of item.topAllowed) {
        lines.push(`    #${top.token.toString().padEnd(5)} ${top.logit.toFixed(3).padStart(9)}  ${JSON.stringify(top.bytes)}`);
      }
    }
    lines.push("");
  }
  return lines.join("\n");
}

type BlockRole = "system" | "context";

function semanticBlockText(text: string, role: BlockRole): string {
  // SYSTEM is a complete ChatML message. CONTEXT deliberately stays raw so
  // multiple resident context blocks can still share one live user/Context
  // frame, preserving the already-working block-store path.
  if (role === "system") return tokenizer.formatMessage("system", text);
  return text;
}

function semanticBlockTokens(text: string, role: BlockRole): number[] {
  return tokenizer.encode(semanticBlockText(text, role), { addBos: false, addEos: false, parseSpecial: true });
}

function updateBudget() {
  try {
    const tokenCount = tokenizer.encodeUserPrompt(promptEl.value).length;
    const maxPossible = Math.max(0, contextCapacity - tokenCount + 1);
    const allowed = Math.min(runtimeMaxNewTokens, maxPossible);
    budgetEl.textContent = `${tokenCount} prompt tokens · up to ${allowed} new`;
    budgetEl.classList.toggle("error", allowed < 1);
  } catch {
    budgetEl.textContent = "tokenization error";
    budgetEl.classList.add("error");
  }
}

function updateBlockBudget() {
  try {
    const count = semanticBlockTokens(blockInputEl.value, blockRoleEl.value as BlockRole).length;
    const chunks = count === 0 ? 0 : Math.ceil(count / runtime.blockCacheMaxTokens);
    const depth = Number(blockDepthEl.value || runtime.blockCacheDepth);
    blockBudgetEl.textContent = `${count} tokens · ${chunks} block${chunks === 1 ? "" : "s"} · depth ${depth}`;
    blockBudgetEl.classList.toggle("error", count < 1);
  } catch {
    blockBudgetEl.textContent = "tokenization error";
    blockBudgetEl.classList.add("error");
  }
}

promptEl.addEventListener("input", updateBudget);
maxTokensEl.addEventListener("input", updateBudget);
blockInputEl.addEventListener("input", updateBlockBudget);
blockRoleEl.addEventListener("change", updateBlockBudget);
blockDepthEl.addEventListener("change", updateBlockBudget);
updateBudget();
updateBlockBudget();

function renderRequestStats(result: GenerateResult, wallMs: number, promptTokens: number) {
  const generated = result.state.generatedCount;
  const decodeTokens = Math.max(0, generated - 1); // first output token is sampled by prefill
  statPrompt.textContent = `${promptTokens} tok`;
  statOutput.textContent = `${generated} tok`;
  statRequestStatus.textContent = statusName(result.state.status);

  if (result.timings) {
    const t = result.timings;
    statPrefill.textContent = `${ms(t.prefillMs)} · ${rate(promptTokens, t.prefillMs)}`;
    statDecode.textContent = `${ms(t.decodeMs)} · ${rate(decodeTokens, t.decodeMs)}`;
    statTotal.textContent = `${ms(t.totalMs)} · ${rate(generated, t.totalMs)}`;
    statCacheRepair.textContent = t.cacheDepth !== undefined
      ? `${t.repairedTokens ?? 0} tok · depth ${t.cacheDepth}`
      : "—";
  } else {
    statPrefill.textContent = "profiling off";
    statDecode.textContent = "profiling off";
    statTotal.textContent = `${ms(wallMs)} · ${rate(generated, wallMs)}`;
    statCacheRepair.textContent = "profiling off";
  }
}

function beginRequest(label: string, promptTokens: number) {
  runEl.disabled = true;
  runStructuredEl.disabled = true;
  runStructuredAllEl.disabled = true;
  stopStructuredAllEl.disabled = !structuredBatchRunning;
  addBlockEl.disabled = true;
  clearBlocksEl.disabled = true;
  runBlocksEl.disabled = true;
  promptEl.disabled = true;
  structuredPromptEl.disabled = true;
  structuredDatasetEl.disabled = true;
  structuredPrevEl.disabled = true;
  structuredRandomEl.disabled = true;
  structuredNextEl.disabled = true;
  structuredMaxTokensEl.disabled = true;
  blockInputEl.disabled = true;
  blockRoleEl.disabled = true;
  blockDepthEl.disabled = true;
  blockOrderEl.disabled = true;
  blockPromptEl.disabled = true;
  maxTokensEl.disabled = true;
  profileEl.disabled = true;
  setStatus(label);
  outputEl.textContent = "…";
  statPrompt.textContent = `${promptTokens} tok`;
  statOutput.textContent = "…";
  statPrefill.textContent = "…";
  statDecode.textContent = "…";
  statTotal.textContent = "…";
  statRequestStatus.textContent = "RUNNING";
}

function endRequest() {
  runEl.disabled = false;
  runStructuredEl.disabled = false;
  runStructuredAllEl.disabled = structuredBatchRunning || !structuredDataset.some((item) => item.program && item.arkType);
  stopStructuredAllEl.disabled = !structuredBatchRunning;
  addBlockEl.disabled = false;
  clearBlocksEl.disabled = storedBlocks.size === 0;
  runBlocksEl.disabled = storedBlocks.size === 0;
  promptEl.disabled = false;
  structuredPromptEl.disabled = false;
  structuredDatasetEl.disabled = structuredDataset.length === 0;
  structuredPrevEl.disabled = structuredDataset.length === 0;
  structuredRandomEl.disabled = !structuredDataset.some((item) => item.program && item.program.summary.fields > 0);
  structuredNextEl.disabled = structuredDataset.length === 0;
  structuredMaxTokensEl.disabled = false;
  if (structuredDataset.length > 0) runStructuredEl.disabled = !selectedDatasetRow().program;
  blockInputEl.disabled = false;
  blockRoleEl.disabled = false;
  blockDepthEl.disabled = false;
  blockOrderEl.disabled = false;
  blockPromptEl.disabled = false;
  maxTokensEl.disabled = false;
  profileEl.disabled = false;
  updateBudget();
  updateBlockBudget();
  }

async function runPrompt(): Promise<void> {
  const prompt = promptEl.value;
  const promptTokens = tokenizer.encodeUserPrompt(prompt);
  const requested = Math.max(1, Math.floor(Number(maxTokensEl.value) || runtimeMaxNewTokens));
  const maxPossible = contextCapacity - promptTokens.length + 1;
  const maxNewTokens = Math.min(requested, runtimeMaxNewTokens, maxPossible);

  if (maxNewTokens < 1) throw new Error(`Prompt has ${promptTokens.length} tokens and does not fit context ${contextCapacity}`);
  if (maxNewTokens !== requested) maxTokensEl.value = String(maxNewTokens);

  beginRequest("inference…", promptTokens.length);
  console.log("[chomato] prompt", prompt);
  console.log(`[chomato] prompt tokens ${promptTokens.length}`, promptTokens);

  const started = performance.now();
  try {
    const result = await runtime.generateTokens(promptTokens, { maxNewTokens, profile: profileEl.checked });
    const elapsed = performance.now() - started;
    const text = tokenizer.decode(result.tokenIds);
    outputEl.textContent = text;
    renderRequestStats(result, elapsed, promptTokens.length);
    setStatus("ready", "ok");
    console.log("\n--- GPU output ---\n" + text + "\n------------------");
    console.log("[chomato] telemetry", { timings: result.timings, tokenIds: result.tokenIds, state: result.state });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    outputEl.textContent = message;
    setStatus("request failed", "error");
    statRequestStatus.textContent = "ERROR";
    console.error(error);
  } finally {
    endRequest();
  }
}

interface ValidationResult {
  ok: boolean;
  value?: unknown;
  error?: string;
}

function validateStructuredText(item: CompiledDatasetRow, text: string): ValidationResult {
  if (!item.arkType) return { ok: false, error: "ArkType validator unavailable for this dataset row" };
  try {
    const value = JSON.parse(text);
    item.arkType.assert(value);
    return { ok: true, value };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) };
  }
}

interface StructuredRunResult {
  kind: "pass" | "validation-fail" | "incomplete" | "decode-fail";
  text: string;
  elapsedMs: number;
  generated: number;
  validation?: ValidationResult;
  error?: string;
  trace: readonly ConstraintTraceStep[];
}

async function executeStructuredRow(
  item: CompiledDatasetRow,
  prompt: string,
  requestedMaxTokens: number,
  renderDetails: boolean,
): Promise<StructuredRunResult> {
  if (!item.program || !item.plan) {
    return {
      kind: "decode-fail",
      text: "",
      elapsedMs: 0,
      generated: 0,
      error: item.error ?? "Selected LayoutPlan is unsupported",
      trace: [],
    };
  }

  const promptTokens = tokenizer.encodeUserPrompt(prompt);
  const maxPossible = contextCapacity - promptTokens.length + 1;
  const maxNewTokens = Math.min(requestedMaxTokens, runtimeMaxNewTokens, maxPossible);
  if (maxNewTokens < 1) {
    return {
      kind: "decode-fail",
      text: "",
      elapsedMs: 0,
      generated: 0,
      error: `Structured prompt has ${promptTokens.length} tokens and does not fit context ${contextCapacity}`,
      trace: [],
    };
  }

  const constraint = createLayoutPlanJsonConstraint({
    plan: item.plan,
    tokens: getConstraintTokenTable(),
    eosToken: tokenizer.eos,
  });

  if (renderDetails) {
    structuredStatusEl.textContent = `running · row #${item.row.id} · ${constraint.program.summary.segments} segments`;
    structuredStatusEl.className = "muted";
    structuredOutputEl.textContent = "…";
    structuredTraceEl.textContent = "CPU constraint → sparse candidate ids → GPU argmax…";
  }

  const started = performance.now();
  try {
    const result = await runtime.generateTokensWithCpuCandidates(promptTokens, constraint, { maxNewTokens });
    const elapsedMs = performance.now() - started;
    const text = tokenizer.decode(result.tokenIds);
    const validation = validateStructuredText(item, text);
    const kind: StructuredRunResult["kind"] = !constraint.done
      ? "incomplete"
      : validation.ok
        ? "pass"
        : "validation-fail";

    if (renderDetails) {
      structuredOutputEl.textContent = text;
      structuredTraceEl.textContent = renderConstraintTrace(constraint.trace);
      const validationText = validation.ok ? "ArkType PASS" : `ArkType FAIL · ${validation.error}`;
      structuredStatusEl.textContent = `${constraint.done ? "complete + EOS" : "stopped"} · ${validationText} · row #${item.row.id} · ${result.state.generatedCount} tok · ${ms(elapsedMs)}`;
      structuredStatusEl.className = kind === "pass" ? "ok" : "error";
      renderRequestStats(result, elapsedMs, promptTokens.length);
      outputEl.textContent = text;
    }

    console.log("[chomato] LayoutPlan constrained output", {
      row: item.row,
      program: constraint.program.summary,
      text,
      validation,
      tokenIds: result.tokenIds,
      state: result.state,
      trace: constraint.trace,
    });

    return {
      kind,
      text,
      elapsedMs,
      generated: result.state.generatedCount,
      validation,
      trace: constraint.trace,
    };
  } catch (error) {
    return {
      kind: "decode-fail",
      text: "",
      elapsedMs: performance.now() - started,
      generated: constraint.trace.length,
      error: error instanceof Error ? error.message : String(error),
      trace: constraint.trace,
    };
  }
}

function renderBatchResults(
  current: number,
  total: number,
  results: Array<{ id: number; task: string; result: StructuredRunResult }>,
  unsupported: number,
): void {
  const counts = { pass: 0, "validation-fail": 0, incomplete: 0, "decode-fail": 0 };
  let tokens = 0;
  let elapsed = 0;
  for (const entry of results) {
    counts[entry.result.kind]++;
    tokens += entry.result.generated;
    elapsed += entry.result.elapsedMs;
  }
  const lines = [
    `progress ${current}/${total}${structuredBatchStopRequested ? " · stopping after current row" : ""}`,
    `PASS ${counts.pass} · validation fail ${counts["validation-fail"]} · incomplete ${counts.incomplete} · decode fail ${counts["decode-fail"]} · unsupported skipped ${unsupported}`,
    `generated ${tokens} tok · ${elapsed > 0 ? rate(tokens, elapsed) : "—"} · ${ms(elapsed)}`,
    "",
  ];
  for (const entry of results) {
    const suffix = entry.result.kind === "validation-fail"
      ? ` · ${entry.result.validation?.error ?? "validation failed"}`
      : entry.result.error
        ? ` · ${entry.result.error}`
        : "";
    lines.push(`${entry.result.kind === "pass" ? "PASS" : "FAIL"} #${entry.id} · ${entry.result.kind} · ${entry.result.generated} tok · ${ms(entry.result.elapsedMs)} · ${entry.task}${suffix}`);
  }
  structuredBatchEl.textContent = lines.join("\n");
}

async function runStructuredPrompt(): Promise<void> {
  const item = selectedDatasetRow();
  if (!item.program) throw new Error(item.error ?? "Selected LayoutPlan is unsupported");
  const prompt = structuredPromptEl.value.trim();
  const requested = Math.max(1, Math.floor(Number(structuredMaxTokensEl.value) || 64));
  const promptTokens = tokenizer.encodeUserPrompt(prompt);

  beginRequest("LayoutPlan CPU constrained decode…", promptTokens.length);
  try {
    const result = await executeStructuredRow(item, prompt, requested, true);
    item.lastResult = result.kind;
    if (result.kind === "pass") {
      setStatus("ready", "ok");
    } else {
      setStatus(`structured ${result.kind}`, "error");
      statRequestStatus.textContent = "ERROR";
      if (result.error) {
        structuredOutputEl.textContent = result.error;
        structuredStatusEl.textContent = `${result.kind} · ${result.error}`;
        structuredStatusEl.className = "error";
      }
      if (result.trace.length > 0) structuredTraceEl.textContent = renderConstraintTrace(result.trace);
    }
  } finally {
    endRequest();
  }
}

async function runAllStructuredRows(): Promise<void> {
  if (structuredBatchRunning) return;
  const runnable = structuredDataset
    .map((item, index) => ({ item, index }))
    .filter(({ item }) => item.program !== null && item.arkType !== null);
  if (runnable.length === 0) throw new Error("No supported dataset rows with ArkType validators");

  const unsupported = structuredDataset.length - runnable.length;
  const requested = Math.max(1, Math.floor(Number(structuredMaxTokensEl.value) || 64));
  const results: Array<{ id: number; task: string; result: StructuredRunResult }> = [];
  structuredBatchRunning = true;
  structuredBatchStopRequested = false;
  beginRequest("structured dataset runner…", 0);
  stopStructuredAllEl.disabled = false;
  structuredBatchEl.textContent = `starting 0/${runnable.length} · unsupported skipped ${unsupported}`;

  try {
    for (let i = 0; i < runnable.length; i++) {
      if (structuredBatchStopRequested) break;
      const { item, index } = runnable[i]!;
      structuredDatasetEl.value = String(index);
      updateStructuredSelection(index);
      const prompt = datasetPrompt(item.row);
      structuredPromptEl.value = prompt;
      structuredStatusEl.textContent = `batch ${i + 1}/${runnable.length} · row #${item.row.id}`;
      structuredStatusEl.className = "muted";

      const result = await executeStructuredRow(item, prompt, requested, true);
      item.lastResult = result.kind;
      results.push({ id: item.row.id, task: item.row.task, result });
      renderBatchResults(i + 1, runnable.length, results, unsupported);
    }

    const failed = results.some((entry) => entry.result.kind !== "pass");
    const stopped = structuredBatchStopRequested;
    structuredStatusEl.textContent = stopped
      ? `batch stopped · ${results.length}/${runnable.length}`
      : `batch complete · ${results.length}/${runnable.length} · ${failed ? "FAILURES" : "ALL PASS"}`;
    structuredStatusEl.className = !stopped && !failed ? "ok" : stopped ? "muted" : "error";
    setStatus(!stopped && !failed ? "ready" : stopped ? "batch stopped" : "batch completed with failures", !stopped && !failed ? "ok" : "error");
  } finally {
    structuredBatchRunning = false;
    stopStructuredAllEl.disabled = true;
    endRequest();
  }
}

interface UiBlock {
  id: number;
  role: BlockRole;
  depth: number;
  text: string;
  tokens: number[];
  cached: Lfm2CachedBlock;
}

const storedBlocks = new Map<number, UiBlock>();
let nextUiBlockId = 1;
const bosBlocksByDepth = new Map<number, Lfm2CachedBlock>();
const contextPrefixBlocksByDepth = new Map<number, Lfm2CachedBlock>();
let lastCacheBuildMs = 0;

function totalBlockVram(): number {
  let bytes = 0;
  for (const item of storedBlocks.values()) bytes += item.cached.gpuBytes;
  return bytes;
}

function escapePreview(text: string): string {
  return text.replace(/\s+/g, " ").trim() || "(whitespace / special tokens)";
}

function renderBlockStore() {
  blockListEl.replaceChildren();
  if (storedBlocks.size === 0) {
    const empty = document.createElement("span");
    empty.className = "muted";
    empty.textContent = "Add blocks to start.";
    blockListEl.append(empty);
  } else {
    for (const item of storedBlocks.values()) {
      const row = document.createElement("div");
      row.className = "block-row";

      const id = document.createElement("span");
      id.className = "block-id";
      id.textContent = `#${item.id}`;

      const tokens = document.createElement("span");
      tokens.textContent = `${item.tokens.length} tok`;

      const depth = document.createElement("span");
      depth.className = "muted";
      depth.textContent = `d${item.depth}`;

      const vram = document.createElement("span");
      vram.className = "vram muted";
      vram.textContent = mib(item.cached.gpuBytes);

      const role = document.createElement("span");
      role.className = "block-role";
      role.textContent = item.role;

      const preview = document.createElement("span");
      preview.className = "preview";
      preview.title = item.text;
      preview.textContent = escapePreview(item.text);

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "secondary";
      remove.textContent = "remove";
      remove.addEventListener("click", () => removeBlock(item.id));

      row.append(id, tokens, depth, vram, role, preview, remove);
      blockListEl.append(row);
    }
  }

  statCacheBlocks.textContent = String(storedBlocks.size);
  statCacheVram.textContent = mib(totalBlockVram());
  statCacheBuild.textContent = lastCacheBuildMs > 0 ? ms(lastCacheBuildMs) : "—";
  clearBlocksEl.disabled = storedBlocks.size === 0;
  runBlocksEl.disabled = storedBlocks.size === 0;
}

async function addBlocks(
  text = blockInputEl.value,
  role: BlockRole = blockRoleEl.value as BlockRole,
  depth = Number(blockDepthEl.value || runtime.blockCacheDepth),
): Promise<UiBlock[]> {
  if (!text.trim()) throw new Error("Block payload is empty");
  const tokens = semanticBlockTokens(text, role);
  if (tokens.length < 1) throw new Error("Block is empty after semantic formatting/tokenization");
  if (!runtime.blockCacheDepths.includes(depth)) {
    throw new Error(`Unsupported cache depth ${depth}; choose ${runtime.blockCacheDepths.join(", ")}`);
  }

  const chunks: number[][] = [];
  for (let start = 0; start < tokens.length; start += runtime.blockCacheMaxTokens) {
    chunks.push(tokens.slice(start, Math.min(start + runtime.blockCacheMaxTokens, tokens.length)));
  }

  addBlockEl.disabled = true;
  runBlocksEl.disabled = true;
  const started = performance.now();
  const built: UiBlock[] = [];
  try {
    for (let index = 0; index < chunks.length; index++) {
      const chunk = chunks[index]!;
      setBlockStatus(`caching ${index + 1}/${chunks.length} · ${chunk.length} tokens · depth ${depth}…`);
      const cached = await runtime.cacheBlock(chunk, { depth });
      const item: UiBlock = {
        id: nextUiBlockId++,
        role,
        depth,
        text: tokenizer.decode(chunk, { skipSpecial: false }),
        tokens: chunk,
        cached,
      };
      storedBlocks.set(item.id, item);
      built.push(item);
      renderBlockStore();
    }

    const elapsed = performance.now() - started;
    lastCacheBuildMs = elapsed;
    blockInputEl.value = "";
    if (!blockOrderEl.value.trim()) blockOrderEl.value = built.map((item) => item.id).join(", ");
    setBlockStatus(
      `added ${built.length} block${built.length === 1 ? "" : "s"} · ${tokens.length} tok · depth ${depth} · ${ms(elapsed)}`,
      "ok",
    );
    renderBlockStore();
    updateBlockBudget();
    console.log("[chomato] cached blocks", {
      ids: built.map((item) => item.id),
      runtimeIds: built.map((item) => item.cached.id),
      role,
      depth,
      tokens: tokens.length,
      chunks: built.length,
      gpuMiB: Number((built.reduce((sum, item) => sum + item.cached.gpuBytes, 0) / 1048576).toFixed(2)),
      cacheMs: Number(elapsed.toFixed(1)),
    });
    return built;
  } catch (error) {
    for (const item of built) {
      runtime.destroyCachedBlock(item.cached);
      storedBlocks.delete(item.id);
    }
    renderBlockStore();
    setBlockStatus(error instanceof Error ? error.message : String(error), "error");
    throw error;
  } finally {
    addBlockEl.disabled = false;
    runBlocksEl.disabled = storedBlocks.size === 0;
  }
}

function removeBlock(id: number): void {
  const item = storedBlocks.get(id);
  if (!item) return;
  runtime.destroyCachedBlock(item.cached);
  storedBlocks.delete(id);
  setBlockStatus(`removed #${id}`);
  renderBlockStore();
}

function clearBlocks(): void {
  for (const item of storedBlocks.values()) runtime.destroyCachedBlock(item.cached);
  storedBlocks.clear();
  lastCacheBuildMs = 0;
  setBlockStatus("no cached blocks");
  renderBlockStore();
}

function parseBlockOrder(value: string): UiBlock[] {
  const ids = value.split(/[\s,;]+/).map((x) => x.trim()).filter(Boolean).map(Number);
  if (ids.length === 0) throw new Error("Enter a block order, e.g. 1, 3, 2");
  const result: UiBlock[] = [];
  for (const id of ids) {
    if (!Number.isInteger(id) || id < 1) throw new Error(`Invalid block id '${id}'`);
    const block = storedBlocks.get(id);
    if (!block) throw new Error(`Block #${id} does not exist`);
    result.push(block);
  }
  return result;
}

async function getBosBlock(depth: number): Promise<Lfm2CachedBlock> {
  let block = bosBlocksByDepth.get(depth);
  if (!block) {
    block = await runtime.cacheBlock([tokenizer.bos], { depth });
    bosBlocksByDepth.set(depth, block);
  }
  return block;
}

async function getContextPrefixBlock(depth: number): Promise<Lfm2CachedBlock> {
  let block = contextPrefixBlocksByDepth.get(depth);
  if (!block) {
    block = await runtime.cacheBlock(
      tokenizer.encode(`<|im_start|>user\nContext:\n`, { addBos: false, addEos: false, parseSpecial: true }),
      { depth },
    );
    contextPrefixBlocksByDepth.set(depth, block);
  }
  return block;
}

function encodeLiveBlockPrompt(prompt: string, hasContext: boolean): number[] {
  const text = hasContext
    ? `\n\nQuery:\n${prompt}\n<|im_end|>\n<|im_start|>assistant\n`
    : tokenizer.formatMessage("user", prompt) + `<|im_start|>assistant\n`;
  return tokenizer.encode(text, { addBos: false, addEos: false, parseSpecial: true });
}

async function runBlockOrder(order = blockOrderEl.value, prompt = blockPromptEl.value): Promise<GenerateResult> {
  const selected = parseBlockOrder(order);
  if (!prompt.trim()) throw new Error("Live prompt is empty");

  // v1 semantic contract:
  //   BOS? -> SYSTEM message blocks -> one shared USER/Context frame -> raw
  //   CONTEXT blocks -> live Query suffix -> ASSISTANT frontier.
  // Keeping context blocks raw preserves the path that already combines facts
  // correctly. Until block grouping is explicit, SYSTEM must precede CONTEXT.
  let seenContext = false;
  for (const block of selected) {
    if (block.role === "context") seenContext = true;
    else if (seenContext) {
      throw new Error("Semantic v1 requires SYSTEM blocks before CONTEXT blocks; grouping comes next");
    }
  }

  const systemBlocks = selected.filter((x) => x.role === "system");
  const contextBlocks = selected.filter((x) => x.role === "context");
  // Structural helper blocks are cached directly at the request frontier.
  // In particular, a depth-5 request now reproduces the original proven path
  // instead of relying on a depth-5 checkpoint taken from a deeper helper.
  const requestedDepth = Math.min(...selected.map((x) => x.cached.cacheDepth));
  const blocks: Lfm2CachedBlock[] = [];
  if (prependBosEl.checked) blocks.push(await getBosBlock(requestedDepth));
  blocks.push(...systemBlocks.map((x) => x.cached));
  if (contextBlocks.length > 0) blocks.push(await getContextPrefixBlock(requestedDepth));
  blocks.push(...contextBlocks.map((x) => x.cached));
  const queryTokens = encodeLiveBlockPrompt(prompt, contextBlocks.length > 0);

  const cachedTokens = blocks.reduce((sum, block) => sum + block.tokenCount, 0);
  const promptTokens = cachedTokens + queryTokens.length;
  const requested = Math.max(1, Math.floor(Number(maxTokensEl.value) || runtimeMaxNewTokens));
  const maxPossible = contextCapacity - promptTokens + 1;
  const maxNewTokens = Math.min(requested, runtimeMaxNewTokens, maxPossible);
  if (maxNewTokens < 1) throw new Error(`Block context + live prompt has ${promptTokens} tokens and fills context ${contextCapacity}`);

  const effectiveDepth = Math.min(...blocks.map((block) => block.cacheDepth));
  beginRequest("block context + live prompt…", promptTokens);
  setBlockStatus(
    `running [${selected.map((x) => `${x.id}:${x.role}@${x.depth}`).join(", ")}] + ${queryTokens.length} live tok · effective depth ${effectiveDepth}`,
  );
  const started = performance.now();
  try {
    const result = await runtime.generateFromBlocksWithQuery(
      blocks, queryTokens, { maxNewTokens, profile: profileEl.checked },
    );
    const elapsed = performance.now() - started;
    const text = tokenizer.decode(result.tokenIds);
    outputEl.textContent = text;
    renderRequestStats(result, elapsed, promptTokens);
    setStatus("ready", "ok");
    setBlockStatus(
      `last query [${selected.map((x) => `${x.id}:${x.role}@${x.depth}`).join(", ")}] · ${cachedTokens} cached + ${queryTokens.length} live tok · effective depth ${result.timings?.cacheDepth ?? effectiveDepth}`,
      "ok",
    );
    renderBlockStore();
    console.log("\n--- GPU block + live query output ---\n" + text + "\n-------------------------------------");
    console.log("[chomato] block live query", {
      uiOrder: selected.map((x) => ({ id: x.id, role: x.role, depth: x.depth })),
      effectiveDepth: result.timings?.cacheDepth ?? effectiveDepth,
      runtimeBlocks: blocks.map((x) => x.id),
      cachedTokens,
      liveQueryTokens: queryTokens.length,
      promptTokens,
      prompt,
      timings: result.timings,
      tokenIds: result.tokenIds,
      state: result.state,
    });
    return result;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    outputEl.textContent = message;
    setStatus("block request failed", "error");
    setBlockStatus(message, "error");
    statRequestStatus.textContent = "ERROR";
    console.error(error);
    throw error;
  } finally {
    endRequest();
  }
}

structuredDatasetEl.addEventListener("change", () => updateStructuredSelection());
structuredPrevEl.addEventListener("click", () => moveStructuredSelection(-1));
structuredRandomEl.addEventListener("click", randomSupportedDatasetRow);
structuredNextEl.addEventListener("click", () => moveStructuredSelection(1));
runStructuredAllEl.addEventListener("click", () => {
  void runAllStructuredRows().catch((error) => {
    const message = error instanceof Error ? error.message : String(error);
    structuredStatusEl.textContent = message;
    structuredStatusEl.className = "error";
    console.error(error);
  });
});
stopStructuredAllEl.addEventListener("click", () => {
  structuredBatchStopRequested = true;
  stopStructuredAllEl.disabled = true;
});

runStructuredEl.addEventListener("click", () => {
  void runStructuredPrompt().catch((error) => {
    const message = error instanceof Error ? error.message : String(error);
    structuredStatusEl.textContent = message;
    structuredStatusEl.className = "error";
    structuredOutputEl.textContent = message;
    console.error(error);
  });
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  void runPrompt();
});
addBlockEl.addEventListener("click", () => void addBlocks().catch(() => {}));
clearBlocksEl.addEventListener("click", clearBlocks);
runBlocksEl.addEventListener("click", () => void runBlockOrder().catch(() => {}));
blockOrderEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    void runBlockOrder().catch(() => {});
  }
});

renderBlockStore();

// Keep the heavy objects resident and expose a console API for experiments.
Object.assign(globalThis, {
  chomato: {
    engine,
    model,
    runtime,
    tokenizer,
    generate: async (prompt: string, maxNewTokens = runtimeMaxNewTokens, profile = true) => {
      const tokens = tokenizer.encodeUserPrompt(prompt);
      return await runtime.generateTokens(tokens, { maxNewTokens, profile });
    },
    cacheTokens: async (tokens: readonly number[], depth = runtime.blockCacheDepth) => await runtime.cacheBlock(tokens, { depth }),
    generateBlocks: async (blocks: readonly Lfm2CachedBlock[], maxNewTokens = runtimeMaxNewTokens, profile = true) =>
      await runtime.generateFromBlocks(blocks, { maxNewTokens, profile }),
    generateBlocksQuery: async (blocks: readonly Lfm2CachedBlock[], query: string, maxNewTokens = runtimeMaxNewTokens, profile = true) =>
      await runtime.generateFromBlocksWithQuery(blocks, encodeLiveBlockPrompt(query, false), { maxNewTokens, profile }),
    destroyBlock: (block: Lfm2CachedBlock) => runtime.destroyCachedBlock(block),
    blockStore: {
      add: async (text: string, role: BlockRole = "context", depth = runtime.blockCacheDepth) => await addBlocks(text, role, depth),
      remove: (id: number) => removeBlock(id),
      clear: () => clearBlocks(),
      list: () => [...storedBlocks.values()].map(({ id, role, depth, text, tokens, cached }) => ({
        id, role, depth, text, tokenCount: tokens.length, gpuBytes: cached.gpuBytes, runtimeId: cached.id,
      })),
      run: async (order: string, prompt = blockPromptEl.value) => await runBlockOrder(order, prompt),
    },
    structuredDataset: () => structuredDataset.map((item, index) => ({
      index,
      id: item.row.id,
      seed: item.row.seed,
      task: item.row.task,
      supported: item.program !== null,
      summary: item.program?.summary,
      error: item.error,
    })),
    constrainLayout: async (index: number, prompt?: string, maxNewTokens = 64) => {
      const item = structuredDataset[index];
      if (!item) throw new Error(`Dataset index ${index} does not exist`);
      if (!item.program) throw new Error(item.error ?? `Dataset index ${index} is unsupported`);
      const constraint = createLayoutPlanJsonConstraint({
        plan: item.plan,
        tokens: getConstraintTokenTable(),
        eosToken: tokenizer.eos,
      });
      const promptTokens = tokenizer.encodeUserPrompt(prompt ?? datasetPrompt(item.row));
      const result = await runtime.generateTokensWithCpuCandidates(promptTokens, constraint, { maxNewTokens });
      return { result, text: tokenizer.decode(result.tokenIds), trace: constraint.trace, program: constraint.program.summary, row: item.row };
    },
    constrainEnum: async (prompt: string, property: string, variants: readonly string[], maxNewTokens = 32) => {
      const constraint = createLiteralEnumJsonConstraint({
        property, variants, tokens: getConstraintTokenTable(), eosToken: tokenizer.eos,
      });
      const promptTokens = tokenizer.encodeUserPrompt(prompt);
      const result = await runtime.generateTokensWithCpuLogits(promptTokens, constraint, { maxNewTokens });
      return { result, text: tokenizer.decode(result.tokenIds), trace: constraint.trace };
    },
    memory: {
      modelBytes: allocatedModelBytes,
      arenaBytes: runtime.arenaBuffer.size,
      kvBytes: runtime.kvBuffer.size,
      convBytes: runtime.convBuffer.size,
    },
  },
});

console.log(`[chomato] ready; model/runtime resident; BlockStore depths=${runtime.blockCacheDepths.join(",")} default=${runtime.blockCacheDepth}`);
console.log("[chomato] runtime buffers", {
  arena: mib(runtime.arenaBuffer.size),
  kv: mib(runtime.kvBuffer.size),
  conv: mib(runtime.convBuffer.size),
});
