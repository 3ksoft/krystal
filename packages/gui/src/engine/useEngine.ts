/**
 * Browser bring-up of the Chomato engine, exposed as engine primitives.
 *
 * The boot chain is deliberately the same one the browser harnesses in
 * packages/webgpu/test/browser use, so this GUI observes the real runtime
 * rather than a parallel implementation:
 *
 *   createWebGpuDevice -> Lfm2GpuModel.open -> Lfm2Forward
 *     -> createLfm2WebGpuTransport -> Engine
 *
 * Two things this layer adds on top of the Engine API:
 *
 * 1. Content. The engine stores blocks as opaque token ids, so the source text
 *    is kept here and a checkpoint snapshots the resolved text of everything it
 *    materializes. That snapshot is taken at creation time and does not follow
 *    the live block list, so a checkpoint stays readable after its blocks are
 *    dropped.
 * 2. Per-operation cost. Only cumulative counters are reported, so every action
 *    diffs engine.debug.stats() around the call. That is what makes "restored
 *    vs recomputed" legible per checkpoint and per generation.
 *
 * There is no global "current context": each panel owns a ContextSelection and
 * passes it in explicitly.
 */
import { reactive, readonly, shallowRef } from "vue";
import { type } from "arktype";
import { Engine, type EngineStats } from "@chomato/engine-ts/transport";
import { compileStructuredGeneration } from "@chomato/engine-ts";
import { BlobSource, HttpRangeSource } from "@chomato/quant";
import {
  createLfm2WebGpuTransport,
  createWebGpuDevice,
  Lfm2Forward,
  Lfm2GpuModel,
  lfm2,
} from "@chomato/webgpu";
import type { Lfm2Tokenizer } from "@chomato/lfm2";
import { contextIssues, type ContextPart } from "./context-rules.ts";
import {
  buildReservedTable,
  EMPTY_RESERVED,
  expandReserved,
  type ReservedTable,
} from "./reserved.ts";

const GIB = 1024 * 1024 * 1024;
const DEFAULT_MODEL_URL = "/models/LFM2.5-1.2B-Instruct-WQ4.wq4";

export type Phase = "idle" | "device" | "model" | "runtime" | "ready" | "busy" | "error";

/** What a panel wants the engine to run against. Panels do not share these. */
export interface ContextSelection {
  checkpoint: number | null;
  /** Ordered: this is the composition order the engine receives. */
  blocks: number[];
}

export function emptySelection(): ContextSelection {
  return { checkpoint: null, blocks: [] };
}

export interface CostDelta {
  prefillTokens: number;
  restoredCheckpointBytes: number;
  checkpointBytes: number;
  checkpointHits: number;
  checkpointMisses: number;
  checkpointCreateUs: number;
  checkpointRestoreUs: number;
  kvBytes: number;
  convBytes: number;
  hiddenBytes: number;
  generatedTokens: number;
  wallMs: number;
}

export interface BlockRow {
  id: number;
  label: string;
  text: string;
  tokens: number[];
  addBos: boolean;
  /** BOS sits at index 0 — correct only when this block starts the context. */
  bosLeading: boolean;
  /** BOS anywhere after index 0. Always wrong: it resets the model mid-context. */
  bosInterior: number;
}

/** One resolved piece of a checkpoint's context, snapshotted at creation. */
export interface CheckpointPart {
  blockId: number;
  label: string;
  text: string;
  tokenCount: number;
}

export interface CheckpointRow {
  id: number;
  label: string;
  /** Checkpoint this one was branched from, or null when rooted at raw blocks. */
  base: number | null;
  /** Blocks appended on top of `base`, in composition order. */
  appended: number[];
  /** Everything this checkpoint materializes, base lineage included. */
  contents: CheckpointPart[];
  text: string;
  position: number;
  bosLeading: boolean;
  bosInterior: number;
  cost: CostDelta;
}

export interface RunRow {
  id: number;
  kind: "structured" | "tokens";
  source: string;
  ok: boolean;
  detail: string;
  contextLabel: string;
  cost: CostDelta;
}

export interface ModelInfo {
  blockCount: number;
  hiddenSize: number;
  feedForwardSize: number;
  vocabSize: number;
  attentionHeads: number;
  headDim: number;
  bosToken: number;
  eosToken: number;
  layers: string[];
  allocatedBytes: number;
  tensorCount: number;
  loadMs: number;
  prepareMs: number;
  contextCapacity: number;
  maxNewTokens: number;
}

/** Shape of the linked GPU constraint program, as reported by its own summary. */
export interface ProgramInfo {
  blobBytes: number;
  blobWords: number;
  sourceNodes: number;
  nodes: number;
  literalNodes: number;
  switchNodes: number;
  stringNodes: number;
  numberNodes: number;
  jumpNodes: number;
  acceptNodes: number;
  edges: number;
  byteTableLength: number;
  maxJsonBytes: number;
  maxTokens: number;
}

function diffStats(before: EngineStats, after: EngineStats, wallMs: number): CostDelta {
  return {
    prefillTokens: after.prefillTokens - before.prefillTokens,
    restoredCheckpointBytes: after.restoredCheckpointBytes - before.restoredCheckpointBytes,
    checkpointBytes: after.checkpointBytes - before.checkpointBytes,
    checkpointHits: after.checkpointHits - before.checkpointHits,
    checkpointMisses: after.checkpointMisses - before.checkpointMisses,
    checkpointCreateUs: after.checkpointCreateUs - before.checkpointCreateUs,
    checkpointRestoreUs: after.checkpointRestoreUs - before.checkpointRestoreUs,
    kvBytes: after.kvBytes - before.kvBytes,
    convBytes: after.convBytes - before.convBytes,
    hiddenBytes: after.hiddenBytes - before.hiddenBytes,
    generatedTokens: after.generatedTokens - before.generatedTokens,
    wallMs,
  };
}

/**
 * Evaluate an arktype definition written by the user.
 *
 * This is a local developer tool driving a local GPU, so the source is
 * evaluated directly rather than through a restricted parser; `type` is the
 * only binding placed in scope.
 */
function evaluateSchema(source: string): unknown {
  const trimmed = source.trim();
  if (!trimmed) throw new Error("Schema is empty");
  const factory = new Function("type", `"use strict"; return (${trimmed});`) as (t: typeof type) => unknown;
  const schema = factory(type);
  if (!schema || typeof (schema as { infer?: unknown }).infer === "undefined") {
    if (typeof schema !== "function") throw new Error("Expression did not produce an arktype type");
  }
  return schema;
}

export function useEngine() {
  const state = reactive({
    phase: "idle" as Phase,
    status: "idle",
    error: null as string | null,
    progress: 0,
    modelUrl: DEFAULT_MODEL_URL,
    model: null as ModelInfo | null,
    blocks: [] as BlockRow[],
    checkpoints: [] as CheckpointRow[],
    runs: [] as RunRow[],
    stats: null as EngineStats | null,
    shaderCoverage: [] as string[],
    lastProgram: null as ProgramInfo | null,
    lastOutput: "",
    /** Reserved vocabulary entries addressable as [-token-K-]. */
    reservedCount: 0,
  });

  const engineRef = shallowRef<Engine | null>(null);
  const forwardRef = shallowRef<Lfm2Forward | null>(null);
  const modelRef = shallowRef<Lfm2GpuModel | null>(null);
  let tokenizer: Lfm2Tokenizer | null = null;
  let reserved: ReservedTable = EMPTY_RESERVED;
  let nextRunId = 1;
  let disposed = false;

  function fail(cause: unknown): never {
    const message = cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause);
    state.error = message;
    state.status = message;
    state.phase = "error";
    throw cause;
  }

  function refreshStats(): void {
    const engine = engineRef.value;
    if (engine) state.stats = engine.debug.stats();
    const forward = forwardRef.value;
    if (forward) state.shaderCoverage = [...forward.executor.shaderCoverage].sort();
  }

  function snapshot(): EngineStats {
    const engine = engineRef.value;
    if (!engine) throw new Error("Engine is not initialized");
    return engine.debug.stats();
  }

  function requireEngine(): Engine {
    const engine = engineRef.value;
    if (!engine) throw new Error("Engine is not initialized");
    return engine;
  }

  // ------------------------------------------------------------- selections

  function block(id: number): BlockRow | undefined {
    return state.blocks.find((row) => row.id === id);
  }

  function checkpoint(id: number): CheckpointRow | undefined {
    return state.checkpoints.find((row) => row.id === id);
  }

  function selectionLabel(selection: ContextSelection): string {
    const parts: string[] = [];
    if (selection.checkpoint !== null) parts.push(`ckpt#${selection.checkpoint}`);
    for (const id of selection.blocks) parts.push(`b#${id}`);
    return parts.length ? parts.join(" + ") : "(empty)";
  }

  function selectionTokens(selection: ContextSelection): number {
    let total = selection.checkpoint !== null ? checkpoint(selection.checkpoint)?.position ?? 0 : 0;
    for (const id of selection.blocks) total += block(id)?.tokens.length ?? 0;
    return total;
  }

  /** Text a selection resolves to, for preview before it is ever executed. */
  function selectionText(selection: ContextSelection): string {
    const parts: string[] = [];
    if (selection.checkpoint !== null) {
      const base = checkpoint(selection.checkpoint);
      if (base?.text) parts.push(base.text);
    }
    for (const id of selection.blocks) {
      const row = block(id);
      if (row) parts.push(row.text);
    }
    return parts.join("");
  }

  /** Structural problems with a composed context. See ./context-rules.ts. */
  function selectionIssues(selection: ContextSelection): string[] {
    const parts: ContextPart[] = [];
    if (selection.checkpoint !== null) {
      const row = checkpoint(selection.checkpoint);
      if (row) parts.push({ what: `ckpt#${row.id}`, bosLeading: row.bosLeading, bosInterior: row.bosInterior });
    }
    for (const id of selection.blocks) {
      const row = block(id);
      if (row) parts.push({ what: `b#${row.id}`, bosLeading: row.bosLeading, bosInterior: row.bosInterior });
    }
    return contextIssues({
      parts,
      tokens: selectionTokens(selection),
      capacity: state.model?.contextCapacity,
    });
  }

  function resolveContext(selection: ContextSelection): { checkpoint?: number; blocks?: number[] } {
    const context: { checkpoint?: number; blocks?: number[] } = {};
    if (selection.checkpoint !== null) context.checkpoint = selection.checkpoint;
    if (selection.blocks.length) context.blocks = [...selection.blocks];
    return context;
  }

  /** Blocks a selection materializes, base checkpoint lineage first. */
  function resolveParts(selection: ContextSelection): CheckpointPart[] {
    const parts: CheckpointPart[] = [];
    if (selection.checkpoint !== null) {
      parts.push(...(checkpoint(selection.checkpoint)?.contents ?? []));
    }
    for (const id of selection.blocks) {
      const row = block(id);
      if (row) {
        parts.push({ blockId: row.id, label: row.label, text: row.text, tokenCount: row.tokens.length });
      }
    }
    return parts;
  }

  // ------------------------------------------------------------------- boot

  async function boot(options: { url?: string; file?: File } = {}): Promise<void> {
    if (engineRef.value) return;
    try {
      state.error = null;
      state.phase = "device";
      state.status = "acquiring WebGPU adapter";

      const { device } = await createWebGpuDevice({
        label: "chomato.gui",
        requiredLimits: {
          maxBufferSize: GIB,
          maxStorageBufferBindingSize: GIB,
          maxComputeWorkgroupsPerDimension: 65535,
          maxStorageBuffersPerShaderStage: 16,
        },
      });
      if (disposed) return;

      // The Sandblaster LFM2 definition owns a device-bound resource graph, so
      // the model must be opened on exactly the device it compiled against.
      state.status = "compiling LFM2 programs";
      const compiled = await lfm2.engine.compile({ device });
      if (compiled.failed) {
        throw new Error(`LFM2 compile failed: ${compiled.failed}/${compiled.total} programs`);
      }
      state.status = `compiled ${compiled.ok}/${compiled.total} programs`;
      if (disposed) return;

      state.phase = "model";
      state.status = "opening WQ4 container";
      const source = options.file
        ? new BlobSource(options.file)
        : await HttpRangeSource.open(options.url ?? state.modelUrl);
      if (options.url) state.modelUrl = options.url;
      if (options.file) state.modelUrl = options.file.name;

      let allocatedBytes = 0;
      const loadStarted = performance.now();
      const model = await Lfm2GpuModel.open(device, source, {
        preload: false,
        drainUploads: true,
        onProgress: (progress) => {
          allocatedBytes = progress.allocatedBytes;
          state.progress = progress.totalBytes > 0 ? progress.uploadedBytes / progress.totalBytes : 0;
          state.status = `uploading tensors ${progress.tensorIndex + 1}/${progress.tensorCount}`;
        },
      });
      if (disposed) return;
      const loadMs = performance.now() - loadStarted;

      state.phase = "runtime";
      state.status = "preloading forward path";
      const prepareStarted = performance.now();
      const forward = new Lfm2Forward(model);
      await forward.prepareAll();
      const prepareMs = performance.now() - prepareStarted;
      if (disposed) return;

      tokenizer = forward.tokenizer;
      reserved = buildReservedTable(
        tokenizer.idToToken,
        (id) => tokenizer!.isSpecialToken(id),
      );
      state.reservedCount = reserved.literals.length;
      const engine = new Engine(createLfm2WebGpuTransport(forward));

      modelRef.value = model;
      forwardRef.value = forward;
      engineRef.value = engine;

      state.model = {
        blockCount: model.config.blockCount,
        hiddenSize: model.config.hiddenSize,
        feedForwardSize: model.config.feedForwardSize,
        vocabSize: model.config.vocabSize,
        attentionHeads: model.config.attentionHeads,
        headDim: model.config.headDim,
        bosToken: model.config.bosToken,
        eosToken: model.config.eosToken,
        layers: [...model.config.layers],
        allocatedBytes,
        tensorCount: model.tensors.size,
        loadMs,
        prepareMs,
        contextCapacity: lfm2.capacities.context,
        maxNewTokens: lfm2.capacities.maxNewTokens,
      };
      state.progress = 1;
      state.status = "ready";
      state.phase = "ready";
      refreshStats();
    } catch (cause) {
      fail(cause);
    }
  }

  async function withBusy<T>(label: string, run: () => Promise<T>): Promise<T> {
    if (state.phase !== "ready") throw new Error(`Engine is ${state.phase}`);
    state.phase = "busy";
    state.status = label;
    try {
      const value = await run();
      state.phase = "ready";
      state.status = "ready";
      refreshStats();
      return value;
    } catch (cause) {
      refreshStats();
      return fail(cause);
    }
  }

  /**
   * Encode with `[-token-K-]` aliases resolved first. The tokenizer resolves
   * special literals itself (parseSpecial defaults on), so expansion only has
   * to produce the literal text.
   */
  function tokenize(text: string, addBos: boolean): number[] {
    if (!tokenizer) throw new Error("Tokenizer is not available");
    const expanded = expandReserved(text, reserved).text;
    return tokenizer.encode(expanded, { addBos, addEos: false });
  }

  /** Tokenization with the alias bookkeeping kept, for the tokenizer window. */
  function inspectText(text: string, addBos: boolean): {
    tokens: number[];
    expanded: string;
    unknownAliases: number[];
  } {
    if (!tokenizer) return { tokens: [], expanded: text, unknownAliases: [] };
    const expansion = expandReserved(text, reserved);
    return {
      tokens: tokenizer.encode(expansion.text, { addBos, addEos: false }),
      expanded: expansion.text,
      unknownAliases: expansion.unknown,
    };
  }

  /** Vocabulary literal for an id, or null when the id is out of range. */
  function tokenLiteral(id: number): string | null {
    return tokenizer?.idToToken[id] ?? null;
  }

  function isSpecial(id: number): boolean {
    return tokenizer?.isSpecialToken(id) ?? false;
  }

  function decode(tokens: readonly number[]): string {
    if (!tokenizer) throw new Error("Tokenizer is not available");
    return tokenizer.decode(tokens, { skipSpecial: false });
  }

  /** Per-token strings, so a block's tokenization can actually be inspected. */
  function tokenPieces(tokens: readonly number[]): string[] {
    if (!tokenizer) return tokens.map((id) => `<${id}>`);
    return tokens.map((id) => {
      const bytes = tokenizer!.tokenBytes(id);
      if (bytes === null || bytes.length === 0) return `<${id}>`;
      return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
    });
  }

  // ----------------------------------------------------------------- blocks

  async function putBlock(input: { text: string; addBos: boolean; label?: string }): Promise<BlockRow> {
    return withBusy("PutBlock", async () => {
      const engine = requireEngine();
      const tokens = tokenize(input.text, input.addBos);
      if (!tokens.length) throw new Error("Block would be empty; nothing to store");
      const id = await engine.putBlock(Uint32Array.from(tokens));
      const bos = state.model?.bosToken;
      const row: BlockRow = {
        id,
        label: input.label?.trim() || `block ${id}`,
        text: input.text,
        tokens,
        addBos: input.addBos,
        bosLeading: bos !== undefined && tokens[0] === bos,
        bosInterior: bos === undefined ? 0 : tokens.slice(1).filter((t) => t === bos).length,
      };
      state.blocks.push(row);
      return row;
    });
  }

  async function dropBlock(id: number): Promise<void> {
    await withBusy("DropBlock", async () => {
      await requireEngine().dropBlock(id);
      state.blocks = state.blocks.filter((row) => row.id !== id);
    });
  }

  // ------------------------------------------------------------ checkpoints

  async function createCheckpoint(selection: ContextSelection, label?: string): Promise<CheckpointRow> {
    return withBusy("Checkpoint", async () => {
      const engine = requireEngine();
      const context = resolveContext(selection);
      if (context.checkpoint === undefined && !context.blocks?.length) {
        throw new Error("Checkpoint needs a context: choose a base checkpoint or at least one block");
      }
      const contents = resolveParts(selection);
      const position = selectionTokens(selection);
      // A checkpoint inherits the BOS shape of whatever it froze, so composing
      // on top of it can be validated the same way as composing on a block.
      const firstPart = selection.checkpoint !== null
        ? checkpoint(selection.checkpoint)
        : block(selection.blocks[0] ?? -1);
      const interior = (selection.checkpoint !== null ? checkpoint(selection.checkpoint)?.bosInterior ?? 0 : 0)
        + selection.blocks
          .map((id, index) => {
            const row = block(id);
            if (!row) return 0;
            const leadingCounts = index === 0 && selection.checkpoint === null ? 0 : (row.bosLeading ? 1 : 0);
            return row.bosInterior + leadingCounts;
          })
          .reduce((a, b) => a + b, 0);

      const before = snapshot();
      const started = performance.now();
      const id = await engine.checkpoint(context);
      const cost = diffStats(before, snapshot(), performance.now() - started);

      const row: CheckpointRow = {
        id,
        label: label?.trim() || `ckpt ${id}`,
        base: selection.checkpoint,
        appended: [...selection.blocks],
        contents,
        text: contents.map((part) => part.text).join(""),
        position,
        bosLeading: firstPart?.bosLeading ?? false,
        bosInterior: interior,
        cost,
      };
      state.checkpoints.push(row);
      return row;
    });
  }

  async function dropCheckpoint(id: number): Promise<void> {
    await withBusy("DropCheckpoint", async () => {
      await requireEngine().dropCheckpoint(id);
      state.checkpoints = state.checkpoints.filter((row) => row.id !== id);
      // Children keep their snapshotted contents; only the broken link is cut.
      for (const row of state.checkpoints) if (row.base === id) row.base = null;
    });
  }

  // ------------------------------------------------------------- generation

  function compileSchema(source: string): ProgramInfo {
    const schema = evaluateSchema(source);
    const compiled = compileStructuredGeneration(schema as never);
    const summary = compiled.program.summary as unknown as Record<string, number>;
    const info: ProgramInfo = {
      blobBytes: summary.blobBytes ?? compiled.program.blob.byteLength,
      blobWords: summary.blobWords ?? compiled.program.blob.length,
      sourceNodes: summary.sourceNodes ?? 0,
      nodes: summary.nodes ?? 0,
      literalNodes: summary.literalNodes ?? 0,
      switchNodes: summary.switchNodes ?? 0,
      stringNodes: summary.stringNodes ?? 0,
      numberNodes: summary.numberNodes ?? 0,
      jumpNodes: summary.jumpNodes ?? 0,
      acceptNodes: summary.acceptNodes ?? 0,
      edges: summary.edges ?? 0,
      byteTableLength: summary.byteLength ?? 0,
      maxJsonBytes: compiled.maxJsonBytes,
      maxTokens: compiled.maxTokens,
    };
    state.lastProgram = info;
    return info;
  }

  async function generateStructured(source: string, selection: ContextSelection): Promise<unknown> {
    return withBusy("Generate (structured)", async () => {
      const engine = requireEngine();
      const schema = evaluateSchema(source);
      compileSchema(source);
      const label = selectionLabel(selection);

      const before = snapshot();
      const started = performance.now();
      let ok = true;
      let detail = "";
      let value: unknown;
      try {
        value = await engine.generate(schema as never, resolveContext(selection));
        detail = JSON.stringify(value);
      } catch (cause) {
        ok = false;
        detail = cause instanceof Error ? cause.message : String(cause);
        throw cause;
      } finally {
        state.runs.unshift({
          id: nextRunId++,
          kind: "structured",
          source: source.trim(),
          ok,
          detail,
          contextLabel: label,
          cost: diffStats(before, snapshot(), performance.now() - started),
        });
      }
      state.lastOutput = detail;
      return value;
    });
  }

  async function generateTokens(maxTokens: number, selection: ContextSelection): Promise<number[]> {
    return withBusy("Generate (tokens)", async () => {
      const engine = requireEngine();
      const label = selectionLabel(selection);
      const collected: number[] = [];

      const before = snapshot();
      const started = performance.now();
      let ok = true;
      let detail = "";
      try {
        const generation = engine.generate(resolveContext(selection), { maxTokens, sampler: "argmax" });
        for await (const token of generation) collected.push(token);
        detail = decode(collected);
      } catch (cause) {
        ok = false;
        detail = cause instanceof Error ? cause.message : String(cause);
        throw cause;
      } finally {
        state.runs.unshift({
          id: nextRunId++,
          kind: "tokens",
          source: `maxTokens=${maxTokens}`,
          ok,
          detail,
          contextLabel: label,
          cost: diffStats(before, snapshot(), performance.now() - started),
        });
      }
      state.lastOutput = detail;
      return collected;
    });
  }

  function resetStats(): void {
    engineRef.value?.debug.resetStats();
    forwardRef.value?.executor.clearShaderCoverage();
    refreshStats();
  }

  async function dispose(): Promise<void> {
    disposed = true;
    try {
      await engineRef.value?.close();
    } catch {
      // Closing a half-initialized engine must not mask the original failure.
    }
    modelRef.value?.destroy();
    engineRef.value = null;
    forwardRef.value = null;
    modelRef.value = null;
    tokenizer = null;
  }

  return {
    state: readonly(state) as typeof state,
    boot,
    block,
    checkpoint,
    putBlock,
    dropBlock,
    createCheckpoint,
    dropCheckpoint,
    compileSchema,
    generateStructured,
    generateTokens,
    selectionLabel,
    selectionTokens,
    selectionText,
    selectionIssues,
    resolveParts,
    refreshStats,
    resetStats,
    tokenize,
    inspectText,
    tokenLiteral,
    isSpecial,
    reservedEntry: (index: number) => ({
      literal: reserved.literals[index - 1] ?? null,
      id: reserved.ids[index - 1] ?? null,
    }),
    decode,
    tokenPieces,
    dispose,
  };
}

export type EngineApi = ReturnType<typeof useEngine>;
