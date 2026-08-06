import { computed, onScopeDispose, readonly, ref, shallowRef } from "vue";
import { Sandblaster } from "@sandblaster/core";
import { HttpRangeSource } from "../../../quant/src/gguf/source.ts";
import { Lfm2Model } from "../../../lfm2/src/model.ts";
import { Lfm2Runtime, type GenerateResult } from "../../../lfm2/src/runtime.ts";
import { Lfm2Tokenizer } from "../../../lfm2/src/tokenizer.ts";
import { $ } from "../../../schema/src/schema.ts";

export type ChomatoPhase =
  | "idle"
  | "webgpu"
  | "model"
  | "runtime"
  | "ready"
  | "generating"
  | "error";

export interface ChomatoConfig {
  modelUrl: string;
  wq4Url?: string;
  contextCapacity: number;
  maxNewTokens: number;
}

export interface ChomatoModelInfo {
  layers: number;
  hiddenSize: number;
  feedForwardSize: number;
  vocabSize: number;
  allocatedBytes: number;
  loadMs: number;
  compileMs: number;
}

export interface ChomatoGenerationStats {
  wallMs: number;
  promptTokens: number;
  generatedTokens: number;
  prefillMs?: number;
  decodeMs?: number;
  readbackMs?: number;
  totalGpuMs?: number;
  decodeTokensPerSecond?: number;
}

const GIB = 1024 * 1024 * 1024;

export function chomatoConfigFromLocation(search = location.search): ChomatoConfig {
  const params = new URLSearchParams(search);
  return {
    modelUrl: params.get("model") ?? "/models/LFM2.5-1.2B-Instruct-F16.gguf",
    wq4Url: params.get("wq4") ?? undefined,
    contextCapacity: Math.max(1, Number(params.get("context") ?? 1024)),
    maxNewTokens: Math.max(1, Number(params.get("tokens") ?? 128)),
  };
}

export function useChomato(initial: Partial<ChomatoConfig> = {}) {
  const defaults = chomatoConfigFromLocation();
  const config: ChomatoConfig = { ...defaults, ...initial };

  const phase = ref<ChomatoPhase>("idle");
  const status = ref("idle");
  const error = ref<string | null>(null);
  const modelProgress = ref(0);
  const prompt = ref(new URLSearchParams(location.search).get("prompt") ?? "Write one sentence about WebGPU.");
  const maxNewTokens = ref(config.maxNewTokens);
  const profile = ref(true);
  const output = ref("");
  const lastResult = shallowRef<GenerateResult | null>(null);
  const modelInfo = shallowRef<ChomatoModelInfo | null>(null);
  const generationStats = shallowRef<ChomatoGenerationStats | null>(null);

  let engine: Awaited<ReturnType<typeof Sandblaster.create>> | null = null;
  let model: Lfm2Model | null = null;
  let tokenizer: Lfm2Tokenizer | null = null;
  let runtime: Lfm2Runtime | null = null;
  let initPromise: Promise<void> | null = null;
  let disposed = false;

  const ready = computed(() => phase.value === "ready");
  const busy = computed(() => !["idle", "ready", "error"].includes(phase.value));
  const canGenerate = computed(() => ready.value && prompt.value.trim().length > 0);

  function setError(cause: unknown): never {
    const message = cause instanceof Error ? cause.message : String(cause);
    error.value = message;
    status.value = message;
    phase.value = "error";
    throw cause;
  }

  async function initialize(): Promise<void> {
    if (runtime) return;
    if (initPromise) return initPromise;

    initPromise = (async () => {
      try {
        error.value = null;
        phase.value = "webgpu";
        status.value = "acquiring WebGPU";

        // Bring-up profile. The runtime currently binds nine storage buffers;
        // keep this explicit here rather than hiding the requirement in Vue UI.
        const requiredLimits: Record<string, number> = {
          maxBufferSize: GIB,
          maxStorageBufferBindingSize: GIB,
          maxComputeWorkgroupsPerDimension: 65535,
          maxStorageBuffersPerShaderStage: 16,
        };
        engine = await Sandblaster.create($).compile({ requiredLimits } as any);
        if (disposed) return;

        phase.value = "model";
        status.value = "opening GGUF";
        const source = await HttpRangeSource.open(config.modelUrl);
        const wq4Source = config.wq4Url ? await HttpRangeSource.open(config.wq4Url) : undefined;

        let allocatedBytes = 0;
        const loadStarted = performance.now();
        model = await Lfm2Model.load(engine.device, source, {
          wq4Source,
          maxPageBytes: 64 * 1024 * 1024,
          drainUploads: true,
          onProgress(progress) {
            allocatedBytes = progress.allocatedBytes;
            modelProgress.value = progress.totalBytes > 0
              ? progress.uploadedBytes / progress.totalBytes
              : 0;
            status.value = `loading model ${Math.floor(modelProgress.value * 100)}%`;
          },
        });
        const loadMs = performance.now() - loadStarted;
        if (disposed) return;

        tokenizer = new Lfm2Tokenizer(model.reader);

        phase.value = "runtime";
        status.value = "compiling runtime";
        const compileStarted = performance.now();
        runtime = await Lfm2Runtime.create(engine, model, {
          contextCapacity: config.contextCapacity,
          maxNewTokens: config.maxNewTokens,
        });
        const compileMs = performance.now() - compileStarted;
        if (disposed) return;

        modelInfo.value = {
          layers: model.config.blockCount,
          hiddenSize: model.config.hiddenSize,
          feedForwardSize: model.config.feedForwardSize,
          vocabSize: model.config.vocabSize,
          allocatedBytes,
          loadMs,
          compileMs,
        };
        modelProgress.value = 1;
        status.value = "ready";
        phase.value = "ready";
      } catch (cause) {
        setError(cause);
      } finally {
        initPromise = null;
      }
    })();

    return initPromise;
  }

  async function generate(): Promise<GenerateResult> {
    if (!runtime || !tokenizer) throw new Error("Chomato is not initialized");
    if (phase.value !== "ready") throw new Error(`Chomato is busy (${phase.value})`);

    const promptTokens = tokenizer.encodeUserPrompt(prompt.value);
    const requested = Math.max(1, Math.floor(maxNewTokens.value || config.maxNewTokens));
    const maxPossible = config.contextCapacity - promptTokens.length + 1;
    const boundedMax = Math.min(requested, config.maxNewTokens, maxPossible);
    if (boundedMax < 1) {
      throw new Error(`Prompt has ${promptTokens.length} tokens and does not fit context ${config.contextCapacity}`);
    }
    maxNewTokens.value = boundedMax;

    phase.value = "generating";
    status.value = "generating";
    error.value = null;
    output.value = "";
    generationStats.value = null;

    const started = performance.now();
    try {
      const result = await runtime.generateTokens(promptTokens, {
        maxNewTokens: boundedMax,
        profile: profile.value,
      });
      const wallMs = performance.now() - started;
      const text = tokenizer.decode(result.tokenIds);

      lastResult.value = result;
      output.value = text;
      const timings = result.timings;
      generationStats.value = {
        wallMs,
        promptTokens: promptTokens.length,
        generatedTokens: result.tokenIds.length,
        prefillMs: timings?.prefillMs,
        decodeMs: timings?.decodeMs,
        readbackMs: timings?.readbackMs,
        totalGpuMs: timings?.totalMs,
        decodeTokensPerSecond: timings?.decodeMs && timings.decodeMs > 0
          ? Math.max(0, result.tokenIds.length - 1) * 1000 / timings.decodeMs
          : undefined,
      };
      status.value = "ready";
      phase.value = "ready";
      return result;
    } catch (cause) {
      return setError(cause);
    }
  }

  function dispose(): void {
    disposed = true;
    runtime?.destroy();
    runtime = null;
    model = null;
    tokenizer = null;
    engine = null;
  }

  onScopeDispose(dispose);

  return {
    config,
    phase: readonly(phase),
    status: readonly(status),
    error: readonly(error),
    modelProgress: readonly(modelProgress),
    prompt,
    maxNewTokens,
    profile,
    output: readonly(output),
    lastResult: readonly(lastResult),
    modelInfo: readonly(modelInfo),
    generationStats: readonly(generationStats),
    ready,
    busy,
    canGenerate,
    initialize,
    generate,
    dispose,
  };
}
