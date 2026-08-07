import type {
  ComputePassRunner,
  SandblasterCommandEncoder,
} from "@sandblaster/core-next";
import {
  lfm2,
  OP_PARAM_BUFFER_BYTES,
  type Lfm2OpParams,
  type Lfm2PassName,
  type Lfm2PassSpec,
  type Lfm2ShaderName,
  type Lfm2Workgroups,
  type Lfm2WeightBinding,
} from "./lfm2";

/**
 * Everything below this import boundary is host-side inference orchestration.
 * Shader resources, includes, entry points and dispatch geometry stay in
 * lfm2.ts; this module deals in semantic pass requests.
 */
export interface Lfm2PassRequest {
  readonly name: Lfm2PassName;
  readonly program: Lfm2PassSpec["program"];
  readonly op: Readonly<Lfm2OpParams>;
  readonly workgroups: Lfm2Workgroups;
  readonly weight: Lfm2WeightBinding;
}

/** Resolve one semantic runtime operation into a concrete GPU pass request. */
export function lfm2Pass(
  name: Lfm2PassName,
  op: Readonly<Lfm2OpParams>,
): Lfm2PassRequest {
  const spec = lfm2.passes[name];
  return {
    name,
    program: spec.program,
    op,
    workgroups: spec.workgroups(op),
    weight: spec.weight,
  };
}

const OP_PARAM_BYTES = 64;

function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

class Lfm2ParamWriter {
  private readonly data: ArrayBuffer;
  private readonly view: DataView;
  private cursor = 0;
  private readonly value = {
    inputOffset: 0, outputOffset: 0, auxOffset: 0, aux2Offset: 0,
    tokenCount: 0, inputDim: 0, outputDim: 0, rowStart: 0,
    rowCount: 0, layerIndex: 0, attentionSlot: 0,
    mode: "prefill" as "prefill" | "decode" | "continuation",
    f0: 0, f1: 0, u0: 0, u1: 0,
    reserved: new Array<number>(48).fill(0),
  };

  constructor(
    private readonly resource: typeof lfm2.resources.op,
    readonly stride: number,
    capacityBytes = OP_PARAM_BUFFER_BYTES,
  ) {
    this.data = new ArrayBuffer(capacityBytes);
    this.view = new DataView(this.data);
  }

  reset(): void {
    this.cursor = 0;
  }

  alloc(params: Readonly<Lfm2OpParams>): number {
    if (this.cursor + this.stride > this.data.byteLength) {
      throw new Error(
        `LFM2 OpParams buffer exhausted at ${this.cursor} B; increase OP_PARAM_BUFFER_BYTES`,
      );
    }

    const offset = this.cursor;
    // Reuse one complete input object so schema defaults do not require an
    // ArkType assert/allocation for every GPU dispatch. The compiled codec owns
    // the actual byte offsets (including autoSorted fields such as `mode`).
    const value = this.value;
    value.inputOffset = params.inputOffset ?? 0;
    value.outputOffset = params.outputOffset ?? 0;
    value.auxOffset = params.auxOffset ?? 0;
    value.aux2Offset = params.aux2Offset ?? 0;
    value.tokenCount = params.tokenCount ?? 0;
    value.inputDim = params.inputDim ?? 0;
    value.outputDim = params.outputDim ?? 0;
    value.rowStart = params.rowStart ?? 0;
    value.rowCount = params.rowCount ?? 0;
    value.layerIndex = params.layerIndex ?? 0;
    value.attentionSlot = params.attentionSlot ?? 0;
    value.mode = params.mode ?? "prefill";
    value.f0 = params.f0 ?? 0;
    value.f1 = params.f1 ?? 0;
    value.u0 = params.u0 ?? 0;
    value.u1 = params.u1 ?? 0;
    this.resource.encodeInto(value, this.view, offset);

    this.cursor += this.stride;
    return offset;
  }

  get usedBytes(): number {
    return this.cursor;
  }

  usedView(): Uint8Array {
    return new Uint8Array(this.data, 0, this.cursor);
  }
}

export type Lfm2WeightPage = GPUBuffer | { readonly buffer: GPUBuffer };

function gpuBuffer(page: Lfm2WeightPage): GPUBuffer {
  return "buffer" in page ? page.buffer : page;
}

/** Runtime-facing wrapper around one Sandblaster compute pass. */
export class Lfm2ComputePass {
  constructor(
    private readonly pass: ComputePassRunner,
    private readonly params: Lfm2ParamWriter,
    private readonly onRun?: (name: Lfm2ShaderName) => void,
  ) {}

  run(
    name: Lfm2PassName,
    op: Readonly<Lfm2OpParams>,
    weightPage?: Lfm2WeightPage,
  ): void {
    this.onRun?.(name);
    const request = lfm2Pass(name, op);
    if (request.weight !== "none" && !weightPage) {
      throw new Error(`${name} requires a ${request.weight} weight page`);
    }

    const opOffset = this.params.alloc(op);
    const overrides = request.weight === "raw"
      ? { weightRaw: gpuBuffer(weightPage!) }
      : request.weight === "f32"
        ? { weight32: gpuBuffer(weightPage!) }
        : undefined;

    // Lfm2PassSpec intentionally erases the program's individual resource map
    // so the scheduler can store all passes in one table. The runtime keys are
    // nevertheless constrained here by request.weight and checked again by
    // Sandblaster against the linked manifest.
    this.pass.run(
      request.program,
      { workgroups: request.workgroups },
      {
        dynamicOffsets: { op: opOffset },
        ...(overrides ? { resources: overrides } : {}),
      } as any,
    );
  }

  /** Dispatch a static AOT program that has no OpParams binding. */
  runStatic(name: "constraint_mask", workgroups: Lfm2Workgroups): void {
    this.onRun?.(name);
    this.pass.run(lfm2.programs[name], { workgroups });
  }
}

export interface Lfm2CommandEncoder {
  readonly gpu: GPUCommandEncoder;
  compute(callback: (pass: Lfm2ComputePass) => void, descriptor?: GPUComputePassDescriptor): void;
}

/**
 * Batched execution bridge for the migrated scheduler.
 *
 * All OpParams records are written once before Sandblaster submits the command
 * buffer. Multiple compute passes and raw copy/clear commands can therefore be
 * interleaved exactly like in the legacy runtime while retaining one queue
 * submit and dynamic uniform offsets per dispatch.
 */
export class Lfm2Executor {
  private readonly params: Lfm2ParamWriter;
  private readonly usedShaders = new Set<Lfm2ShaderName>();

  constructor(readonly definition = lfm2) {
    const recordBytes = definition.resources.op.compiledInfo.physicalStride;
    if (recordBytes !== OP_PARAM_BYTES) {
      throw new Error(
        `LFM2 OpParams ABI changed: expected ${OP_PARAM_BYTES} B, got ${recordBytes} B`,
      );
    }
    // Regression guard for the weight placeholder contract: pass.ts overrides
    // weightRaw/weight32 with real tensor pages, so they must link as
    // RUNTIME-sized WGSL arrays (declared as count>1 scalar buffers in
    // lfm2-definition.ts). If Sandblaster's linker ever emits a fixed-length
    // type here, weight reads past it become out-of-bounds and inference
    // silently degrades to garbage/NaN instead of failing loudly.
    for (const programName of ["embedding_wq4", "matmul_wq4", "rms_norm"] as const) {
      const manifest = lfm2.programs[programName].manifest;
      for (const binding of manifest.bindings) {
        if (binding.name !== "weightRaw" && binding.name !== "weight32") continue;
        if (!binding.wgslType.startsWith("array<")) {
          throw new Error(
            `LFM2 weight binding '${binding.name}' must be a runtime-sized WGSL array, ` +
            `got '${binding.wgslType}' (${programName})`,
          );
        }
      }
    }
    const alignment = Number(
      definition.engine.device.limits.minUniformBufferOffsetAlignment ?? 256,
    );
    this.params = new Lfm2ParamWriter(
      definition.resources.op,
      align(recordBytes, alignment),
    );
  }

  get shaderCoverage(): readonly Lfm2ShaderName[] {
    return [...this.usedShaders];
  }

  clearShaderCoverage(): void {
    this.usedShaders.clear();
  }

  submit(callback: (encoder: Lfm2CommandEncoder) => void): void {
    this.params.reset();
    const { engine, resources } = this.definition;

    engine.submit((sandblasterEncoder: SandblasterCommandEncoder) => {
      const encoder: Lfm2CommandEncoder = {
        gpu: sandblasterEncoder.gpu,
        compute: (computeCallback, descriptor = {}) => {
          sandblasterEncoder.compute(descriptor, (pass) => {
            computeCallback(new Lfm2ComputePass(pass, this.params, (name) => this.usedShaders.add(name)));
          });
        },
      };

      callback(encoder);

      if (this.params.usedBytes > 0) {
        // queue.writeBuffer is ordered before the command buffer submitted by
        // engine.submit() after this callback returns. Every dispatch therefore
        // observes its own 256-byte record selected by dynamic offset.
        engine.device.queue.writeBuffer(
          resources.op.gpu,
          0,
          this.params.usedView(),
        );
      }
    });
  }
}

/** Runtime-facing GPU definition. */
export const gpu = lfm2;
