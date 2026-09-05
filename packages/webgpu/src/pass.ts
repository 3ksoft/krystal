import type {
  ComputePassRunner,
  SandblasterCommandEncoder,
} from "@sandblaster/core";
import {
  krystal,
  OP_PARAM_BUFFER_BYTES,
  type KrystalOpParams,
  type KrystalPassName,
  type KrystalPassSpec,
  type KrystalProgramName,
  type KrystalWorkgroups,
  type KrystalWeightBinding,
} from "./krystal";

/**
 * Everything below this import boundary is host-side inference orchestration.
 * Shader resources, includes, entry points and dispatch geometry stay in
 * krystal.ts; this module deals in semantic pass requests.
 */
interface KrystalPassRequest {
  readonly name: KrystalPassName;
  readonly program: KrystalPassSpec["program"];
  readonly op: Readonly<KrystalOpParams>;
  readonly workgroups: KrystalWorkgroups;
  readonly weight: KrystalWeightBinding;
}

/** Resolve one semantic runtime operation into a concrete GPU pass request. */
function krystalPass(
  name: KrystalPassName,
  op: Readonly<KrystalOpParams>,
): KrystalPassRequest {
  const spec = krystal.passes[name];
  return {
    name,
    program: spec.program,
    op,
    workgroups: spec.workgroups(op),
    weight: spec.weight,
  };
}

const OP_PARAM_BYTES = 96;

function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

class KrystalParamWriter {
  private readonly data: ArrayBuffer;
  private readonly view: DataView;
  private cursor = 0;
  private readonly value = {
    inputOffset: 0, outputOffset: 0, auxOffset: 0, aux2Offset: 0,
    aux3Offset: 0, aux4Offset: 0, aux5Offset: 0, aux6Offset: 0,
    tokenCount: 0, inputDim: 0, outputDim: 0, rowStart: 0,
    rowCount: 0, layerIndex: 0, attentionSlot: 0,
    mode: "prefill" as "prefill" | "decode" | "continuation",
    f0: 0, f1: 0, u0: 0, u1: 0, u2: 0, u3: 0, u4: 0, u5: 0,
    reserved: new Array<number>(44).fill(0),
  };

  constructor(
    private readonly resource: typeof krystal.resources.op,
    readonly stride: number,
    capacityBytes = OP_PARAM_BUFFER_BYTES,
  ) {
    this.data = new ArrayBuffer(capacityBytes);
    this.view = new DataView(this.data);
  }

  reset(): void {
    this.cursor = 0;
  }

  alloc(params: Readonly<KrystalOpParams>): number {
    if (this.cursor + this.stride > this.data.byteLength) {
      throw new Error(
        `Krystal OpParams buffer exhausted at ${this.cursor} B; increase OP_PARAM_BUFFER_BYTES`,
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
    value.aux3Offset = params.aux3Offset ?? 0;
    value.aux4Offset = params.aux4Offset ?? 0;
    value.aux5Offset = params.aux5Offset ?? 0;
    value.aux6Offset = params.aux6Offset ?? 0;
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
    value.u2 = params.u2 ?? 0;
    value.u3 = params.u3 ?? 0;
    value.u4 = params.u4 ?? 0;
    value.u5 = params.u5 ?? 0;
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

type KrystalWeightPage = GPUBuffer | { readonly buffer: GPUBuffer };

function gpuBuffer(page: KrystalWeightPage): GPUBuffer {
  return "buffer" in page ? page.buffer : page;
}

/** Runtime-facing wrapper around one Sandblaster compute pass. */
export class KrystalComputePass {
  constructor(
    private readonly pass: ComputePassRunner,
    private readonly params: KrystalParamWriter,
    private readonly onRun?: (name: KrystalProgramName) => void,
  ) {}

  run(
    name: KrystalPassName,
    op: Readonly<KrystalOpParams>,
    weightPage?: KrystalWeightPage,
  ): void {
    this.onRun?.(name);
    const request = krystalPass(name, op);
    if (request.weight !== "none" && !weightPage) {
      throw new Error(`${name} requires a ${request.weight} weight page`);
    }

    const opOffset = this.params.alloc(op);
    const overrides = request.weight === "raw"
      ? { weightRaw: gpuBuffer(weightPage!) }
      : request.weight === "f32"
        ? { weight32: gpuBuffer(weightPage!) }
        : undefined;

    // KrystalPassSpec intentionally erases the program's individual resource map
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

}

export interface KrystalCommandEncoder {
  readonly gpu: GPUCommandEncoder;
  compute(callback: (pass: KrystalComputePass) => void, descriptor?: GPUComputePassDescriptor): void;
}

/**
 * Batched execution bridge for the migrated scheduler.
 *
 * All OpParams records are written once before Sandblaster submits the command
 * buffer. Multiple compute passes and raw copy/clear commands can therefore be
 * interleaved exactly like in the legacy runtime while retaining one queue
 * submit and dynamic uniform offsets per dispatch.
 */
export class KrystalExecutor {
  private readonly params: KrystalParamWriter;
  private readonly usedShaders = new Set<KrystalProgramName>();

  constructor(readonly definition = krystal) {
    const recordBytes = definition.resources.op.compiledInfo.physicalStride;
    if (recordBytes !== OP_PARAM_BYTES) {
      throw new Error(
        `Krystal OpParams ABI changed: expected ${OP_PARAM_BYTES} B, got ${recordBytes} B`,
      );
    }
    // Regression guard for the weight placeholder contract: pass.ts overrides
    // weight32 with real tensor pages, so it must link as a RUNTIME-sized WGSL
    // array (declared as a count>1 scalar buffer in krystal-definition.ts). If
    // Sandblaster's linker ever emits a fixed-length type here, weight reads
    // past it become out-of-bounds and training silently degrades to
    // garbage/NaN instead of failing loudly.
    for (const programName of [
      "krystal_field_embed", "matmul_backward_input", "sgd_step",
    ] as const) {
      const manifest = krystal.programs[programName].manifest;
      for (const binding of manifest.bindings) {
        if (binding.name !== "weight32") continue;
        if (!binding.wgslType.startsWith("array<")) {
          throw new Error(
            `Krystal weight binding '${binding.name}' must be a runtime-sized WGSL array, ` +
            `got '${binding.wgslType}' (${programName})`,
          );
        }
      }
    }
    const alignment = Number(
      definition.engine.device.limits.minUniformBufferOffsetAlignment ?? 256,
    );
    this.params = new KrystalParamWriter(
      definition.resources.op,
      align(recordBytes, alignment),
    );
  }

  get shaderCoverage(): readonly KrystalProgramName[] {
    return [...this.usedShaders];
  }

  clearShaderCoverage(): void {
    this.usedShaders.clear();
  }

  submit(callback: (encoder: KrystalCommandEncoder) => void): void {
    this.params.reset();
    const { engine, resources } = this.definition;

    engine.submit((sandblasterEncoder: SandblasterCommandEncoder) => {
      const encoder: KrystalCommandEncoder = {
        gpu: sandblasterEncoder.gpu,
        compute: (computeCallback, descriptor = {}) => {
          sandblasterEncoder.compute(descriptor, (pass) => {
            computeCallback(new KrystalComputePass(pass, this.params, (name) => this.usedShaders.add(name)));
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
export const gpu = krystal;
