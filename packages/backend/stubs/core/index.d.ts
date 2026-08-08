// EXPERIMENTAL LOCAL STUB — see package.json.
// Type declarations for the @sandblaster/core API surface consumed by
// chomato's packages/webgpu (lfm2-definition.ts, pass.ts, forward.ts, lfm2.ts).
// Any GPUBuffer/GPUDevice-typed slots use `any` so this file does not depend on
// @webgpu/types; the real webgpu sources are still checked against the real
// @webgpu/types via the harness's triple-slash reference.

export interface BufferResourceUse {
  resource: unknown;
  group?: number;
  buffer: {
    type: "read-only-storage" | "storage" | "uniform";
    hasDynamicOffset?: boolean;
    minBindingSize?: number;
  };
  representation: string;
  offset?: number;
  size?: number;
}

export interface AnyComputeHandle {
  readonly kind: "compute";
  readonly label: string;
  /** Linked binding manifest; the runtime reads it for ABI regression guards. */
  readonly manifest: {
    readonly bindings: readonly {
      readonly name: string;
      readonly wgslType: string;
    }[];
  };
}

export interface ComputePassRunner {
  run(
    program: AnyComputeHandle,
    options: { workgroups: readonly number[] },
    bind?: unknown,
  ): void;
}

export interface SandblasterCommandEncoder {
  readonly gpu: any;
  compute(descriptor: unknown, callback: (pass: ComputePassRunner) => void): void;
}

export interface BufferResource<T = unknown> {
  readonly name: string;
  readonly compiledInfo: { physicalStride: number; byteSize: number };
  readonly gpu: any;
  write(value: T): void;
  encodeInto(value: T, view: DataView, offset: number): void;
  readback(options?: { dropIfBusy?: boolean }): Promise<T>;
}

/**
 * A buffer resource whose record type is supplied by the caller (artifacts
 * cannot carry type parameters). Mirror of the real core's TypedBufferResource.
 */
export interface TypedBufferResource<V = any>
  extends Omit<BufferResource<V>, "encodeInto" | "write" | "readback"> {
  encodeInto(value: V, view: DataView, offset?: number): void;
  write(value: V, options?: { index?: number }): void;
  readback(options?: { dropIfBusy?: boolean }): Promise<V | V[] | null>;
}

export interface StubDevice {
  readonly queue: {
    writeBuffer(buffer: any, offset: number, data: any): void;
    onSubmittedWorkDone(): Promise<void>;
    submit(commandBuffers: unknown[]): void;
  };
  createBuffer(options: unknown): any;
  createCommandEncoder(options?: unknown): any;
  readonly limits: {
    minUniformBufferOffsetAlignment: number;
    maxStorageBufferBindingSize: number;
    maxBufferSize: number;
    maxComputeWorkgroupsPerDimension: number;
  };
}

export interface TypeHandle<T = unknown> {
  readonly name: string;
  assert(value: unknown): T;
}

export interface SandblasterEngine {
  readonly device: StubDevice;
  type<T = unknown>(nameOrShape: string | object): TypeHandle<T>;
  buffer<T = unknown>(
    type: TypeHandle<T>,
    options: {
      label?: string;
      size?: number;
      count?: number;
      value?: T;
      readback?: boolean;
    },
  ): BufferResource<T>;
  compute(options: unknown): AnyComputeHandle;
  submit(callback: (encoder: SandblasterCommandEncoder) => void): void;
  deserialize(json: string): void;
  /** Look up an artifact-created buffer by label or numeric resource id. */
  resource<V = any>(key: string | number): TypedBufferResource<V>;
  /** Look up an artifact-created compute program by its linked label. */
  computeProgram(label: string): AnyComputeHandle;
  /** Static-compilable no-op compile; the real engine compiles via the shim. */
  compile(options?: unknown): Promise<{ status: string; failed: number; total: number }>;
}

export declare class Sandblaster {
  static create(schema: unknown, options: unknown): SandblasterEngine;
  /** Build an engine from a linked artifact (no ArkType scope). */
  static fromArtifact(serialized: string | object, options?: unknown): SandblasterEngine;
}
