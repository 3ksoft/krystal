import {
  compileStructuredGeneration,
  isGeneratableSchema,
  type GeneratableSchema,
  type InferGeneratable,
} from "./structured.ts";
import type { v1_0_0 as ABI } from "@chomato/bridge/types";
import {
  CONTEXT_FLAG_STRUCTURED, MAX_CONTEXT_BLOCKS } from "@chomato/bridge/constants";

export type OperationId = number;
export type BlockId = number;
export type CheckpointId = number;
export type TokenId = number;

export interface Context {
  readonly checkpoint?: CheckpointId;
  readonly blocks?: readonly BlockId[];
}

export interface GenerateOptions {
  readonly maxTokens: number;
  /** Sampling strategy. The transport/backends currently only implement argmax. */
  readonly sampler?: "argmax";
}

export interface EngineStats {
  /** Total commands sent through the transport. */
  commands: number;
  blocksPut: number;
  /** Tokens stored via PutBlock. */
  contextTokens: number;
  blocksDropped: number;
  checkpointsCreated: number;
  checkpointsDropped: number;
  generations: number;
  /** Tokens emitted by generations. */
  generatedTokens: number;
  cancellations: number;
  /**
   * Tokens that had to be computed fresh during Generate — context tokens NOT
   * covered by a checkpoint. A checkpoint is already-materialized KV/conv
   * state, so its prefix must never be re-prefilled.
   */
  prefillTokens: number;
  /** Generations that used a checkpoint successfully. */
  checkpointHits: number;
  /** Generations that requested a checkpoint but could not restore one. */
  checkpointMisses: number;
  /** Bytes of physical checkpoint state restored/copied from reusable state. */
  restoredCheckpointBytes: number;
  /** Total physical bytes materialized into checkpoints since resetStats(). */
  checkpointBytes: number;
  /** Snapshot KV bytes materialized since resetStats(). */
  kvBytes: number;
  /** Corresponding live KV capacity represented by created checkpoints. */
  kvCapacityBytes: number;
  /** Snapshot recurrent convolution bytes materialized since resetStats(). */
  convBytes: number;
  /** Snapshot last-hidden bytes materialized since resetStats(). */
  hiddenBytes: number;
  /** Backend-reported checkpoint materialization wall time, in microseconds. */
  checkpointCreateUs: number;
  /** Backend-reported physical restore time, in microseconds; 0 means unavailable. */
  checkpointRestoreUs: number;
}

export function emptyEngineStats(): EngineStats {
  return {
    commands: 0,
    blocksPut: 0,
    contextTokens: 0,
    blocksDropped: 0,
    checkpointsCreated: 0,
    checkpointsDropped: 0,
    generations: 0,
    generatedTokens: 0,
    cancellations: 0,
    prefillTokens: 0,
    checkpointHits: 0,
    checkpointMisses: 0,
    restoredCheckpointBytes: 0,
    checkpointBytes: 0,
    kvBytes: 0,
    kvCapacityBytes: 0,
    convBytes: 0,
    hiddenBytes: 0,
    checkpointCreateUs: 0,
    checkpointRestoreUs: 0,
  };
}

export interface EngineDebug {
  /** Snapshot of operation counters since creation or the last resetStats(). */
  stats(): EngineStats;
  resetStats(): void;
}

export interface TransportFrame<T> {
  readonly message: T;
  readonly payload?: Uint8Array;
}

/**
 * Decoded transport boundary used by engine-ts.
 *
 * WebGPU can pass these objects in-process with no codec. A native transport
 * encodes the same message using @chomato/bridge and appends frame.payload.
 */
export interface EngineTransport {
  send(frame: TransportFrame<ABI.EngineCommand>): void | Promise<void>;
  subscribe(listener: (frame: TransportFrame<ABI.EngineEvent>) => void): () => void;
  close?(): void | Promise<void>;
}

export class EngineOperationError extends Error {
  constructor(
    readonly operation: OperationId,
    readonly code: ABI.ErrorCode,
    message: string,
  ) {
    super(message);
    this.name = "EngineOperationError";
  }
}

class Deferred<T> {
  readonly promise: Promise<T>;
  resolve!: (value: T | PromiseLike<T>) => void;
  reject!: (reason?: unknown) => void;

  constructor() {
    this.promise = new Promise<T>((resolve, reject) => {
      this.resolve = resolve;
      this.reject = reject;
    });
  }
}

class TokenQueue {
  private readonly values: TokenId[] = [];
  private readonly waiters: Array<Deferred<IteratorResult<TokenId>>> = [];
  private ended = false;
  private error: unknown = null;

  push(token: TokenId): void {
    if (this.ended) return;
    const waiter = this.waiters.shift();
    if (waiter) waiter.resolve({ value: token, done: false });
    else this.values.push(token);
  }

  finish(): void {
    if (this.ended) return;
    this.ended = true;
    for (const waiter of this.waiters.splice(0)) waiter.resolve({ value: undefined, done: true });
  }

  fail(error: unknown): void {
    if (this.ended) return;
    this.error = error;
    this.ended = true;
    for (const waiter of this.waiters.splice(0)) waiter.reject(error);
  }

  async next(): Promise<IteratorResult<TokenId>> {
    if (this.values.length) return { value: this.values.shift()!, done: false };
    if (this.error) throw this.error;
    if (this.ended) return { value: undefined, done: true };
    const waiter = new Deferred<IteratorResult<TokenId>>();
    this.waiters.push(waiter);
    return waiter.promise;
  }
}

export interface Generation extends AsyncIterable<TokenId> {
  readonly id: OperationId;
  cancel(): Promise<void>;
}

/**
 * Decode UTF-8, rejecting malformed input.
 *
 * Equivalent to `new TextDecoder("utf-8", { fatal: true })`, but built from
 * one-argument constructors so it also compiles for runtimes whose TextDecoder
 * lib does not model the options bag (the scriptc-built native exe among them).
 * A non-fatal decode substitutes U+FFFD for malformed sequences, so re-encoding
 * differs from the input exactly when the input was not valid UTF-8 — input
 * that legitimately contains U+FFFD round-trips unchanged.
 */
function decodeUtf8Strict(bytes: Uint8Array): string {
  const text = new TextDecoder().decode(bytes);
  const reencoded = new TextEncoder().encode(text);
  if (reencoded.length !== bytes.length) {
    throw new TypeError("Structured payload is not valid UTF-8");
  }
  for (let i = 0; i < reencoded.length; i++) {
    if (reencoded[i] !== bytes[i]) throw new TypeError("Structured payload is not valid UTF-8");
  }
  return text;
}

const littleEndian = (() => {
  const u16 = new Uint16Array([0x0102]);
  return new Uint8Array(u16.buffer)[0] === 0x02;
})();

function concatStructuredPayload(blocks: Uint8Array | undefined, constraint: Uint8Array): Uint8Array {
  const blockBytes = blocks?.byteLength ?? 0;
  const result = new Uint8Array(blockBytes + constraint.byteLength);
  if (blocks) result.set(blocks, 0);
  result.set(constraint, blockBytes);
  return result;
}

/** Encode a u32 vector exactly as the bridge bulk payload. */
export function encodeU32Payload(values: Uint32Array): Uint8Array {
  if (littleEndian) return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  const bytes = new Uint8Array(values.byteLength);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < values.length; i++) view.setUint32(i * 4, values[i]!, true);
  return bytes;
}

/** Decode a bridge u32 payload. A copy is returned so alignment is irrelevant. */
export function decodeU32Payload(payload: Uint8Array | undefined, count: number): Uint32Array {
  const expected = count * 4;
  if ((payload?.byteLength ?? 0) !== expected) {
    throw new Error(`Invalid u32 payload: expected ${expected} bytes, got ${payload?.byteLength ?? 0}`);
  }
  if (count === 0) return new Uint32Array(0);
  const result = new Uint32Array(count);
  const view = new DataView(payload!.buffer, payload!.byteOffset, payload!.byteLength);
  for (let i = 0; i < count; i++) result[i] = view.getUint32(i * 4, true);
  return result;
}

function encodeContext(context: Context, flags = 0): { ref: ABI.ContextRef; payload?: Uint8Array } {
  const checkpoint = context.checkpoint ?? 0;
  if (!Number.isInteger(checkpoint) || checkpoint < 0 || checkpoint > 0xffff_ffff) {
    throw new RangeError(`Invalid checkpoint id ${checkpoint}`);
  }

  const blocks = context.blocks ?? [];
  if (blocks.length > MAX_CONTEXT_BLOCKS) {
    throw new RangeError(`Context has ${blocks.length} blocks; max is ${MAX_CONTEXT_BLOCKS}`);
  }
  const ids = new Uint32Array(blocks.length);
  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i]!;
    if (!Number.isInteger(block) || block <= 0 || block > 0xffff_ffff) {
      throw new RangeError(`Invalid block id ${block}`);
    }
    ids[i] = block;
  }

  return {
    ref: { checkpoint, blockCount: ids.length, reserved: flags },
    payload: ids.length ? encodeU32Payload(ids) : undefined,
  };
}

function decodeFailure(event: ABI.Failed, payload: Uint8Array | undefined): EngineOperationError {
  if (event.messageBytes > (payload?.byteLength ?? 0)) {
    return new EngineOperationError(
      event.operation,
      event.code,
      `Malformed engine error payload: expected ${event.messageBytes} bytes, got ${payload?.byteLength ?? 0}`,
    );
  }
  const message = new TextDecoder().decode(payload?.subarray(0, event.messageBytes) ?? new Uint8Array());
  return new EngineOperationError(event.operation, event.code, message || event.code);
}

export class Engine {
  private readonly structured = new Map<OperationId, Deferred<Uint8Array>>();
  private nextOperation = 1;
  private nextResource = 1;
  private readonly pending = new Map<OperationId, Deferred<void>>();
  private readonly generations = new Map<OperationId, TokenQueue>();
  private readonly unsubscribe: () => void;
  private closed = false;
  private stats: EngineStats = emptyEngineStats();

  /**
   * Client-known lifecycle counters plus backend-reported execution facts.
   * prefill/checkpoint counters are updated only from ExecutionStats events;
   * engine-ts never infers backend work from the requested Context.
   */
  readonly debug: EngineDebug = {
    stats: () => ({ ...this.stats }),
    resetStats: () => {
      this.stats = emptyEngineStats();
    },
  };

  constructor(readonly transport: EngineTransport) {
    this.unsubscribe = transport.subscribe((frame) => this.onEvent(frame));
  }

  private allocateOperation(): OperationId {
    const id = this.nextOperation++ >>> 0;
    if (id === 0) throw new Error("Operation id space exhausted");
    return id;
  }

  private allocateResource(): number {
    const id = this.nextResource++ >>> 0;
    if (id === 0) throw new Error("Resource id space exhausted");
    return id;
  }

  private assertOpen(): void {
    if (this.closed) throw new Error("Engine is closed");
  }

  private async request(command: ABI.EngineCommand, payload?: Uint8Array): Promise<void> {
    this.assertOpen();
    const operation = command.operation;
    const deferred = new Deferred<void>();
    this.pending.set(operation, deferred);
    try {
      await this.transport.send({ message: command, payload });
    } catch (error) {
      this.pending.delete(operation);
      deferred.reject(error);
    }
    return deferred.promise;
  }

  private onEvent(frame: TransportFrame<ABI.EngineEvent>): void {
    const event = frame.message;
    if (event.kind === "TokenEmitted") {
      this.stats.generatedTokens++;
      this.generations.get(event.operation)?.push(event.token);
      return;
    }

    if (event.kind === "ExecutionStats") {
      this.stats.prefillTokens += event.prefillTokens;
      this.stats.checkpointHits += event.checkpointHits;
      this.stats.checkpointMisses += event.checkpointMisses;
      this.stats.restoredCheckpointBytes += event.restoredBytes;
      this.stats.checkpointBytes += event.checkpointBytes;
      this.stats.kvBytes += event.kvBytes;
      this.stats.kvCapacityBytes += event.kvCapacityBytes;
      this.stats.convBytes += event.convBytes;
      this.stats.hiddenBytes += event.hiddenBytes;
      this.stats.checkpointCreateUs += event.checkpointCreateUs;
      this.stats.checkpointRestoreUs += event.checkpointRestoreUs;
      return;
    }

    if (event.kind === "Completed") {
      const structured = this.structured.get(event.operation);
      if (structured) {
        this.structured.delete(event.operation);
        structured.resolve(frame.payload?.slice() ?? new Uint8Array());
        return;
      }
      const generation = this.generations.get(event.operation);
      if (generation) {
        generation.finish();
        this.generations.delete(event.operation);
        return;
      }
      const pending = this.pending.get(event.operation);
      if (pending) {
        this.pending.delete(event.operation);
        pending.resolve();
      }
      return;
    }

    const error = decodeFailure(event, frame.payload);
    const structured = this.structured.get(event.operation);
    if (structured) {
      this.structured.delete(event.operation);
      structured.reject(error);
      return;
    }
    const generation = this.generations.get(event.operation);
    if (generation) {
      generation.fail(error);
      this.generations.delete(event.operation);
      return;
    }
    const pending = this.pending.get(event.operation);
    if (pending) {
      this.pending.delete(event.operation);
      pending.reject(error);
    }
  }

  async putBlock(tokens: Uint32Array | readonly number[]): Promise<BlockId> {
    const values = tokens instanceof Uint32Array ? tokens : Uint32Array.from(tokens);
    const operation = this.allocateOperation();
    const block = this.allocateResource();
    this.stats.blocksPut++;
    this.stats.contextTokens += values.length;
    this.stats.commands++;
    await this.request(
      { kind: "PutBlock", operation, block, tokenCount: values.length },
      encodeU32Payload(values),
    );
    return block;
  }

  async dropBlock(block: BlockId): Promise<void> {
    const operation = this.allocateOperation();
    this.stats.blocksDropped++;
    this.stats.commands++;
    await this.request({ kind: "DropBlock", operation, block });
  }

  async checkpoint(context: Context): Promise<CheckpointId> {
    const operation = this.allocateOperation();
    const checkpoint = this.allocateResource();
    const encoded = encodeContext(context);
    this.stats.checkpointsCreated++;
    this.stats.commands++;
    await this.request(
      { kind: "CreateCheckpoint", operation, checkpoint, context: encoded.ref },
      encoded.payload,
    );
    return checkpoint;
  }

  async dropCheckpoint(checkpoint: CheckpointId): Promise<void> {
    const operation = this.allocateOperation();
    this.stats.checkpointsDropped++;
    this.stats.commands++;
    await this.request({ kind: "DropCheckpoint", operation, checkpoint });
  }

  generate(context: Context, options: GenerateOptions): Generation;
  generate<S extends GeneratableSchema>(schema: S, context: Context): Promise<InferGeneratable<S>>;
  generate(
    schemaOrContext: GeneratableSchema | Context,
    contextOrOptions: Context | GenerateOptions,
  ): Generation | Promise<unknown> {
    if (isGeneratableSchema(schemaOrContext)) {
      return this.generateStructured(schemaOrContext, contextOrOptions as Context);
    }
    return this.generateTokens(schemaOrContext as Context, contextOrOptions as GenerateOptions);
  }

  private async generateStructured<S extends GeneratableSchema>(
    schema: S,
    context: Context,
  ): Promise<InferGeneratable<S>> {
    this.assertOpen();
    const compiled = compileStructuredGeneration(schema);
    const operation = this.allocateOperation();
    const encoded = encodeContext(context, CONTEXT_FLAG_STRUCTURED);
    const payload = concatStructuredPayload(encoded.payload, encodeU32Payload(compiled.program.blob));
    const pending = new Deferred<Uint8Array>();

    this.stats.generations++;
    this.stats.commands++;
    this.structured.set(operation, pending);

    try {
      await Promise.resolve(this.transport.send({
        message: {
          kind: "Generate",
          operation,
          context: encoded.ref,
          maxTokens: compiled.maxTokens,
        },
        payload,
      }));
      const bytes = await pending.promise;
      const json = decodeUtf8Strict(bytes);
      return JSON.parse(json) as InferGeneratable<S>;
    } catch (error) {
      this.structured.delete(operation);
      throw error;
    }
  }

  private generateTokens(context: Context, options: GenerateOptions): Generation {
    this.assertOpen();
    if (!Number.isInteger(options.maxTokens) || options.maxTokens < 1 || options.maxTokens > 0xffff_ffff) {
      throw new RangeError(`Invalid maxTokens ${options.maxTokens}`);
    }

    const operation = this.allocateOperation();
    const queue = new TokenQueue();
    const encoded = encodeContext(context);
    this.stats.generations++;
    this.stats.commands++;
    this.generations.set(operation, queue);

    const started: Promise<void> = Promise.resolve(
      this.transport.send({
        message: {
          kind: "Generate",
          operation,
          context: encoded.ref,
          maxTokens: options.maxTokens,
        },
        payload: encoded.payload,
      }),
    ).catch((error) => {
      queue.fail(error);
      this.generations.delete(operation);
    });

    let cancelled = false;
    const cancel = async (): Promise<void> => {
      if (cancelled || !this.generations.has(operation)) return;
      cancelled = true;
      await started;
      if (!this.generations.has(operation)) return;
      const cancelOperation = this.allocateOperation();
      this.stats.cancellations++;
      this.stats.commands++;
      await this.request({ kind: "Cancel", operation: cancelOperation, target: operation });
      queue.finish();
      this.generations.delete(operation);
    };

    return {
      id: operation,
      cancel,
      [Symbol.asyncIterator](): AsyncIterator<TokenId> {
        return {
          async next() {
            await started;
            return queue.next();
          },
          async return() {
            await cancel();
            return { value: undefined, done: true };
          },
        };
      },
    };
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    this.unsubscribe();
    const error = new Error("Engine closed");
    for (const pending of this.pending.values()) pending.reject(error);
    for (const generation of this.generations.values()) generation.fail(error);
    for (const structured of this.structured.values()) structured.reject(error);
    this.pending.clear();
    this.generations.clear();
    this.structured.clear();
    await this.transport.close?.();
  }
}

/** Zero-codec transport for an engine implementation in the same JS process. */
export class InProcessTransport implements EngineTransport {
  private readonly listeners = new Set<(frame: TransportFrame<ABI.EngineEvent>) => void>();

  constructor(
    readonly handler: (
      frame: TransportFrame<ABI.EngineCommand>,
      emit: (frame: TransportFrame<ABI.EngineEvent>) => void,
    ) => void | Promise<void>,
    private readonly onClose?: () => void | Promise<void>,
  ) {}

  send(frame: TransportFrame<ABI.EngineCommand>): void | Promise<void> {
    return this.handler(frame, (event) => {
      for (const listener of this.listeners) listener(event);
    });
  }

  subscribe(listener: (frame: TransportFrame<ABI.EngineEvent>) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  close(): void | Promise<void> {
    this.listeners.clear();
    return this.onClose?.();
  }
}

/** Helpers useful to in-process/native transport adapters. */
export function completed(operation: OperationId): TransportFrame<ABI.EngineEvent> {
  return { message: { kind: "Completed", operation } };
}

export function tokenEmitted(operation: OperationId, token: TokenId): TransportFrame<ABI.EngineEvent> {
  return { message: { kind: "TokenEmitted", operation, token } };
}

export function executionStats(
  operation: OperationId,
  stats: {
    readonly prefillTokens?: number;
    readonly checkpointHits?: number;
    readonly checkpointMisses?: number;
    readonly restoredBytes?: number;
    readonly checkpointBytes?: number;
    readonly kvBytes?: number;
    readonly kvCapacityBytes?: number;
    readonly convBytes?: number;
    readonly hiddenBytes?: number;
    readonly checkpointCreateUs?: number;
    readonly checkpointRestoreUs?: number;
  },
): TransportFrame<ABI.EngineEvent> {
  return {
    message: {
      kind: "ExecutionStats",
      operation,
      prefillTokens: stats.prefillTokens ?? 0,
      checkpointHits: stats.checkpointHits ?? 0,
      checkpointMisses: stats.checkpointMisses ?? 0,
      restoredBytes: stats.restoredBytes ?? 0,
      checkpointBytes: stats.checkpointBytes ?? 0,
      kvBytes: stats.kvBytes ?? 0,
      kvCapacityBytes: stats.kvCapacityBytes ?? 0,
      convBytes: stats.convBytes ?? 0,
      hiddenBytes: stats.hiddenBytes ?? 0,
      checkpointCreateUs: stats.checkpointCreateUs ?? 0,
      checkpointRestoreUs: stats.checkpointRestoreUs ?? 0,
    },
  };
}

export function failed(
  operation: OperationId,
  code: ABI.ErrorCode,
  message: string,
): TransportFrame<ABI.EngineEvent> {
  const payload = new TextEncoder().encode(message);
  if (payload.byteLength > 0xffff) throw new RangeError("Bridge error message exceeds 65535 UTF-8 bytes");
  return {
    message: { kind: "Failed", operation, code, messageBytes: payload.byteLength, reserved: 0 },
    payload,
  };
}

export function completedWithPayload(
  operation: OperationId,
  payload: Uint8Array,
): TransportFrame<ABI.EngineEvent> {
  return { message: { kind: "Completed", operation }, payload };
}
