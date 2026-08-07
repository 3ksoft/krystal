import { CONTEXT_FLAG_STRUCTURED } from "@chomato/bridge/constants";
import {
  completedWithPayload,
  InProcessTransport,
  completed,
  decodeU32Payload,
  executionStats,
  failed,
  tokenEmitted,
  type EngineTransport,
  type TransportFrame,
} from "@chomato/engine-ts/transport";
import type { v1_0_0 as ABI } from "@chomato/bridge/types";
export interface Lfm2RuntimeCheckpoint {
  readonly position: number;
  readonly byteLength: number;
  readonly kvBytes: number;
  readonly kvCapacityBytes: number;
  readonly convBytes: number;
  readonly hiddenBytes: number;
  readonly createUs: number;
  readonly creationRestoredBytes: number;
  destroy(): void;
}

export interface Lfm2RuntimeGenerationResult {
  readonly tokens: number[];
  readonly execution: {
    readonly prefillTokens: number;
    readonly restoredCheckpointBytes: number;
    readonly checkpointRestoreUs: number;
  };
}

export interface Lfm2GenerationRuntime {
  generateGreedy(
    promptTokens: Uint32Array | readonly number[],
    options?: { readonly maxNewTokens?: number; readonly resetState?: boolean },
  ): Promise<Lfm2RuntimeGenerationResult>;
  createCheckpoint(
    tailTokens: Uint32Array | readonly number[],
    base?: Lfm2RuntimeCheckpoint,
  ): Promise<Lfm2RuntimeCheckpoint>;
  generateGreedyFromCheckpoint(
    checkpoint: Lfm2RuntimeCheckpoint,
    tailTokens: Uint32Array | readonly number[],
    options?: { readonly maxNewTokens?: number },
  ): Promise<Lfm2RuntimeGenerationResult>;
  generateStructured(
    promptTokens: Uint32Array | readonly number[],
    constraintBlob: Uint32Array,
    options: { readonly maxNewTokens: number },
  ): Promise<Lfm2RuntimeGenerationResult & { readonly text: string }>;
  generateStructuredFromCheckpoint(
    checkpoint: Lfm2RuntimeCheckpoint,
    tailTokens: Uint32Array | readonly number[],
    constraintBlob: Uint32Array,
    options: { readonly maxNewTokens: number },
  ): Promise<Lfm2RuntimeGenerationResult & { readonly text: string }>;
}

function decodeConstraintWords(payload: Uint8Array | undefined, byteOffset: number): Uint32Array {
  const total = payload?.byteLength ?? 0;
  if (byteOffset < 0 || byteOffset > total) throw new Error(`Invalid structured payload offset ${byteOffset}/${total}`);
  const bytes = total - byteOffset;
  if (bytes <= 0 || (bytes & 3) !== 0) throw new Error(`Invalid constraint payload size ${bytes}`);
  const result = new Uint32Array(bytes >>> 2);
  const view = new DataView(payload!.buffer, payload!.byteOffset + byteOffset, bytes);
  for (let i = 0; i < result.length; i++) result[i] = view.getUint32(i * 4, true);
  return result;
}

function concatTokens(parts: readonly Uint32Array[]): Uint32Array {
  let length = 0;
  for (const part of parts) length += part.length;
  const result = new Uint32Array(length);
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.length;
  }
  return result;
}

/**
 * In-process implementation of the portable Chomato engine protocol.
 *
 * Checkpoints are exact physical GPU snapshots. The backend keeps the immutable
 * token prefix for context composition, while Lfm2GenerationRuntime owns the
 * actual KV/conv/last-hidden state. engine-ts and the bridge remain transport-agnostic.
 */
export class Lfm2WebGpuEngineBackend {
  private readonly blocks = new Map<number, Uint32Array>();
  private readonly checkpoints = new Map<number, {
    readonly tokens: Uint32Array;
    readonly state: Lfm2RuntimeCheckpoint;
  }>();
  private readonly cancelled = new Set<number>();

  constructor(readonly forward: Lfm2GenerationRuntime) {}

  private resolveContext(
    context: ABI.ContextRef,
    payload: Uint8Array | undefined,
  ): {
    readonly checkpoint?: { readonly tokens: Uint32Array; readonly state: Lfm2RuntimeCheckpoint };
    readonly appended: Uint32Array;
    readonly full: Uint32Array;
  } {
    const blockIds = decodeU32Payload(payload, context.blockCount);
    const parts: Uint32Array[] = [];
    let checkpoint: { readonly tokens: Uint32Array; readonly state: Lfm2RuntimeCheckpoint } | undefined;

    if (context.checkpoint !== 0) {
      checkpoint = this.checkpoints.get(context.checkpoint);
      if (!checkpoint) throw new Error(`Checkpoint ${context.checkpoint} not found`);
    }

    for (const blockId of blockIds) {
      const block = this.blocks.get(blockId);
      if (!block) throw new Error(`Block ${blockId} not found`);
      parts.push(block);
    }

    const appended = concatTokens(parts);
    const full = checkpoint ? concatTokens([checkpoint.tokens, appended]) : appended;
    return { checkpoint, appended, full };
  }

  private async execute(
    frame: TransportFrame<ABI.EngineCommand>,
    emit: (frame: TransportFrame<ABI.EngineEvent>) => void,
  ): Promise<void> {
    const command = frame.message;

    switch (command.kind) {
      case "PutBlock": {
        if (this.blocks.has(command.block)) throw new Error(`Block ${command.block} already exists`);
        const tokens = decodeU32Payload(frame.payload, command.tokenCount);
        this.blocks.set(command.block, tokens);
        emit(completed(command.operation));
        return;
      }

      case "DropBlock": {
        if (!this.blocks.delete(command.block)) throw new Error(`Block ${command.block} not found`);
        emit(completed(command.operation));
        return;
      }

      case "CreateCheckpoint": {
        if (this.checkpoints.has(command.checkpoint)) {
          throw new Error(`Checkpoint ${command.checkpoint} already exists`);
        }
        const context = this.resolveContext(command.context, frame.payload);
        if (context.full.length === 0) throw new Error("Checkpoint context is empty");
        const state = await this.forward.createCheckpoint(
          context.appended,
          context.checkpoint?.state,
        );
        this.checkpoints.set(command.checkpoint, {
          tokens: context.full.slice(),
          state,
        });
        emit(executionStats(command.operation, {
          prefillTokens: context.appended.length,
          checkpointHits: context.checkpoint ? 1 : 0,
          checkpointMisses: 0,
          restoredBytes: state.creationRestoredBytes,
          checkpointBytes: state.byteLength,
          kvBytes: state.kvBytes,
          kvCapacityBytes: state.kvCapacityBytes,
          convBytes: state.convBytes,
          hiddenBytes: state.hiddenBytes,
          checkpointCreateUs: state.createUs,
          checkpointRestoreUs: 0,
        }));
        emit(completed(command.operation));
        return;
      }

      case "DropCheckpoint": {
        const checkpoint = this.checkpoints.get(command.checkpoint);
        if (!checkpoint) throw new Error(`Checkpoint ${command.checkpoint} not found`);
        this.checkpoints.delete(command.checkpoint);
        checkpoint.state.destroy();
        emit(completed(command.operation));
        return;
      }

      case "Generate": {
        if ((command.context.reserved & CONTEXT_FLAG_STRUCTURED) !== 0) {
          const blockBytes = command.context.blockCount * 4;
          const totalBytes = frame.payload?.byteLength ?? 0;
          if (totalBytes < blockBytes) {
            throw new Error(`Structured Generate payload has ${totalBytes} bytes; block ids require ${blockBytes}`);
          }
          const blockPayload = blockBytes > 0
            ? frame.payload!.subarray(0, blockBytes)
            : new Uint8Array(0);
          const constraintBlob = decodeConstraintWords(frame.payload, blockBytes);
          const context = this.resolveContext(command.context, blockPayload);
          if (context.full.length === 0) throw new Error("Generation context is empty");
          this.cancelled.delete(command.operation);

          const result = context.checkpoint
            ? await this.forward.generateStructuredFromCheckpoint(
                context.checkpoint.state,
                context.appended,
                constraintBlob,
                { maxNewTokens: command.maxTokens },
              )
            : await this.forward.generateStructured(
                context.appended,
                constraintBlob,
                { maxNewTokens: command.maxTokens },
              );

          if (this.cancelled.has(command.operation)) {
            this.cancelled.delete(command.operation);
            emit(failed(command.operation, "Cancelled", "Generation cancelled"));
            return;
          }

          emit(executionStats(command.operation, {
            prefillTokens: result.execution.prefillTokens,
            checkpointHits: context.checkpoint ? 1 : 0,
            checkpointMisses: 0,
            restoredBytes: result.execution.restoredCheckpointBytes,
          }));
          emit(completedWithPayload(command.operation, new TextEncoder().encode(result.text)));
          return;
        }

        const context = this.resolveContext(command.context, frame.payload);
        if (context.full.length === 0) throw new Error("Generation context is empty");
        this.cancelled.delete(command.operation);

        const result = context.checkpoint
          ? await this.forward.generateGreedyFromCheckpoint(
              context.checkpoint.state,
              context.appended,
              { maxNewTokens: command.maxTokens },
            )
          : await this.forward.generateGreedy(context.appended, {
              maxNewTokens: command.maxTokens,
              resetState: true,
            });

        if (this.cancelled.has(command.operation)) {
          this.cancelled.delete(command.operation);
          emit(failed(command.operation, "Cancelled", "Generation cancelled"));
          return;
        }

        emit(executionStats(command.operation, {
          prefillTokens: result.execution.prefillTokens,
          checkpointHits: context.checkpoint ? 1 : 0,
          checkpointMisses: 0,
          restoredBytes: result.execution.restoredCheckpointBytes,
          checkpointRestoreUs: result.execution.checkpointRestoreUs,
        }));
        for (const token of result.tokens) emit(tokenEmitted(command.operation, token));
        emit(completed(command.operation));
        return;
      }

      case "Cancel": {
        this.cancelled.add(command.target);
        emit(completed(command.operation));
        return;
      }
    }
  }

  close(): void {
    for (const checkpoint of this.checkpoints.values()) checkpoint.state.destroy();
    this.checkpoints.clear();
    this.blocks.clear();
    this.cancelled.clear();
  }

  handle = async (
    frame: TransportFrame<ABI.EngineCommand>,
    emit: (frame: TransportFrame<ABI.EngineEvent>) => void,
  ): Promise<void> => {
    try {
      await this.execute(frame, emit);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const code: ABI.ErrorCode = message.includes("not found") ? "NotFound" : "InternalError";
      if (frame.message.kind === "Generate" && frame.message.context.checkpoint !== 0) {
        emit(executionStats(frame.message.operation, { checkpointMisses: 1 }));
      }
      emit(failed(frame.message.operation, code, message));
    }
  };
}

/** Create the local WebGPU transport consumed by @chomato/engine-ts. */
export function createLfm2WebGpuTransport(forward: Lfm2GenerationRuntime): EngineTransport {
  const backend = new Lfm2WebGpuEngineBackend(forward);
  return new InProcessTransport(backend.handle, () => backend.close());
}
