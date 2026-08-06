import {
  InProcessTransport,
  completed,
  decodeU32Payload,
  failed,
  tokenEmitted,
  type EngineTransport,
  type TransportFrame,
} from "@chomato/engine-ts/transport";
import type { v1_0_0 as ABI } from "@chomato/bridge/types";
export interface Lfm2GenerationRuntime {
  generateGreedy(
    promptTokens: Uint32Array | readonly number[],
    options?: { readonly maxNewTokens?: number; readonly resetState?: boolean },
  ): Promise<{ readonly tokens: number[] }>;
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
 * Checkpoints are exact but currently recomputable: the transport stores the
 * immutable token prefix represented by a checkpoint and normal prefill is
 * replayed for generation. Replacing this with KV/conv state snapshots does
 * not change the bridge or engine-ts API.
 */
export class Lfm2WebGpuEngineBackend {
  private readonly blocks = new Map<number, Uint32Array>();
  private readonly checkpoints = new Map<number, Uint32Array>();
  private readonly cancelled = new Set<number>();

  constructor(readonly forward: Lfm2GenerationRuntime) {}

  private contextTokens(
    context: ABI.ContextRef,
    payload: Uint8Array | undefined,
  ): Uint32Array {
    const blockIds = decodeU32Payload(payload, context.blockCount);
    const parts: Uint32Array[] = [];

    if (context.checkpoint !== 0) {
      const checkpoint = this.checkpoints.get(context.checkpoint);
      if (!checkpoint) throw new Error(`Checkpoint ${context.checkpoint} not found`);
      parts.push(checkpoint);
    }

    for (const blockId of blockIds) {
      const block = this.blocks.get(blockId);
      if (!block) throw new Error(`Block ${blockId} not found`);
      parts.push(block);
    }

    return concatTokens(parts);
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
        const tokens = this.contextTokens(command.context, frame.payload);
        this.checkpoints.set(command.checkpoint, tokens.slice());
        emit(completed(command.operation));
        return;
      }

      case "DropCheckpoint": {
        if (!this.checkpoints.delete(command.checkpoint)) {
          throw new Error(`Checkpoint ${command.checkpoint} not found`);
        }
        emit(completed(command.operation));
        return;
      }

      case "Generate": {
        const prompt = this.contextTokens(command.context, frame.payload);
        if (prompt.length === 0) throw new Error("Generation context is empty");
        this.cancelled.delete(command.operation);

        const result = await this.forward.generateGreedy(prompt, {
          maxNewTokens: command.maxTokens,
          resetState: true,
        });

        if (this.cancelled.has(command.operation)) {
          this.cancelled.delete(command.operation);
          emit(failed(command.operation, "Cancelled", "Generation cancelled"));
          return;
        }

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

  handle = async (
    frame: TransportFrame<ABI.EngineCommand>,
    emit: (frame: TransportFrame<ABI.EngineEvent>) => void,
  ): Promise<void> => {
    try {
      await this.execute(frame, emit);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const code: ABI.ErrorCode = message.includes("not found") ? "NotFound" : "InternalError";
      emit(failed(frame.message.operation, code, message));
    }
  };
}

/** Create the local WebGPU transport consumed by @chomato/engine-ts. */
export function createLfm2WebGpuTransport(forward: Lfm2GenerationRuntime): EngineTransport {
  const backend = new Lfm2WebGpuEngineBackend(forward);
  return new InProcessTransport(backend.handle);
}
