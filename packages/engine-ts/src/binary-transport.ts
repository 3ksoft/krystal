import type { v1_0_0 as ABI } from "@chomato/bridge/types";
import {
  decodeEventBody,
  decodeFrameHeader,
  encodeFrame,
} from "@chomato/bridge/wire";
import { SIZEOF_FrameHeader } from "@chomato/bridge/layout";
import type { EngineTransport, TransportFrame } from "./transport";

export interface BinaryChannel {
  send(bytes: Uint8Array): void | Promise<void>;
  subscribe(listener: (bytes: Uint8Array) => void): () => void;
  close?(): void | Promise<void>;
}

export interface BinaryEngineTransportOptions {
  readonly maxPayloadBytes?: number;
}

function concat(a: Uint8Array, b: Uint8Array): Uint8Array {
  if (a.byteLength === 0) return b.slice();
  const result = new Uint8Array(a.byteLength + b.byteLength);
  result.set(a, 0);
  result.set(b, a.byteLength);
  return result;
}

/**
 * Stream-framed binary implementation of EngineTransport.
 * Works over stdio, sockets, WebSockets, native messaging, etc. once wrapped in
 * the tiny BinaryChannel interface.
 */
export class BinaryEngineTransport implements EngineTransport {
  private readonly listeners = new Set<(frame: TransportFrame<ABI.EngineEvent>) => void>();
  private readonly unsubscribe: () => void;
  private readonly maxPayloadBytes: number;
  private buffered = new Uint8Array();

  constructor(
    readonly channel: BinaryChannel,
    options: BinaryEngineTransportOptions = {},
  ) {
    this.maxPayloadBytes = options.maxPayloadBytes ?? 64 * 1024 * 1024;
    this.unsubscribe = channel.subscribe((bytes) => this.receive(bytes));
  }

  send(frame: TransportFrame<ABI.EngineCommand>): void | Promise<void> {
    return this.channel.send(encodeFrame("command", frame));
  }

  subscribe(listener: (frame: TransportFrame<ABI.EngineEvent>) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private receive(bytes: Uint8Array): void {
    this.buffered = concat(this.buffered, bytes);
    while (this.buffered.byteLength >= SIZEOF_FrameHeader) {
      const header = decodeFrameHeader(this.buffered);
      if (header.direction !== "event") throw new Error(`Expected event frame, got ${header.direction}`);
      if (header.payloadBytes > this.maxPayloadBytes) {
        throw new Error(`Bridge payload ${header.payloadBytes} exceeds limit ${this.maxPayloadBytes}`);
      }
      if (this.buffered.byteLength < header.frameBytes) return;

      const frame = this.buffered.subarray(0, header.frameBytes);
      const bodyStart = SIZEOF_FrameHeader;
      const body = frame.subarray(bodyStart, bodyStart + header.bodyBytes);
      const payload = header.payloadBytes ? frame.subarray(bodyStart + header.bodyBytes).slice() : undefined;
      const message = decodeEventBody(body);
      for (const listener of this.listeners) listener({ message, payload });
      this.buffered = this.buffered.subarray(header.frameBytes).slice();
    }
  }

  async close(): Promise<void> {
    this.unsubscribe();
    this.listeners.clear();
    await this.channel.close?.();
  }
}
