import { CHOMATO_BRIDGE_MAGIC, CHOMATO_BRIDGE_VERSION } from "./constants";
import { v1_0_0 as ABI } from "../generated/bridge.types";
import {
  SIZEOF_EngineCommand,
  SIZEOF_EngineEvent,
  SIZEOF_FrameHeader,
} from "../generated/bridge.layout";

export interface WireFrame<T> {
  readonly message: T;
  readonly payload?: Uint8Array;
}

function writeContext(view: DataView, offset: number, context: ABI.ContextRef): void {
  view.setUint32(offset, context.checkpoint, true);
  view.setUint16(offset + 4, context.blockCount, true);
  view.setUint16(offset + 6, context.reserved, true);
}

function readContext(view: DataView, offset: number): ABI.ContextRef {
  return {
    checkpoint: view.getUint32(offset, true),
    blockCount: view.getUint16(offset + 4, true),
    reserved: view.getUint16(offset + 6, true),
  };
}

function enumName<T extends string>(values: Record<T, number>, encoded: number, label: string): T {
  for (const [name, value] of Object.entries(values) as Array<[T, number]>) {
    if (value === encoded) return name;
  }
  throw new Error(`Unknown ${label} value ${encoded}`);
}

export function encodeCommandBody(command: ABI.EngineCommand): Uint8Array {
  const bytes = new Uint8Array(SIZEOF_EngineCommand);
  const view = new DataView(bytes.buffer);
  view.setUint8(0, ABI.EngineCommandTag[command.kind]);
  const base = 1;

  switch (command.kind) {
    case "Cancel":
      view.setUint32(base, command.operation, true);
      view.setUint32(base + 4, command.target, true);
      break;
    case "CreateCheckpoint":
      view.setUint32(base, command.operation, true);
      view.setUint32(base + 4, command.checkpoint, true);
      writeContext(view, base + 8, command.context);
      break;
    case "DropBlock":
      view.setUint32(base, command.operation, true);
      view.setUint32(base + 4, command.block, true);
      break;
    case "DropCheckpoint":
      view.setUint32(base, command.operation, true);
      view.setUint32(base + 4, command.checkpoint, true);
      break;
    case "Generate":
      view.setUint32(base, command.operation, true);
      writeContext(view, base + 4, command.context);
      view.setUint32(base + 12, command.maxTokens, true);
      view.setFloat32(base + 16, command.temperature, true);
      view.setUint32(base + 20, command.seed, true);
      view.setUint16(base + 24, command.topK, true);
      view.setUint8(base + 26, ABI.Sampler[command.sampler]);
      view.setUint8(base + 27, command.reserved);
      break;
    case "PutBlock":
      view.setUint32(base, command.operation, true);
      view.setUint32(base + 4, command.block, true);
      view.setUint32(base + 8, command.tokenCount, true);
      break;
  }
  return bytes;
}

export function decodeCommandBody(bytes: Uint8Array): ABI.EngineCommand {
  if (bytes.byteLength !== SIZEOF_EngineCommand) {
    throw new Error(`EngineCommand body must be ${SIZEOF_EngineCommand} bytes, got ${bytes.byteLength}`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const kind = enumName(ABI.EngineCommandTag, view.getUint8(0), "EngineCommandTag");
  const base = 1;
  switch (kind) {
    case "Cancel": return { kind, operation: view.getUint32(base, true), target: view.getUint32(base + 4, true) };
    case "CreateCheckpoint": return { kind, operation: view.getUint32(base, true), checkpoint: view.getUint32(base + 4, true), context: readContext(view, base + 8) };
    case "DropBlock": return { kind, operation: view.getUint32(base, true), block: view.getUint32(base + 4, true) };
    case "DropCheckpoint": return { kind, operation: view.getUint32(base, true), checkpoint: view.getUint32(base + 4, true) };
    case "Generate": return {
      kind,
      operation: view.getUint32(base, true),
      context: readContext(view, base + 4),
      maxTokens: view.getUint32(base + 12, true),
      temperature: view.getFloat32(base + 16, true),
      seed: view.getUint32(base + 20, true),
      topK: view.getUint16(base + 24, true),
      sampler: enumName(ABI.Sampler, view.getUint8(base + 26), "Sampler"),
      reserved: view.getUint8(base + 27),
    };
    case "PutBlock": return { kind, operation: view.getUint32(base, true), block: view.getUint32(base + 4, true), tokenCount: view.getUint32(base + 8, true) };
  }
}

export function encodeEventBody(event: ABI.EngineEvent): Uint8Array {
  const bytes = new Uint8Array(SIZEOF_EngineEvent);
  const view = new DataView(bytes.buffer);
  view.setUint8(0, ABI.EngineEventTag[event.kind]);
  const base = 1;
  switch (event.kind) {
    case "Completed":
      view.setUint32(base, event.operation, true);
      break;
    case "TokenEmitted":
      view.setUint32(base, event.operation, true);
      view.setUint32(base + 4, event.token, true);
      break;
    case "ExecutionStats":
      view.setUint32(base, event.operation, true);
      view.setUint32(base + 4, event.prefillTokens, true);
      view.setUint32(base + 8, event.checkpointHits, true);
      view.setUint32(base + 12, event.checkpointMisses, true);
      view.setUint32(base + 16, event.restoredBytes, true);
      view.setUint32(base + 20, event.checkpointBytes, true);
      view.setUint32(base + 24, event.kvBytes, true);
      view.setUint32(base + 28, event.kvCapacityBytes, true);
      view.setUint32(base + 32, event.convBytes, true);
      view.setUint32(base + 36, event.hiddenBytes, true);
      view.setUint32(base + 40, event.checkpointCreateUs, true);
      view.setUint32(base + 44, event.checkpointRestoreUs, true);
      break;
    case "Failed":
      view.setUint32(base, event.operation, true);
      view.setUint16(base + 4, event.messageBytes, true);
      view.setUint8(base + 6, ABI.ErrorCode[event.code]);
      view.setUint8(base + 7, event.reserved);
      break;
  }
  return bytes;
}

export function decodeEventBody(bytes: Uint8Array): ABI.EngineEvent {
  if (bytes.byteLength !== SIZEOF_EngineEvent) {
    throw new Error(`EngineEvent body must be ${SIZEOF_EngineEvent} bytes, got ${bytes.byteLength}`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const kind = enumName(ABI.EngineEventTag, view.getUint8(0), "EngineEventTag");
  const base = 1;
  switch (kind) {
    case "Completed": return { kind, operation: view.getUint32(base, true) };
    case "TokenEmitted": return { kind, operation: view.getUint32(base, true), token: view.getUint32(base + 4, true) };
    case "ExecutionStats": return {
      kind,
      operation: view.getUint32(base, true),
      prefillTokens: view.getUint32(base + 4, true),
      checkpointHits: view.getUint32(base + 8, true),
      checkpointMisses: view.getUint32(base + 12, true),
      restoredBytes: view.getUint32(base + 16, true),
      checkpointBytes: view.getUint32(base + 20, true),
      kvBytes: view.getUint32(base + 24, true),
      kvCapacityBytes: view.getUint32(base + 28, true),
      convBytes: view.getUint32(base + 32, true),
      hiddenBytes: view.getUint32(base + 36, true),
      checkpointCreateUs: view.getUint32(base + 40, true),
      checkpointRestoreUs: view.getUint32(base + 44, true),
    };
    case "Failed": return {
      kind,
      operation: view.getUint32(base, true),
      messageBytes: view.getUint16(base + 4, true),
      code: enumName(ABI.ErrorCode, view.getUint8(base + 6), "ErrorCode"),
      reserved: view.getUint8(base + 7),
    };
  }
}

export function encodeFrame<T extends ABI.EngineCommand | ABI.EngineEvent>(
  direction: "command" | "event",
  frame: WireFrame<T>,
): Uint8Array {
  const body = direction === "command"
    ? encodeCommandBody(frame.message as ABI.EngineCommand)
    : encodeEventBody(frame.message as ABI.EngineEvent);
  const payload = frame.payload ?? new Uint8Array();
  const bytes = new Uint8Array(SIZEOF_FrameHeader + body.byteLength + payload.byteLength);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, CHOMATO_BRIDGE_MAGIC, true);
  view.setUint16(4, CHOMATO_BRIDGE_VERSION, true);
  view.setUint8(6, ABI.FrameDirection[direction]);
  view.setUint8(7, 0);
  view.setUint32(8, body.byteLength, true);
  view.setUint32(12, payload.byteLength, true);
  bytes.set(body, SIZEOF_FrameHeader);
  bytes.set(payload, SIZEOF_FrameHeader + body.byteLength);
  return bytes;
}

export interface DecodedFrameHeader {
  readonly direction: "command" | "event";
  readonly flags: number;
  readonly bodyBytes: number;
  readonly payloadBytes: number;
  readonly frameBytes: number;
}

export function decodeFrameHeader(bytes: Uint8Array): DecodedFrameHeader {
  if (bytes.byteLength < SIZEOF_FrameHeader) throw new Error("Incomplete Chomato frame header");
  const view = new DataView(bytes.buffer, bytes.byteOffset, SIZEOF_FrameHeader);
  const magic = view.getUint32(0, true);
  const version = view.getUint16(4, true);
  if (magic !== CHOMATO_BRIDGE_MAGIC) throw new Error(`Invalid Chomato bridge magic 0x${magic.toString(16)}`);
  if (version !== CHOMATO_BRIDGE_VERSION) throw new Error(`Unsupported Chomato bridge version ${version}`);
  const direction = enumName(ABI.FrameDirection, view.getUint8(6), "FrameDirection");
  const bodyBytes = view.getUint32(8, true);
  const payloadBytes = view.getUint32(12, true);
  return {
    direction,
    flags: view.getUint8(7),
    bodyBytes,
    payloadBytes,
    frameBytes: SIZEOF_FrameHeader + bodyBytes + payloadBytes,
  };
}

export function decodeFrame(bytes: Uint8Array): WireFrame<ABI.EngineCommand | ABI.EngineEvent> & { direction: "command" | "event" } {
  const header = decodeFrameHeader(bytes);
  if (bytes.byteLength !== header.frameBytes) {
    throw new Error(`Frame length mismatch: expected ${header.frameBytes}, got ${bytes.byteLength}`);
  }
  const bodyStart = SIZEOF_FrameHeader;
  const body = bytes.subarray(bodyStart, bodyStart + header.bodyBytes);
  const payload = header.payloadBytes ? bytes.subarray(bodyStart + header.bodyBytes) : undefined;
  const message = header.direction === "command" ? decodeCommandBody(body) : decodeEventBody(body);
  return { direction: header.direction, message, payload };
}
