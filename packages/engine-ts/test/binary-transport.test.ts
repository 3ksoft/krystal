import { expect, test } from "bun:test";
import { decodeFrame, encodeFrame } from "@chomato/bridge/wire";
import { BinaryEngineTransport, type BinaryChannel } from "../src/binary-transport";
import { Engine } from "../src/transport";

class LoopbackNativeChannel implements BinaryChannel {
  private readonly listeners = new Set<(bytes: Uint8Array) => void>();

  subscribe(listener: (bytes: Uint8Array) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private receive(bytes: Uint8Array): void {
    // Deliberately split a frame to exercise stream reassembly.
    const split = Math.min(7, bytes.byteLength);
    for (const listener of this.listeners) {
      listener(bytes.subarray(0, split));
      listener(bytes.subarray(split));
    }
  }

  send(bytes: Uint8Array): void {
    const frame = decodeFrame(bytes);
    expect(frame.direction).toBe("command");
    const command = frame.message;
    if (command.kind === "PutBlock") {
      expect(frame.payload?.byteLength).toBe(command.tokenCount * 4);
      this.receive(encodeFrame("event", { message: { kind: "Completed", operation: command.operation } }));
      return;
    }
    if (command.kind === "Generate") {
      this.receive(encodeFrame("event", { message: { kind: "TokenEmitted", operation: command.operation, token: 777 } }));
      this.receive(encodeFrame("event", { message: { kind: "Completed", operation: command.operation } }));
      return;
    }
    this.receive(encodeFrame("event", { message: { kind: "Completed", operation: command.operation } }));
  }
}

test("binary transport frames the same engine protocol over arbitrary chunks", async () => {
  const engine = new Engine(new BinaryEngineTransport(new LoopbackNativeChannel()));
  const block = await engine.putBlock(Uint32Array.of(1, 2, 3));
  const generated: number[] = [];
  for await (const token of engine.generate({ blocks: [block] }, { maxTokens: 1 })) generated.push(token);
  expect(generated).toEqual([777]);
  await engine.close();
});
