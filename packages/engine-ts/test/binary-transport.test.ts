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

// A dropped or misaligned sampling field does not fail the frame — it silently
// turns a seeded generation into a different one on the far side of the
// transport, which is exactly the failure the seed exists to prevent.
test("Generate carries its sampler configuration across the wire", () => {
  const message = {
    kind: "Generate",
    operation: 9,
    context: { checkpoint: 3, blockCount: 2, reserved: 0 },
    maxTokens: 64,
    sampler: "topk",
    temperature: 0.75,
    topK: 40,
    seed: 0xdead_beef,
    reserved: 0,
  } as const;

  const frame = decodeFrame(encodeFrame("command", { message }));
  expect(frame.message).toEqual(message);
});

test("binary transport frames the same engine protocol over arbitrary chunks", async () => {
  const engine = new Engine(new BinaryEngineTransport(new LoopbackNativeChannel()));
  const block = await engine.putBlock(Uint32Array.of(1, 2, 3));
  const generated: number[] = [];
  for await (const token of engine.generate({ blocks: [block] }, { maxTokens: 1 })) generated.push(token);
  expect(generated).toEqual([777]);
  await engine.close();
});
