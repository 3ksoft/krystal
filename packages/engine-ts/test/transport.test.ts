import { expect, test } from "bun:test";
import {
  Engine,
  InProcessTransport,
  completed,
  decodeU32Payload,
  failed,
  tokenEmitted,
} from "../src/transport";

test("engine-ts correlates blocks, checkpoints and token generation", async () => {
  const seen: string[] = [];
  const transport = new InProcessTransport(async (frame, emit) => {
    const command = frame.message;
    seen.push(command.kind);
    if (command.kind === "PutBlock") {
      expect(Array.from(decodeU32Payload(frame.payload, command.tokenCount))).toEqual([10, 11, 12]);
      emit(completed(command.operation));
      return;
    }
    if (command.kind === "CreateCheckpoint") {
      expect(command.context.checkpoint).toBe(0);
      expect(Array.from(decodeU32Payload(frame.payload, command.context.blockCount))).toEqual([1]);
      emit(completed(command.operation));
      return;
    }
    if (command.kind === "Generate") {
      expect(command.context.checkpoint).toBe(2);
      expect(command.context.blockCount).toBe(0);
      emit(tokenEmitted(command.operation, 90));
      emit(tokenEmitted(command.operation, 91));
      emit(completed(command.operation));
      return;
    }
    emit(completed(command.operation));
  });

  const engine = new Engine(transport);
  const block = await engine.putBlock(Uint32Array.of(10, 11, 12));
  const checkpoint = await engine.checkpoint({ blocks: [block] });
  const tokens: number[] = [];
  for await (const token of engine.generate({ checkpoint }, { maxTokens: 2 })) tokens.push(token);

  expect(block).toBe(1);
  expect(checkpoint).toBe(2);
  expect(tokens).toEqual([90, 91]);
  expect(seen).toEqual(["PutBlock", "CreateCheckpoint", "Generate"]);
  await engine.close();
});

test("engine-ts surfaces Failed events as EngineOperationError", async () => {
  const transport = new InProcessTransport((frame, emit) => {
    emit(failed(frame.message.operation, "NotFound", "missing resource"));
  });
  const engine = new Engine(transport);
  await expect(engine.dropBlock(123)).rejects.toMatchObject({
    name: "EngineOperationError",
    code: "NotFound",
    message: "missing resource",
  });
  await engine.close();
});
