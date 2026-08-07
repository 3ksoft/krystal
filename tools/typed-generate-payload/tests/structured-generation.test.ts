import { expect, test } from "bun:test";
import { type } from "arktype";
import { loadModel } from "../src";

const MODEL = "./models/LFM2.5-1.2B-Instruct-WQ4.wq4";
const BOS = 1;

function tokens(...values: number[]): Uint32Array {
  return new Uint32Array(values);
}

test("typed generate returns a root value through the public engine API", async () => {
  const model = await loadModel(MODEL);
  const engine = model.engine;

  try {
    const context = await engine.putBlock(tokens(BOS));
    const result = await engine.generate(type("'ok'"), { blocks: [context] });
    expect(result).toBe("ok");
  } finally {
    await model.dispose();
  }
}, 30_000);

test("typed generate composes with an exact context checkpoint", async () => {
  const model = await loadModel(MODEL);
  const engine = model.engine;

  try {
    const prefix = await engine.putBlock(tokens(BOS));
    const checkpoint = await engine.checkpoint({ blocks: [prefix] });
    const result = await engine.generate(type("'ok'"), { checkpoint });
    expect(result).toBe("ok");
  } finally {
    await model.dispose();
  }
}, 30_000);
