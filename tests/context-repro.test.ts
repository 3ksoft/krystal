// Repro: does context actually reach the model?
//
// A value is stated in one block and asked for in another. If the composed
// context is really in the model's KV state, structured generation must return
// that value. The controls are what make the failure diagnosable:
//
//   single   one block containing the whole prompt   -> is the prompt itself sound?
//   blocks   the same text split across two blocks   -> does block composition work?
//   ckpt     first block frozen into a checkpoint    -> does checkpoint restore work?
//   blind    the question with no context at all     -> what does the model say uninformed?
//
// "blind" is the discriminator: if an informed run matches it, the context did
// not reach the model rather than merely being misread.
import { expect, test } from "bun:test";
import { type } from "arktype";
import { loadModel } from "../src";

const MODEL = "./models/LFM2.5-1.2B-Instruct-WQ4.wq4";

const FACTS = "const alfa = 43.2;\nconst gamma = 21.7;\n";
const QUESTION = "provide value for alfa";
const Answer = type({ alfa: "number" });

/** ChatML framing split so the facts land in their own block. */
function parts(tokenizer: {
  formatMessage(role: "system" | "user", content: string): string;
  encode(text: string, options?: { addBos?: boolean; addEos?: boolean; parseSpecial?: boolean }): number[];
}) {
  const head = `<|im_start|>user\n${FACTS}`;
  const tail = `${QUESTION}<|im_end|>\n<|im_start|>assistant\n`;
  return {
    head: Uint32Array.from(tokenizer.encode(head, { addBos: true, addEos: false, parseSpecial: true })),
    tail: Uint32Array.from(tokenizer.encode(tail, { addBos: false, addEos: false, parseSpecial: true })),
    whole: Uint32Array.from(tokenizer.encode(head + tail, { addBos: true, addEos: false, parseSpecial: true })),
    blind: Uint32Array.from(
      tokenizer.encode(`<|im_start|>user\n${tail}`, { addBos: true, addEos: false, parseSpecial: true }),
    ),
  };
}

test("context reaches the model through blocks and through a checkpoint", async () => {
  const model = await loadModel(MODEL);
  const engine = model.engine;
  const tokenizer = model.forward?.tokenizer;
  if (!tokenizer) throw new Error("real GPU engine required for this repro");

  const p = parts(tokenizer);
  const results: Record<string, unknown> = {};

  try {
    // 1. Whole prompt in a single block — establishes that the prompt works.
    const whole = await engine.putBlock(p.whole);
    results.single = await engine.generate(Answer, { blocks: [whole] });

    // 2. Same tokens, split across two blocks.
    const head = await engine.putBlock(p.head);
    const tail = await engine.putBlock(p.tail);
    results.blocks = await engine.generate(Answer, { blocks: [head, tail] });

    // 3. Facts frozen into a checkpoint, question appended on top.
    const checkpoint = await engine.checkpoint({ blocks: [head] });
    results.ckpt = await engine.generate(Answer, { checkpoint, blocks: [tail] });

    // 4. No context at all.
    const blind = await engine.putBlock(p.blind);
    results.blind = await engine.generate(Answer, { blocks: [blind] });

    console.log("[context-repro]", JSON.stringify({
      tokens: { head: p.head.length, tail: p.tail.length, whole: p.whole.length, blind: p.blind.length },
      results,
      stats: engine.debug.stats(),
    }, null, 2));

    expect((results.single as { alfa: number }).alfa).toBe(43.2);
    expect((results.blocks as { alfa: number }).alfa).toBe(43.2);
    expect((results.ckpt as { alfa: number }).alfa).toBe(43.2);
  } finally {
    await model.dispose();
  }
}, 180_000);
