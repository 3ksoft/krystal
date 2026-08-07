// Reserved-vocabulary addressing against the real model.
//
// No GPU is involved: this reads the WQ4 container's tokenizer metadata only,
// so it verifies that [-token-K-] resolves to an actual single vocabulary entry
// rather than being silently tokenized as literal text.
import { expect, test } from "bun:test";
import { open } from "node:fs/promises";
import { Lfm2Tokenizer } from "../packages/lfm2/src/tokenizer.ts";
import type { RandomAccessSource } from "../packages/quant/src/gguf/source.ts";
import { Wq4Reader } from "../packages/quant/src/wq4/reader.ts";
import { buildReservedTable, expandReserved } from "../packages/gui/src/engine/reserved.ts";

const MODEL = "./models/LFM2.5-1.2B-Instruct-WQ4.wq4";

class NodeFileSource implements RandomAccessSource {
  readonly size: number;
  private constructor(private readonly handle: Awaited<ReturnType<typeof open>>, size: number) {
    this.size = size;
  }
  static async open(path: string): Promise<NodeFileSource> {
    const handle = await open(path, "r");
    return new NodeFileSource(handle, (await handle.stat()).size);
  }
  async read(offset: number, length: number): Promise<Uint8Array> {
    const buffer = new Uint8Array(length);
    await this.handle.read(buffer, 0, length, offset);
    return buffer;
  }
}

async function tokenizer(): Promise<Lfm2Tokenizer> {
  const reader = await Wq4Reader.open(await NodeFileSource.open(MODEL));
  return new Lfm2Tokenizer(reader as never);
}

function tableFor(tk: Lfm2Tokenizer) {
  return buildReservedTable(tk.idToToken, (id) => tk.isSpecialToken(id));
}

test("[-token-K-] resolves to exactly one reserved vocabulary entry", async () => {
  const tk = await tokenizer();
  const table = tableFor(tk);

  expect(table.literals.length).toBeGreaterThan(300);

  for (const index of [1, 2, 50, table.literals.length]) {
    const expanded = expandReserved(`[-token-${index}-]`, table);
    expect(expanded.unknown).toEqual([]);

    const ids = tk.encode(expanded.text, { addBos: false, addEos: false, parseSpecial: true });
    expect(ids).toEqual([table.ids[index - 1]!]);
    expect(tk.isSpecialToken(ids[0]!)).toBe(true);
  }
});

test("aliases survive being embedded in ordinary text", async () => {
  const tk = await tokenizer();
  const table = tableFor(tk);

  const expanded = expandReserved("alfa [-token-1-] gamma [-token-2-]", table);
  const ids = tk.encode(expanded.text, { addBos: false, addEos: false, parseSpecial: true });

  expect(ids).toContain(table.ids[0]!);
  expect(ids).toContain(table.ids[1]!);
  // Ordinary words still tokenize normally around them.
  expect(ids.length).toBeGreaterThan(4);
});

test("an out-of-range alias stays literal instead of vanishing", async () => {
  const tk = await tokenizer();
  const table = tableFor(tk);
  const beyond = table.literals.length + 1;

  const expanded = expandReserved(`[-token-${beyond}-]`, table);
  expect(expanded.unknown).toEqual([beyond]);
  expect(expanded.text).toBe(`[-token-${beyond}-]`);

  // It must tokenize as plain text, never as some unrelated special token.
  const ids = tk.encode(expanded.text, { addBos: false, addEos: false, parseSpecial: true });
  expect(ids.length).toBeGreaterThan(1);
  expect(ids.every((id) => !tk.isSpecialToken(id))).toBe(true);
});

test("reserved ids are disjoint from the named special tokens", async () => {
  const tk = await tokenizer();
  const table = tableFor(tk);

  for (const named of ["<|im_start|>", "<|im_end|>", "<|startoftext|>"]) {
    const id = tk.tokenToId.get(named);
    expect(id).toBeDefined();
    expect(table.ids).not.toContain(id!);
  }
});
