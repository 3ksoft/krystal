import type { GgufValue } from "../../gguf/src/types.ts";
import type { GgufReader } from "../../gguf/src/reader.ts";

const TOKEN_TYPE_CONTROL = 3;
const TOKEN_TYPE_USER_DEFINED = 4;

function asStringArray(value: GgufValue, key: string): string[] {
  if (!Array.isArray(value) || value.some((v) => typeof v !== "string")) {
    throw new Error(`${key} must be a string array`);
  }
  return value as string[];
}

function asNumberArray(value: GgufValue, key: string): number[] {
  if (!Array.isArray(value)) throw new Error(`${key} must be an array`);
  return value.map((v) => {
    if (typeof v === "number") return v;
    if (typeof v === "bigint") return Number(v);
    throw new Error(`${key} contains a non-number`);
  });
}

function asNumber(value: GgufValue, key: string): number {
  if (typeof value === "number") return value;
  if (typeof value === "bigint") return Number(value);
  throw new Error(`${key} must be numeric`);
}

function byteToUnicode(): string[] {
  const table = new Array<string>(256);
  const used = new Array<boolean>(256).fill(false);
  const ranges: Array<[number, number]> = [[0x21, 0x7e], [0xa1, 0xac], [0xae, 0xff]];
  for (const [lo, hi] of ranges) {
    for (let b = lo; b <= hi; b++) {
      table[b] = String.fromCodePoint(b);
      used[b] = true;
    }
  }
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!used[b]) table[b] = String.fromCodePoint(256 + n++);
  }
  return table;
}

const LETTER_RE = /^\p{L}$/u;
const NUMBER_RE = /^\p{N}$/u;
const WS_RE = /^\s$/u;
const isLetter = (c: string) => LETTER_RE.test(c);
const isNumber = (c: string) => NUMBER_RE.test(c);
const isWhitespace = (c: string) => WS_RE.test(c);

/**
 * LFM2 uses the Llama-3-style byte-level BPE pre-tokenizer. This scanner is a
 * direct, regex-free implementation of its ordered alternatives:
 *
 *   '(?i:[sdmt]|ll|ve|re)
 *   | [^\r\n\p{L}\p{N}]?\p{L}+
 *   | \p{N}{1,3}
 *   | ?[^\s\p{L}\p{N}]+[\r\n]*
 *   | \s*[\r\n]
 *   | \s+(?!\S)
 *   | \s+
 */
export function pretokenizeLfm2(text: string): string[] {
  const c = Array.from(text);
  const pieces: string[] = [];
  let i = 0;

  const contraction = (at: number): number | null => {
    if (c[at] !== "'") return null;
    const c1 = c[at + 1]?.toLowerCase();
    if (!c1) return null;
    if (c1 === "s" || c1 === "d" || c1 === "m" || c1 === "t") return 2;
    const c2 = c[at + 2]?.toLowerCase();
    if ((c1 === "l" && c2 === "l") || (c1 === "v" && c2 === "e") || (c1 === "r" && c2 === "e")) return 3;
    return null;
  };

  const letters = (at: number): number | null => {
    let j = at;
    const cur = c[at]!;
    const prefixable = cur !== "\r" && cur !== "\n" && !isLetter(cur) && !isNumber(cur);
    if (prefixable && c[at + 1] !== undefined && isLetter(c[at + 1]!)) j++;
    const start = j;
    while (c[j] !== undefined && isLetter(c[j]!)) j++;
    return j > start ? j - at : null;
  };

  const numbers = (at: number): number | null => {
    let j = at;
    while (j < at + 3 && c[j] !== undefined && isNumber(c[j]!)) j++;
    return j > at ? j - at : null;
  };

  const punct = (at: number): number | null => {
    const isPunct = (ch: string) => !isWhitespace(ch) && !isLetter(ch) && !isNumber(ch);
    let j = at;
    if (c[at] === " " && c[at + 1] !== undefined && isPunct(c[at + 1]!)) j++;
    const start = j;
    while (c[j] !== undefined && isPunct(c[j]!)) j++;
    if (j === start) return null;
    while (c[j] === "\r" || c[j] === "\n") j++;
    return j - at;
  };

  const newlineRun = (at: number): number | null => {
    let j = at;
    while (c[j] !== undefined && isWhitespace(c[j]!)) j++;
    let lastNl = -1;
    for (let p = at; p < j; p++) if (c[p] === "\r" || c[p] === "\n") lastNl = p;
    return lastNl >= 0 ? lastNl + 1 - at : null;
  };

  const trailingWs = (at: number): number | null => {
    let j = at;
    while (c[j] !== undefined && isWhitespace(c[j]!)) j++;
    if (j === at) return null;
    if (j >= c.length) return j - at;
    return j - at > 1 ? j - 1 - at : null;
  };

  const ws = (at: number): number | null => {
    let j = at;
    while (c[j] !== undefined && isWhitespace(c[j]!)) j++;
    return j > at ? j - at : null;
  };

  while (i < c.length) {
    const length = contraction(i) ?? letters(i) ?? numbers(i) ?? punct(i) ?? newlineRun(i) ?? trailingWs(i) ?? ws(i) ?? 1;
    pieces.push(c.slice(i, i + length).join(""));
    i += length;
  }
  return pieces;
}

export interface TokenizeOptions {
  addBos?: boolean;
  addEos?: boolean;
  parseSpecial?: boolean;
}

/** Pure TypeScript byte-level BPE tokenizer built entirely from GGUF metadata. */
export class Lfm2Tokenizer {
  readonly idToToken: string[];
  readonly tokenToId: Map<string, number>;
  readonly bos: number;
  readonly eos: number;
  readonly addBosByDefault: boolean;
  readonly addEosByDefault: boolean;

  private readonly mergeRank = new Map<string, number>();
  private readonly specials: Array<{ literal: string; id: number }>;
  private readonly specialIds = new Set<number>();
  private readonly byteEncoder = byteToUnicode();
  private readonly byteDecoder = new Map<string, number>();
  private readonly textEncoder = new TextEncoder();

  constructor(reader: GgufReader) {
    const model = reader.metadata<string>("tokenizer.ggml.model");
    if (model !== "gpt2") throw new Error(`Expected GGUF gpt2/BPE tokenizer, got '${model}'`);

    const pre = reader.metadata<string>("tokenizer.ggml.pre");
    if (pre !== "lfm2") throw new Error(`Expected LFM2 pre-tokenizer, got '${pre}'`);

    this.idToToken = asStringArray(reader.metadata("tokenizer.ggml.tokens"), "tokenizer.ggml.tokens");
    this.tokenToId = new Map(this.idToToken.map((token, id) => [token, id]));

    const merges = asStringArray(reader.metadata("tokenizer.ggml.merges"), "tokenizer.ggml.merges");
    for (let rank = 0; rank < merges.length; rank++) {
      const merge = merges[rank]!;
      const separator = merge.indexOf(" ");
      if (separator <= 0) continue;
      this.mergeRank.set(`${merge.slice(0, separator)}\0${merge.slice(separator + 1)}`, rank);
    }

    const tokenTypes = asNumberArray(reader.metadata("tokenizer.ggml.token_type"), "tokenizer.ggml.token_type");
    this.specials = tokenTypes
      .map((type, id) => ({ type, id, literal: this.idToToken[id] }))
      .filter((x): x is { type: number; id: number; literal: string } =>
        x.literal !== undefined && (x.type === TOKEN_TYPE_CONTROL || x.type === TOKEN_TYPE_USER_DEFINED)
      )
      .map(({ id, literal }) => ({ id, literal }))
      .sort((a, b) => b.literal.length - a.literal.length);
    for (const special of this.specials) this.specialIds.add(special.id);

    this.bos = asNumber(reader.metadata("tokenizer.ggml.bos_token_id"), "tokenizer.ggml.bos_token_id");
    this.eos = asNumber(reader.metadata("tokenizer.ggml.eos_token_id"), "tokenizer.ggml.eos_token_id");
    this.addBosByDefault = Boolean(reader.metadata("tokenizer.ggml.add_bos_token"));
    this.addEosByDefault = Boolean(reader.metadata("tokenizer.ggml.add_eos_token"));

    this.byteEncoder.forEach((char, byte) => this.byteDecoder.set(char, byte));
  }

  encode(text: string, options: TokenizeOptions = {}): number[] {
    const addBos = options.addBos ?? this.addBosByDefault;
    const addEos = options.addEos ?? this.addEosByDefault;
    const parseSpecial = options.parseSpecial ?? true;
    const ids: number[] = [];
    if (addBos) ids.push(this.bos);

    if (!parseSpecial) {
      this.bpeChunk(text, ids);
    } else {
      let rest = text;
      while (rest.length > 0) {
        let hit: { pos: number; length: number; id: number } | null = null;
        for (const special of this.specials) {
          const pos = rest.indexOf(special.literal);
          if (pos < 0) continue;
          if (!hit || pos < hit.pos || (pos === hit.pos && special.literal.length > hit.length)) {
            hit = { pos, length: special.literal.length, id: special.id };
          }
        }
        if (!hit) {
          this.bpeChunk(rest, ids);
          break;
        }
        this.bpeChunk(rest.slice(0, hit.pos), ids);
        ids.push(hit.id);
        rest = rest.slice(hit.pos + hit.length);
      }
    }

    if (addEos) ids.push(this.eos);
    return ids;
  }

  isSpecialToken(id: number): boolean {
    return this.specialIds.has(id);
  }

  /**
   * Exact byte payload represented by one ordinary BPE token. Control/user
   * defined special tokens are not byte-level vocabulary entries and return
   * null so constrained decoders can handle them explicitly (for example EOS).
   */
  tokenBytes(id: number): Uint8Array | null {
    const token = this.idToToken[id];
    if (token === undefined || this.specialIds.has(id)) return null;
    const bytes: number[] = [];
    for (const char of Array.from(token)) {
      const byte = this.byteDecoder.get(char);
      if (byte === undefined) return null;
      bytes.push(byte);
    }
    return new Uint8Array(bytes);
  }

  decode(ids: readonly number[], options: { skipSpecial?: boolean } = {}): string {
    const skipSpecial = options.skipSpecial ?? true;
    const bytes: number[] = [];
    const decoder = new TextDecoder();
    let result = "";

    const flush = () => {
      if (bytes.length === 0) return;
      result += decoder.decode(new Uint8Array(bytes));
      bytes.length = 0;
    };

    for (const id of ids) {
      const token = this.idToToken[id];
      if (token === undefined) continue;
      if (this.specialIds.has(id)) {
        flush();
        if (!skipSpecial) result += token;
        continue;
      }
      for (const char of Array.from(token)) {
        const byte = this.byteDecoder.get(char);
        if (byte !== undefined) bytes.push(byte);
      }
    }
    flush();
    return result;
  }

  /** One complete ChatML message. Useful for resident semantic blocks. */
  formatMessage(role: "system" | "user", content: string): string {
    return `<|im_start|>${role}\n${content}<|im_end|>\n`;
  }

  /** Minimal text-only ChatML used by LFM2.5 Instruct. */
  formatUserPrompt(content: string, system?: string): string {
    let prompt = "";
    if (system) prompt += this.formatMessage("system", system);
    prompt += this.formatMessage("user", content);
    prompt += `<|im_start|>assistant\n`;
    return prompt;
  }

  encodeUserPrompt(content: string, system?: string): number[] {
    return this.encode(this.formatUserPrompt(content, system), { addBos: true, addEos: false, parseSpecial: true });
  }

  private bpeChunk(text: string, out: number[]): void {
    for (const piece of pretokenizeLfm2(text)) {
      const encodedBytes = this.textEncoder.encode(piece);
      let mapped = "";
      for (const byte of encodedBytes) mapped += this.byteEncoder[byte]!;
      for (const symbol of this.bpe(mapped)) {
        const id = this.tokenToId.get(symbol);
        if (id !== undefined) {
          out.push(id);
          continue;
        }
        // The vocab contains every atomic byte symbol; this is only a safety net.
        for (const char of Array.from(symbol)) {
          const fallback = this.tokenToId.get(char);
          if (fallback === undefined) throw new Error(`Tokenizer cannot encode atomic symbol ${JSON.stringify(char)}`);
          out.push(fallback);
        }
      }
    }
  }

  private bpe(word: string): string[] {
    const symbols = Array.from(word);
    if (symbols.length < 2) return symbols;
    while (true) {
      let bestIndex = -1;
      let bestRank = Number.POSITIVE_INFINITY;
      for (let i = 0; i + 1 < symbols.length; i++) {
        const rank = this.mergeRank.get(`${symbols[i]}\0${symbols[i + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestIndex < 0) break;
      symbols.splice(bestIndex, 2, symbols[bestIndex]! + symbols[bestIndex + 1]!);
    }
    return symbols;
  }
}
