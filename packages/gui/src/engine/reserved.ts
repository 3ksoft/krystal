/**
 * Friendly addressing for the model's unused reserved vocabulary.
 *
 * LFM2.5 ships 377 `<|reserved_N|>` entries. Their ids are almost, but not
 * quite, contiguous — ids 128..132 in the middle of the range are named
 * audio/text tokens — and the numbering starts at `<|reserved_4|>`, not 1.
 * Addressing them by that internal numbering means memorising both quirks, so
 * `[-token-K-]` is a dense 1-based index into whatever reserved tokens the
 * loaded model actually has.
 *
 * The tokenizer already resolves special literals when parseSpecial is on, so
 * expansion only has to produce the literal text; no id surgery is involved.
 */

const ALIAS = /\[-token-(\d+)-\]/g;
const RESERVED_LITERAL = /^<\|reserved_\d+\|>$/;

export interface ReservedTable {
  /** Literal at 1-based alias index: literals[0] is `[-token-1-]`. */
  literals: string[];
  ids: number[];
}

/**
 * Build the alias table for a vocabulary.
 *
 * Matching the `<|reserved_N|>` spelling is not sufficient. In LFM2.5,
 * `<|reserved_4|>` and `<|reserved_5|>` carry token_type NORMAL rather than
 * CONTROL, so the tokenizer does not resolve them and they tokenize as eleven
 * characters of literal text. Only entries the tokenizer itself treats as
 * special can be addressed, so the predicate decides membership.
 */
export function buildReservedTable(
  vocab: readonly string[],
  isSpecial: (id: number) => boolean,
): ReservedTable {
  const literals: string[] = [];
  const ids: number[] = [];
  for (let id = 0; id < vocab.length; id++) {
    const literal = vocab[id]!;
    if (RESERVED_LITERAL.test(literal) && isSpecial(id)) {
      literals.push(literal);
      ids.push(id);
    }
  }
  return { literals, ids };
}

export const EMPTY_RESERVED: ReservedTable = { literals: [], ids: [] };

export interface Expansion {
  text: string;
  /** Alias indices that the loaded model has no reserved token for. */
  unknown: number[];
  /** 1-based alias indices actually substituted, in order of appearance. */
  used: number[];
}

/**
 * Replace `[-token-K-]` with its reserved literal.
 *
 * Out-of-range aliases are left verbatim rather than dropped: silently removing
 * them would change the prompt in a way that is invisible in the token view.
 */
export function expandReserved(text: string, table: ReservedTable): Expansion {
  const unknown: number[] = [];
  const used: number[] = [];
  const out = text.replace(ALIAS, (whole, digits: string) => {
    const index = Number(digits);
    const literal = index >= 1 ? table.literals[index - 1] : undefined;
    if (literal === undefined) {
      if (!unknown.includes(index)) unknown.push(index);
      return whole;
    }
    used.push(index);
    return literal;
  });
  return { text: out, unknown, used };
}

/** Inverse of expandReserved, for showing stored text back in alias form. */
export function collapseReserved(text: string, table: ReservedTable): string {
  if (!table.literals.length) return text;
  return text.replace(/<\|reserved_\d+\|>/g, (literal) => {
    const at = table.literals.indexOf(literal);
    return at < 0 ? literal : `[-token-${at + 1}-]`;
  });
}

export function aliasFor(table: ReservedTable, id: number): string | null {
  const at = table.ids.indexOf(id);
  return at < 0 ? null : `[-token-${at + 1}-]`;
}
