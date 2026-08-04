export interface TokenByteTableEntry {
  readonly id: number;
  readonly bytes: Uint8Array | null;
  readonly special: boolean;
}

export interface ConstraintTraceStep {
  step: number;
  decode: boolean;
  prefixBefore: string;
  prefixAfter: string;
  survivingCandidates: string[];
  allowedCount: number;
  selectedToken: number;
  selectedBytes: string;
  topBefore: Array<{ token: number; logit: number; bytes: string; allowed: boolean }>;
  topAllowed: Array<{ token: number; logit: number; bytes: string }>;
  complete: boolean;
}

const decoder = new TextDecoder();
const encoder = new TextEncoder();

function startsWithBytes(value: Uint8Array, prefix: Uint8Array): boolean {
  if (prefix.length > value.length) return false;
  for (let i = 0; i < prefix.length; i++) {
    if (value[i] !== prefix[i]) return false;
  }
  return true;
}

function concatBytes(a: Uint8Array, b: Uint8Array): Uint8Array {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function extendsWithBytes(candidate: Uint8Array, prefix: Uint8Array, suffix: Uint8Array): boolean {
  if (prefix.length + suffix.length > candidate.length) return false;
  for (let i = 0; i < prefix.length; i++) {
    if (candidate[i] !== prefix[i]) return false;
  }
  for (let i = 0; i < suffix.length; i++) {
    if (candidate[prefix.length + i] !== suffix[i]) return false;
  }
  return true;
}

function displayBytes(bytes: Uint8Array | null): string {
  if (!bytes) return "<special>";
  let out = "";
  for (const byte of bytes) {
    if (byte >= 0x20 && byte <= 0x7e) out += String.fromCharCode(byte);
    else out += `\\x${byte.toString(16).padStart(2, "0")}`;
  }
  return out;
}

function topK(logits: Float32Array, count: number): Array<{ token: number; logit: number }> {
  const best: Array<{ token: number; logit: number }> = [];
  for (let token = 0; token < logits.length; token++) {
    const logit = logits[token]!;
    let at = best.length;
    while (at > 0) {
      const previous = best[at - 1]!;
      if (previous.logit > logit || (previous.logit === logit && previous.token < token)) break;
      at--;
    }
    if (at < count) {
      best.splice(at, 0, { token, logit });
      if (best.length > count) best.pop();
    }
  }
  return best;
}

/**
 * Slow, deliberately transparent CPU oracle for a finite byte language.
 *
 * A token is allowed iff appending its exact byte payload keeps the generated
 * prefix as a prefix of at least one candidate. This naturally handles BPE
 * tokens that cross JSON syntax / enum boundaries without any token-string
 * special cases. Once a candidate is complete, EOS is the only allowed token.
 */
export class FiniteStringConstraint {
  readonly candidates: readonly string[];
  readonly trace: ConstraintTraceStep[] = [];

  private readonly candidateBytes: readonly Uint8Array[];
  private prefix = new Uint8Array(0);
  private pending: {
    step: number;
    decode: boolean;
    prefixBefore: string;
    allowed: Uint8Array;
    topBefore: Array<{ token: number; logit: number; bytes: string; allowed: boolean }>;
    topAllowed: Array<{ token: number; logit: number; bytes: string }>;
  } | null = null;
  private eosAccepted = false;

  constructor(
    candidates: readonly string[],
    private readonly tokens: readonly TokenByteTableEntry[],
    readonly eosToken: number,
  ) {
    if (candidates.length < 1) throw new Error("FiniteStringConstraint requires at least one candidate");
    this.candidates = [...new Set(candidates)];
    this.candidateBytes = this.candidates.map((value) => encoder.encode(value));
    if (!tokens[eosToken]) throw new Error(`EOS token ${eosToken} is outside token table`);
  }

  get generatedPrefix(): string {
    return decoder.decode(this.prefix);
  }

  get complete(): boolean {
    return this.candidateBytes.some((candidate) => candidate.length === this.prefix.length && startsWithBytes(candidate, this.prefix));
  }

  get done(): boolean {
    return this.eosAccepted;
  }

  survivingCandidates(): string[] {
    return this.candidates.filter((_, index) => startsWithBytes(this.candidateBytes[index]!, this.prefix));
  }

  process(logits: Float32Array, context: { step: number; decode: boolean }): void {
    if (logits.length !== this.tokens.length) {
      throw new Error(`Logit/token table mismatch: ${logits.length} vs ${this.tokens.length}`);
    }
    if (this.pending) throw new Error("Constraint process() called twice without accept()");

    const prefixBefore = this.generatedPrefix;
    const allowed = new Uint8Array(logits.length);
    let allowedCount = 0;

    if (this.complete) {
      allowed[this.eosToken] = 1;
      allowedCount = 1;
    } else {
      for (let id = 0; id < this.tokens.length; id++) {
        const entry = this.tokens[id]!;
        if (entry.special || !entry.bytes || entry.bytes.length === 0) continue;
        if (this.candidateBytes.some((candidate) => extendsWithBytes(candidate, this.prefix, entry.bytes!))) {
          allowed[id] = 1;
          allowedCount++;
        }
      }
    }

    if (allowedCount === 0) {
      throw new Error(`Constraint dead-end after ${JSON.stringify(prefixBefore)}`);
    }

    const before = topK(logits, 10).map(({ token, logit }) => ({
      token,
      logit,
      bytes: token === this.eosToken ? "<eos>" : displayBytes(this.tokens[token]?.bytes ?? null),
      allowed: allowed[token] === 1,
    }));

    for (let id = 0; id < logits.length; id++) {
      if (allowed[id] === 0) logits[id] = Number.NEGATIVE_INFINITY;
    }
    const topAllowed = topK(logits, 10)
      .filter(({ logit }) => Number.isFinite(logit))
      .map(({ token, logit }) => ({
        token,
        logit,
        bytes: token === this.eosToken ? "<eos>" : displayBytes(this.tokens[token]?.bytes ?? null),
      }));

    this.pending = { step: context.step, decode: context.decode, prefixBefore, allowed, topBefore: before, topAllowed };
  }

  accept(tokenId: number, _context: { step: number; decode: boolean }): void {
    const pending = this.pending;
    if (!pending) throw new Error("Constraint accept() called before process()");
    this.pending = null;
    if (pending.allowed[tokenId] !== 1) {
      throw new Error(`Runtime selected masked token ${tokenId}`);
    }

    let selectedBytes = "<eos>";
    if (tokenId === this.eosToken) {
      if (!this.complete) throw new Error("EOS selected before finite constraint completed");
      this.eosAccepted = true;
    } else {
      const bytes = this.tokens[tokenId]?.bytes;
      if (!bytes) throw new Error(`Selected token ${tokenId} has no byte payload`);
      selectedBytes = displayBytes(bytes);
      this.prefix = concatBytes(this.prefix, bytes);
      if (this.survivingCandidates().length === 0) {
        throw new Error(`Selected token ${tokenId} left the finite language`);
      }
    }

    this.trace.push({
      step: pending.step,
      decode: pending.decode,
      prefixBefore: pending.prefixBefore,
      prefixAfter: this.generatedPrefix,
      survivingCandidates: this.survivingCandidates(),
      allowedCount: pending.allowed.reduce((sum, value) => sum + value, 0),
      selectedToken: tokenId,
      selectedBytes,
      topBefore: pending.topBefore,
      topAllowed: pending.topAllowed,
      complete: this.complete,
    });
  }

  shouldStop(): boolean {
    return this.done;
  }
}

/** First vertical slice used by the web lab: exact JSON syntax + one enum field. */
export function createLiteralEnumJsonConstraint(options: {
  property: string;
  variants: readonly string[];
  tokens: readonly TokenByteTableEntry[];
  eosToken: number;
}): FiniteStringConstraint {
  if (options.variants.length < 1) throw new Error("Enum must contain at least one variant");
  const candidates = options.variants.map((variant) => JSON.stringify({ [options.property]: variant }));
  return new FiniteStringConstraint(candidates, options.tokens, options.eosToken);
}
