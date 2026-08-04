import type { LayoutPlan } from "@schema-pop/schema";

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
  stateBefore: string;
  stateAfter: string;
  survivingCandidates?: string[];
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

interface PendingMask {
  step: number;
  decode: boolean;
  prefixBefore: string;
  stateBefore: string;
  allowed: Uint8Array;
  topBefore: Array<{ token: number; logit: number; bytes: string; allowed: boolean }>;
  topAllowed: Array<{ token: number; logit: number; bytes: string }>;
}

function maskAndTrace(
  logits: Float32Array,
  tokens: readonly TokenByteTableEntry[],
  allowed: Uint8Array,
  eosToken?: number,
): Pick<PendingMask, "topBefore" | "topAllowed"> {
  const before = topK(logits, 10).map(({ token, logit }) => ({
    token,
    logit,
    bytes: token === eosToken ? "<eos>" : displayBytes(tokens[token]?.bytes ?? null),
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
      bytes: token === eosToken ? "<eos>" : displayBytes(tokens[token]?.bytes ?? null),
    }));

  return { topBefore: before, topAllowed };
}

/**
 * Slow, deliberately transparent CPU oracle for a finite byte language.
 * Kept as a tiny reference implementation beside the LayoutPlan machine.
 */
export class FiniteStringConstraint {
  readonly candidates: readonly string[];
  readonly trace: ConstraintTraceStep[] = [];

  private readonly candidateBytes: readonly Uint8Array[];
  private prefix = new Uint8Array(0);
  private pending: PendingMask | null = null;
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

    if (allowedCount === 0) throw new Error(`Constraint dead-end after ${JSON.stringify(prefixBefore)}`);
    const traced = maskAndTrace(logits, this.tokens, allowed, this.eosToken);
    this.pending = {
      step: context.step,
      decode: context.decode,
      prefixBefore,
      stateBefore: `finite prefix ${JSON.stringify(prefixBefore)}`,
      allowed,
      ...traced,
    };
  }

  accept(tokenId: number, _context: { step: number; decode: boolean }): void {
    const pending = this.pending;
    if (!pending) throw new Error("Constraint accept() called before process()");
    this.pending = null;
    if (pending.allowed[tokenId] !== 1) throw new Error(`Runtime selected masked token ${tokenId}`);

    let selectedBytes = "<eos>";
    if (tokenId === this.eosToken) {
      if (!this.complete) throw new Error("EOS selected before finite constraint completed");
      this.eosAccepted = true;
    } else {
      const bytes = this.tokens[tokenId]?.bytes;
      if (!bytes) throw new Error(`Selected token ${tokenId} has no byte payload`);
      selectedBytes = displayBytes(bytes);
      this.prefix = concatBytes(this.prefix, bytes);
      if (this.survivingCandidates().length === 0) throw new Error(`Selected token ${tokenId} left the finite language`);
    }

    this.trace.push({
      step: pending.step,
      decode: pending.decode,
      prefixBefore: pending.prefixBefore,
      prefixAfter: this.generatedPrefix,
      stateBefore: pending.stateBefore,
      stateAfter: `finite prefix ${JSON.stringify(this.generatedPrefix)}`,
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

/** First vertical slice retained for direct finite-language tests. */
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

// ---------------------------------------------------------------------------
// LayoutPlan -> canonical compact JSON byte program
// ---------------------------------------------------------------------------

type LiteralSegment = {
  kind: "literal";
  bytes: Uint8Array;
  text: string;
  label: string;
};

type ChoiceSegment = {
  kind: "choice";
  alternatives: readonly Uint8Array[];
  texts: readonly string[];
  label: string;
};

type StringSegment = {
  kind: "string";
  minLength: number;
  maxLength: number;
  label: string;
};

type NumberSegment = {
  kind: "number";
  integer: boolean;
  min?: number;
  max?: number;
  step?: number;
  maxChars: number;
  label: string;
};

type JsonSegment = LiteralSegment | ChoiceSegment | StringSegment | NumberSegment;

export interface LayoutConstraintProgramSummary {
  rootType: string;
  segments: number;
  fields: number;
  optionalIncluded: number;
  optionalSkipped: number;
  enums: number;
  strings: number;
  numbers: number;
  booleans: number;
  arrays: number;
}

export interface LayoutConstraintProgram {
  readonly segments: readonly JsonSegment[];
  readonly summary: Readonly<LayoutConstraintProgramSummary>;
}

export class UnsupportedLayoutPlanError extends Error {
  constructor(readonly path: string, reason: string) {
    super(`[layout-constraint] ${path}: ${reason}`);
    this.name = "UnsupportedLayoutPlanError";
  }
}

function literalSegment(text: string, label: string): LiteralSegment {
  return { kind: "literal", bytes: encoder.encode(text), text, label };
}

function choiceSegment(texts: readonly string[], label: string): ChoiceSegment {
  const unique = [...new Set(texts)];
  if (unique.length === 0) throw new UnsupportedLayoutPlanError(label, "empty choice");
  return { kind: "choice", alternatives: unique.map((value) => encoder.encode(value)), texts: unique, label };
}

/**
 * Compile the analyzed LayoutPlan directly. The compiler deliberately emits a
 * canonical, whitespace-free JSON shape with struct fields in analyzed order.
 * Optional fields are included when their inner type is supported; an
 * unsupported optional field is omitted because omission is schema-valid.
 * Required unsupported fields reject the record instead of guessing.
 */
export function compileLayoutPlanProgram(
  plan: LayoutPlan,
  options: { rootType?: string } = {},
): LayoutConstraintProgram {
  const rootType = options.rootType ?? "value";
  const types = new Map<string, any>((plan.types as readonly any[]).map((type) => [type.name, type]));
  const summary: LayoutConstraintProgramSummary = {
    rootType,
    segments: 0,
    fields: 0,
    optionalIncluded: 0,
    optionalSkipped: 0,
    enums: 0,
    strings: 0,
    numbers: 0,
    booleans: 0,
    arrays: 0,
  };

  const compileFieldType = (
    field: any,
    path: string,
    out: JsonSegment[],
    visited: Set<string>,
  ): void => {
    switch (field?.kind) {
      case "primitive": {
        const primitiveType = String(field.type ?? "");
        const name = String(field.name ?? "").toLowerCase();
        if (primitiveType === "boolean" || name === "bool" || name === "boolean") {
          out.push(choiceSegment(["true", "false"], `bool ${path}`));
          summary.booleans++;
          return;
        }
        if (primitiveType === "number" || /^(u|i|f)\d+$/.test(name) || ["float", "double", "usize"].includes(name)) {
          const integer = !(field.isFloat || name === "f32" || name === "f64" || name === "float" || name === "double");
          out.push({
            kind: "number",
            integer,
            min: typeof field.min === "number" ? field.min : undefined,
            max: typeof field.max === "number" ? field.max : undefined,
            step: typeof field.step === "number" && field.step > 0 ? field.step : undefined,
            maxChars: 32,
            label: `number ${path}`,
          });
          summary.numbers++;
          return;
        }
        throw new UnsupportedLayoutPlanError(path, `primitive '${field.name ?? field.type ?? "?"}'`);
      }

      case "string": {
        const exactLength = typeof field.exactLength === "number" ? field.exactLength : undefined;
        const maxLength = exactLength ?? (typeof field.maxLength === "number" ? field.maxLength : undefined);
        if (maxLength === undefined || !Number.isFinite(maxLength)) {
          throw new UnsupportedLayoutPlanError(path, "string has no finite maxLength/exactLength in LayoutPlan");
        }
        out.push({
          kind: "string",
          minLength: exactLength ?? (typeof field.minLength === "number" ? field.minLength : 0),
          maxLength,
          label: `string ${path}`,
        });
        summary.strings++;
        return;
      }

      case "reference": {
        const target = types.get(field.name);
        if (!target) throw new UnsupportedLayoutPlanError(path, `missing referenced type '${field.name}'`);
        if (visited.has(field.name)) throw new UnsupportedLayoutPlanError(path, `cyclic reference '${field.name}'`);
        const next = new Set(visited);
        next.add(field.name);
        if (target.kind === "enum") {
          out.push(choiceSegment(target.variants.map((variant: any) => JSON.stringify(String(variant.name))), `enum ${path}`));
          summary.enums++;
          return;
        }
        if (target.kind === "struct") {
          compileStruct(target, path, out, next);
          return;
        }
        if (target.kind === "alias") {
          compileFieldType(target.type, path, out, next);
          return;
        }
        throw new UnsupportedLayoutPlanError(path, `referenced type '${field.name}' has kind '${target.kind}'`);
      }

      case "optional":
        compileFieldType(field.inner, path, out, visited);
        return;

      case "array": {
        const exactLength = typeof field.exactLength === "number" ? field.exactLength : undefined;
        if (exactLength === undefined) {
          throw new UnsupportedLayoutPlanError(path, "array has no exactLength in LayoutPlan");
        }
        out.push(literalSegment("[", `array-open ${path}`));
        for (let i = 0; i < exactLength; i++) {
          if (i > 0) out.push(literalSegment(",", `array-comma ${path}[${i}]`));
          compileFieldType(field.item, `${path}[${i}]`, out, visited);
        }
        out.push(literalSegment("]", `array-close ${path}`));
        summary.arrays++;
        return;
      }

      case "inlineStruct":
        compileStruct(field, path, out, visited);
        return;

      case "unit":
        throw new UnsupportedLayoutPlanError(path, "unit carries no JSON value semantics");

      default:
        throw new UnsupportedLayoutPlanError(path, `field kind '${field?.kind ?? "?"}'`);
    }
  };

  const compileStruct = (
    struct: any,
    path: string,
    out: JsonSegment[],
    visited: Set<string>,
  ): void => {
    out.push(literalSegment("{", `object-open ${path}`));
    let emitted = 0;
    for (const fieldPlan of struct.fields ?? []) {
      const fieldPath = `${path}.${fieldPlan.name}`;
      const fieldType = fieldPlan.type;
      let valueSegments: JsonSegment[] = [];
      if (fieldType?.kind === "optional") {
        try {
          compileFieldType(fieldType.inner, fieldPath, valueSegments, visited);
          summary.optionalIncluded++;
        } catch (error) {
          if (!(error instanceof UnsupportedLayoutPlanError)) throw error;
          summary.optionalSkipped++;
          continue;
        }
      } else {
        compileFieldType(fieldType, fieldPath, valueSegments, visited);
      }

      const propertyPrefix = `${emitted > 0 ? "," : ""}${JSON.stringify(String(fieldPlan.name))}:`;
      out.push(literalSegment(propertyPrefix, `field ${fieldPath}`));
      out.push(...valueSegments);
      emitted++;
      summary.fields++;
    }
    out.push(literalSegment("}", `object-close ${path}`));
  };

  const root = types.get(rootType);
  if (!root) throw new UnsupportedLayoutPlanError(rootType, `root type not found; available: ${[...types.keys()].join(", ")}`);

  const segments: JsonSegment[] = [];
  const visited = new Set<string>([rootType]);
  if (root.kind === "struct") compileStruct(root, rootType, segments, visited);
  else if (root.kind === "enum") {
    segments.push(choiceSegment(root.variants.map((variant: any) => JSON.stringify(String(variant.name))), `enum ${rootType}`));
    summary.enums++;
  } else if (root.kind === "alias") compileFieldType(root.type, rootType, segments, visited);
  else throw new UnsupportedLayoutPlanError(rootType, `top-level kind '${root.kind}'`);

  summary.segments = segments.length;
  return { segments, summary };
}

type LiteralLocal = { kind: "literal"; offset: number };
type ChoiceLocal = { kind: "choice"; prefix: number[] };
type StringLocal = {
  kind: "string";
  phase: "open" | "body" | "escape" | "unicode";
  length: number;
  unicodeRemaining: number;
};
type NumberLocal = { kind: "number"; text: string };
type SegmentLocal = LiteralLocal | ChoiceLocal | StringLocal | NumberLocal;

type ProgramState = {
  segment: number;
  local: SegmentLocal | null;
};

function cloneProgramState(state: ProgramState): ProgramState {
  if (!state.local) return { segment: state.segment, local: null };
  if (state.local.kind === "choice") return { segment: state.segment, local: { kind: "choice", prefix: [...state.local.prefix] } };
  return { segment: state.segment, local: { ...state.local } } as ProgramState;
}

function ensureLocal(segment: JsonSegment, state: ProgramState): SegmentLocal {
  if (state.local) return state.local;
  switch (segment.kind) {
    case "literal": state.local = { kind: "literal", offset: 0 }; break;
    case "choice": state.local = { kind: "choice", prefix: [] }; break;
    case "string": state.local = { kind: "string", phase: "open", length: 0, unicodeRemaining: 0 }; break;
    case "number": state.local = { kind: "number", text: "" }; break;
  }
  return state.local;
}

function advanceSegment(state: ProgramState): void {
  state.segment++;
  state.local = null;
}

function isHex(byte: number): boolean {
  return (byte >= 0x30 && byte <= 0x39) || (byte >= 0x41 && byte <= 0x46) || (byte >= 0x61 && byte <= 0x66);
}

function isJsonNumberPrefix(text: string, integer: boolean): boolean {
  if (text.length === 0) return true;
  let i = 0;
  if (text[i] === "-") {
    i++;
    if (i === text.length) return true;
  }
  if (text[i] === "0") {
    i++;
  } else if (text[i] && text[i]! >= "1" && text[i]! <= "9") {
    i++;
    while (i < text.length && text[i]! >= "0" && text[i]! <= "9") i++;
  } else {
    return false;
  }
  if (i === text.length) return true;
  if (integer) return false;
  if (text[i] === ".") {
    i++;
    if (i === text.length) return true;
    if (text[i]! < "0" || text[i]! > "9") return false;
    while (i < text.length && text[i]! >= "0" && text[i]! <= "9") i++;
  }
  if (i === text.length) return true;
  if (text[i] === "e" || text[i] === "E") {
    i++;
    if (i === text.length) return true;
    if (text[i] === "+" || text[i] === "-") {
      i++;
      if (i === text.length) return true;
    }
    if (text[i]! < "0" || text[i]! > "9") return false;
    while (i < text.length && text[i]! >= "0" && text[i]! <= "9") i++;
  }
  return i === text.length;
}

function isJsonNumberComplete(segment: NumberSegment, text: string): boolean {
  const pattern = segment.integer
    ? /^-?(?:0|[1-9]\d*)$/
    : /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/;
  if (!pattern.test(text)) return false;
  const value = Number(text);
  if (!Number.isFinite(value)) return false;
  if (segment.min !== undefined && value < segment.min) return false;
  if (segment.max !== undefined && value > segment.max) return false;
  if (segment.step !== undefined) {
    const base = segment.min ?? 0;
    const scaled = (value - base) / segment.step;
    if (Math.abs(scaled - Math.round(scaled)) > 1e-9 * Math.max(1, Math.abs(scaled))) return false;
  }
  return true;
}

function choiceMatchesPrefix(segment: ChoiceSegment, prefix: readonly number[]): boolean {
  return segment.alternatives.some((candidate) => {
    if (prefix.length > candidate.length) return false;
    for (let i = 0; i < prefix.length; i++) if (candidate[i] !== prefix[i]) return false;
    return true;
  });
}

function choiceComplete(segment: ChoiceSegment, prefix: readonly number[]): boolean {
  return segment.alternatives.some((candidate) => {
    if (prefix.length !== candidate.length) return false;
    for (let i = 0; i < prefix.length; i++) if (candidate[i] !== prefix[i]) return false;
    return true;
  });
}

function feedByte(program: LayoutConstraintProgram, state: ProgramState, byte: number): boolean {
  // Number completion does not consume the delimiter. When it completes, retry
  // the same byte against the next segment. No other segment needs pushback.
  for (let retry = 0; retry < 2; retry++) {
    const segment = program.segments[state.segment];
    if (!segment) return false;
    const local = ensureLocal(segment, state);

    if (segment.kind === "literal") {
      const literal = local as LiteralLocal;
      if (segment.bytes[literal.offset] !== byte) return false;
      literal.offset++;
      if (literal.offset === segment.bytes.length) advanceSegment(state);
      return true;
    }

    if (segment.kind === "choice") {
      const choice = local as ChoiceLocal;
      choice.prefix.push(byte);
      if (!choiceMatchesPrefix(segment, choice.prefix)) return false;
      if (choiceComplete(segment, choice.prefix)) advanceSegment(state);
      return true;
    }

    if (segment.kind === "string") {
      const string = local as StringLocal;
      if (string.phase === "open") {
        if (byte !== 0x22) return false;
        string.phase = "body";
        return true;
      }
      if (string.phase === "escape") {
        if (byte === 0x75) {
          string.phase = "unicode";
          string.unicodeRemaining = 4;
          return true;
        }
        if (![0x22, 0x5c, 0x2f, 0x62, 0x66, 0x6e, 0x72, 0x74].includes(byte)) return false;
        if (string.length >= segment.maxLength) return false;
        string.length++;
        string.phase = "body";
        return true;
      }
      if (string.phase === "unicode") {
        if (!isHex(byte)) return false;
        string.unicodeRemaining--;
        if (string.unicodeRemaining === 0) {
          if (string.length >= segment.maxLength) return false;
          string.length++;
          string.phase = "body";
        }
        return true;
      }
      if (byte === 0x22) {
        if (string.length < segment.minLength) return false;
        advanceSegment(state);
        return true;
      }
      if (byte === 0x5c) {
        if (string.length >= segment.maxLength) return false;
        string.phase = "escape";
        return true;
      }
      if (byte < 0x20 || string.length >= segment.maxLength) return false;
      string.length++;
      return true;
    }

    const number = local as NumberLocal;
    const char = byte <= 0x7f ? String.fromCharCode(byte) : "";
    if (char && number.text.length < segment.maxChars) {
      const next = number.text + char;
      if (isJsonNumberPrefix(next, segment.integer)) {
        if (next.length === segment.maxChars && !isJsonNumberComplete(segment, next)) return false;
        number.text = next;
        return true;
      }
    }
    if (!isJsonNumberComplete(segment, number.text)) return false;
    advanceSegment(state);
  }
  return false;
}

function feedBytes(program: LayoutConstraintProgram, state: ProgramState, bytes: Uint8Array): boolean {
  for (const byte of bytes) if (!feedByte(program, state, byte)) return false;
  return true;
}

function describeProgramState(program: LayoutConstraintProgram, state: ProgramState): string {
  const segment = program.segments[state.segment];
  if (!segment) return "<complete>";
  const local = ensureLocal(segment, state);
  switch (segment.kind) {
    case "literal":
      return `${segment.label} · literal ${JSON.stringify(segment.text)} @ ${(local as LiteralLocal).offset}/${segment.bytes.length}`;
    case "choice": {
      const prefix = new Uint8Array((local as ChoiceLocal).prefix);
      const alive = segment.texts.filter((_, index) => startsWithBytes(segment.alternatives[index]!, prefix));
      return `${segment.label} · prefix ${JSON.stringify(displayBytes(prefix))} · ${alive.length} choices`;
    }
    case "string": {
      const value = local as StringLocal;
      return `${segment.label} · ${value.phase} · len ${value.length}/${segment.maxLength}`;
    }
    case "number": {
      const value = local as NumberLocal;
      const range = `${segment.min ?? "-∞"}..${segment.max ?? "+∞"}${segment.step ? ` step ${segment.step}` : ""}`;
      return `${segment.label} · ${JSON.stringify(value.text)} · ${range}`;
    }
  }
}

/**
 * Token-level CPU oracle driven directly by LayoutPlan. It scans every vocab
 * token against a clone of the byte-state machine, masks illegal logits, then
 * advances the real machine with the selected token.
 */
export class LayoutPlanJsonConstraint {
  readonly trace: ConstraintTraceStep[] = [];
  readonly program: LayoutConstraintProgram;

  private state: ProgramState = { segment: 0, local: null };
  private prefix = new Uint8Array(0);
  private pending: PendingMask | null = null;
  private eosAccepted = false;

  constructor(
    plan: LayoutPlan,
    private readonly tokens: readonly TokenByteTableEntry[],
    readonly eosToken: number,
    options: { rootType?: string } = {},
  ) {
    this.program = compileLayoutPlanProgram(plan, options);
    if (!tokens[eosToken]) throw new Error(`EOS token ${eosToken} is outside token table`);
  }

  get generatedPrefix(): string {
    return decoder.decode(this.prefix);
  }

  get complete(): boolean {
    return this.state.segment >= this.program.segments.length;
  }

  get done(): boolean {
    return this.eosAccepted;
  }

  process(logits: Float32Array, context: { step: number; decode: boolean }): void {
    if (logits.length !== this.tokens.length) throw new Error(`Logit/token table mismatch: ${logits.length} vs ${this.tokens.length}`);
    if (this.pending) throw new Error("Constraint process() called twice without accept()");

    const allowed = new Uint8Array(logits.length);
    let allowedCount = 0;
    if (this.complete) {
      allowed[this.eosToken] = 1;
      allowedCount = 1;
    } else {
      for (let id = 0; id < this.tokens.length; id++) {
        const entry = this.tokens[id]!;
        if (entry.special || !entry.bytes || entry.bytes.length === 0) continue;
        const candidateState = cloneProgramState(this.state);
        if (feedBytes(this.program, candidateState, entry.bytes)) {
          allowed[id] = 1;
          allowedCount++;
        }
      }
    }

    const prefixBefore = this.generatedPrefix;
    const stateBefore = describeProgramState(this.program, cloneProgramState(this.state));
    if (allowedCount === 0) throw new Error(`LayoutPlan constraint dead-end after ${JSON.stringify(prefixBefore)} · ${stateBefore}`);
    const traced = maskAndTrace(logits, this.tokens, allowed, this.eosToken);
    this.pending = { step: context.step, decode: context.decode, prefixBefore, stateBefore, allowed, ...traced };
  }

  accept(tokenId: number, _context: { step: number; decode: boolean }): void {
    const pending = this.pending;
    if (!pending) throw new Error("Constraint accept() called before process()");
    this.pending = null;
    if (pending.allowed[tokenId] !== 1) throw new Error(`Runtime selected masked token ${tokenId}`);

    let selectedBytes = "<eos>";
    if (tokenId === this.eosToken) {
      if (!this.complete) throw new Error("EOS selected before LayoutPlan constraint completed");
      this.eosAccepted = true;
    } else {
      const bytes = this.tokens[tokenId]?.bytes;
      if (!bytes) throw new Error(`Selected token ${tokenId} has no byte payload`);
      selectedBytes = displayBytes(bytes);
      if (!feedBytes(this.program, this.state, bytes)) throw new Error(`Selected token ${tokenId} failed replay against LayoutPlan machine`);
      this.prefix = concatBytes(this.prefix, bytes);
    }

    this.trace.push({
      step: pending.step,
      decode: pending.decode,
      prefixBefore: pending.prefixBefore,
      prefixAfter: this.generatedPrefix,
      stateBefore: pending.stateBefore,
      stateAfter: describeProgramState(this.program, cloneProgramState(this.state)),
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

export function createLayoutPlanJsonConstraint(options: {
  plan: LayoutPlan;
  tokens: readonly TokenByteTableEntry[];
  eosToken: number;
  rootType?: string;
}): LayoutPlanJsonConstraint {
  return new LayoutPlanJsonConstraint(options.plan, options.tokens, options.eosToken, { rootType: options.rootType });
}
