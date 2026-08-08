import { isJsonNumberComplete as isJsonNumberCompleteExact } from "./json-number.ts";
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

// ArrayBuffer-backed explicitly: freshly allocated here, and the bare
// `Uint8Array` alias widens to ArrayBufferLike, which will not assign back to
// the ArrayBuffer-backed fields this feeds.
function concatBytes(a: Uint8Array, b: Uint8Array): Uint8Array<ArrayBuffer> {
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
// LayoutPlan -> canonical compact JSON constraint graph
// ---------------------------------------------------------------------------

export type LiteralNode = {
  kind: "literal";
  bytes: Uint8Array;
  text: string;
  label: string;
  next: number;
};

export type ChoiceNode = {
  kind: "choice";
  alternatives: readonly Uint8Array[];
  texts: readonly string[];
  label: string;
  next: number;
};

export type StringNode = {
  kind: "string";
  minLength: number;
  maxLength: number;
  label: string;
  next: number;
};

export type NumberNode = {
  kind: "number";
  integer: boolean;
  min?: number;
  max?: number;
  step?: number;
  maxChars: number;
  label: string;
  next: number;
};

export type SplitNode = {
  kind: "split";
  targets: readonly number[];
  label: string;
};

/** Epsilon control-flow barrier used to keep deterministic GPU linking local. */
export type JumpNode = {
  kind: "jump";
  next: number;
  label: string;
};

export type AcceptNode = {
  kind: "accept";
  label: string;
};

export type JsonNode = LiteralNode | ChoiceNode | StringNode | NumberNode | SplitNode | JumpNode | AcceptNode;

export interface LayoutConstraintProgramSummary {
  rootType: string;
  /** Number of graph nodes. Kept as `segments` for GUI/backward compatibility. */
  segments: number;
  fields: number;
  optionalIncluded: number;
  /** Always zero now: unsupported optional fields reject the whole plan. */
  optionalSkipped: number;
  enums: number;
  strings: number;
  numbers: number;
  booleans: number;
  arrays: number;
}

export interface LayoutConstraintProgram {
  readonly nodes: readonly JsonNode[];
  readonly entry: number;
  readonly accept: number;
  readonly summary: Readonly<LayoutConstraintProgramSummary>;
}

export class UnsupportedLayoutPlanError extends Error {
  constructor(readonly path: string, reason: string) {
    super(`[layout-constraint] ${path}: ${reason}`);
    this.name = "UnsupportedLayoutPlanError";
  }
}

/**
 * Compile the analyzed LayoutPlan directly into a finite control-flow graph.
 * Struct fields keep analyzed order. Optional fields become runtime branches:
 * the model may emit the field or skip it, while required fields cannot be
 * skipped. Unsupported optional payloads are NOT silently dropped; they make
 * the whole record unsupported, because otherwise the compiled language would
 * be a strict subset of the schema without saying so.
 */
export function compileLayoutPlanProgram(
  plan: LayoutPlan,
  options: { rootType?: string } = {},
): LayoutConstraintProgram {
  const rootType = options.rootType ?? "value";
  const types = new Map<string, any>((plan.types as readonly any[]).map((type) => [type.name, type]));
  const nodes: JsonNode[] = [];
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
  const counted = new Set<string>();

  const countOnce = (key: string, fn: () => void): void => {
    if (counted.has(key)) return;
    counted.add(key);
    fn();
  };

  const addNode = (node: JsonNode): number => {
    const id = nodes.length;
    nodes.push(node);
    return id;
  };

  const literal = (text: string, label: string, next: number): number =>
    addNode({ kind: "literal", bytes: encoder.encode(text), text, label, next });

  const choice = (texts: readonly string[], label: string, next: number): number => {
    const unique = [...new Set(texts)];
    if (unique.length === 0) throw new UnsupportedLayoutPlanError(label, "empty choice");
    return addNode({
      kind: "choice",
      alternatives: unique.map((value) => encoder.encode(value)),
      texts: unique,
      label,
      next,
    });
  };

  let compileFieldType: (
    field: any,
    path: string,
    next: number,
    visited: Set<string>,
  ) => number;

  const compileStruct = (
    struct: any,
    path: string,
    next: number,
    visited: Set<string>,
  ): number => {
    const fields = [...(struct.fields ?? [])] as any[];
    const close = literal("}", `object-close ${path}`, next);
    const memo = new Map<string, number>();

    const compileFields = (index: number, emitted: boolean): number => {
      if (index >= fields.length) return close;
      const memoKey = `${index}:${emitted ? 1 : 0}`;
      const cached = memo.get(memoKey);
      if (cached !== undefined) return cached;

      const fieldPlan = fields[index]!;
      const fieldPath = `${path}.${fieldPlan.name}`;
      const fieldType = fieldPlan.type;
      const optional = fieldType?.kind === "optional";
      const valueType = optional ? fieldType.inner : fieldType;

      // Compile the value first. Any unsupported inner type propagates even for
      // optional fields; silently skipping it would lie about supported schema.
      const afterPresent = compileFields(index + 1, true);
      const valueEntry = compileFieldType(valueType, fieldPath, afterPresent, visited);
      const propertyPrefix = `${emitted ? "," : ""}${JSON.stringify(String(fieldPlan.name))}:`;
      const presentEntry = literal(propertyPrefix, `field ${fieldPath}`, valueEntry);

      countOnce(`field:${fieldPath}`, () => {
        summary.fields++;
        if (optional) summary.optionalIncluded++;
      });

      let entry = presentEntry;
      if (optional) {
        const skippedEntry = compileFields(index + 1, emitted);
        entry = addNode({
          kind: "split",
          targets: [presentEntry, skippedEntry],
          label: `optional ${fieldPath}`,
        });
      }

      memo.set(memoKey, entry);
      return entry;
    };

    const fieldsEntry = compileFields(0, false);
    return literal("{", `object-open ${path}`, fieldsEntry);
  };

  compileFieldType = (
    field: any,
    path: string,
    next: number,
    visited: Set<string>,
  ): number => {
    switch (field?.kind) {
      case "primitive": {
        const primitiveType = String(field.type ?? "");
        const name = String(field.name ?? "").toLowerCase();
        if (primitiveType === "boolean" || name === "bool" || name === "boolean") {
          countOnce(`bool:${path}`, () => summary.booleans++);
          return choice(["true", "false"], `bool ${path}`, next);
        }
        if (primitiveType === "number" || /^(u|i|f)\d+$/.test(name) || ["float", "double", "usize"].includes(name)) {
          const integer = !(field.isFloat || name === "f32" || name === "f64" || name === "float" || name === "double");
          countOnce(`number:${path}`, () => summary.numbers++);
          return addNode({
            kind: "number",
            integer,
            min: typeof field.min === "number" ? field.min : undefined,
            max: typeof field.max === "number" ? field.max : undefined,
            step: typeof field.step === "number" && field.step > 0 ? field.step : undefined,
            maxChars: 32,
            label: `number ${path}`,
            next,
          });
        }
        throw new UnsupportedLayoutPlanError(path, `primitive '${field.name ?? field.type ?? "?"}'`);
      }

      case "string": {
        const exactLength = typeof field.exactLength === "number" ? field.exactLength : undefined;
        const maxLength = exactLength ?? (typeof field.maxLength === "number" ? field.maxLength : undefined);
        if (maxLength === undefined || !Number.isFinite(maxLength)) {
          throw new UnsupportedLayoutPlanError(path, "string has no finite maxLength/exactLength in LayoutPlan");
        }
        countOnce(`string:${path}`, () => summary.strings++);
        return addNode({
          kind: "string",
          minLength: exactLength ?? (typeof field.minLength === "number" ? field.minLength : 0),
          maxLength,
          label: `string ${path}`,
          next,
        });
      }

      case "reference": {
        const target = types.get(field.name);
        if (!target) throw new UnsupportedLayoutPlanError(path, `missing referenced type '${field.name}'`);
        if (visited.has(field.name)) throw new UnsupportedLayoutPlanError(path, `cyclic reference '${field.name}'`);
        const nextVisited = new Set(visited);
        nextVisited.add(field.name);
        if (target.kind === "enum") {
          countOnce(`enum:${path}`, () => summary.enums++);
          return choice(
            target.variants.map((variant: any) => JSON.stringify(String(variant.name))),
            `enum ${path}`,
            next,
          );
        }
        if (target.kind === "struct") return compileStruct(target, path, next, nextVisited);
        if (target.kind === "alias") return compileFieldType(target.type, path, next, nextVisited);
        throw new UnsupportedLayoutPlanError(path, `referenced type '${field.name}' has kind '${target.kind}'`);
      }

      case "optional":
        throw new UnsupportedLayoutPlanError(path, "optional is only supported as a struct field");

      case "array": {
        const exactLength = typeof field.exactLength === "number" ? field.exactLength : undefined;
        if (exactLength === undefined) {
          throw new UnsupportedLayoutPlanError(path, "array has no exactLength in LayoutPlan");
        }
        countOnce(`array:${path}`, () => summary.arrays++);
        let entry = literal("]", `array-close ${path}`, next);
        for (let i = exactLength - 1; i >= 0; i--) {
          entry = compileFieldType(field.item, `${path}[${i}]`, entry, visited);
          if (i > 0) entry = literal(",", `array-comma ${path}[${i}]`, entry);
        }
        return literal("[", `array-open ${path}`, entry);
      }

      case "inlineStruct":
        return compileStruct(field, path, next, visited);

      case "unit":
        throw new UnsupportedLayoutPlanError(path, "unit carries no JSON value semantics");

      default:
        throw new UnsupportedLayoutPlanError(path, `field kind '${field?.kind ?? "?"}'`);
    }
  };

  const accept = addNode({ kind: "accept", label: "<complete>" });
  const root = types.get(rootType);
  if (!root) throw new UnsupportedLayoutPlanError(rootType, `root type not found; available: ${[...types.keys()].join(", ")}`);

  let entry: number;
  const visited = new Set<string>([rootType]);
  if (root.kind === "struct") entry = compileStruct(root, rootType, accept, visited);
  else if (root.kind === "enum") {
    countOnce(`enum:${rootType}`, () => summary.enums++);
    entry = choice(root.variants.map((variant: any) => JSON.stringify(String(variant.name))), `enum ${rootType}`, accept);
  } else if (root.kind === "alias") entry = compileFieldType(root.type, rootType, accept, visited);
  else throw new UnsupportedLayoutPlanError(rootType, `top-level kind '${root.kind}'`);

  summary.segments = nodes.length;
  return { nodes, entry, accept, summary };
}

type LiteralLocal = { kind: "literal"; offset: number };
type ChoiceLocal = { kind: "choice"; prefix: number[] };
type StringLocal = {
  kind: "string";
  phase: "open" | "body" | "escape";
  length: number;
};
type NumberLocal = { kind: "number"; text: string };
type NodeLocal = LiteralLocal | ChoiceLocal | StringLocal | NumberLocal;

type BranchState = {
  node: number;
  local: NodeLocal | null;
};

type ProgramState = readonly BranchState[];

function cloneBranchState(state: BranchState): BranchState {
  if (!state.local) return { node: state.node, local: null };
  if (state.local.kind === "choice") return { node: state.node, local: { kind: "choice", prefix: [...state.local.prefix] } };
  return { node: state.node, local: { ...state.local } } as BranchState;
}

function cloneProgramState(state: ProgramState): BranchState[] {
  return state.map(cloneBranchState);
}

function localKey(local: NodeLocal | null): string {
  if (!local) return "-";
  switch (local.kind) {
    case "literal": return `l:${local.offset}`;
    case "choice": return `c:${local.prefix.join(".")}`;
    case "string": return `s:${local.phase}:${local.length}`;
    case "number": return `n:${local.text}`;
  }
}

function normalizeProgramState(program: LayoutConstraintProgram, state: ProgramState): BranchState[] {
  const queue = cloneProgramState(state);
  const out: BranchState[] = [];
  const seenSplit = new Set<number>();
  const seenConcrete = new Set<string>();

  while (queue.length > 0) {
    const branch = queue.pop()!;
    const node = program.nodes[branch.node];
    if (!node) continue;
    if (node.kind === "split" || node.kind === "jump") {
      if (seenSplit.has(branch.node)) continue;
      seenSplit.add(branch.node);
      if (node.kind === "jump") queue.push({ node: node.next, local: null });
      else for (const target of node.targets) queue.push({ node: target, local: null });
      continue;
    }
    const key = `${branch.node}|${localKey(branch.local)}`;
    if (seenConcrete.has(key)) continue;
    seenConcrete.add(key);
    out.push(branch);
  }
  return out;
}

function ensureLocal(node: Exclude<JsonNode, SplitNode | JumpNode | AcceptNode>, state: BranchState): NodeLocal {
  if (state.local) return state.local;
  switch (node.kind) {
    case "literal": state.local = { kind: "literal", offset: 0 }; break;
    case "choice": state.local = { kind: "choice", prefix: [] }; break;
    case "string": state.local = { kind: "string", phase: "open", length: 0 }; break;
    case "number": state.local = { kind: "number", text: "" }; break;
  }
  return state.local;
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

function isJsonNumberComplete(node: NumberNode, text: string): boolean {
  return isJsonNumberCompleteExact(text, {
    integer: node.integer,
    minText: node.min === undefined ? undefined : String(node.min),
    maxText: node.max === undefined ? undefined : String(node.max),
    step: node.step,
  });
}

function choiceMatchesPrefix(node: ChoiceNode, prefix: readonly number[]): boolean {
  return node.alternatives.some((candidate) => {
    if (prefix.length > candidate.length) return false;
    for (let i = 0; i < prefix.length; i++) if (candidate[i] !== prefix[i]) return false;
    return true;
  });
}

function choiceComplete(node: ChoiceNode, prefix: readonly number[]): boolean {
  return node.alternatives.some((candidate) => {
    if (prefix.length !== candidate.length) return false;
    for (let i = 0; i < prefix.length; i++) if (candidate[i] !== prefix[i]) return false;
    return true;
  });
}

function feedByteFromBranch(
  program: LayoutConstraintProgram,
  initial: BranchState,
  byte: number,
  depth = 0,
): BranchState[] {
  if (depth > 128) throw new Error("LayoutPlan constraint exceeded control-flow retry limit");
  const starts = normalizeProgramState(program, [initial]);
  const results: BranchState[] = [];

  for (const start of starts) {
    const node = program.nodes[start.node];
    if (!node || node.kind === "accept" || node.kind === "split" || node.kind === "jump") continue;
    const state = cloneBranchState(start);
    const local = ensureLocal(node, state);

    if (node.kind === "literal") {
      const literalState = local as LiteralLocal;
      if (node.bytes[literalState.offset] !== byte) continue;
      literalState.offset++;
      if (literalState.offset === node.bytes.length) {
        results.push(...normalizeProgramState(program, [{ node: node.next, local: null }]));
      } else results.push(state);
      continue;
    }

    if (node.kind === "choice") {
      const choiceState = local as ChoiceLocal;
      choiceState.prefix.push(byte);
      if (!choiceMatchesPrefix(node, choiceState.prefix)) continue;
      if (choiceComplete(node, choiceState.prefix)) {
        results.push(...normalizeProgramState(program, [{ node: node.next, local: null }]));
      } else results.push(state);
      continue;
    }

    if (node.kind === "string") {
      const stringState = local as StringLocal;
      if (stringState.phase === "open") {
        if (byte !== 0x22) continue;
        stringState.phase = "body";
        results.push(state);
        continue;
      }
      if (stringState.phase === "escape") {
        // \uXXXX is deliberately not accepted — see NO_UNICODE_ESCAPE.
        if (![0x22, 0x5c, 0x2f, 0x62, 0x66, 0x6e, 0x72, 0x74].includes(byte)) continue;
        if (stringState.length >= node.maxLength) continue;
        stringState.length++;
        stringState.phase = "body";
        results.push(state);
        continue;
      }
      if (byte === 0x22) {
        if (stringState.length < node.minLength) continue;
        results.push(...normalizeProgramState(program, [{ node: node.next, local: null }]));
        continue;
      }
      if (byte === 0x5c) {
        if (stringState.length >= node.maxLength) continue;
        stringState.phase = "escape";
        results.push(state);
        continue;
      }
      if (byte < 0x20 || stringState.length >= node.maxLength) continue;
      stringState.length++;
      results.push(state);
      continue;
    }

    const numberState = local as NumberLocal;
    const char = byte <= 0x7f ? String.fromCharCode(byte) : "";
    if (char && numberState.text.length < node.maxChars) {
      const nextText = numberState.text + char;
      if (isJsonNumberPrefix(nextText, node.integer)) {
        if (nextText.length === node.maxChars && !isJsonNumberComplete(node, nextText)) continue;
        numberState.text = nextText;
        results.push(state);
        continue;
      }
    }
    if (!isJsonNumberComplete(node, numberState.text)) continue;
    // Number completion does not consume the delimiter. Retry this byte from
    // every epsilon-expanded continuation branch.
    const continuations = normalizeProgramState(program, [{ node: node.next, local: null }]);
    for (const continuation of continuations) {
      results.push(...feedByteFromBranch(program, continuation, byte, depth + 1));
    }
  }

  return normalizeProgramState(program, results);
}

function feedByte(program: LayoutConstraintProgram, state: ProgramState, byte: number): BranchState[] {
  const normalized = normalizeProgramState(program, state);
  const next: BranchState[] = [];
  for (const branch of normalized) next.push(...feedByteFromBranch(program, branch, byte));
  return normalizeProgramState(program, next);
}

function feedBytes(program: LayoutConstraintProgram, state: ProgramState, bytes: Uint8Array): BranchState[] {
  let current = normalizeProgramState(program, state);
  for (const byte of bytes) {
    current = feedByte(program, current, byte);
    if (current.length === 0) break;
  }
  return current;
}

function describeBranch(program: LayoutConstraintProgram, branch: BranchState): string {
  const node = program.nodes[branch.node];
  if (!node) return `<invalid node ${branch.node}>`;
  if (node.kind === "accept") return "<complete>";
  if (node.kind === "split") return `${node.label} · split ${node.targets.length}`;
  if (node.kind === "jump") return `${node.label} · jump`;
  const local = ensureLocal(node, branch);
  switch (node.kind) {
    case "literal":
      return `${node.label} · literal ${JSON.stringify(node.text)} @ ${(local as LiteralLocal).offset}/${node.bytes.length}`;
    case "choice": {
      const prefix = new Uint8Array((local as ChoiceLocal).prefix);
      const alive = node.texts.filter((_, index) => startsWithBytes(node.alternatives[index]!, prefix));
      return `${node.label} · prefix ${JSON.stringify(displayBytes(prefix))} · ${alive.length} choices`;
    }
    case "string": {
      const value = local as StringLocal;
      return `${node.label} · ${value.phase} · len ${value.length}/${node.maxLength}`;
    }
    case "number": {
      const value = local as NumberLocal;
      const range = `${node.min ?? "-∞"}..${node.max ?? "+∞"}${node.step ? ` step ${node.step}` : ""}`;
      return `${node.label} · ${JSON.stringify(value.text)} · ${range}`;
    }
  }
}

function describeProgramState(program: LayoutConstraintProgram, state: ProgramState): string {
  const normalized = normalizeProgramState(program, state);
  if (normalized.length === 0) return "<dead-end>";
  const descriptions = normalized.slice(0, 4).map((branch) => describeBranch(program, cloneBranchState(branch)));
  if (normalized.length > descriptions.length) descriptions.push(`… +${normalized.length - descriptions.length} branches`);
  return descriptions.join(" OR ");
}

function programComplete(program: LayoutConstraintProgram, state: ProgramState): boolean {
  return normalizeProgramState(program, state).some((branch) => branch.node === program.accept);
}

/**
 * Token-level CPU oracle driven directly by LayoutPlan. It scans every vocab
 * token against a clone of the byte-state NFA, masks illegal logits, then
 * advances the real machine with the selected token.
 */
export class LayoutPlanJsonConstraint {
  readonly trace: ConstraintTraceStep[] = [];
  readonly program: LayoutConstraintProgram;

  private state: BranchState[];
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
    this.state = normalizeProgramState(this.program, [{ node: this.program.entry, local: null }]);
    if (!tokens[eosToken]) throw new Error(`EOS token ${eosToken} is outside token table`);
  }

  get generatedPrefix(): string {
    return decoder.decode(this.prefix);
  }

  get complete(): boolean {
    return programComplete(this.program, this.state);
  }

  get done(): boolean {
    return this.eosAccepted;
  }

  private prepareAllowed(context: { step: number; decode: boolean }): { allowed: Uint8Array; ids: Uint32Array; prefixBefore: string; stateBefore: string } {
    if (this.pending) throw new Error("Constraint candidates/process called twice without accept()");

    const allowed = new Uint8Array(this.tokens.length);
    const ids: number[] = [];

    if (this.complete) {
      allowed[this.eosToken] = 1;
      ids.push(this.eosToken);
    } else {
      for (let id = 0; id < this.tokens.length; id++) {
        const entry = this.tokens[id]!;
        if (entry.special || !entry.bytes || entry.bytes.length === 0) continue;
        const candidateState = feedBytes(this.program, this.state, entry.bytes);
        if (candidateState.length > 0) {
          allowed[id] = 1;
          ids.push(id);
        }
      }
    }

    const prefixBefore = this.generatedPrefix;
    const stateBefore = describeProgramState(this.program, this.state);
    if (ids.length === 0) throw new Error(`LayoutPlan constraint dead-end after ${JSON.stringify(prefixBefore)} · ${stateBefore}`);
    return { allowed, ids: Uint32Array.from(ids), prefixBefore, stateBefore };
  }

  /** Fast diagnostic path: return sparse legal token ids without touching logits. */
  candidates(context: { step: number; decode: boolean }): Uint32Array {
    const prepared = this.prepareAllowed(context);
    this.pending = {
      step: context.step,
      decode: context.decode,
      prefixBefore: prepared.prefixBefore,
      stateBefore: prepared.stateBefore,
      allowed: prepared.allowed,
      topBefore: [],
      topAllowed: [],
    };
    return prepared.ids;
  }

  process(logits: Float32Array, context: { step: number; decode: boolean }): void {
    if (logits.length !== this.tokens.length) throw new Error(`Logit/token table mismatch: ${logits.length} vs ${this.tokens.length}`);
    const prepared = this.prepareAllowed(context);
    const traced = maskAndTrace(logits, this.tokens, prepared.allowed, this.eosToken);
    this.pending = {
      step: context.step,
      decode: context.decode,
      prefixBefore: prepared.prefixBefore,
      stateBefore: prepared.stateBefore,
      allowed: prepared.allowed,
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
      if (!this.complete) throw new Error("EOS selected before LayoutPlan constraint completed");
      this.eosAccepted = true;
    } else {
      const bytes = this.tokens[tokenId]?.bytes;
      if (!bytes) throw new Error(`Selected token ${tokenId} has no byte payload`);
      const nextState = feedBytes(this.program, this.state, bytes);
      if (nextState.length === 0) throw new Error(`Selected token ${tokenId} failed replay against LayoutPlan machine`);
      selectedBytes = displayBytes(bytes);
      this.state = nextState;
      this.prefix = concatBytes(this.prefix, bytes);
    }

    this.trace.push({
      step: pending.step,
      decode: pending.decode,
      prefixBefore: pending.prefixBefore,
      prefixAfter: this.generatedPrefix,
      stateBefore: pending.stateBefore,
      stateAfter: describeProgramState(this.program, this.state),
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

export * from "./transport";

export * from "./binary-transport";

export * from "./gpu-constraint.ts";

export * from "./json-schema-constraint.ts";
export * from "./structured.ts";
