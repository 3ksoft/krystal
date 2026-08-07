import type {
  ChoiceNode,
  JsonNode,
  LayoutConstraintProgram,
  NumberNode,
  TokenByteTableEntry,
} from "./index.ts";
import { isJsonNumberComplete } from "./json-number.ts";

/** Must stay in lock-step with packages/schema/src/schema.ts. */
export const GPU_CONSTRAINT_ABI = {
  version: 2,
  invalidNode: 0xffff_ffff,
  maxNodes: 0xffff,
  maxSwitchEdges: 0xff,
  maxLiteralBytes: 0xffff,
  maxStringLength: 0xffff,
  maxNumberChars: 32,
  maxTokenBytes: 0xff,
  maxVocabSize: 0x1_0000,
  headerWords: 12,
  nodeWords: 12,
  tokenizerHeaderWords: 8,
  tokenizerEntryWords: 2,
} as const;

export const GPU_CONSTRAINT_NODE_KIND = {
  literal: 0,
  switch: 1,
  string: 2,
  number: 3,
  accept: 4,
  jump: 5,
} as const;

export const GPU_CONSTRAINT_NUMBER_FLAGS = {
  integer: 1 << 0,
  hasMin: 1 << 1,
  hasMax: 1 << 2,
  hasStep: 1 << 3,
} as const;

export const GPU_CONSTRAINT_EDGE_FLAGS = {
  replayByte: 1 << 24,
} as const;

export const GPU_CONSTRAINT_TOKEN_META = {
  lengthMask: 0xff,
  special: 1 << 8,
} as const;

type GpuNodeKind = keyof typeof GPU_CONSTRAINT_NODE_KIND;

type MutableGpuNode = {
  kind: GpuNodeKind;
  next: number;
  dataOffset: number;
  dataCount: number;
  args: [number, number, number, number, number, number, number, number];
};

type StaticPath = {
  bytes: number[];
  endpoint: number;
  /** The final dispatch byte selects this endpoint but must be replayed there. */
  replay?: boolean;
};

type TrieEndpoint = {
  source: number;
  replay: boolean;
};

type TrieNode = {
  endpoint?: TrieEndpoint;
  children: Map<number, TrieNode>;
};

export interface GpuConstraintProgramSummary {
  readonly sourceNodes: number;
  readonly nodes: number;
  readonly literalNodes: number;
  readonly switchNodes: number;
  readonly stringNodes: number;
  readonly numberNodes: number;
  readonly acceptNodes: number;
  readonly jumpNodes: number;
  readonly edges: number;
  readonly byteLength: number;
  readonly blobWords: number;
  readonly blobBytes: number;
}

/**
 * Upload-ready structured decoder program.
 *
 * Offsets in `header` address the concatenated `blob` in u32 words. `byteWords`
 * store the raw byte pool packed little-endian, four bytes per u32.
 */
export interface GpuConstraintProgram {
  readonly entryNode: number;
  readonly acceptNode: number;
  readonly header: Uint32Array;
  readonly nodes: Uint32Array;
  readonly edges: Uint32Array;
  readonly byteWords: Uint32Array;
  readonly blob: Uint32Array;
  readonly summary: Readonly<GpuConstraintProgramSummary>;
}

export interface GpuConstraintTokenizer {
  readonly header: Uint32Array;
  readonly entries: Uint32Array;
  readonly byteWords: Uint32Array;
  readonly blob: Uint32Array;
  readonly byteLength: number;
}

function fail(reason: string): never {
  throw new Error(`[gpu-constraint] ${reason}`);
}

function assertU32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > 0xffff_ffff) {
    fail(`${label}=${value} is outside u32`);
  }
  return value >>> 0;
}

function concatByteArrays(parts: readonly Uint8Array[]): Uint8Array {
  let length = 0;
  for (const part of parts) length += part.length;
  const out = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

function packBytes(bytes: Uint8Array): Uint32Array {
  const words = new Uint32Array(Math.ceil(bytes.length / 4));
  for (let i = 0; i < bytes.length; i++) {
    words[i >>> 2] = (words[i >>> 2]! | (bytes[i]! << ((i & 3) * 8))) >>> 0;
  }
  return words;
}

function nodeArgs(): MutableGpuNode["args"] {
  return [0, 0, 0, 0, 0, 0, 0, 0];
}

function createTrie(): TrieNode {
  return { children: new Map() };
}

function appendTriePath(root: TrieNode, path: StaticPath, label: string): void {
  if (path.bytes.length === 0) fail(`${label}: epsilon path cannot be encoded as a byte switch`);
  let node = root;
  for (const byte of path.bytes) {
    let child = node.children.get(byte);
    if (!child) {
      child = createTrie();
      node.children.set(byte, child);
    }
    node = child;
  }
  const endpoint = { source: path.endpoint, replay: path.replay === true };
  if (node.endpoint !== undefined
    && (node.endpoint.source !== endpoint.source || node.endpoint.replay !== endpoint.replay)) {
    fail(`${label}: identical byte prefix reaches incompatible source endpoints`);
  }
  node.endpoint = endpoint;
}

function validatePrefixFreeTrie(node: TrieNode, label: string): void {
  if (node.endpoint !== undefined && node.children.size > 0) {
    fail(`${label}: one static alternative is a byte-prefix of another`);
  }
  for (const child of node.children.values()) validatePrefixFreeTrie(child, label);
}

/**
 * Link the CPU LayoutConstraintProgram into the deterministic byte VM ABI used
 * by WGSL. Static NFA regions (`literal`, `choice`, `split`) are collapsed into
 * literal runs and sparse byte tries. Dynamic JSON lexers (`string`, `number`)
 * remain explicit instructions.
 */
export function linkGpuConstraintProgram(program: LayoutConstraintProgram): GpuConstraintProgram {
  const sourceNodes = program.nodes;
  const gpuNodes: MutableGpuNode[] = [];
  const edges: number[] = [];
  const byteParts: Uint8Array[] = [];
  let byteLength = 0;
  const sourceMemo = new Map<number, number>();

  const appendBytes = (bytes: Uint8Array): { offset: number; length: number } => {
    if (bytes.length === 0) fail("zero-length literal run");
    if (bytes.length > GPU_CONSTRAINT_ABI.maxLiteralBytes) {
      fail(`literal run is ${bytes.length} bytes; max is ${GPU_CONSTRAINT_ABI.maxLiteralBytes}`);
    }
    const offset = byteLength;
    byteParts.push(bytes);
    byteLength += bytes.length;
    if (byteLength > 0xffff_ffff) fail("byte pool exceeds u32 address space");
    return { offset, length: bytes.length };
  };

  const reserveNode = (kind: GpuNodeKind): number => {
    if (gpuNodes.length >= GPU_CONSTRAINT_ABI.maxNodes) {
      fail(`node count exceeds ${GPU_CONSTRAINT_ABI.maxNodes}`);
    }
    const id = gpuNodes.length;
    gpuNodes.push({
      kind,
      next: GPU_CONSTRAINT_ABI.invalidNode,
      dataOffset: 0,
      dataCount: 0,
      args: nodeArgs(),
    });
    return id;
  };

  const collectStaticPaths = (
    sourceId: number,
    prefix: number[],
    out: StaticPath[],
    stack: Set<number>,
  ): void => {
    const source = sourceNodes[sourceId] as JsonNode | undefined;
    if (!source) fail(`source node ${sourceId} does not exist`);

    if (source.kind === "choice" || source.kind === "string" || source.kind === "number" || source.kind === "jump" || source.kind === "accept") {
      out.push({ bytes: prefix, endpoint: sourceId });
      return;
    }

    if (stack.has(sourceId)) fail(`cycle detected while expanding static source node ${sourceId}`);
    const nextStack = new Set(stack);
    nextStack.add(sourceId);

    if (source.kind === "literal") {
      collectStaticPaths(source.next, [...prefix, ...source.bytes], out, nextStack);
      return;
    }

    for (const target of source.targets) collectStaticPaths(target, [...prefix], out, nextStack);
  };

  let compileSource: (sourceId: number) => number;

  const compileTrie = (trie: TrieNode, label: string): number => {
    if (trie.endpoint !== undefined) {
      if (trie.children.size !== 0) fail(`${label}: terminal trie node still has children`);
      if (trie.endpoint.replay) fail(`${label}: replay endpoint must be reached through a switch edge`);
      return compileSource(trie.endpoint.source);
    }
    if (trie.children.size === 0) fail(`${label}: empty trie node`);

    if (trie.children.size === 1) {
      const [[_firstByte, firstChild]] = trie.children;
      // A replay edge is a zero-consumption dispatch guard. It cannot be
      // collapsed into a literal because the destination lexer still needs to
      // consume the discriminating byte itself.
      if (!(firstChild.endpoint?.replay && firstChild.children.size === 0)) {
        const bytes: number[] = [];
        let cursor = trie;
        while (cursor.endpoint === undefined && cursor.children.size === 1) {
          const [[byte, child]] = cursor.children;
          if (child.endpoint?.replay && child.children.size === 0) break;
          bytes.push(byte);
          cursor = child;
        }
        if (bytes.length > 0) {
          const id = reserveNode("literal");
          const stored = appendBytes(Uint8Array.from(bytes));
          const node = gpuNodes[id]!;
          node.dataOffset = stored.offset;
          node.dataCount = stored.length;
          node.next = compileTrie(cursor, label);
          return id;
        }
      }
    }

    if (trie.children.size > GPU_CONSTRAINT_ABI.maxSwitchEdges) {
      fail(`${label}: switch has ${trie.children.size} edges; max is ${GPU_CONSTRAINT_ABI.maxSwitchEdges}`);
    }

    const id = reserveNode("switch");
    const node = gpuNodes[id]!;
    const packedChildren: number[] = [];

    const sorted = [...trie.children.entries()].sort((a, b) => a[0] - b[0]);
    for (const [byte, child] of sorted) {
      const replay = child.endpoint?.replay === true && child.children.size === 0;
      const target = replay
        ? compileSource(child.endpoint!.source)
        : compileTrie(child, label);
      if (target >= 0x1_0000) fail(`${label}: edge target ${target} does not fit packed u16`);
      packedChildren.push((byte | (target << 8) | (replay ? GPU_CONSTRAINT_EDGE_FLAGS.replayByte : 0)) >>> 0);
    }

    // Children may recursively emit their own switch edges. Record this
    // switch's range only after they are fully linked, otherwise nested tries
    // point at the first descendant edge instead of their own table slice.
    node.dataOffset = edges.length;
    node.dataCount = packedChildren.length;
    edges.push(...packedChildren);
    return id;
  };

  const compileStaticAlternatives = (paths: StaticPath[], label: string): number => {
    const dedup = new Map<string, StaticPath>();
    for (const path of paths) {
      const key = `${path.bytes.join(",")}|${path.endpoint}`;
      dedup.set(key, path);
    }
    const unique = [...dedup.values()];
    if (unique.length === 0) fail(`${label}: static expansion produced no alternatives`);

    let expanded = unique;
    const epsilon = expanded.filter((path) => path.bytes.length === 0);
    if (epsilon.length > 0) {
      const endpoints = new Set(epsilon.map((path) => path.endpoint));
      if (epsilon.length === expanded.length && endpoints.size === 1) {
        return compileSource(epsilon[0]!.endpoint);
      }

      // A split such as bounded `array<string>` branches between `]` and a
      // dynamic lexer. Guard the lexer by its legal first byte(s), then replay
      // that byte at the destination. The high byte of the packed switch edge
      // carries this zero-consumption/replay flag.
      const guarded: StaticPath[] = [];
      for (const path of expanded) {
        if (path.bytes.length !== 0) {
          guarded.push(path);
          continue;
        }
        const endpoint = sourceNodes[path.endpoint] as JsonNode | undefined;
        if (!endpoint) fail(`${label}: epsilon endpoint ${path.endpoint} does not exist`);
        if (endpoint.kind === "string") {
          guarded.push({ bytes: [0x22], endpoint: path.endpoint, replay: true });
          continue;
        }
        if (endpoint.kind === "number") {
          guarded.push({ bytes: [0x2d], endpoint: path.endpoint, replay: true });
          for (let byte = 0x30; byte <= 0x39; byte++) {
            guarded.push({ bytes: [byte], endpoint: path.endpoint, replay: true });
          }
          continue;
        }
        if (endpoint.kind === "choice") {
          for (const alternative of endpoint.alternatives) {
            guarded.push({ bytes: [...alternative], endpoint: endpoint.next });
          }
          continue;
        }
        fail(`${label}: epsilon endpoint ${path.endpoint} (${endpoint.kind}) cannot be determinized`);
      }
      expanded = guarded;
    }

    const trie = createTrie();
    for (const path of expanded) appendTriePath(trie, path, label);
    validatePrefixFreeTrie(trie, label);
    return compileTrie(trie, label);
  };

  const compileChoice = (sourceId: number, source: ChoiceNode): number => {
    const paths = source.alternatives.map((alternative) => ({
      bytes: [...alternative],
      endpoint: source.next,
    }));
    return compileStaticAlternatives(paths, `choice source ${sourceId} (${source.label})`);
  };

  const compileNumber = (sourceId: number, source: NumberNode): number => {
    if (source.step !== undefined) {
      fail(`number source ${sourceId} uses step=${source.step}; exact GPU step/multipleOf is not implemented in ABI v1`);
    }
    if (source.maxChars > GPU_CONSTRAINT_ABI.maxNumberChars) {
      fail(`number source ${sourceId} maxChars=${source.maxChars}; max is ${GPU_CONSTRAINT_ABI.maxNumberChars}`);
    }
    const id = reserveNode("number");
    sourceMemo.set(sourceId, id);
    const node = gpuNodes[id]!;
    node.next = compileSource(source.next);

    let flags = 0;
    if (source.integer) flags |= GPU_CONSTRAINT_NUMBER_FLAGS.integer;
    if (source.min !== undefined) flags |= GPU_CONSTRAINT_NUMBER_FLAGS.hasMin;
    if (source.max !== undefined) flags |= GPU_CONSTRAINT_NUMBER_FLAGS.hasMax;
    if (source.step !== undefined) flags |= GPU_CONSTRAINT_NUMBER_FLAGS.hasStep;
    node.args[0] = flags >>> 0;
    node.args[1] = source.maxChars >>> 0;

    // Bounds are stored as canonical ASCII decimal strings rather than f64
    // bits. WGSL has no f64, while decimal comparison can preserve the exact
    // schema boundary (including values such as 999999999999.99).
    const encodeBound = (value: number | undefined): [number, number] => {
      if (value === undefined) return [0, 0];
      if (!Number.isFinite(value)) fail(`number source ${sourceId} has non-finite bound ${value}`);
      const encoded = new TextEncoder().encode(String(value));
      const stored = appendBytes(encoded);
      return [stored.offset, stored.length];
    };
    [node.args[2], node.args[3]] = encodeBound(source.min);
    [node.args[4], node.args[5]] = encodeBound(source.max);
    [node.args[6], node.args[7]] = encodeBound(source.step);
    return id;
  };

  compileSource = (sourceId: number): number => {
    const cached = sourceMemo.get(sourceId);
    if (cached !== undefined) return cached;
    const source = sourceNodes[sourceId] as JsonNode | undefined;
    if (!source) fail(`source node ${sourceId} does not exist`);

    if (source.kind === "accept") {
      const id = reserveNode("accept");
      sourceMemo.set(sourceId, id);
      return id;
    }

    if (source.kind === "string") {
      if (source.maxLength > GPU_CONSTRAINT_ABI.maxStringLength) {
        fail(`string source ${sourceId} maxLength=${source.maxLength}; max is ${GPU_CONSTRAINT_ABI.maxStringLength}`);
      }
      if (source.minLength < 0 || source.minLength > source.maxLength) {
        fail(`string source ${sourceId} has invalid range ${source.minLength}..${source.maxLength}`);
      }
      const id = reserveNode("string");
      sourceMemo.set(sourceId, id);
      const node = gpuNodes[id]!;
      node.next = compileSource(source.next);
      node.args[0] = assertU32(source.minLength, `string ${sourceId}.minLength`);
      node.args[1] = assertU32(source.maxLength, `string ${sourceId}.maxLength`);
      return id;
    }

    if (source.kind === "number") return compileNumber(sourceId, source);

    if (source.kind === "jump") {
      const id = reserveNode("jump");
      sourceMemo.set(sourceId, id);
      gpuNodes[id]!.next = compileSource(source.next);
      return id;
    }

    if (source.kind === "literal") {
      const id = reserveNode("literal");
      sourceMemo.set(sourceId, id);
      const stored = appendBytes(source.bytes);
      const node = gpuNodes[id]!;
      node.dataOffset = stored.offset;
      node.dataCount = stored.length;
      node.next = compileSource(source.next);
      return id;
    }

    if (source.kind === "choice") {
      const id = compileChoice(sourceId, source);
      sourceMemo.set(sourceId, id);
      return id;
    }

    const paths: StaticPath[] = [];
    for (const target of source.targets) collectStaticPaths(target, [], paths, new Set([sourceId]));
    const id = compileStaticAlternatives(paths, `split source ${sourceId} (${source.label})`);
    sourceMemo.set(sourceId, id);
    return id;
  };

  const entryNode = compileSource(program.entry);
  const acceptNode = compileSource(program.accept);

  const packedNodes = new Uint32Array(gpuNodes.length * GPU_CONSTRAINT_ABI.nodeWords);
  const counts: Record<GpuNodeKind, number> = {
    literal: 0,
    switch: 0,
    string: 0,
    number: 0,
    accept: 0,
    jump: 0,
  };
  for (let id = 0; id < gpuNodes.length; id++) {
    const node = gpuNodes[id]!;
    counts[node.kind]++;
    const at = id * GPU_CONSTRAINT_ABI.nodeWords;
    packedNodes[at + 0] = GPU_CONSTRAINT_NODE_KIND[node.kind];
    packedNodes[at + 1] = node.next >>> 0;
    packedNodes[at + 2] = node.dataOffset >>> 0;
    packedNodes[at + 3] = node.dataCount >>> 0;
    for (let arg = 0; arg < 8; arg++) packedNodes[at + 4 + arg] = node.args[arg]! >>> 0;
  }

  const packedEdges = Uint32Array.from(edges);
  const rawBytes = concatByteArrays(byteParts);
  const byteWords = packBytes(rawBytes);
  const header = new Uint32Array(GPU_CONSTRAINT_ABI.headerWords);
  const nodeWordOffset = GPU_CONSTRAINT_ABI.headerWords;
  const edgeWordOffset = nodeWordOffset + packedNodes.length;
  const byteWordOffset = edgeWordOffset + packedEdges.length;
  header[0] = GPU_CONSTRAINT_ABI.version;
  header[1] = 0;
  header[2] = entryNode;
  header[3] = acceptNode;
  header[4] = nodeWordOffset;
  header[5] = gpuNodes.length;
  header[6] = edgeWordOffset;
  header[7] = packedEdges.length;
  header[8] = byteWordOffset;
  header[9] = rawBytes.length;

  const blob = new Uint32Array(byteWordOffset + byteWords.length);
  blob.set(header, 0);
  blob.set(packedNodes, nodeWordOffset);
  blob.set(packedEdges, edgeWordOffset);
  blob.set(byteWords, byteWordOffset);

  return {
    entryNode,
    acceptNode,
    header,
    nodes: packedNodes,
    edges: packedEdges,
    byteWords,
    blob,
    summary: {
      sourceNodes: sourceNodes.length,
      nodes: gpuNodes.length,
      literalNodes: counts.literal,
      switchNodes: counts.switch,
      stringNodes: counts.string,
      numberNodes: counts.number,
      acceptNodes: counts.accept,
      jumpNodes: counts.jump,
      edges: packedEdges.length,
      byteLength: rawBytes.length,
      blobWords: blob.length,
      blobBytes: blob.byteLength,
    },
  };
}

/** Pack one model-global token-id -> raw-byte table for the GPU byte VM. */
export function linkGpuConstraintTokenizer(
  tokens: readonly TokenByteTableEntry[],
  eosToken: number,
): GpuConstraintTokenizer {
  if (tokens.length > GPU_CONSTRAINT_ABI.maxVocabSize) {
    fail(`vocab has ${tokens.length} entries; max is ${GPU_CONSTRAINT_ABI.maxVocabSize}`);
  }
  if (!Number.isInteger(eosToken) || eosToken < 0 || eosToken >= tokens.length) {
    fail(`EOS token ${eosToken} is outside vocab of ${tokens.length}`);
  }

  const entries = new Uint32Array(tokens.length * GPU_CONSTRAINT_ABI.tokenizerEntryWords);
  const byteParts: Uint8Array[] = [];
  let byteLength = 0;

  for (let tokenId = 0; tokenId < tokens.length; tokenId++) {
    const token = tokens[tokenId]!;
    if (token.id !== tokenId) fail(`token table index ${tokenId} contains id=${token.id}`);
    const bytes = token.bytes ?? new Uint8Array(0);
    if (bytes.length > GPU_CONSTRAINT_ABI.maxTokenBytes) {
      fail(`token ${tokenId} has ${bytes.length} bytes; max is ${GPU_CONSTRAINT_ABI.maxTokenBytes}`);
    }
    const at = tokenId * GPU_CONSTRAINT_ABI.tokenizerEntryWords;
    entries[at] = byteLength;
    entries[at + 1] = (bytes.length | (token.special ? GPU_CONSTRAINT_TOKEN_META.special : 0)) >>> 0;
    if (bytes.length > 0) {
      byteParts.push(bytes);
      byteLength += bytes.length;
    }
  }

  const rawBytes = concatByteArrays(byteParts);
  const byteWords = packBytes(rawBytes);
  const header = new Uint32Array(GPU_CONSTRAINT_ABI.tokenizerHeaderWords);
  const entryWordOffset = GPU_CONSTRAINT_ABI.tokenizerHeaderWords;
  const byteWordOffset = entryWordOffset + entries.length;
  header[0] = tokens.length;
  header[1] = eosToken;
  header[2] = entryWordOffset;
  header[3] = byteWordOffset;
  header[4] = rawBytes.length;

  const blob = new Uint32Array(byteWordOffset + byteWords.length);
  blob.set(header, 0);
  blob.set(entries, entryWordOffset);
  blob.set(byteWords, byteWordOffset);

  return { header, entries, byteWords, blob, byteLength: rawBytes.length };
}

// ---------------------------------------------------------------------------
// Upload-blob reference VM
// ---------------------------------------------------------------------------

/**
 * Exact 64-byte decoder state mirrored by the planned WGSL implementation.
 *
 * words:
 *   0  current node
 *   1  local0 (literal cursor / string phase / number text length)
 *   2  local1 (string decoded length)
 *   3  local2 (string unicode hex digits remaining)
 *   4  ConstraintDecoderStatus (low 8 bits; running = 0)
 *   5  errorCode
 *   6  reserved0
 *   7  reserved1
 *   8..15 numberText, packed little-endian (32 ASCII bytes)
 */
export const GPU_CONSTRAINT_STATE = {
  words: 16,
  byteLength: 64,
  node: 0,
  local0: 1,
  local1: 2,
  local2: 3,
  status: 4,
  errorCode: 5,
  reserved0: 6,
  reserved1: 7,
  numberWordOffset: 8,
  numberWords: 8,
} as const;

export type GpuConstraintDecoderState = Uint32Array;

export function createGpuConstraintDecoderState(
  program: GpuConstraintProgram,
): GpuConstraintDecoderState {
  const state = new Uint32Array(GPU_CONSTRAINT_STATE.words);
  state[GPU_CONSTRAINT_STATE.node] = program.entryNode;
  return state;
}

export function cloneGpuConstraintDecoderState(
  state: GpuConstraintDecoderState,
): GpuConstraintDecoderState {
  if (state.length !== GPU_CONSTRAINT_STATE.words) {
    fail(`decoder state has ${state.length} words; expected ${GPU_CONSTRAINT_STATE.words}`);
  }
  return state.slice();
}

function gpuNodeWord(program: GpuConstraintProgram, nodeId: number, word: number): number {
  if (nodeId >= program.header[5]!) fail(`decoder references node ${nodeId}, but program has ${program.header[5]} nodes`);
  return program.blob[program.header[4]! + nodeId * GPU_CONSTRAINT_ABI.nodeWords + word]!;
}

function gpuProgramByte(program: GpuConstraintProgram, byteOffset: number): number {
  if (byteOffset >= program.header[9]!) fail(`decoder byte offset ${byteOffset} exceeds pool length ${program.header[9]}`);
  const word = program.blob[program.header[8]! + (byteOffset >>> 2)]!;
  return (word >>> ((byteOffset & 3) * 8)) & 0xff;
}

function gpuNumberByte(state: GpuConstraintDecoderState, offset: number): number {
  const word = state[GPU_CONSTRAINT_STATE.numberWordOffset + (offset >>> 2)]!;
  return (word >>> ((offset & 3) * 8)) & 0xff;
}

function setGpuNumberByte(state: GpuConstraintDecoderState, offset: number, byte: number): void {
  const wordIndex = GPU_CONSTRAINT_STATE.numberWordOffset + (offset >>> 2);
  const shift = (offset & 3) * 8;
  const mask = (0xff << shift) >>> 0;
  state[wordIndex] = ((state[wordIndex]! & ~mask) | ((byte & 0xff) << shift)) >>> 0;
}

function gpuNumberText(state: GpuConstraintDecoderState): string {
  const length = state[GPU_CONSTRAINT_STATE.local0]!;
  let out = "";
  for (let i = 0; i < length; i++) out += String.fromCharCode(gpuNumberByte(state, i));
  return out;
}

function gpuProgramAscii(
  program: GpuConstraintProgram,
  byteOffset: number,
  byteLength: number,
): string {
  let out = "";
  for (let i = 0; i < byteLength; i++) out += String.fromCharCode(gpuProgramByte(program, byteOffset + i));
  return out;
}

function gpuIsHex(byte: number): boolean {
  return (byte >= 0x30 && byte <= 0x39)
    || (byte >= 0x41 && byte <= 0x46)
    || (byte >= 0x61 && byte <= 0x66);
}

const GPU_NUMBER_PHASE = {
  start: 0,
  sign: 1,
  zero: 2,
  integer: 3,
  dot: 4,
  fraction: 5,
  exponent: 6,
  exponentSign: 7,
  exponentDigits: 8,
} as const;

function gpuNumberNextPhase(phase: number, byte: number, integer: boolean): number | undefined {
  const digit = byte >= 0x30 && byte <= 0x39;
  const nonZeroDigit = byte >= 0x31 && byte <= 0x39;

  if (phase === GPU_NUMBER_PHASE.start) {
    if (byte === 0x2d) return GPU_NUMBER_PHASE.sign;
    if (byte === 0x30) return GPU_NUMBER_PHASE.zero;
    if (nonZeroDigit) return GPU_NUMBER_PHASE.integer;
    return undefined;
  }
  if (phase === GPU_NUMBER_PHASE.sign) {
    if (byte === 0x30) return GPU_NUMBER_PHASE.zero;
    if (nonZeroDigit) return GPU_NUMBER_PHASE.integer;
    return undefined;
  }
  if (phase === GPU_NUMBER_PHASE.zero) {
    if (!integer && byte === 0x2e) return GPU_NUMBER_PHASE.dot;
    if (!integer && (byte === 0x65 || byte === 0x45)) return GPU_NUMBER_PHASE.exponent;
    return undefined;
  }
  if (phase === GPU_NUMBER_PHASE.integer) {
    if (digit) return GPU_NUMBER_PHASE.integer;
    if (!integer && byte === 0x2e) return GPU_NUMBER_PHASE.dot;
    if (!integer && (byte === 0x65 || byte === 0x45)) return GPU_NUMBER_PHASE.exponent;
    return undefined;
  }
  if (phase === GPU_NUMBER_PHASE.dot) {
    return digit ? GPU_NUMBER_PHASE.fraction : undefined;
  }
  if (phase === GPU_NUMBER_PHASE.fraction) {
    if (digit) return GPU_NUMBER_PHASE.fraction;
    if (byte === 0x65 || byte === 0x45) return GPU_NUMBER_PHASE.exponent;
    return undefined;
  }
  if (phase === GPU_NUMBER_PHASE.exponent) {
    if (byte === 0x2b || byte === 0x2d) return GPU_NUMBER_PHASE.exponentSign;
    return digit ? GPU_NUMBER_PHASE.exponentDigits : undefined;
  }
  if (phase === GPU_NUMBER_PHASE.exponentSign) {
    return digit ? GPU_NUMBER_PHASE.exponentDigits : undefined;
  }
  if (phase === GPU_NUMBER_PHASE.exponentDigits) {
    return digit ? GPU_NUMBER_PHASE.exponentDigits : undefined;
  }
  return undefined;
}

function gpuNumberPhaseComplete(phase: number): boolean {
  return phase === GPU_NUMBER_PHASE.zero
    || phase === GPU_NUMBER_PHASE.integer
    || phase === GPU_NUMBER_PHASE.fraction
    || phase === GPU_NUMBER_PHASE.exponentDigits;
}

function gpuNumberComplete(
  program: GpuConstraintProgram,
  state: GpuConstraintDecoderState,
  nodeId: number,
): boolean {
  const phase = state[GPU_CONSTRAINT_STATE.local1]!;
  if (!gpuNumberPhaseComplete(phase)) return false;

  const flags = gpuNodeWord(program, nodeId, 4);
  const minText = (flags & GPU_CONSTRAINT_NUMBER_FLAGS.hasMin) !== 0
    ? gpuProgramAscii(program, gpuNodeWord(program, nodeId, 6), gpuNodeWord(program, nodeId, 7))
    : undefined;
  const maxText = (flags & GPU_CONSTRAINT_NUMBER_FLAGS.hasMax) !== 0
    ? gpuProgramAscii(program, gpuNodeWord(program, nodeId, 8), gpuNodeWord(program, nodeId, 9))
    : undefined;

  // ABI v1 linker rejects step/multipleOf before upload. Keep malformed blobs
  // fail-closed even though the shared CPU oracle still supports step.
  if ((flags & GPU_CONSTRAINT_NUMBER_FLAGS.hasStep) !== 0) return false;
  return isJsonNumberComplete(gpuNumberText(state), {
    integer: (flags & GPU_CONSTRAINT_NUMBER_FLAGS.integer) !== 0,
    minText,
    maxText,
  });
}

function gpuGoto(state: GpuConstraintDecoderState, node: number): void {
  state[GPU_CONSTRAINT_STATE.node] = node >>> 0;
  state[GPU_CONSTRAINT_STATE.local0] = 0;
  state[GPU_CONSTRAINT_STATE.local1] = 0;
  state[GPU_CONSTRAINT_STATE.local2] = 0;
}

function gpuFindSwitchEdge(
  program: GpuConstraintProgram,
  nodeId: number,
  byte: number,
): { target: number; replay: boolean } | undefined {
  const edgeOffset = gpuNodeWord(program, nodeId, 2);
  const edgeCount = gpuNodeWord(program, nodeId, 3);
  let lo = 0;
  let hi = edgeCount;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    const packed = program.blob[program.header[6]! + edgeOffset + mid]!;
    const edgeByte = packed & 0xff;
    if (edgeByte < byte) lo = mid + 1;
    else hi = mid;
  }
  if (lo >= edgeCount) return undefined;
  const packed = program.blob[program.header[6]! + edgeOffset + lo]!;
  if ((packed & 0xff) !== byte) return undefined;
  return {
    target: (packed >>> 8) & 0xffff,
    replay: (packed & GPU_CONSTRAINT_EDGE_FLAGS.replayByte) !== 0,
  };
}

/**
 * Feed one raw byte into a decoder-state clone. This reads the upload `blob`
 * rather than the friendly linker arrays, so it is also a serialization/ABI
 * reference for the WGSL implementation.
 */
export function feedGpuConstraintByte(
  program: GpuConstraintProgram,
  state: GpuConstraintDecoderState,
  byte: number,
): boolean {
  if (state.length !== GPU_CONSTRAINT_STATE.words) {
    fail(`decoder state has ${state.length} words; expected ${GPU_CONSTRAINT_STATE.words}`);
  }
  if (!Number.isInteger(byte) || byte < 0 || byte > 0xff) fail(`invalid byte ${byte}`);

  // Only number completion retries the same delimiter byte against its next
  // node. A hard cap protects malformed/cyclic programs and maps cleanly to a
  // bounded WGSL loop.
  for (let retry = 0; retry < 64; retry++) {
    const nodeId = state[GPU_CONSTRAINT_STATE.node]!;
    const kind = gpuNodeWord(program, nodeId, 0);
    const next = gpuNodeWord(program, nodeId, 1);

    if (kind === GPU_CONSTRAINT_NODE_KIND.accept) return false;

    if (kind === GPU_CONSTRAINT_NODE_KIND.jump) {
      gpuGoto(state, next);
      continue;
    }

    if (kind === GPU_CONSTRAINT_NODE_KIND.literal) {
      const cursor = state[GPU_CONSTRAINT_STATE.local0]!;
      const byteOffset = gpuNodeWord(program, nodeId, 2);
      const byteCount = gpuNodeWord(program, nodeId, 3);
      if (cursor >= byteCount) fail(`literal node ${nodeId} cursor ${cursor} exceeds ${byteCount}`);
      if (gpuProgramByte(program, byteOffset + cursor) !== byte) return false;
      const advanced = cursor + 1;
      if (advanced === byteCount) gpuGoto(state, next);
      else state[GPU_CONSTRAINT_STATE.local0] = advanced;
      return true;
    }

    if (kind === GPU_CONSTRAINT_NODE_KIND.switch) {
      const edge = gpuFindSwitchEdge(program, nodeId, byte);
      if (edge === undefined) return false;
      gpuGoto(state, edge.target);
      if (edge.replay) continue;
      return true;
    }

    if (kind === GPU_CONSTRAINT_NODE_KIND.string) {
      const phase = state[GPU_CONSTRAINT_STATE.local0]!;
      const length = state[GPU_CONSTRAINT_STATE.local1]!;
      const minLength = gpuNodeWord(program, nodeId, 4);
      const maxLength = gpuNodeWord(program, nodeId, 5);

      // 0=open, 1=body, 2=escape, 3=unicode
      if (phase === 0) {
        if (byte !== 0x22) return false;
        state[GPU_CONSTRAINT_STATE.local0] = 1;
        return true;
      }
      if (phase === 2) {
        if (byte === 0x75) {
          state[GPU_CONSTRAINT_STATE.local0] = 3;
          state[GPU_CONSTRAINT_STATE.local2] = 4;
          return true;
        }
        if (![0x22, 0x5c, 0x2f, 0x62, 0x66, 0x6e, 0x72, 0x74].includes(byte)) return false;
        if (length >= maxLength) return false;
        state[GPU_CONSTRAINT_STATE.local1] = length + 1;
        state[GPU_CONSTRAINT_STATE.local0] = 1;
        return true;
      }
      if (phase === 3) {
        if (!gpuIsHex(byte)) return false;
        const remaining = state[GPU_CONSTRAINT_STATE.local2]! - 1;
        state[GPU_CONSTRAINT_STATE.local2] = remaining;
        if (remaining === 0) {
          if (length >= maxLength) return false;
          state[GPU_CONSTRAINT_STATE.local1] = length + 1;
          state[GPU_CONSTRAINT_STATE.local0] = 1;
        }
        return true;
      }
      if (phase !== 1) fail(`string node ${nodeId} has invalid phase ${phase}`);

      if (byte === 0x22) {
        if (length < minLength) return false;
        gpuGoto(state, next);
        return true;
      }
      if (byte === 0x5c) {
        if (length >= maxLength) return false;
        state[GPU_CONSTRAINT_STATE.local0] = 2;
        return true;
      }
      if (byte < 0x20 || length >= maxLength) return false;
      state[GPU_CONSTRAINT_STATE.local1] = length + 1;
      return true;
    }

    if (kind === GPU_CONSTRAINT_NODE_KIND.number) {
      const length = state[GPU_CONSTRAINT_STATE.local0]!;
      const phase = state[GPU_CONSTRAINT_STATE.local1]!;
      const flags = gpuNodeWord(program, nodeId, 4);
      const maxChars = gpuNodeWord(program, nodeId, 5);
      const integer = (flags & GPU_CONSTRAINT_NUMBER_FLAGS.integer) !== 0;
      const nextPhase = length < maxChars ? gpuNumberNextPhase(phase, byte, integer) : undefined;

      if (nextPhase !== undefined) {
        setGpuNumberByte(state, length, byte);
        state[GPU_CONSTRAINT_STATE.local0] = length + 1;
        state[GPU_CONSTRAINT_STATE.local1] = nextPhase;
        if (length + 1 === maxChars && !gpuNumberComplete(program, state, nodeId)) return false;
        return true;
      }

      if (!gpuNumberComplete(program, state, nodeId)) return false;
      gpuGoto(state, next);
      continue; // retry delimiter against continuation
    }

    fail(`node ${nodeId} has unknown GPU kind ${kind}`);
  }

  fail("decoder exceeded 64 delimiter retries");
}

/** Transactionally feed bytes, committing state only when the whole payload survives. */
export function feedGpuConstraintBytes(
  program: GpuConstraintProgram,
  state: GpuConstraintDecoderState,
  bytes: Uint8Array,
): boolean {
  const candidate = cloneGpuConstraintDecoderState(state);
  for (const byte of bytes) {
    if (!feedGpuConstraintByte(program, candidate, byte)) return false;
  }
  state.set(candidate);
  return true;
}

export function gpuConstraintComplete(
  program: GpuConstraintProgram,
  state: GpuConstraintDecoderState,
): boolean {
  return state[GPU_CONSTRAINT_STATE.node] === program.acceptNode;
}


/**
 * CPU reference for the exact packed mask produced by constraint_mask.wgsl.
 * It intentionally reads the tokenizer upload blob rather than the original
 * TokenByteTableEntry[] so GPU/CPU parity also covers serialization.
 */
export function gpuConstraintMaskReference(
  program: GpuConstraintProgram,
  tokenizer: GpuConstraintTokenizer,
  state: GpuConstraintDecoderState,
): Uint32Array {
  if (state.length !== GPU_CONSTRAINT_STATE.words) {
    fail(`decoder state has ${state.length} words; expected ${GPU_CONSTRAINT_STATE.words}`);
  }

  const vocabSize = tokenizer.header[0]!;
  const eosToken = tokenizer.header[1]!;
  const entryWordOffset = tokenizer.header[2]!;
  const byteWordOffset = tokenizer.header[3]!;
  const byteLength = tokenizer.header[4]!;
  const mask = new Uint32Array(Math.ceil(vocabSize / 32));
  const complete = gpuConstraintComplete(program, state);

  const tokenizerByte = (offset: number): number => {
    if (offset >= byteLength) fail(`tokenizer byte offset ${offset} exceeds pool length ${byteLength}`);
    const word = tokenizer.blob[byteWordOffset + (offset >>> 2)]!;
    return (word >>> ((offset & 3) * 8)) & 0xff;
  };

  for (let token = 0; token < vocabSize; token++) {
    let allowed = false;
    if (complete) {
      allowed = token === eosToken;
    } else if (token !== eosToken) {
      const entry = entryWordOffset + token * GPU_CONSTRAINT_ABI.tokenizerEntryWords;
      const offset = tokenizer.blob[entry]!;
      const meta = tokenizer.blob[entry + 1]!;
      const length = meta & GPU_CONSTRAINT_TOKEN_META.lengthMask;
      const special = (meta & GPU_CONSTRAINT_TOKEN_META.special) !== 0;
      if (!special && length > 0) {
        const candidate = cloneGpuConstraintDecoderState(state);
        allowed = true;
        for (let i = 0; i < length; i++) {
          if (!feedGpuConstraintByte(program, candidate, tokenizerByte(offset + i))) {
            allowed = false;
            break;
          }
        }
      }
    }

    if (allowed) mask[token >>> 5] = (mask[token >>> 5]! | (1 << (token & 31))) >>> 0;
  }

  return mask;
}
