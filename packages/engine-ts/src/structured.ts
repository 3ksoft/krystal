import type { JsonNode, LayoutConstraintProgram } from "./index.ts";
import { compileJsonSchemaProgram } from "./json-schema-constraint.ts";
import {
  GPU_CONSTRAINT_ABI,
  GPU_CONSTRAINT_NODE_KIND,
  linkGpuConstraintProgram,
  type GpuConstraintProgram,
  type GpuConstraintProgramSummary,
} from "./gpu-constraint.ts";

/** Minimal structural contract implemented by ArkType Type values. */
export interface GeneratableSchema {
  readonly infer?: unknown;
  toJsonSchema(): unknown;
}

/** Preserve ArkType's `typeof schema.infer` without importing ArkType at runtime. */
export type InferGeneratable<S> = S extends { readonly infer: infer T } ? T : unknown;

export interface CompiledStructuredGeneration {
  readonly program: GpuConstraintProgram;
  /** Conservative upper bound for the strict JSON value, excluding EOS. */
  readonly maxJsonBytes: number;
  /** Decode budget including the final strict-root EOS token. */
  readonly maxTokens: number;
}

export function isGeneratableSchema(value: unknown): value is GeneratableSchema {
  return (
    (typeof value === "object" && value !== null) || typeof value === "function"
  ) && typeof (value as { toJsonSchema?: unknown }).toJsonSchema === "function";
}

function checkedAdd(a: number, b: number, label: string): number {
  const value = a + b;
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`[structured] ${label} exceeds the u32 decode budget`);
  }
  return value;
}

/**
 * Conservative maximum number of UTF-8/JSON bytes accepted by an acyclic
 * constraint program. This is used only to bound decode work; the VM remains
 * the source of truth for the language itself.
 *
 * NO_UNICODE_ESCAPE. Strings cost 2 bytes per maxLength unit, which is the
 * worst case once `\\uXXXX` is out of the accepted language: the string VM
 * charges one length unit per body byte, so the only form that spends more
 * bytes than units is a two-byte escape (`\\n`, `\\"`, ...). Allowing `\\uXXXX`
 * would raise the ceiling to 6 and it bought nothing — the VM's body phase
 * accepts every byte >= 0x20, raw UTF-8 included, so `é` and `\\u00e9` were two
 * spellings of a string that stays reachable either way. What is genuinely lost
 * is escaping control characters with no short form (`\\u0001`); the five that
 * matter (`\\b \\f \\n \\r \\t`) keep their escapes.
 *
 * The three implementations of this rule must agree: engine-ts/src/index.ts,
 * engine-ts/src/gpu-constraint.ts and webgpu/src/shaders/includes/constraint-vm.wgsl.
 */
export function estimateLayoutConstraintMaxBytes(program: LayoutConstraintProgram): number {
  const memo = new Map<number, number>();
  const visiting = new Set<number>();

  const visit = (id: number): number => {
    const cached = memo.get(id);
    if (cached !== undefined) return cached;
    if (visiting.has(id)) throw new Error(`[structured] constraint graph contains a cycle at node ${id}`);
    const node = program.nodes[id] as JsonNode | undefined;
    if (!node) throw new Error(`[structured] constraint node ${id} does not exist`);

    visiting.add(id);
    let value: number;
    switch (node.kind) {
      case "accept":
        value = 0;
        break;
      case "jump":
        value = visit(node.next);
        break;
      case "split": {
        let best = 0;
        for (const target of node.targets) best = Math.max(best, visit(target));
        value = best;
        break;
      }
      case "literal":
        value = checkedAdd(node.bytes.length, visit(node.next), `literal ${node.label}`);
        break;
      case "choice": {
        let longest = 0;
        for (const alternative of node.alternatives) longest = Math.max(longest, alternative.length);
        value = checkedAdd(longest, visit(node.next), `choice ${node.label}`);
        break;
      }
      case "string": {
        const encoded = checkedAdd(2, node.maxLength * 2, `string ${node.label}`);
        value = checkedAdd(encoded, visit(node.next), `string continuation ${node.label}`);
        break;
      }
      case "number":
        value = checkedAdd(node.maxChars, visit(node.next), `number ${node.label}`);
        break;
    }
    visiting.delete(id);
    memo.set(id, value);
    return value;
  };

  return visit(program.entry);
}

/** Compile one public schema into the exact upload blob and a bounded decode budget. */
export function compileStructuredGeneration(schema: GeneratableSchema): CompiledStructuredGeneration {
  const cpu = compileJsonSchemaProgram(schema.toJsonSchema());
  const program = linkGpuConstraintProgram(cpu);
  const maxJsonBytes = estimateLayoutConstraintMaxBytes(cpu);
  const maxTokens = checkedAdd(maxJsonBytes, 1, "structured generation + EOS");
  return { program, maxJsonBytes, maxTokens };
}

function countGpuKinds(nodes: Uint32Array): Omit<GpuConstraintProgramSummary, "sourceNodes" | "edges" | "byteLength" | "blobWords" | "blobBytes"> {
  const counts = {
    nodes: Math.floor(nodes.length / GPU_CONSTRAINT_ABI.nodeWords),
    literalNodes: 0,
    switchNodes: 0,
    stringNodes: 0,
    numberNodes: 0,
    acceptNodes: 0,
    jumpNodes: 0,
  };
  for (let i = 0; i < counts.nodes; i++) {
    const kind = nodes[i * GPU_CONSTRAINT_ABI.nodeWords]!;
    if (kind === GPU_CONSTRAINT_NODE_KIND.literal) counts.literalNodes++;
    else if (kind === GPU_CONSTRAINT_NODE_KIND.switch) counts.switchNodes++;
    else if (kind === GPU_CONSTRAINT_NODE_KIND.string) counts.stringNodes++;
    else if (kind === GPU_CONSTRAINT_NODE_KIND.number) counts.numberNodes++;
    else if (kind === GPU_CONSTRAINT_NODE_KIND.accept) counts.acceptNodes++;
    else if (kind === GPU_CONSTRAINT_NODE_KIND.jump) counts.jumpNodes++;
    else throw new Error(`[structured] GPU program contains unknown node kind ${kind}`);
  }
  return counts;
}

/**
 * Rehydrate the upload-only program object on the backend side. The bridge
 * transports exactly `program.blob`; no JS graph or schema crosses that seam.
 */
export function gpuConstraintProgramFromBlob(input: Uint32Array): GpuConstraintProgram {
  const blob = input.slice();
  if (blob.length < GPU_CONSTRAINT_ABI.headerWords) {
    throw new Error(`[structured] GPU constraint blob has ${blob.length} words; header needs ${GPU_CONSTRAINT_ABI.headerWords}`);
  }
  const header = blob.slice(0, GPU_CONSTRAINT_ABI.headerWords);
  if (header[0] !== GPU_CONSTRAINT_ABI.version) {
    throw new Error(`[structured] GPU constraint ABI ${header[0]} is unsupported; expected ${GPU_CONSTRAINT_ABI.version}`);
  }

  const entryNode = header[2]!;
  const acceptNode = header[3]!;
  const nodeWordOffset = header[4]!;
  const nodeCount = header[5]!;
  const edgeWordOffset = header[6]!;
  const edgeCount = header[7]!;
  const byteWordOffset = header[8]!;
  const byteLength = header[9]!;
  const nodeWords = nodeCount * GPU_CONSTRAINT_ABI.nodeWords;
  const byteWordsLength = Math.ceil(byteLength / 4);

  if (nodeWordOffset !== GPU_CONSTRAINT_ABI.headerWords) {
    throw new Error(`[structured] invalid node word offset ${nodeWordOffset}`);
  }
  if (edgeWordOffset !== nodeWordOffset + nodeWords) {
    throw new Error(`[structured] invalid edge word offset ${edgeWordOffset}`);
  }
  if (byteWordOffset !== edgeWordOffset + edgeCount) {
    throw new Error(`[structured] invalid byte word offset ${byteWordOffset}`);
  }
  const expectedWords = byteWordOffset + byteWordsLength;
  if (blob.length !== expectedWords) {
    throw new Error(`[structured] GPU constraint blob has ${blob.length} words; header describes ${expectedWords}`);
  }
  if (entryNode >= nodeCount || acceptNode >= nodeCount) {
    throw new Error(`[structured] GPU constraint entry/accept node is outside node table`);
  }

  const nodes = blob.slice(nodeWordOffset, edgeWordOffset);
  const edges = blob.slice(edgeWordOffset, byteWordOffset);
  const byteWords = blob.slice(byteWordOffset);
  const kinds = countGpuKinds(nodes);
  return {
    entryNode,
    acceptNode,
    header,
    nodes,
    edges,
    byteWords,
    blob,
    summary: {
      sourceNodes: 0,
      ...kinds,
      edges: edgeCount,
      byteLength,
      blobWords: blob.length,
      blobBytes: blob.byteLength,
    },
  };
}
