/**
 * A brain, written down.
 *
 * Weights are only meaningful together with two other things: the geometry they
 * were shaped for, and the token→row mapping they learned their meanings under.
 * A blob that carries neither can be loaded into the wrong brain and will not
 * fail — it will simply denote something else, quietly, for the rest of that
 * creature's life. So both ride along and both are CHECKED, and a mismatch is
 * refused rather than coerced.
 *
 * The layout is the order `createBrainForwardWeights` builds them in, which is
 * the one order both directions walk. There is no per-array name in the file:
 * the geometry in the header already determines every length, so a name would
 * be a second source of truth for something the header has to get right anyway.
 */
import type { BrainForwardConfig, BrainForwardWeights } from "../forward/model.ts";

const MAGIC = 0x4b525931; // "KRY1"
const VERSION = 1;
/** magic, version, 7 geometry numbers, tokenRows length, weight element count. */
const HEADER_WORDS = 11;

/** Every weight array, in the one order both directions walk. */
function arraysOf(weights: BrainForwardWeights): Float32Array[] {
  const out: Float32Array[] = [weights.embeddings];
  for (const block of weights.enc) out.push(block.wq, block.wk, block.wv, block.w1, block.w2);
  out.push(weights.pool);
  for (const block of weights.mixer) out.push(block.wq, block.wk, block.wv, block.w1, block.w2);
  out.push(weights.selector.wq, weights.selector.wk, weights.decisionHeadWh, weights.valueHeadWv);
  return out;
}

const geometryOf = (config: BrainForwardConfig): number[] => [
  config.hiddenSize,
  config.ffnSize,
  config.encoderBlocks,
  config.mixerBlocks,
  config.headCount,
  config.headDim,
  config.routeKindCount,
];

/** Why a checkpoint was refused, or null when it fits. */
export type CheckpointRefusal =
  | "not a krystal checkpoint"
  | "a later checkpoint format"
  | "a different geometry"
  | "a different vocabulary"
  | "truncated";

export function encodeCheckpoint(
  weights: BrainForwardWeights,
  config: BrainForwardConfig,
  tokenRows: Uint32Array,
): Uint8Array {
  const arrays = arraysOf(weights);
  const elements = arrays.reduce((total, array) => total + array.length, 0);
  const words = HEADER_WORDS + tokenRows.length + elements;
  const buffer = new ArrayBuffer(words * 4);
  const header = new Uint32Array(buffer, 0, HEADER_WORDS);
  header.set([MAGIC, VERSION, ...geometryOf(config), tokenRows.length, elements]);
  new Uint32Array(buffer, HEADER_WORDS * 4, tokenRows.length).set(tokenRows);
  const floats = new Float32Array(buffer, (HEADER_WORDS + tokenRows.length) * 4);
  let cursor = 0;
  for (const array of arrays) {
    floats.set(array, cursor);
    cursor += array.length;
  }
  return new Uint8Array(buffer);
}

/**
 * Read a checkpoint into weights that already exist.
 *
 * In place rather than returning fresh arrays, because a session hands out its
 * weights: replacing them would leave anything holding the old ones training a
 * brain nobody is thinking with.
 */
export function decodeCheckpoint(
  bytes: Uint8Array,
  weights: BrainForwardWeights,
  config: BrainForwardConfig,
  tokenRows: Uint32Array,
): CheckpointRefusal | null {
  if (bytes.byteLength < HEADER_WORDS * 4) return "truncated";
  // A copy when the view is not word-aligned: a Uint8Array from a file or a
  // network read may start anywhere, and a typed-array view over it would
  // throw rather than read the numbers that are plainly there.
  const aligned =
    bytes.byteOffset % 4 === 0 ? bytes : new Uint8Array(bytes.slice().buffer);
  const header = new Uint32Array(aligned.buffer, aligned.byteOffset, HEADER_WORDS);
  if (header[0] !== MAGIC) return "not a krystal checkpoint";
  if (header[1]! > VERSION) return "a later checkpoint format";
  const geometry = geometryOf(config);
  for (let index = 0; index < geometry.length; index++)
    if (header[2 + index] !== geometry[index]) return "a different geometry";
  const rowCount = header[9]!;
  const elements = header[10]!;
  if (rowCount !== tokenRows.length) return "a different vocabulary";
  const words = HEADER_WORDS + rowCount + elements;
  if (aligned.byteLength < words * 4) return "truncated";

  const rows = new Uint32Array(aligned.buffer, aligned.byteOffset + HEADER_WORDS * 4, rowCount);
  for (let index = 0; index < rowCount; index++)
    if (rows[index] !== tokenRows[index]) return "a different vocabulary";

  const arrays = arraysOf(weights);
  if (arrays.reduce((total, array) => total + array.length, 0) !== elements) return "a different geometry";
  const floats = new Float32Array(aligned.buffer, aligned.byteOffset + (HEADER_WORDS + rowCount) * 4, elements);
  let cursor = 0;
  for (const array of arrays) {
    array.set(floats.subarray(cursor, cursor + array.length));
    cursor += array.length;
  }
  return null;
}
