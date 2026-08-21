/**
 * The frame the GPU tests think about, built the way a host now builds one.
 *
 * The old tests reached for `buildFixtureFrame` and a compiled vocabulary —
 * both gone with the world they described. What the model actually needs is
 * what `packHostFrame` takes: records of token ids, one of them a question. So
 * the tests supply that and nothing else, which is also the point of the
 * exercise: a frame is now sized by what was sent, not by a fixed geometry.
 */
import { packHostFrame, type HostRecord } from "../../krystal/src/host/frame";
import { compileActiveFrame, type ActiveFrame } from "../../krystal/src/forward/masks";
import { BRAIN_FORWARD_CONFIG, type BrainForwardConfig } from "../../krystal/src/forward/model";
import type { v1_0_0 } from "../../schema/generated/krystal.types";

/** A tiny manifest: token id N sits in row N % 64. */
export const TEST_TOKEN_ROWS = Uint32Array.from({ length: 4096 }, (_, id) => id % 64);

export const TEST_CONFIG: BrainForwardConfig = { ...BRAIN_FORWARD_CONFIG, tokenRows: TEST_TOKEN_ROWS };

/** Project a token buffer into embedding rows, as the runner does on upload. */
export function toEmbeddingRows(tokenIds: ArrayLike<number>, table: Uint32Array = TEST_TOKEN_ROWS): Uint32Array {
  const rows = new Uint32Array(tokenIds.length);
  for (let i = 0; i < tokenIds.length; i++) rows[i] = table[tokenIds[i]!]!;
  return rows;
}

/** Two things, a feeling and a question about them. */
export const TEST_RECORDS: readonly HostRecord[] = [
  { schemaId: 1, band: 3, tokens: [10, 11, 12] }, // an apple
  { schemaId: 1, band: 3, tokens: [10, 13] }, // a stone
  { schemaId: 2, band: 2, tokens: [20, 21] }, // a comfort
  { schemaId: 9, query: true, tokens: [30] }, // what next?
];

export function testFrame(records: readonly HostRecord[] = TEST_RECORDS): {
  frame: v1_0_0.BrainFrameGpu;
  active: ActiveFrame;
  recordMask: Float32Array;
} {
  const packed = packHostFrame(records);
  return { frame: packed.gpu, active: compileActiveFrame(packed.gpu), recordMask: packed.recordMask };
}
