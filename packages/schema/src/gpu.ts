/**
 * CPU -> GPU uniform ABI used by LFM2 compute kernels.
 *
 * This is intentionally housed in @chomato/schema because it crosses the
 * host/GPU boundary. The current writer is explicit; schema-pop should replace
 * this hand-written layout together with the matching WGSL declaration.
 */
export interface Lfm2OpParams {
  inputOffset?: number;
  outputOffset?: number;
  auxOffset?: number;
  aux2Offset?: number;
  tokenCount?: number;
  inputDim?: number;
  outputDim?: number;
  rowStart?: number;
  rowCount?: number;
  layerIndex?: number;
  attentionSlot?: number;
  mode?: number;
  f0?: number;
  f1?: number;
  u0?: number;
  u1?: number;
}

/** 12 u32 + 2 f32 + 2 u32 = 64 bytes. */
export const LFM2_OP_PARAMS_BYTES = 64;

export function serializeLfm2OpParams(view: DataView, state: Lfm2OpParams, offset = 0): void {
  let p = offset;
  const u32 = (v = 0) => { view.setUint32(p, v >>> 0, true); p += 4; };
  const f32 = (v = 0) => { view.setFloat32(p, v, true); p += 4; };

  u32(state.inputOffset);
  u32(state.outputOffset);
  u32(state.auxOffset);
  u32(state.aux2Offset);
  u32(state.tokenCount);
  u32(state.inputDim);
  u32(state.outputDim);
  u32(state.rowStart);
  u32(state.rowCount);
  u32(state.layerIndex);
  u32(state.attentionSlot);
  u32(state.mode);
  f32(state.f0);
  f32(state.f1);
  u32(state.u0);
  u32(state.u1);
}
