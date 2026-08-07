export const PARAM_BYTES = 64;
// ~250 dispatches per decode step at 256 B stride; 128 MiB covers the full
// 1024-step budget with headroom (same sizing as the main LFM2 runtime).
export const PARAM_BUFFER_BYTES = 128 * 1024 * 1024;
export const HEAD_DIM = 64;
export const KV_HEADS = 8;
export const QUERY_HEADS = 32;
export const KV_DIM = KV_HEADS * HEAD_DIM;
export const MAX_BRINGUP_CONTEXT = 1024;

export const BLOCK_EXACT_DEPTH = 2;
export const BLOCK_DEFAULT_CACHE_DEPTH = 2;
export const BLOCK_CACHE_MAX_TOKENS = 256;
export const BLOCK_BOUNDARY_HISTORY = 4;
export const BLOCK_BOUNDARY_REPAIR = 4;
export const BLOCK_TAIL_REBUILD_TOKENS = 5;
export const BLOCK_REPAIR_CAPACITY = BLOCK_BOUNDARY_HISTORY + BLOCK_BOUNDARY_REPAIR;
