fn arena_index(base: u32, tokenIndex: u32, dim: u32, stride: u32) -> u32 {
  return base + tokenIndex * stride + dim;
}

const WG: u32 = 64u;
/** Output rows per matmul_wq4 workgroup. Mirrored by MATMUL_WQ4_ROWS in lfm2-definition.ts. */
const MATMUL_ROWS: u32 = 8u;
const ARGMAX_WG: u32 = 256u;
const MAX_CONTEXT: u32 = 1024u;
const HEAD_DIM: u32 = 64u;
const KV_HEADS: u32 = 8u;
const QUERY_HEADS: u32 = 32u;
const KV_DIM: u32 = KV_HEADS * HEAD_DIM;
