fn arena_index(base: u32, tokenIndex: u32, dim: u32, stride: u32) -> u32 {
  return base + tokenIndex * stride + dim;
}

const WG: u32 = 64u;
// MATMUL_ROWS lives in the matmul-rows / matmul-rows-wide includes, because it
// is the one constant that differs between two programs built from the same
// matmul_wq4 body.
const ARGMAX_WG: u32 = 256u;
const MAX_CONTEXT: u32 = 1024u;
const HEAD_DIM: u32 = 64u;
const KV_HEADS: u32 = 8u;
const QUERY_HEADS: u32 = 32u;
const KV_DIM: u32 = KV_HEADS * HEAD_DIM;
