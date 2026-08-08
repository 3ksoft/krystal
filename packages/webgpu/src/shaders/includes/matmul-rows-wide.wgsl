// Output rows per matmul_wq4 workgroup — wide variant.
//
// Mirrored by MATMUL_WQ4_ROWS_WIDE in lfm2-layout.ts.
//
// One tiling does not fit the model. Measured on an RTX 3060 with
// tests/matmul-variants.test.ts (GiB/s, rows8 vs rows16):
//
//   lm_head      2048x52428   118.3   156.5
//   ffn_gate/up  2048x8192    113.5   134.3
//   conv_in_proj 2048x6144    111.8   121.2
//   ffn_down     8192x2048    138.2   134.3
//   attn_q/out   2048x2048     85.1    76.9
//
// The split is at the output width, not the reduction length: wider tiles pay
// off until the row count stops covering the launch, and at 2048 rows they
// stop. Those first three shapes carry 68% of the model's weight bytes, which
// is what makes two programs worth their cost.
const MATMUL_ROWS: u32 = 16u;
