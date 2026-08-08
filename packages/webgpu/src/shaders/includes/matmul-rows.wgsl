// Output rows per matmul_wq4 workgroup — narrow variant.
//
// Mirrored by MATMUL_WQ4_ROWS in lfm2-layout.ts, which divides the launch
// count by it. See matmul-rows-wide.wgsl for why there are two.
const MATMUL_ROWS: u32 = 8u;
