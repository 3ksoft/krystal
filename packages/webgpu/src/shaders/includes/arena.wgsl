fn rope_component(base: u32, tokenIndex: u32, head: u32, d: u32, headCount: u32) -> f32 {
  let dim = headCount * HEAD_DIM;
  return arena[base + tokenIndex * dim + head * HEAD_DIM + d];
}
