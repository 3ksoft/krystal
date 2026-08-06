fn token_position(tokenIndex: u32) -> u32 {
  if (op.mode == 1u) { return runtime.position; }
  if (op.mode == 2u) { return op.u1 + tokenIndex; }
  return tokenIndex;
}
