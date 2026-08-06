fn load_f16(index: u32) -> f32 {
  let packed = weightRaw[index >> 1u];
  let pair = unpack2x16float(packed);
  return select(pair.x, pair.y, (index & 1u) != 0u);
}

fn load_wq4(index: u32) -> f32 {
  let block = index / 32u;
  let lane = index % 32u;
  let baseU32 = block * 5u;
  let packed = weightRaw[baseU32 + lane / 8u];
  let shift = (lane % 8u) * 4u;
  let expVal = bitcast<i32>(weightRaw[baseU32 + 4u]);
  return (f32((packed >> shift) & 0x0Fu) - 8.0) * exp2(f32(expVal));
}
