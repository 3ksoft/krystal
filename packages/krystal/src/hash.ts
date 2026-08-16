/**
 * Deterministic hashing for frozen layout plans and fixture manifests.
 *
 * These hashes are compile-time compatibility guards, not cryptographic
 * digests: any change to the serialized content must change the hash so an
 * older runtime fails the version check instead of misreading buffers.
 *
 * Implementation: two independent FNV-1a 32-bit passes over the same bytes.
 * The second pass uses a different offset basis so `planHashLo` and
 * `planHashHi` do not collide trivially. JS bitwise operators are 32-bit, so
 * one FNV-1a pass is exact with plain `| 0` arithmetic.
 */

export const FNV1A_OFFSET_LO = 0x811c9dc5;
export const FNV1A_OFFSET_HI = 0x9747b28c;
const FNV1A_PRIME = 0x01000193;

export function fnv1a32(bytes: Uint8Array, seed = FNV1A_OFFSET_LO): number {
  let hash = seed | 0;
  for (let i = 0; i < bytes.length; i++) {
    hash ^= bytes[i]!;
    hash = Math.imul(hash, FNV1A_PRIME);
  }
  return hash | 0;
}

/** Little-endian u32 bytes of a non-negative integer. */
export function u32le(value: number): Uint8Array {
  const out = new Uint8Array(4);
  out[0] = value & 0xff;
  out[1] = (value >>> 8) & 0xff;
  out[2] = (value >>> 16) & 0xff;
  out[3] = (value >>> 24) & 0xff;
  return out;
}

/** Serialize a flat list of u32 values into one little-endian byte stream. */
export function u32sBytes(values: readonly number[]): Uint8Array {
  const out = new Uint8Array(values.length * 4);
  for (let i = 0; i < values.length; i++) {
    out.set(u32le(values[i]!), i * 4);
  }
  return out;
}

export interface U64Hash {
  lo: number;
  hi: number;
}

/**
 * Two independent FNV-1a passes (different offset bases) over the same byte
 * stream. `lo`/`hi` are 32-bit; callers store them in `*HashLo`/`*HashHi`.
 */
export function hashU64(bytes: Uint8Array): U64Hash {
  return {
    lo: fnv1a32(bytes, FNV1A_OFFSET_LO),
    hi: fnv1a32(bytes, FNV1A_OFFSET_HI),
  };
}

export function hashU32s(values: readonly number[]): U64Hash {
  return hashU64(u32sBytes(values));
}
