import {} from "./env";
import { scope } from "arktype";
import { wgsl } from "@schema-pop/schema";

/**
 * Physical dispatch ABI — the layer BELOW semantic records.
 *
 * Nothing here knows what a record, a band or a relation is. This is the
 * uniform header a compute pass reads to find its tensor regions, and the only
 * contract shared by every Krystal kernel. Semantics live in
 * `krystal-engine-schema.ts`, and the separation is deliberate: a threshold or
 * a role that leaked down here would be a meaning encoded in a dispatch
 * parameter, invisible to everything that reasons about meanings.
 *
 * Excluded from the schema-pop layout build on purpose — see build.ts, whose
 * only target is the brain ABI.
 */

export const schema = scope({
  ...wgsl.import(),

  KrystalMode: "'prefill' | 'decode' | 'continuation'",

  /**
   * One dispatch's parameters, selected by dynamic uniform offset.
   *
   * The field order IS the ABI: `pass.ts` asserts the encoded record size
   * against `OP_PARAM_BYTES` and every shader indexes these by position.
   */
  OpParams: {
    inputOffset: "u32 = 0",
    outputOffset: "u32 = 0",
    auxOffset: "u32 = 0",
    aux2Offset: "u32 = 0",

    // Extended offsets for ops that move more than four tensor regions
    // (training attention reads Q/K/V/mask and writes out/P/dQ/dK/dV).
    aux3Offset: "u32 = 0",
    aux4Offset: "u32 = 0",
    aux5Offset: "u32 = 0",
    aux6Offset: "u32 = 0",

    tokenCount: "u32 = 0",
    inputDim: "u32 = 0",
    outputDim: "u32 = 0",
    rowStart: "u32 = 0",

    rowCount: "u32 = 0",
    layerIndex: "u32 = 0",
    attentionSlot: "u32 = 0",
    mode: "KrystalMode = 'prefill'",

    f0: "f32 = 0",
    f1: "f32 = 0",
    u0: "u32 = 0",
    u1: "u32 = 0",

    // Extra scalar u32s for ops that carry more than two small integers
    // (krystal_field_embed passes six embedding-table bases).
    u2: "u32 = 0",
    u3: "u32 = 0",
    u4: "u32 = 0",
    u5: "u32 = 0",
  },
});

export const $ = schema;
