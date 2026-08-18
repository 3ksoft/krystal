// Dataset for the same-word attention-bias assay (docs/word_attention_bias.md).
//
// One visible, near, edible Apple whose record carries the identical flat token
// multiset {SOME, MUCH, RED, YELLOW} in every example. Only token-to-word
// membership differs, and it alone decides the gold intent:
//
//   MUCH with RED -> EAT      SOME with RED -> LOOK
//
// Everything a flat representation could key on is randomized per example:
// physical token positions (word members may be non-contiguous), local word
// ids, the record's slot inside the vision band, and the runtime ref.
import {
  BRAIN_FIXED_RECORDS,
  BRAIN_FRAME_BANDS,
  BRAIN_LIMITS,
  INVALID_U32,
  RECORD_FLAGS,
  TOKEN_FLAGS,
} from "../packages/schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../packages/schema/generated/krystal.types.ts";
import { PAD_TOKEN_ID } from "../packages/krystal/src/frame/packer.ts";
import { ACTION_INTENT_SCHEMA_ID } from "../packages/krystal/src/fixtures/frame.ts";
import { fixtureTokenId } from "../packages/krystal/src/fixtures/vocabulary.ts";
import { INVALID_WORD_ID } from "../packages/krystal/src/forward/masks.ts";
import { mulberry32 } from "../packages/krystal/src/forward/model.ts";
import { mix32 } from "../packages/krystal/src/bridge/comfort.ts";

const W = BRAIN_LIMITS.recordWidth;
const APPLE_SCHEMA_ID = 2;
const HOMEOSTASIS_SCHEMA_ID = 3;
const VISION = BRAIN_FRAME_BANDS.find((b) => b.kind === "vision")!;

export const CATALOG_ACTIONS = ["LOOK", "EAT", "MOVE_TOWARDS", "WAIT", "CRY", "LAUGH"] as const;
export type WordBindAction = "EAT" | "LOOK";

export interface WordBindExample {
  readonly frame: v1_0_0.BrainFrame;
  /** Frame-token index -> local word id; absent entries carry no word. */
  readonly wordIds: Record<number, number>;
  readonly gold: { readonly action: WordBindAction; readonly refToken: number };
  /** Audit surface: what a flat/positional model could try to key on. */
  readonly audit: {
    readonly slot: number;
    readonly refToken: number;
    /** Physical position (0..7) of each token inside the apple record. */
    readonly positions: Readonly<Record<string, number>>;
    readonly localWordIds: readonly number[];
    readonly tokenMultiset: readonly string[];
  };
}

function tokenMeta(roleToken: number, flags = 0): v1_0_0.BrainTokenMeta {
  return { fieldId: 0, roleToken, flags, referenceBinding: INVALID_U32 };
}

function refBinding(localTokenIndex: number, tokenId: number, generation: number): v1_0_0.BrainReferenceBinding {
  return {
    localTokenIndex, fieldId: 0, flags: 0, reserved0: 0,
    handle: { tokenId, generation, kind: "entity", status: "live" },
  };
}

interface Spec {
  slot: number;
  band: v1_0_0.BrainBandKind;
  schemaId: number;
  tokens: number[];
  refs?: v1_0_0.BrainReferenceBinding[];
  source: v1_0_0.RecordSource;
  flags: number;
}

function assemble(specs: readonly Spec[], tick: number): v1_0_0.BrainFrame {
  const records: v1_0_0.BrainRecordSlot[] = [];
  for (let slot = 0; slot < BRAIN_LIMITS.frameRecordSlots; slot++) {
    records.push({
      header: {
        schemaId: 0, band: "system", source: "runtime", flags: 0, tokenCount: 0,
        referenceCount: 0, observedAt: 0, revision: 0, primaryReference: INVALID_U32,
        continuationRecord: INVALID_U32, salience: 0, freshness: 0,
      },
      tokens: new Array<number>(W).fill(PAD_TOKEN_ID),
      tokenMeta: new Array<v1_0_0.BrainTokenMeta>(W).fill(tokenMeta(0, TOKEN_FLAGS.padding)),
      references: new Array<v1_0_0.BrainReferenceBinding>(BRAIN_LIMITS.maxReferencesPerRecord)
        .fill(refBinding(INVALID_U32, 0, 0)),
    });
  }
  let activeRecordCount = 0;
  let activeTokenCount = 0;
  for (const spec of specs) {
    const record = records[spec.slot]!;
    const tokenCount = spec.tokens.filter((t) => t !== PAD_TOKEN_ID).length;
    record.header = {
      schemaId: spec.schemaId, band: spec.band, source: spec.source, flags: spec.flags,
      tokenCount, referenceCount: spec.refs?.length ?? 0, observedAt: tick, revision: 1,
      primaryReference: spec.refs?.[0]?.handle.tokenId ?? INVALID_U32,
      continuationRecord: INVALID_U32, salience: 0.5, freshness: 1,
    };
    record.tokens = spec.tokens.slice();
    record.tokenMeta = spec.tokens.map((t) =>
      t === PAD_TOKEN_ID ? tokenMeta(0, TOKEN_FLAGS.padding) : tokenMeta(t, TOKEN_FLAGS.structural));
    if (spec.refs) record.references = spec.refs.slice();
    activeRecordCount++;
    activeTokenCount += tokenCount;
  }
  const bands = BRAIN_FRAME_BANDS.map((band) => {
    const active = specs.filter((s) => s.band === band.kind);
    return {
      kind: band.kind, activeRecords: active.length,
      activeTokens: active.reduce((n, s) => n + s.tokens.filter((t) => t !== PAD_TOKEN_ID).length, 0),
      overflowRecords: 0, truncatedRecords: 0, revision: 0, flags: 0, reserved0: 0,
    } satisfies v1_0_0.BrainBandState;
  });
  return {
    header: {
      tokenAbiVersion: 0, architectureVersion: 2, layoutVersion: 1, tick, snapshot: 1,
      activeRecordCount, activeTokenCount,
      activeQueryRecord: BRAIN_FIXED_RECORDS.primaryQuery,
      actorRecord: BRAIN_FIXED_RECORDS.actor,
      frameRevision: 1, memoryRevision: 0, intentRevision: 0, flags: 0,
    },
    bands, records,
  };
}

/** Shuffle in place with the supplied stream (Fisher-Yates). */
function shuffle<T>(items: T[], rng: () => number): T[] {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [items[i], items[j]] = [items[j]!, items[i]!];
  }
  return items;
}

export interface WordBindOptions {
  /** Scramble token-to-word membership while preserving word count and sizes
   *  (profile P2 — the randomized-membership control). */
  readonly scrambleWords?: boolean;
  /** Apply a random bijection to the local word ids (profile P3). */
  readonly bijectWordIds?: boolean;
}

/**
 * One counterfactual PAIR. Both variants share a single nuisance draw, so the
 * two frames are byte-identical — same tokens at the same physical positions,
 * same record slot, same runtime ref, same local word-id values. The only
 * difference is which word RED belongs to.
 *
 * That makes the flat representation non-predictive BY CONSTRUCTION rather
 * than merely on average: no statistic of the token tape can separate the
 * labels, because the tapes are equal. (Deriving the nuisance stream from
 * `seed` alone — never from the variant — is what buys this; seeding it with
 * `seed*2 + variant` silently correlated the variant with every draw.)
 */
export function wordBindPair(seed: number, options: WordBindOptions = {}): [WordBindExample, WordBindExample] {
  const rng = mulberry32(mix32((seed >>> 0) ^ 0x9e37_79b9) >>> 0);
  const APPLE = fixtureTokenId("APPLE");
  const RED = fixtureTokenId("RED");
  const YELLOW = fixtureTokenId("YELLOW");
  const SOME = fixtureTokenId("SOME");
  const MUCH = fixtureTokenId("MUCH");

  // Nuisance drawn ONCE per pair, before any variant is considered.
  const refToken = 0xe00 + Math.floor(rng() * 0x100);
  const slot = VISION.recordOffset + Math.floor(rng() * VISION.recordCapacity);
  const free = shuffle([2, 3, 4, 5, 6, 7], rng).slice(0, 4);
  const order = shuffle([RED, YELLOW, SOME, MUCH], rng);
  const posOf = new Map<number, number>();
  for (let i = 0; i < order.length; i++) posOf.set(order[i]!, free[i]!);
  const idPool = shuffle([3, 5, 11, 19, 27], rng).slice(0, 2);
  const localWordIds = options.bijectWordIds ? shuffle([41, 53, 61, 7, 23], rng).slice(0, 2) : idPool;

  const tokens = new Array<number>(W).fill(PAD_TOKEN_ID);
  tokens[0] = APPLE;
  tokens[1] = refToken;
  for (const [token, pos] of posOf) tokens[pos] = token;

  const signal = [fixtureTokenId("COMFORT"), fixtureTokenId("FEEL_BAD"), fixtureTokenId("SEVERE")];
  const specs: Spec[] = [
    {
      slot: BRAIN_FIXED_RECORDS.homeostasisSummary, band: "homeostasis",
      schemaId: HOMEOSTASIS_SCHEMA_ID,
      tokens: [...signal, ...new Array<number>(W - 3).fill(PAD_TOKEN_ID)],
      source: "homeostasis", flags: RECORD_FLAGS.occupied | RECORD_FLAGS.fixed,
    },
    {
      slot, band: "vision", schemaId: APPLE_SCHEMA_ID, tokens,
      refs: [refBinding(1, refToken, 1)],
      source: "sensor", flags: RECORD_FLAGS.occupied,
    },
    {
      slot: BRAIN_FIXED_RECORDS.primaryQuery, band: "query", schemaId: HOMEOSTASIS_SCHEMA_ID,
      tokens: [...signal, ...new Array<number>(W - 3).fill(PAD_TOKEN_ID)],
      source: "query", flags: RECORD_FLAGS.occupied | RECORD_FLAGS.fixed | RECORD_FLAGS.query,
    },
  ];
  for (let i = 0; i < CATALOG_ACTIONS.length; i++) {
    const token = fixtureTokenId(CATALOG_ACTIONS[i]!);
    specs.push({
      slot: BRAIN_FIXED_RECORDS.catalogBase + i, band: "catalog",
      schemaId: ACTION_INTENT_SCHEMA_ID,
      tokens: [token, ...new Array<number>(W - 1).fill(PAD_TOKEN_ID)],
      source: "creator",
      flags: RECORD_FLAGS.occupied | RECORD_FLAGS.fixed | RECORD_FLAGS.creatorAuthored,
    });
  }

  const names: Record<number, string> = { [RED]: "RED", [YELLOW]: "YELLOW", [SOME]: "SOME", [MUCH]: "MUCH" };
  const build = (variant: 0 | 1): WordBindExample => {
    // variant 0: (MUCH,RED) + (SOME,YELLOW) -> EAT
    // variant 1: (SOME,RED) + (MUCH,YELLOW) -> LOOK
    let membership: [number, number][] = variant === 0
      ? [[MUCH, 0], [RED, 0], [SOME, 1], [YELLOW, 1]]
      : [[SOME, 0], [RED, 0], [MUCH, 1], [YELLOW, 1]];
    if (options.scrambleWords) {
      // Preserve word count and sizes, destroy the binding (profile P2). The
      // scramble uses its own stream so it cannot alias with the variant.
      const scr = mulberry32(mix32(((seed >>> 0) ^ 0x5bf0_3635) + variant) >>> 0);
      const toks = shuffle([RED, YELLOW, SOME, MUCH], scr);
      membership = [[toks[0]!, 0], [toks[1]!, 0], [toks[2]!, 1], [toks[3]!, 1]];
    }
    const wordIds: Record<number, number> = {};
    for (const [token, word] of membership) wordIds[slot * W + posOf.get(token)!] = localWordIds[word]!;
    const redWord = membership.find(([t]) => t === RED)![1];
    const mates = membership.filter(([, w]) => w === redWord).map(([t]) => t);
    const action: WordBindAction = mates.includes(MUCH) ? "EAT" : "LOOK";
    return {
      frame: assemble(specs, 10),
      wordIds,
      gold: { action, refToken },
      audit: {
        slot, refToken,
        positions: Object.fromEntries([...posOf].map(([t, p]) => [names[t]!, p])),
        localWordIds,
        tokenMultiset: [...posOf.keys()].map((t) => names[t]!).sort(),
      },
    };
  };
  return [build(0), build(1)];
}

/** A balanced set: every semantic pair contributes both counterfactuals. */
export function wordBindPairs(seeds: readonly number[], options: WordBindOptions = {}): WordBindExample[] {
  return seeds.flatMap((seed) => wordBindPair(seed, options));
}
