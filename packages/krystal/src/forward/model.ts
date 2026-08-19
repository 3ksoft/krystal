/**
 * Krystal brain forward model graph (M2b).
 *
 * This module owns the model-graph spec that packages/krystal is responsible
 * for (concerns answer 30): dimensions, the concatenated embedding-table
 * layout, per-block weight pages and the deterministic initialization
 * convention (answer 23). The GPU runner in packages/webgpu validates against
 * this spec and dispatches the same ops; the CPU oracle implements the same
 * math for parity.
 *
 * First profile (answers 9/10/11/13/14):
 *   - hidden size 128, 4 full heads x 32, FFN 384;
 *   - 2 shared local record/query encoder blocks, 2 query-to-record mixer
 *     blocks; plain bidirectional self-attention + ReLU FFN, no ShortConv;
 *   - record-local position embeddings 0..7, no RoPE, no E_recordIndex;
 *   - shared record/query encoder weights; E_stream distinguishes roles.
 *
 * Weight layout (all matrices stored [outDim, inDim] row-major, matching
 * matmul_f32's y = x @ W^T):
 *   embeddings  one page, six concatenated tables (see EMBEDDING_TABLES)
 *   enc[b]      { wq, wk, wv [H,H], w1 [FFN,H], w2 [H,FFN] } per block
 *   pool        [2, H] qk row 0, qv row 1
 *   mixer[b]    same block shape as enc[b]
 */
import { BRAIN_LIMITS, KRYSTAL_ABI } from "../../../schema/src/krystal-engine-schema.ts";
import { FIXTURE_TOKEN_ROWS } from "../fixtures/vocabulary.ts";

export interface BrainForwardConfig {
  readonly hiddenSize: number;
  readonly headCount: number;
  readonly headDim: number;
  readonly ffnSize: number;
  readonly encoderBlocks: number;
  readonly mixerBlocks: number;
  /** Embedding table row counts (see BRAIN_FORWARD_CONFIG for the projection). */
  readonly tokenSpace: number;
  readonly fieldSpace: number;
  readonly schemaSpace: number;
  readonly bandSpace: number;
  readonly streamSpace: number;
  readonly posSpace: number;
  /** Typed decision head class count (route kinds, architecture v2 §9). */
  readonly routeKindCount: number;
  /**
   * Token id -> embedding row, for this profile's vocabulary.
   *
   * Carried here rather than reached for as a module-level constant, because
   * the mapping is a property of the vocabulary an agent was compiled against.
   * A global made the forward pass silently correct for exactly one world: any
   * other grammar assigns different rows, so the same token would train a
   * different vector with nothing to signal the mismatch.
   */
  readonly tokenRows: Uint32Array;
}

export const BRAIN_FORWARD_CONFIG: BrainForwardConfig = {
  hiddenSize: 128,
  headCount: 4,
  headDim: 32,
  ffnSize: 384,
  encoderBlocks: 2,
  mixerBlocks: 2,
  // Embedding rows, NOT the id space: semantic symbols occupy one row each at
  // their manifest index, and the whole reference half folds into a shared pool
  // of `refEmbeddingRows`. Indexing by id instead would cost 0x8000 rows to
  // carry a few hundred live symbols.
  tokenSpace: KRYSTAL_ABI.semanticEmbeddingRows + KRYSTAL_ABI.refEmbeddingRows,
  // Field roles are ordinary semantic tokens, so they project through the same
  // table and need the same capacity.
  fieldSpace: KRYSTAL_ABI.semanticEmbeddingRows + KRYSTAL_ABI.refEmbeddingRows,
  schemaSpace: 0x100, // maxRecordSchemas
  bandSpace: BRAIN_LIMITS.frameBands, // one embedding row per frame band
  streamSpace: 2, // record / query
  posSpace: BRAIN_LIMITS.recordWidth, // learned record-local positions
  routeKindCount: 4, // DIRECT / ACTION / ALU / NONE (provisional fixture set)
  // This constant is the FIXTURE profile, and naming its vocabulary here says
  // so. An agent built from a simulation grammar passes its own
  // `CompiledGrammar.tokenRows` instead.
  tokenRows: FIXTURE_TOKEN_ROWS,
};

export interface EmbeddingTableLayout {
  readonly name: string;
  readonly rows: number;
}

/**
 * Table order and row counts of the concatenated embeddings page, derived from
 * the forward config so the two cannot disagree: a mismatch surfaces only as an
 * "embedding row N outside table K" failure inside a training step.
 */
export const EMBEDDING_TABLES: readonly EmbeddingTableLayout[] = [
  { name: "token", rows: BRAIN_FORWARD_CONFIG.tokenSpace },
  { name: "field", rows: BRAIN_FORWARD_CONFIG.fieldSpace },
  { name: "schema", rows: BRAIN_FORWARD_CONFIG.schemaSpace },
  { name: "band", rows: BRAIN_FORWARD_CONFIG.bandSpace },
  { name: "stream", rows: BRAIN_FORWARD_CONFIG.streamSpace },
  { name: "pos", rows: BRAIN_FORWARD_CONFIG.posSpace },
] as const;

/** Base word offset of each table inside the concatenated embeddings page. */
export function embeddingTableBases(config: BrainForwardConfig): {
  token: number;
  field: number;
  schema: number;
  band: number;
  stream: number;
  pos: number;
} {
  const bases: number[] = [];
  let cursor = 0;
  for (const table of EMBEDDING_TABLES) {
    bases.push(cursor);
    cursor += table.rows * config.hiddenSize;
  }
  return {
    token: bases[0]!,
    field: bases[1]!,
    schema: bases[2]!,
    band: bases[3]!,
    stream: bases[4]!,
    pos: bases[5]!,
  };
}

/** Total f32 elements of the concatenated embeddings page. */
export function embeddingsPageElements(config: BrainForwardConfig): number {
  return EMBEDDING_TABLES.reduce((sum, table) => sum + table.rows * config.hiddenSize, 0);
}

export interface BlockWeights {
  readonly wq: Float32Array; // [H,H]
  readonly wk: Float32Array; // [H,H]
  readonly wv: Float32Array; // [H,H]
  readonly w1: Float32Array; // [FFN,H]
  readonly w2: Float32Array; // [H,FFN]
}

export interface SelectorWeights {
  /** Query projection [H,H] and key projection [H,H] (answer 26 scoring). */
  readonly wq: Float32Array;
  readonly wk: Float32Array;
}

export interface BrainForwardWeights {
  /** Concatenated embedding tables (see EMBEDDING_TABLES + embeddingTableBases). */
  readonly embeddings: Float32Array;
  readonly enc: readonly BlockWeights[];
  /** Pool queries [2, H]: key row 0, value row 1. */
  readonly pool: Float32Array;
  readonly mixer: readonly BlockWeights[];
  /**
   * Shared selector projections for catalog selection (concerns answer 26:
   * score = dot(Wq*query, Wk*key)/sqrt(H)). The first forward shares one pair
   * across selector slots; per-slot projections are a later ablation.
   */
  readonly selector: SelectorWeights;
  /**
   * Typed decision head weights [routeKindCount, 3H]: logits over route kinds
   * from the concatenated gathered context (query output + intent gather +
   * argument gather; architecture v2 §12.9). Row-major like matmul weights.
   */
  readonly decisionHeadWh: Float32Array;
  /**
   * Value head [1, 3H]: predicted change in valence for the next tick, from the
   * same concatenated context the decision head reads.
   *
   * Structurally it is a decision head with a single class, so it reuses that
   * forward and backward exactly; only the loss differs — squared error against
   * an observed number rather than cross-entropy against a label.
   *
   * It reads the 3H context rather than the query output alone so that the
   * prediction MAY condition on the intent and argument about to be chosen,
   * without being forced to. Whether the creature actually learns to connect
   * "eat that" with "things improve" is the question worth asking, and wiring
   * the connection in by hand would answer it in advance.
   *
   * The target needs no labelling: next tick's valence is simply observed. That
   * is what makes live play in the simulation trainable at all once the gold
   * curriculum stops.
   */
  readonly valueHeadWv: Float32Array;
}

/** Deterministic mulberry32 PRNG (same helper as the training tests). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function xavierUniform(out: number, input: number, rand: () => number): Float32Array {
  const limit = Math.sqrt(6 / (out + input));
  const values = new Float32Array(out * input);
  for (let i = 0; i < values.length; i++) values[i] = (rand() * 2 - 1) * limit;
  return values;
}

function normalTable(rows: number, h: number, rand: () => number, std: number): Float32Array {
  const values = new Float32Array(rows * h);
  // Box-Muller; deterministic from the seeded PRNG.
  for (let i = 0; i < values.length; i += 2) {
    const u1 = Math.max(rand(), 1e-12);
    const u2 = rand();
    const r = Math.sqrt(-2 * Math.log(u1));
    values[i] = r * Math.cos(2 * Math.PI * u2) * std;
    if (i + 1 < values.length) values[i + 1] = r * Math.sin(2 * Math.PI * u2) * std;
  }
  return values;
}

function createBlockWeights(h: number, ffn: number, rand: () => number): BlockWeights {
  return {
    wq: xavierUniform(h, h, rand),
    wk: xavierUniform(h, h, rand),
    wv: xavierUniform(h, h, rand),
    w1: xavierUniform(ffn, h, rand),
    w2: xavierUniform(h, ffn, rand),
  };
}

/**
 * Deterministic weight initialization (answer 23): Xavier/Glorot uniform for
 * matrices, normal std=0.02 for the token/metadata embeddings, zero biases
 * (this graph has no bias terms). Generate once on the host and upload the
 * identical arrays to CPU/GPU tests.
 */
export function createBrainForwardWeights(
  config: BrainForwardConfig = BRAIN_FORWARD_CONFIG,
  seed = 1337,
): BrainForwardWeights {
  const rand = mulberry32(seed);
  const { hiddenSize: h, ffnSize: ffn, encoderBlocks, mixerBlocks } = config;

  const embeddings = new Float32Array(embeddingsPageElements(config));
  let cursor = 0;
  for (const table of EMBEDDING_TABLES) {
    const values = normalTable(table.rows, h, rand, 0.02);
    embeddings.set(values, cursor);
    cursor += values.length;
  }

  const enc: BlockWeights[] = [];
  for (let b = 0; b < encoderBlocks; b++) enc.push(createBlockWeights(h, ffn, rand));

  const pool = normalTable(2, h, rand, 0.02);

  const mixer: BlockWeights[] = [];
  for (let b = 0; b < mixerBlocks; b++) mixer.push(createBlockWeights(h, ffn, rand));

  return {
    embeddings,
    enc,
    pool,
    mixer,
    selector: { wq: xavierUniform(h, h, rand), wk: xavierUniform(h, h, rand) },
    decisionHeadWh: xavierUniform(config.routeKindCount, 3 * h, rand),
    valueHeadWv: xavierUniform(1, 3 * h, rand),
  };
}

/** Query stream id (E_stream distinguishes record vs query encoder roles). */
export const STREAM_RECORD = 0;
export const STREAM_QUERY = 1;

export function validateBrainForwardWeights(
  config: BrainForwardConfig,
  weights: BrainForwardWeights,
): void {
  const { hiddenSize: h, ffnSize: ffn, encoderBlocks, mixerBlocks } = config;
  const bases = embeddingTableBases(config);
  const tokenEnd = bases.token + config.tokenSpace * h;
  const fieldEnd = bases.field + config.fieldSpace * h;
  const schemaEnd = bases.schema + config.schemaSpace * h;
  const bandEnd = bases.band + config.bandSpace * h;
  const streamEnd = bases.stream + config.streamSpace * h;
  const posEnd = bases.pos + config.posSpace * h;
  const require = (condition: boolean, message: string): void => {
    if (!condition) throw new Error(`BrainForwardWeights: ${message}`);
  };
  require(weights.embeddings.length === embeddingsPageElements(config), "embeddings page size");
  require(tokenEnd <= weights.embeddings.length, "token table overflow");
  require(fieldEnd <= weights.embeddings.length, "field table overflow");
  require(schemaEnd <= weights.embeddings.length, "schema table overflow");
  require(bandEnd <= weights.embeddings.length, "band table overflow");
  require(streamEnd <= weights.embeddings.length, "stream table overflow");
  require(posEnd <= weights.embeddings.length, "pos table overflow");
  require(weights.enc.length === encoderBlocks, "encoder block count");
  require(weights.mixer.length === mixerBlocks, "mixer block count");
  require(weights.pool.length === 2 * h, "pool query page size");
  for (const [label, blocks] of [
    ["enc", weights.enc],
    ["mixer", weights.mixer],
  ] as const) {
    for (const block of blocks) {
      require(block.wq.length === h * h, `${label}.wq size`);
      require(block.wk.length === h * h, `${label}.wk size`);
      require(block.wv.length === h * h, `${label}.wv size`);
      require(block.w1.length === ffn * h, `${label}.w1 size`);
      require(block.w2.length === h * ffn, `${label}.w2 size`);
    }
  }
  require(weights.selector.wq.length === h * h, "selector.wq size");
  require(weights.selector.wk.length === h * h, "selector.wk size");
  require(
    weights.decisionHeadWh.length === config.routeKindCount * 3 * h,
    "decision head weight size",
  );
  require(weights.valueHeadWv.length === 3 * h, "value head weight size");
}

/** Typed brain-stream id list per record slot (from band id). */
export function compileStreamIds(
  bandIds: readonly number[],
  queryBandIndex: number,
): Uint32Array {
  const streamIds = new Uint32Array(bandIds.length);
  for (let slot = 0; slot < bandIds.length; slot++) {
    streamIds[slot] = bandIds[slot] === queryBandIndex ? STREAM_QUERY : STREAM_RECORD;
  }
  return streamIds;
}

