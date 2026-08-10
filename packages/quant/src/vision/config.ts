/**
 * Typed vision-tower configuration parsed from the mmproj sidecar's embedded
 * `clip.*` GGUF metadata (see docs/decisions/0009-vision-language-integration.md).
 *
 * The parse function is metadata-accessor-agnostic so the same code serves the
 * GGUF reader (mmproj source), the WQ4 sidecar (runtime path), and synthetic
 * test fixtures.
 */
export interface VisionConfig {
  /** `clip.vision.embedding_length` — tower hidden size. */
  readonly hiddenSize: number;
  /** `clip.vision.block_count` — transformer block count. */
  readonly blockCount: number;
  /** `clip.vision.attention.head_count` — attention heads. */
  readonly attentionHeads: number;
  /** hiddenSize / attentionHeads. */
  readonly headDim: number;
  /** `clip.vision.patch_size` — patch size in pixels. */
  readonly patchSize: number;
  /** `clip.vision.feed_forward_length` — MLP intermediate size. */
  readonly feedForwardSize: number;
  /** `clip.vision.attention.layer_norm_epsilon`. */
  readonly layerNormEpsilon: number;
  /** `clip.vision.projector.scale_factor` — pixel-unshuffle factor (2). */
  readonly projectorScaleFactor: number;
  /** `clip.vision.projection_dim` — projector/LM hidden size (2048). */
  readonly projectorHiddenSize: number;
  /** `clip.vision.image_mean` — per-channel normalization mean. */
  readonly imageMean: readonly number[];
  /** `clip.vision.image_std` — per-channel normalization std. */
  readonly imageStd: readonly number[];
  /** `clip.use_gelu` — MLP uses GELU (tanh-approximated) instead of SiLU. */
  readonly useGelu: boolean;
  /** Side length of the learned position-embedding grid (16). */
  readonly positionEmbeddingGrid: number;
  /** NaFlex max patch count the tower pads to (1024 for this model). */
  readonly maxPatches: number;
  /** Image placeholder token id (396); absent unless derivable/configured. */
  readonly imageTokenId?: number;
}

export interface ParseVisionConfigOptions {
  /** Position-embedding grid side; defaults to sqrt of the embedding count. */
  readonly positionEmbeddingGrid?: number;
  /** NaFlex max patch count; defaults to 1024. */
  readonly maxPatches?: number;
  /** Image placeholder token id; 396 for LFM2.5-VL. */
  readonly imageTokenId?: number;
}

function asNumber(value: unknown, key: string): number {
  if (typeof value === "number") return value;
  if (typeof value === "bigint") return Number(value);
  throw new Error(`Vision config: ${key} must be numeric, got ${String(value)}`);
}

function asNumberArray(value: unknown, key: string): number[] {
  if (!Array.isArray(value)) throw new Error(`Vision config: ${key} must be an array`);
  return value.map((item) => asNumber(item, key));
}

function asBool(value: unknown, key: string): boolean {
  if (typeof value === "boolean") return value;
  throw new Error(`Vision config: ${key} must be a boolean, got ${String(value)}`);
}

/**
 * Parse and validate the vision config from a metadata accessor. `get` must
 * throw (or return undefined) for missing keys; missing required keys are
 * reported with their names.
 */
export function parseVisionConfig(
  get: (key: string) => unknown,
  options: ParseVisionConfigOptions = {},
): VisionConfig {
  const missing: string[] = [];
  const required = (key: string): unknown => {
    let value: unknown;
    try {
      value = get(key);
    } catch {
      value = undefined;
    }
    if (value === undefined) missing.push(key);
    return value;
  };

  const hiddenSize = asNumber(required("clip.vision.embedding_length"), "clip.vision.embedding_length");
  const blockCount = asNumber(required("clip.vision.block_count"), "clip.vision.block_count");
  const attentionHeads = asNumber(required("clip.vision.attention.head_count"), "clip.vision.attention.head_count");
  const patchSize = asNumber(required("clip.vision.patch_size"), "clip.vision.patch_size");
  const feedForwardSize = asNumber(required("clip.vision.feed_forward_length"), "clip.vision.feed_forward_length");
  const layerNormEpsilon = asNumber(
    required("clip.vision.attention.layer_norm_epsilon"),
    "clip.vision.attention.layer_norm_epsilon",
  );
  const projectorScaleFactor = asNumber(
    required("clip.vision.projector.scale_factor"),
    "clip.vision.projector.scale_factor",
  );
  const projectorHiddenSize = asNumber(required("clip.vision.projection_dim"), "clip.vision.projection_dim");
  const imageMean = asNumberArray(required("clip.vision.image_mean"), "clip.vision.image_mean");
  const imageStd = asNumberArray(required("clip.vision.image_std"), "clip.vision.image_std");
  const useGelu = asBool(required("clip.use_gelu"), "clip.use_gelu");

  if (missing.length > 0) {
    throw new Error(`Vision config: missing required metadata keys: ${missing.join(", ")}`);
  }
  if (imageMean.length !== 3 || imageStd.length !== 3) {
    throw new Error("Vision config: image mean/std must have 3 channels");
  }
  if (hiddenSize % attentionHeads !== 0) {
    throw new Error(`Vision config: hidden ${hiddenSize} not divisible by heads ${attentionHeads}`);
  }
  if (projectorScaleFactor < 1 || !Number.isInteger(projectorScaleFactor)) {
    throw new Error(`Vision config: projector scale factor must be a positive integer, got ${projectorScaleFactor}`);
  }
  if (patchSize < 1 || !Number.isInteger(patchSize)) {
    throw new Error(`Vision config: patch size must be a positive integer, got ${patchSize}`);
  }

  return {
    hiddenSize,
    blockCount,
    attentionHeads,
    headDim: hiddenSize / attentionHeads,
    patchSize,
    feedForwardSize,
    layerNormEpsilon,
    projectorScaleFactor,
    projectorHiddenSize,
    imageMean,
    imageStd,
    useGelu,
    positionEmbeddingGrid: options.positionEmbeddingGrid ?? 16,
    maxPatches: options.maxPatches ?? 1024,
    ...(options.imageTokenId !== undefined ? { imageTokenId: options.imageTokenId } : {}),
  };
}

/** Derive the `<image>` placeholder id from a tokenizer token list (or 396). */
export function findImageTokenId(tokens: readonly string[], fallback = 396): number {
  const index = tokens.findIndex((token) => token === "<image>");
  return index >= 0 ? index : fallback;
}
