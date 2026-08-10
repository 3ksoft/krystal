/**
 * Host-side image processor for the LFM2.5-VL v0 envelope (ADA-0009).
 *
 * v0 contract: one image, fit into `targetSize` preserving aspect ratio,
 * dims rounded to patch multiples, center-padded with zeros (zeros are the
 * normalized mean, so padding is neutral), patchified into `(channel, h, w)`
 * w-fastest vectors, and the learned position embeddings resized bilinearly
 * (align_corners=False, matching `F.interpolate`) to the image grid.
 *
 * Input is decoded RGBA (byte decode is a caller concern: canvas /
 * ImageDecoder in the browser, an image library in headless runs). No model
 * weights are needed here except the position-embedding base grid.
 */

export interface VisionProcessorConfig {
  readonly hiddenSize: number;
  readonly patchSize: number;
  readonly imageMean: readonly number[];
  readonly imageStd: readonly number[];
  /** Side of the learned position-embedding grid (16 for this model). */
  readonly positionEmbeddingGrid: number;
  /** NaFlex patch capacity the output is padded to (1024). */
  readonly maxPatches: number;
  /** Pixel-unshuffle factor; fit grid aligned to patchSize*factor (default 1). */
  readonly projectorScaleFactor?: number;
}

export interface EncodeImageOptions {
  /** Square target the image is fit into; default 512. */
  readonly targetSize?: number;
}

export interface ProcessedImage {
  /** [maxPatches, 3*patchSize^2] patch tokens (padded with zeros). */
  readonly patches: Float32Array;
  /** [maxPatches, hiddenSize] resized position embeddings (padded with zeros). */
  readonly posEmb: Float32Array;
  /** Number of valid (non-padding) patches. */
  readonly patchCount: number;
  /** Patch grid of the resized (pre-pad) image. */
  readonly gridH: number;
  readonly gridW: number;
  /** [maxPatches], 1 = valid. */
  readonly paddingMask: Uint8Array;
  /** Resized (pre-pad) pixel dimensions. */
  readonly imageW: number;
  readonly imageH: number;
}

/**
 * Fit `(w, h)` into `target` keeping aspect ratio, dims rounded to multiples
 * of `align` (patchSize for the grid, patchSize*factor for the unshuffle tail).
 */
export function fitSize(width: number, height: number, target: number, align: number): { w: number; h: number } {
  const scale = Math.min(target / width, target / height);
  const round = (value: number) => {
    const patches = Math.round(value / align);
    return Math.min(target, Math.max(align, patches * align));
  };
  return { w: round(scale * width), h: round(scale * height) };
}

/**
 * Bilinear resize (align_corners=False, same as `F.interpolate(mode="bilinear")`)
 * with normalization into `out` ([th*tw*3] floats, RGB).
 */
export function resizeBilinearToRgb(
  rgba: Uint8Array,
  width: number,
  height: number,
  tw: number,
  th: number,
  mean: readonly number[],
  std: readonly number[],
  out: Float32Array,
): void {
  if (rgba.length < width * height * 4) throw new Error("resizeBilinearToRgb: RGBA buffer too small");
  if (out.length < tw * th * 3) throw new Error("resizeBilinearToRgb: output buffer too small");
  const scaleX = width / tw;
  const scaleY = height / th;

  for (let y = 0; y < th; y++) {
    // torch clamps the source coordinate to 0 instead of extrapolating
    // (area_pixel_compute_source_index), so must we.
    let srcY = (y + 0.5) * scaleY - 0.5;
    if (srcY < 0) srcY = 0;
    const y0 = Math.floor(srcY);
    const y1 = Math.min(height - 1, y0 + 1);
    const fy = srcY - y0;
    const rowBase = y * tw * 3;
    for (let x = 0; x < tw; x++) {
      let srcX = (x + 0.5) * scaleX - 0.5;
      if (srcX < 0) srcX = 0;
      const x0 = Math.floor(srcX);
      const x1 = Math.min(width - 1, x0 + 1);
      const fx = srcX - x0;
      const p00 = (y0 * width + x0) * 4;
      const p01 = (y0 * width + x1) * 4;
      const p10 = (y1 * width + x0) * 4;
      const p11 = (y1 * width + x1) * 4;
      const outBase = rowBase + x * 3;
      for (let c = 0; c < 3; c++) {
        const v00 = rgba[p00 + c]!;
        const v01 = rgba[p01 + c]!;
        const v10 = rgba[p10 + c]!;
        const v11 = rgba[p11 + c]!;
        const top = v00 + (v01 - v00) * fx;
        const bottom = v10 + (v11 - v10) * fx;
        const value = (top + (bottom - top) * fy) / 255;
        out[outBase + c] = (value - mean[c]!) / std[c]!;
      }
    }
  }
}

/**
 * Patchify a normalized RGB buffer `[gridH*16*gridW*16*3]` into patch tokens
 * `[gridH*gridW, 3*patchSize^2]` in `(channel, h, w)` order with `w` fastest
 * (matching the flattening of the `v.patch_embd` conv kernel).
 */
export function patchify(rgb: Float32Array, gridH: number, gridW: number, patchSize: number): Float32Array {
  const patchDim = 3 * patchSize * patchSize;
  const out = new Float32Array(gridH * gridW * patchDim);
  const pixelsPerRow = gridW * patchSize;
  for (let py = 0; py < gridH; py++) {
    for (let px = 0; px < gridW; px++) {
      const patchBase = (py * gridW + px) * patchDim;
      for (let c = 0; c < 3; c++) {
        for (let i = 0; i < patchSize; i++) {
          for (let j = 0; j < patchSize; j++) {
            const rgbIndex = ((py * patchSize + i) * pixelsPerRow + (px * patchSize + j)) * 3 + c;
            const outIndex = patchBase + (c * patchSize * patchSize + i * patchSize + j);
            out[outIndex] = rgb[rgbIndex]!;
          }
        }
      }
    }
  }
  return out;
}

/**
 * Resize a square learned position-embedding grid `[baseSide^2, hidden]` to a
 * `gridH x gridW` grid using bilinear interpolation with align_corners=False.
 * Out-of-range source coordinates are clamped (upsampling only in practice).
 */
export function resizePositionEmbedding(
  posEmbBase: Float32Array,
  baseSide: number,
  gridH: number,
  gridW: number,
  hidden: number,
): Float32Array {
  if (posEmbBase.length < baseSide * baseSide * hidden) {
    throw new Error("resizePositionEmbedding: base grid buffer too small");
  }
  const out = new Float32Array(gridH * gridW * hidden);
  const scaleY = baseSide / gridH;
  const scaleX = baseSide / gridW;

  for (let y = 0; y < gridH; y++) {
    // Same torch-compatible source clamping as resizeBilinearToRgb.
    let srcY = (y + 0.5) * scaleY - 0.5;
    if (srcY < 0) srcY = 0;
    const y0 = Math.floor(srcY);
    const y1 = Math.min(baseSide - 1, y0 + 1);
    const fy = srcY - y0;
    const rowBase = y * gridW * hidden;
    for (let x = 0; x < gridW; x++) {
      let srcX = (x + 0.5) * scaleX - 0.5;
      if (srcX < 0) srcX = 0;
      const x0 = Math.floor(srcX);
      const x1 = Math.min(baseSide - 1, x0 + 1);
      const fx = srcX - x0;
      const base = rowBase + x * hidden;
      const b00 = (y0 * baseSide + x0) * hidden;
      const b01 = (y0 * baseSide + x1) * hidden;
      const b10 = (y1 * baseSide + x0) * hidden;
      const b11 = (y1 * baseSide + x1) * hidden;
      for (let d = 0; d < hidden; d++) {
        const v00 = posEmbBase[b00 + d]!;
        const v01 = posEmbBase[b01 + d]!;
        const v10 = posEmbBase[b10 + d]!;
        const v11 = posEmbBase[b11 + d]!;
        const top = v00 + (v01 - v00) * fx;
        const bottom = v10 + (v11 - v10) * fx;
        out[base + d] = top + (bottom - top) * fy;
      }
    }
  }
  return out;
}

/** Full v0 encode: fit → resize+normalize → patchify → pos-emb resize → pad. */
export function encodeImage(
  rgba: Uint8Array,
  width: number,
  height: number,
  posEmbBase: Float32Array,
  config: VisionProcessorConfig,
  options: EncodeImageOptions = {},
): ProcessedImage {
  const target = options.targetSize ?? 512;
  const { hiddenSize, patchSize, imageMean, imageStd, positionEmbeddingGrid, maxPatches } = config;
  // Align to patch*factor: the pixel-unshuffle tail needs an even grid, and
  // llama.cpp's LFM2 preprocessor aligns to patch_size * n_merge as well.
  const { w, h } = fitSize(width, height, target, patchSize * (config.projectorScaleFactor ?? 1));
  const gridW = w / patchSize;
  const gridH = h / patchSize;
  const patchCount = gridW * gridH;
  if (patchCount > maxPatches) throw new Error(`encodeImage: ${gridH}x${gridW} grid exceeds maxPatches ${maxPatches}`);

  const rgb = new Float32Array(w * h * 3);
  resizeBilinearToRgb(rgba, width, height, w, h, imageMean, imageStd, rgb);

  const patchDim = 3 * patchSize * patchSize;
  const patches = new Float32Array(maxPatches * patchDim);
  const rawPatches = patchify(rgb, gridH, gridW, patchSize);
  patches.set(rawPatches);

  const posEmb = new Float32Array(maxPatches * hiddenSize);
  const rawPosEmb = resizePositionEmbedding(posEmbBase, positionEmbeddingGrid, gridH, gridW, hiddenSize);
  posEmb.set(rawPosEmb);
  // Padded positions get zero pos-emb (HF repeats the first embedding there).
  // The difference has no effect on output: padded patches are masked in
  // attention and never reach the pixel-unshuffle/projector.

  const paddingMask = new Uint8Array(maxPatches);
  paddingMask.fill(1, 0, patchCount);

  return { patches, posEmb, patchCount, gridH, gridW, paddingMask, imageW: w, imageH: h };
}
