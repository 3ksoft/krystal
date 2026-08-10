/**
 * M3 — image → chat integration (ADA-0009).
 *
 * `VisionLfm2Session` composes the existing pieces into the v0 VL envelope:
 *
 *   image RGBA → encodeImage (fit/resize/normalize/patchify + pos-emb)
 *             → VisionTower.run (27-block SigLIP2 tower + pixel-unshuffle
 *               projector) → [imageTokens x 2048] image embeddings
 *   prompt with the `<image>` marker → tokenizer → token sequence
 *             → each marker expanded to `imageTokens` copies of the
 *               placeholder token (396)
 *   combined sequence → Lfm2Forward.generateWithImageEmbeddings — embed the
 *             whole sequence, overwrite the placeholder rows in hiddenA with
 *             the tower embeddings, then run the existing LFM2 layer stack
 *             over the combined sequence (image tokens are ordinary sequence
 *             positions; causal attention and RoPE need no changes).
 *
 * The tokenizer is built from the text WQ4 metadata with a tolerant adapter:
 * the VL GGUF conversion omits `tokenizer.ggml.add_eos_token`, which the
 * strict WQ4 metadata accessor rejects. `addEosByDefault` falls back to false
 * (LFM2 does not append EOS), matching the reference text path.
 *
 * Vision weights come from whichever source is handed in, detected by file
 * magic: the exact-F32 load of the F16 mmproj GGUF (the M2 differential
 * oracle path) or the WQ4 sidecar `LFM2.5-VL-mmproj-WQ4.wq4` (M1 conversion;
 * host-dequant in the shared vision loader, raw-F16 `ffn_down` pass-through).
 * The tower and the LM must share one GPUDevice — both run on the global
 * Sandblaster definition's device via Lfm2Forward/VisionTower.
 */
import type { RandomAccessSource } from "../../../quant/src/gguf/source.ts";
import { GgufReader } from "../../../quant/src/gguf/reader.ts";
import { WQ4_MAGIC, Wq4Reader } from "../../../quant/src/wq4/reader.ts";
import { Lfm2Tokenizer } from "../../../lfm2/src/tokenizer.ts";
import { Lfm2GpuModel } from "../model.ts";
import { Lfm2Forward, type Lfm2Sampling } from "../forward.ts";
import { lfm2 } from "../lfm2.ts";
import {
  parseVisionConfig,
  type VisionConfig,
} from "../../../quant/src/vision/config.ts";
import { loadVisionWeights, visionTensorSource } from "../../../quant/src/vision/weights.ts";
import type { VisionReferenceWeights } from "../../../quant/src/vision/reference.ts";
import { encodeImage, type ProcessedImage } from "./processor.ts";
import { VisionTower } from "./tower.ts";

export interface VisionLfm2SessionOptions {
  readonly device: GPUDevice;
  /** WQ4 text backbone source (models/LFM2.5-VL-1.6B-WQ4.wq4). */
  readonly textSource: RandomAccessSource;
  /**
   * Vision tower + projector source: the F16 GGUF
   * (models/mmproj-LFM2.5-VL-1.6b-F16.gguf) or the WQ4 sidecar
   * (models/LFM2.5-VL-mmproj-WQ4.wq4); format is auto-detected by magic.
   */
  readonly visionSource: RandomAccessSource;
  /** Literal image placeholder in prompts; must map to a special token (default "<image>"). */
  readonly imageMarker?: string;
  /** Square target the image is fit into before patching; default 512 (v0 envelope). */
  readonly targetSize?: number;
}

export interface VisionChatRequest {
  /** Decoded RGBA pixels (byte decode is a caller concern). */
  readonly rgba: Uint8Array;
  readonly width: number;
  readonly height: number;
  /** Prompt text; may contain the image marker. Prepended if absent. */
  readonly prompt: string;
  readonly system?: string;
  readonly maxNewTokens?: number;
  readonly sampling?: Lfm2Sampling;
}

export interface VisionChatResult {
  /** Decoded generated text (special tokens skipped). */
  readonly text: string;
  readonly tokens: number[];
  /** Tower output token count (expansion factor for the placeholder). */
  readonly imageTokens: number;
  /** First expanded placeholder position in the combined prompt sequence. */
  readonly imageStart: number;
  /** Processed (fit + rounded) patch grid of the image. */
  readonly grid: { readonly w: number; readonly h: number };
}

/**
 * Open the vision source as either a GGUF (F16 mmproj) or a WQ4 v3 sidecar,
 * detected by the 4-byte file magic, and load the typed vision config +
 * F32-dequantized weights from it. The decoded weight layout is identical for
 * both formats (see the weights module doc), so nothing downstream needs to
 * know which source was used.
 */
async function openVisionWeights(
  source: RandomAccessSource,
): Promise<{ visionConfig: VisionConfig; visionWeights: VisionReferenceWeights }> {
  const head = await source.read(0, 4);
  const magic = new DataView(head.buffer, head.byteOffset, 4).getUint32(0, true);
  if (magic === WQ4_MAGIC) {
    const reader = await Wq4Reader.open(source);
    const visionConfig = parseVisionConfig((key) => reader.metadataValue(key));
    const visionWeights = await loadVisionWeights(visionTensorSource(reader), visionConfig);
    return { visionConfig, visionWeights };
  }
  const reader = await GgufReader.open(source);
  const visionConfig = parseVisionConfig((key) => reader.metadata(key));
  const visionWeights = await loadVisionWeights(visionTensorSource(reader), visionConfig);
  return { visionConfig, visionWeights };
}

export class VisionLfm2Session {
  readonly textModel: Lfm2GpuModel;
  readonly forward: Lfm2Forward;
  readonly tower: VisionTower;
  readonly tokenizer: Lfm2Tokenizer;
  readonly visionConfig: VisionConfig;
  readonly visionWeights: VisionReferenceWeights;
  readonly imageMarker: string;
  readonly targetSize: number;
  private readonly placeholderId: number;
  private destroyed = false;

  private constructor(
    textModel: Lfm2GpuModel,
    forward: Lfm2Forward,
    tower: VisionTower,
    tokenizer: Lfm2Tokenizer,
    visionConfig: VisionConfig,
    visionWeights: VisionReferenceWeights,
    imageMarker: string,
    targetSize: number,
    placeholderId: number,
  ) {
    this.textModel = textModel;
    this.forward = forward;
    this.tower = tower;
    this.tokenizer = tokenizer;
    this.visionConfig = visionConfig;
    this.visionWeights = visionWeights;
    this.imageMarker = imageMarker;
    this.targetSize = targetSize;
    this.placeholderId = placeholderId;
  }

  static async create(options: VisionLfm2SessionOptions): Promise<VisionLfm2Session> {
    const { device, textSource, visionSource } = options;
    const imageMarker = options.imageMarker ?? "<image>";
    const targetSize = options.targetSize ?? 512;

    // The Sandblaster LFM2 definition must be compiled on this device before
    // Lfm2Forward can read resource layouts. compile() runs once per process;
    // a pre-compiled engine (e.g. the GUI's text path) stays untouched — the
    // Lfm2Forward device check below catches a compile-vs-session mismatch.
    if (lfm2.engine.state !== "ready") {
      const compiled = await lfm2.engine.compile({ device });
      if (compiled.failed) {
        throw new Error(`LFM2 compile failed ${compiled.failed}/${compiled.total}`);
      }
    }

    const textModel = await Lfm2GpuModel.open(device, textSource, { preload: false });
    const forward = new Lfm2Forward(textModel);

    const { visionConfig, visionWeights } = await openVisionWeights(visionSource);

    if (visionConfig.projectorHiddenSize !== textModel.config.hiddenSize) {
      await visionSource.close?.();
      textModel.destroy();
      throw new Error(
        `VisionLfm2Session: projector dim ${visionConfig.projectorHiddenSize} != LM hidden ${textModel.config.hiddenSize}`,
      );
    }

    // Tolerant tokenizer metadata: the VL WQ4 omits tokenizer.ggml.add_eos_token.
    const tokenizer = new Lfm2Tokenizer({
      metadata: (key: string) => {
        try {
          return textModel.metadata(key);
        } catch {
          return undefined;
        }
      },
    } as any);

    const placeholderId = tokenizer.tokenToId.get(imageMarker);
    if (placeholderId === undefined || !tokenizer.isSpecialToken(placeholderId)) {
      await visionSource.close?.();
      textModel.destroy();
      throw new Error(
        `VisionLfm2Session: image marker ${JSON.stringify(imageMarker)} is not a special vocabulary token`,
      );
    }

    const tower = await VisionTower.create({ device, config: visionConfig, weights: visionWeights });
    return new VisionLfm2Session(
      textModel,
      forward,
      tower,
      tokenizer,
      visionConfig,
      visionWeights,
      imageMarker,
      targetSize,
      placeholderId,
    );
  }

  /**
   * Run the vision tower over one image. `imageTokens` is the count the
   * placeholder must expand to: floor(gridH/factor) * floor(gridW/factor).
   */
  async embedImage(
    rgba: Uint8Array,
    width: number,
    height: number,
  ): Promise<{ readonly image: ProcessedImage; readonly embeddings: Float32Array; readonly imageTokens: number }> {
    if (this.destroyed) throw new Error("VisionLfm2Session is destroyed");
    const factor = this.visionConfig.projectorScaleFactor;
    const image = encodeImage(rgba, width, height, this.visionWeights.posEmb, this.visionConfig, {
      targetSize: this.targetSize,
    });
    const imageTokens = Math.floor(image.gridH / factor) * Math.floor(image.gridW / factor);
    if (imageTokens < 1) {
      throw new Error(`VisionLfm2Session: grid ${image.gridH}x${image.gridW} yields no image tokens`);
    }
    const embeddings = await this.tower.run(image);
    if (embeddings.length !== imageTokens * this.textModel.config.hiddenSize) {
      throw new Error(
        `VisionLfm2Session: tower returned ${embeddings.length} floats, expected ${imageTokens} x ${this.textModel.config.hiddenSize}`,
      );
    }
    return { image, embeddings, imageTokens };
  }

  /**
   * Chat about one image. The prompt may contain the image marker (default
   * "<image>"); it is prepended when absent. v0: exactly one image per turn.
   */
  async chat(request: VisionChatRequest): Promise<VisionChatResult> {
    if (this.destroyed) throw new Error("VisionLfm2Session is destroyed");
    const content = request.prompt.includes(this.imageMarker)
      ? request.prompt
      : `${this.imageMarker}\n${request.prompt}`;

    // BOS + ChatML + special-token parse (same template as the text runtime).
    const raw = this.tokenizer.encodeUserPrompt(content, request.system);
    let markers = 0;
    for (const id of raw) if (id === this.placeholderId) markers++;
    if (markers !== 1) {
      throw new Error(
        `VisionLfm2Session: expected exactly one ${JSON.stringify(this.imageMarker)} marker in the prompt, found ${markers}`,
      );
    }

    const { embeddings, imageTokens, image } = await this.embedImage(
      request.rgba,
      request.width,
      request.height,
    );

    // Expand the single marker into `imageTokens` placeholder copies.
    const expanded: number[] = [];
    let imageStart = -1;
    for (const id of raw) {
      if (id === this.placeholderId) {
        imageStart = expanded.length;
        for (let i = 0; i < imageTokens; i++) expanded.push(this.placeholderId);
      } else {
        expanded.push(id);
      }
    }

    const result = await this.forward.generateWithImageEmbeddings(expanded, embeddings, {
      imageStart,
      maxNewTokens: request.maxNewTokens,
      sampling: request.sampling,
    });
    const text = this.tokenizer.decode(result.tokens, { skipSpecial: true });
    return {
      text,
      tokens: result.tokens,
      imageTokens,
      imageStart,
      grid: { w: image.gridW, h: image.gridH },
    };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.tower.destroy();
    this.textModel.destroy();
  }
}
