/**
 * VL CHAT — manual-test window for the M3 vision-language flow (ADA-0009).
 *
 * The message being composed is the planned `ContextBlock` shape: text and
 * image blocks the VL layer will eventually turn into LM context slots. The
 * v0 envelope (the one the session implements today) is simplified — an image
 * block is represented in the prompt by the `<image>` marker, which the
 * session expands into a run of placeholders whose rows are then overwritten
 * with the tower embeddings. So this window is a prototype of the future image
 * block, not its final API; exactly one image per turn for now.
 *
 * Session lifecycle: the VL session reuses the device the main engine booted
 * on (Lfm2Forward rejects a model opened on a different device than the one
 * the global Sandblaster definition was compiled on), so LOAD is gated on the
 * engine being ready. Both run on the same shared arena, so a VL chat and a
 * main-engine operation must not overlap — they are mutually exclusive in
 * practice because the panel owns its own busy flag and the engine gates its
 * own ops; do not click both at once.
 */
import { computed, defineComponent, inject, onBeforeUnmount, ref, shallowRef } from "vue";
import { HttpRangeSource } from "@chomato/quant";
import { VisionLfm2Session, type VisionChatResult } from "@chomato/webgpu";
import { ENGINE_KEY } from "../engine/key.ts";
import { VL_TEXT_MODEL_URL, VL_VISION_MODEL_URL } from "../engine/model-source.ts";
import { fmt, TosWindow } from "../ui/tos.ts";

/**
 * The message composition model this window edits.
 *
 * This is the shape the VL layer is planned to accept directly: text blocks
 * pass through, image blocks are decoded to embeddings that replace the
 * block's slot in the LM context. v0 renders the image block as the `<image>`
 * marker instead (see header comment).
 */
export type ContextBlock =
  | { type: "text"; text: string }
  | { type: "image"; rgba: Uint8Array; width: number; height: number };

const IMAGE_MARKER = "<image>";
const DEFAULT_MAX_TOKENS = 64;

/** Decode an image file to RGBA without adding a decoder dependency. */
async function decodeImage(file: File): Promise<Extract<ContextBlock, { type: "image" }>> {
  const bitmap = await createImageBitmap(file);
  try {
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("2d canvas context unavailable");
    context.drawImage(bitmap, 0, 0);
    const data = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
    return {
      type: "image",
      // The typed array aliases the ImageData buffer; the panel holds the
      // block for the duration of the turn, which is all the session needs.
      rgba: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
      width: bitmap.width,
      height: bitmap.height,
    };
  } finally {
    bitmap.close();
  }
}

export default defineComponent({
  name: "VlPanel",
  components: { TosWindow },
  setup() {
    const api = inject(ENGINE_KEY)!;

    const blocks = ref<ContextBlock[]>([]);
    const draft = ref("");
    const maxTokens = ref(DEFAULT_MAX_TOKENS);

    const session = shallowRef<VisionLfm2Session | null>(null);
    const sessionState = ref<"idle" | "loading" | "ready" | "error">("idle");
    const sessionError = ref<string | null>(null);
    const sessionMeta = ref<{ grid: string; tokens: number; ms: number } | null>(null);

    const busy = ref(false);
    const lastResult = ref<VisionChatResult | null>(null);
    const sendError = ref<string | null>(null);

    // Set when the window unmounts (it is toggleable). Guards the async
    // session create: a component torn down mid-load must not receive a live
    // session that would never be destroyed.
    let disposed = false;

    // Data URLs for image block previews, cached by block identity.
    const thumbCache = new WeakMap<object, string>();

    const engineReady = computed(() => api.state.phase === "ready");
    const imageBlocks = computed(() => blocks.value.filter((b) => b.type === "image"));
    const canSend = computed(
      () => session.value !== null && sessionState.value === "ready" && !busy.value
        && imageBlocks.value.length === 1 && blocks.value.length > 0
    );
    const tooManyImages = computed(() => imageBlocks.value.length > 1);

    function thumbUrl(block: Extract<ContextBlock, { type: "image" }>): string {
      const cached = thumbCache.get(block);
      if (cached) return cached;
      const canvas = document.createElement("canvas");
      canvas.width = block.width;
      canvas.height = block.height;
      const context = canvas.getContext("2d");
      if (!context) return "";
      const clamped = new Uint8ClampedArray(block.rgba.length);
      clamped.set(block.rgba);
      const data = new ImageData(clamped, block.width, block.height);
      context.putImageData(data, 0, 0);
      const url = canvas.toDataURL("image/png");
      thumbCache.set(block, url);
      return url;
    }

    function addText(): void {
      const text = draft.value.trim();
      if (!text) return;
      blocks.value = [...blocks.value, { type: "text", text }];
      draft.value = "";
    }

    async function addImage(event: Event): Promise<void> {
      const input = event.target as HTMLInputElement;
      const file = input.files?.[0];
      input.value = "";
      if (!file) return;
      try {
        const block = await decodeImage(file);
        blocks.value = [...blocks.value, block];
      } catch (cause) {
        sendError.value = cause instanceof Error ? cause.message : String(cause);
      }
    }

    function removeBlock(index: number): void {
      blocks.value = blocks.value.filter((_, i) => i !== index);
    }

    function clearMessage(): void {
      blocks.value = [];
      draft.value = "";
      lastResult.value = null;
      sendError.value = null;
    }

    async function loadSession(): Promise<void> {
      if (sessionState.value === "loading" || session.value) return;
      const device = api.getDevice();
      if (!device) {
        sendError.value = "boot the model first — the VL session reuses the engine device";
        return;
      }
      sessionState.value = "loading";
      sessionError.value = null;
      sendError.value = null;
      try {
        const created = await VisionLfm2Session.create({
          device,
          textSource: await HttpRangeSource.open(VL_TEXT_MODEL_URL),
          visionSource: await HttpRangeSource.open(VL_VISION_MODEL_URL),
        });
        if (disposed) {
          created.destroy();
          return;
        }
        session.value = created;
        sessionState.value = "ready";
      } catch (cause) {
        if (disposed) return;
        sessionError.value = cause instanceof Error ? cause.message : String(cause);
        sessionState.value = "error";
        session.value = null;
      }
    }

    /** Compose the block list into the v0 prompt: images become the marker. */
    function composePrompt(): string {
      return blocks.value
        .map((block) => (block.type === "image" ? IMAGE_MARKER : block.text))
        .join("\n");
    }

    async function send(): Promise<void> {
      if (!canSend.value) return;
      const current = session.value!;
      const image = imageBlocks.value[0]!;
      busy.value = true;
      sendError.value = null;
      const started = performance.now();
      try {
        lastResult.value = await current.chat({
          rgba: image.rgba,
          width: image.width,
          height: image.height,
          prompt: composePrompt(),
          maxNewTokens: Math.max(1, Math.min(1024, Math.floor(maxTokens.value))),
        });
        sessionMeta.value = {
          grid: `${lastResult.value.grid.w}×${lastResult.value.grid.h}`,
          tokens: lastResult.value.imageTokens,
          ms: performance.now() - started,
        };
      } catch (cause) {
        sendError.value = cause instanceof Error ? cause.message : String(cause);
      } finally {
        busy.value = false;
      }
    }

    onBeforeUnmount(() => {
      disposed = true;
      session.value?.destroy();
      session.value = null;
    });

    return {
      api, blocks, draft, maxTokens, session, sessionState, sessionError, sessionMeta,
      busy, lastResult, sendError,
      engineReady, canSend, imageBlocks, tooManyImages,
      addText, addImage, removeBlock, clearMessage, loadSession, send,
      thumbUrl, fmt,
    };
  },
  template: `
    <TosWindow title="VL" icon="◉" :span="12">
      <div class="stack">
        <div class="row row--tight" style="align-items:center">
          <span class="muted">session</span>
          <span
            class="statusbar__phase"
            :class="{ 'statusbar__phase--live': sessionState === 'loading' }"
          >{{ sessionState.toUpperCase() }}</span>
          <template v-if="!session">
            <button
              class="btn btn--default"
              type="button"
              :disabled="!engineReady || sessionState === 'loading'"
              :title="engineReady ? 'Load the VL text backbone + mmproj (shared engine device)' : 'Boot the model first'"
              @click="loadSession"
            >{{ sessionState === 'loading' ? 'LOADING…' : 'LOAD VL' }}</button>
            <span class="muted" v-if="!engineReady" title="The VL session reuses the engine device, so any model must be booted first. It then loads its own VL text + mmproj, independent of the MODEL window's model.">
              boot any model first — VL loads its own 1.6B + mmproj
            </span>
            <span class="muted" v-else>loads its own 1.6B + mmproj, not the MODEL window's model</span>
          </template>
          <template v-else>
            <span class="muted">1.6B WQ4 + mmproj F16, target 512</span>
            <span class="menubar__spacer"></span>
            <button class="btn" type="button" @click="clearMessage">CLEAR</button>
          </template>
        </div>
        <div v-if="sessionError" class="pre" style="border:1px solid var(--fg);padding:4px">
          session error: {{ sessionError }}
        </div>

        <template v-if="session && sessionState === 'ready'">
          <div v-if="!blocks.length" class="list__empty" style="padding:var(--cell-x)">
            compose a message: add text and one image, then send
          </div>
          <div v-else class="list" style="border:0;max-height:220px">
            <div
              v-for="(block, index) in blocks"
              :key="index"
              class="list__row"
              style="cursor:default;align-items:flex-start;padding:4px"
            >
              <span class="num muted" style="width:4ch">{{ index + 1 }}</span>
              <template v-if="block.type === 'text'">
                <span class="grow pre" style="white-space:pre-wrap">{{ block.text }}</span>
              </template>
              <template v-else>
                <img class="vl-thumb" :src="thumbUrl(block)" :alt="'image ' + block.width + 'x' + block.height" />
                <span class="muted">{{ block.width }}×{{ block.height }} px</span>
              </template>
              <button class="btn" type="button" title="remove block" @click="removeBlock(index)">✕</button>
            </div>
          </div>

          <div class="row">
            <input
              class="grow"
              type="text"
              v-model="draft"
              placeholder="text block…"
              @keyup.enter="addText"
            />
            <button class="btn" type="button" :disabled="!draft.trim()" @click="addText">ADD TEXT</button>
            <label class="btn" for="vl-image" style="cursor:pointer" title="Add an image block">
              ADD IMAGE<input id="vl-image" type="file" accept="image/*" style="display:none" @change="addImage" />
            </label>
          </div>

          <div class="row">
            <button
              class="btn btn--default"
              type="button"
              :disabled="!canSend"
              @click="send"
            >{{ busy ? 'GENERATING…' : 'SEND' }}</button>
            <label for="vl-max-tokens" class="muted">maxTokens</label>
            <input id="vl-max-tokens" type="number" min="1" max="1024" v-model.number="maxTokens" style="width:8ch" />
            <span class="muted" v-if="!imageBlocks.length">add one image</span>
            <span class="muted" v-else-if="tooManyImages">v0: one image per turn</span>
          </div>
          <div v-if="sendError" class="pre" style="border:1px solid var(--fg);padding:4px">{{ sendError }}</div>

          <div v-if="lastResult">
            <div class="row">
              <span class="muted grow">answer</span>
              <span v-if="sessionMeta" class="muted">
                grid {{ sessionMeta.grid }} · {{ fmt.int(sessionMeta.tokens) }} image tokens · {{ fmt.ms(sessionMeta.ms) }}
              </span>
            </div>
            <pre class="pre" style="border:1px solid var(--fg);padding:4px;min-height:var(--cell-y)">{{ lastResult.text || ' ' }}</pre>
          </div>
        </template>
      </div>
    </TosWindow>
  `,
});
