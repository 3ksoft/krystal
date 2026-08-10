import { computed, defineComponent, inject, ref } from "vue";
import { ENGINE_KEY } from "../engine/key.ts";
import { fmt, TosWindow } from "../ui/tos.ts";

export default defineComponent({
  name: "BlocksPanel",
  components: { TosWindow },
  setup() {
    const api = inject(ENGINE_KEY)!;
    const text = ref("const alfa = 43.2;\nconst gamma = 21.7;");
    const label = ref("");
    const inspecting = ref<number | null>(null);
    const view = ref<"text" | "tokens">("text");

    /**
     * BOS belongs to the composed sequence, not to each block: a second BOS
     * further along resets the model and the earlier blocks stop being visible
     * to it. Default it on only for the block that will plausibly start a
     * context, and let it be overridden per block.
     */
    const bosOverride = ref<boolean | null>(null);
    const addBos = computed(() => bosOverride.value ?? api.state.blocks.length === 0);

    /**
     * LFM2.5 is an Instruct model, so a context split across blocks normally
     * wants its ChatML turn opened by the first block and closed by the last.
     */
    const WRAPS = {
      raw: { label: "raw", pre: "", post: "" },
      open: { label: "open user turn", pre: "<|im_start|>user\n", post: "\n" },
      close: { label: "close → assistant", pre: "", post: "<|im_end|>\n<|im_start|>assistant\n" },
      full: { label: "full user turn", pre: "<|im_start|>user\n", post: "<|im_end|>\n<|im_start|>assistant\n" },
    } as const;
    const wrap = ref<keyof typeof WRAPS>("raw");
    const wrapped = computed(() => `${WRAPS[wrap.value].pre}${text.value}${WRAPS[wrap.value].post}`);

    const ready = computed(() => api.state.phase === "ready");
    const preview = computed(() => {
      if (!ready.value || !text.value) return null;
      try {
        return api.tokenize(wrapped.value, addBos.value).length;
      } catch {
        return null;
      }
    });

    /** Inspection is local to this window; it selects nothing for the engine. */
    const inspected = computed(() =>
      inspecting.value === null ? null : api.block(inspecting.value) ?? null
    );

    const pieces = computed(() => {
      const row = inspected.value;
      if (!row) return [];
      return api.tokenPieces(row.tokens).map((piece, index) => ({
        id: row.tokens[index]!,
        piece,
        special: piece.startsWith("<") && piece.endsWith(">"),
      }));
    });

    async function put(): Promise<void> {
      try {
        const row = await api.putBlock({ text: wrapped.value, addBos: addBos.value, label: label.value });
        label.value = "";
        inspecting.value = row.id;
        // Fall back to the automatic BOS decision for the next block.
        bosOverride.value = null;
      } catch {
        // Surfaced through state.error.
      }
    }

    async function putImage(event: Event): Promise<void> {
      const input = event.target as HTMLInputElement;
      const file = input.files?.[0];
      input.value = "";
      if (!file) return;
      try {
        const row = await api.putImageBlock(file);
        inspecting.value = row.id;
      } catch {
        // Surfaced through state.error.
      }
    }

    async function drop(id: number): Promise<void> {
      try {
        await api.dropBlock(id);
        if (inspecting.value === id) inspecting.value = null;
      } catch {
        // Surfaced through state.error.
      }
    }

    return {
      api, text, label, addBos, bosOverride, wrap, WRAPS, wrapped,
      ready, preview, inspecting, inspected, view, pieces, put, putImage, drop, fmt,
    };
  },
  template: `
    <TosWindow title="BLOCKS" icon="▒" :span="4">
      <div class="stack">
        <textarea v-model="text" rows="3" spellcheck="false" :disabled="!ready"></textarea>
        <div class="field">
          <label for="block-label">name</label>
          <input id="block-label" class="grow" type="text" v-model="label" placeholder="optional" :disabled="!ready" />
        </div>
        <div class="field">
          <label for="block-wrap">wrap</label>
          <select id="block-wrap" class="grow" v-model="wrap" :disabled="!ready">
            <option v-for="(entry, key) in WRAPS" :key="key" :value="key">{{ entry.label }}</option>
          </select>
        </div>
        <div class="row">
          <button class="btn" type="button" :class="{ 'btn--on': addBos }" :disabled="!ready"
            :title="'BOS belongs to the first element of a context only'"
            @click="bosOverride = !addBos">
            {{ addBos ? '[x]' : '[ ]' }} BOS
          </button>
          <button class="btn btn--default" type="button" :disabled="!ready || !text" @click="put">PUT BLOCK</button>
          <label class="btn" for="block-image" :class="{ 'btn--default': api.state.modelKind === 'vl' }">
            PUT IMAGE<input id="block-image" type="file" accept="image/*" style="display:none" @change="putImage" />
          </label>
          <span class="muted" v-if="preview !== null">{{ preview }} tok</span>
          <span class="muted" v-if="bosOverride === null">auto</span>
        </div>

        <div class="list" style="max-height:150px">
          <div v-if="!api.state.blocks.length" class="list__empty">no blocks stored</div>
          <div
            v-for="row in api.state.blocks"
            :key="row.id"
            class="list__row"
            :class="{ 'list__row--on': inspecting === row.id }"
            @click="inspecting = inspecting === row.id ? null : row.id"
          >
            <span style="width:5ch">#{{ row.id }}</span>
            <span class="grow" style="overflow:hidden;text-overflow:ellipsis">{{ row.label }}</span>
            <span class="num muted">{{ row.tokens.length }}t</span>
            <button class="btn" type="button" title="DropBlock" :disabled="!ready" @click.stop="drop(row.id)">DEL</button>
          </div>
        </div>

        <template v-if="inspected">
          <div class="row">
            <span class="muted grow">#{{ inspected.id }} contents</span>
            <button class="btn" type="button" :class="{ 'btn--on': view === 'text' }" @click="view = 'text'">TEXT</button>
            <button class="btn" type="button" :class="{ 'btn--on': view === 'tokens' }" @click="view = 'tokens'">TOKENS</button>
          </div>
          <pre v-if="view === 'text'" class="pre content">{{ inspected.text }}</pre>
          <div v-else class="pieces">
            <span
              v-for="(cell, index) in pieces"
              :key="index"
              class="pieces__cell"
              :class="{ 'pieces__cell--special': cell.special }"
              :title="'id ' + cell.id"
            >{{ cell.piece }}</span>
          </div>
          <dl class="kv">
            <dt>tokens</dt><dd>{{ fmt.int(inspected.tokens.length) }}</dd>
            <dt>chars</dt><dd>{{ fmt.int(inspected.text.length) }}</dd>
            <dt>BOS</dt><dd>{{ inspected.addBos ? 'yes' : 'no' }}</dd>
          </dl>
        </template>
        <div v-else class="muted">select a block to inspect its contents</div>
      </div>
    </TosWindow>
  `,
});
