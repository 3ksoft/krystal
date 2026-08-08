import { computed, defineComponent, inject, ref } from "vue";
import { ENGINE_KEY } from "../engine/key.ts";
import { fmt, TosMeter, TosWindow } from "../ui/tos.ts";

import { DEFAULT_IS_REMOTE, MODEL_DOWNLOAD_URL } from "../engine/model-source.ts";

export default defineComponent({
  name: "ModelPanel",
  components: { TosWindow, TosMeter },
  setup() {
    const api = inject(ENGINE_KEY)!;
    const url = ref(api.state.modelUrl);
    const file = ref<File | null>(null);

    const loading = computed(() => ["device", "model", "runtime"].includes(api.state.phase));
    const booted = computed(() => api.state.model !== null);

    async function boot(): Promise<void> {
      try {
        await api.boot(file.value ? { file: file.value } : { url: url.value });
      } catch {
        // Surfaced through state.error by the alert in App.
      }
    }

    function pick(event: Event): void {
      const input = event.target as HTMLInputElement;
      file.value = input.files?.[0] ?? null;
    }

    return { api, url, file, loading, booted, boot, pick, fmt, MODEL_DOWNLOAD_URL, DEFAULT_IS_REMOTE };
  },
  template: `
    <TosWindow title="MODEL" icon="▛" :span="4">
      <div class="stack">
        <template v-if="!booted">
          <div class="field">
            <label for="model-url">url</label>
            <input id="model-url" class="grow" type="text" v-model="url" :disabled="loading || !!file" />
          </div>
          <div class="field">
            <label for="model-file">file</label>
            <input id="model-file" class="grow" type="file" accept=".wq4" :disabled="loading" @change="pick" />
          </div>
          <div class="row">
            <button class="btn btn--default" type="button" :disabled="loading" @click="boot">
              {{ loading ? 'LOADING…' : 'LOAD MODEL' }}
            </button>
            <span class="muted">WQ4 v3, self-contained</span>
          </div>
          <div v-if="DEFAULT_IS_REMOTE && !file" class="muted">
            the url streams ~700 MB from
            <a :href="MODEL_DOWNLOAD_URL" target="_blank" rel="noreferrer">HuggingFace</a>;
            pick a local copy above to skip the download
          </div>
          <TosMeter v-if="loading" :value="api.state.progress" />
        </template>

        <template v-else>
          <dl class="kv">
            <dt>source</dt><dd style="text-align:left">{{ api.state.modelUrl }}</dd>
            <dt>tensors</dt><dd>{{ fmt.int(api.state.model.tensorCount) }}</dd>
            <dt>VRAM</dt><dd>{{ fmt.bytes(api.state.model.allocatedBytes) }}</dd>
            <dt>hidden</dt><dd>{{ fmt.int(api.state.model.hiddenSize) }}</dd>
            <dt>ffn</dt><dd>{{ fmt.int(api.state.model.feedForwardSize) }}</dd>
            <dt>vocab</dt><dd>{{ fmt.int(api.state.model.vocabSize) }}</dd>
            <dt>heads</dt><dd>{{ api.state.model.attentionHeads }} × {{ api.state.model.headDim }}</dd>
            <dt>bos / eos</dt><dd>{{ api.state.model.bosToken }} / {{ api.state.model.eosToken }}</dd>
            <dt>context</dt><dd>{{ fmt.int(api.state.model.contextCapacity) }}</dd>
            <dt>max new</dt><dd>{{ fmt.int(api.state.model.maxNewTokens) }}</dd>
            <dt>upload</dt><dd>{{ fmt.ms(api.state.model.loadMs) }}</dd>
            <dt>prepare</dt><dd>{{ fmt.ms(api.state.model.prepareMs) }}</dd>
          </dl>

          <div>
            <div class="muted">layers &mdash; <span class="layers__cell" style="display:inline-grid;vertical-align:middle;width:14px;height:14px">C</span> short-conv,
              <span class="layers__cell layers__cell--attn" style="display:inline-grid;vertical-align:middle;width:14px;height:14px">A</span> attention</div>
            <div class="layers" style="margin-top:4px">
              <div
                v-for="(kind, index) in api.state.model.layers"
                :key="index"
                class="layers__cell"
                :class="{ 'layers__cell--attn': kind === 'attention' }"
                :title="'block ' + index + ' — ' + kind"
              >{{ kind === 'attention' ? 'A' : 'C' }}</div>
            </div>
          </div>
        </template>
      </div>
    </TosWindow>
  `,
});
