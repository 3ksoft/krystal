import { defineComponent, onMounted, reactive } from "vue";
import { useChomato } from "./composables/useChomato.ts";

const ms = (value?: number) => value === undefined ? "—" : value < 1000 ? `${value.toFixed(1)} ms` : `${(value / 1000).toFixed(2)} s`;
const gib = (bytes?: number) => bytes === undefined ? "—" : `${(bytes / 1073741824).toFixed(2)} GiB`;

export default defineComponent({
  name: "App",
  setup() {
    const chomato = reactive(useChomato());
    onMounted(() => void chomato.initialize());

    const submit = async () => {
      if (!chomato.canGenerate) return;
      try {
        await chomato.generate();
      } catch (error) {
        console.error(error);
      }
    };

    return { chomato, submit, ms, gib };
  },
  template: `
    <main class="shell">
      <header>
        <div>
          <h1>Chomato</h1>
          <p class="muted">minimal WebGPU inference harness</p>
        </div>
        <span class="status" :data-phase="chomato.phase">{{ chomato.status }}</span>
      </header>

      <section class="card model">
        <div><span class="label">model</span><strong>{{ chomato.config.modelUrl }}</strong></div>
        <div><span class="label">context</span><strong>{{ chomato.config.contextCapacity }}</strong></div>
        <div><span class="label">loaded</span><strong>{{ Math.round(chomato.modelProgress * 100) }}%</strong></div>
        <template v-if="chomato.modelInfo">
          <div><span class="label">shape</span><strong>{{ chomato.modelInfo.layers }}L / {{ chomato.modelInfo.hiddenSize }}H / {{ chomato.modelInfo.vocabSize }}V</strong></div>
          <div><span class="label">VRAM</span><strong>{{ gib(chomato.modelInfo.allocatedBytes) }}</strong></div>
          <div><span class="label">load / compile</span><strong>{{ ms(chomato.modelInfo.loadMs) }} / {{ ms(chomato.modelInfo.compileMs) }}</strong></div>
        </template>
      </section>

      <form class="card" @submit.prevent="submit">
        <label class="stack">
          <span class="label">Prompt</span>
          <textarea v-model="chomato.prompt" rows="7" :disabled="chomato.busy"></textarea>
        </label>
        <div class="controls">
          <label>tokens <input v-model.number="chomato.maxNewTokens" type="number" min="1" :max="chomato.config.maxNewTokens" /></label>
          <label><input v-model="chomato.profile" type="checkbox" /> profile</label>
          <button type="submit" :disabled="!chomato.canGenerate">Generate</button>
        </div>
      </form>

      <section class="card output">
        <span class="label">Output</span>
        <pre>{{ chomato.output || (chomato.ready ? 'Ready.' : '') }}</pre>
      </section>

      <section v-if="chomato.generationStats" class="card stats">
        <div><span class="label">prompt</span><strong>{{ chomato.generationStats.promptTokens }} tok</strong></div>
        <div><span class="label">generated</span><strong>{{ chomato.generationStats.generatedTokens }} tok</strong></div>
        <div><span class="label">wall</span><strong>{{ ms(chomato.generationStats.wallMs) }}</strong></div>
        <div><span class="label">prefill</span><strong>{{ ms(chomato.generationStats.prefillMs) }}</strong></div>
        <div><span class="label">decode</span><strong>{{ ms(chomato.generationStats.decodeMs) }}</strong></div>
        <div><span class="label">decode rate</span><strong>{{ chomato.generationStats.decodeTokensPerSecond?.toFixed(1) ?? '—' }} tok/s</strong></div>
      </section>

      <section v-if="chomato.error" class="card error"><pre>{{ chomato.error }}</pre></section>
    </main>
  `,
});
