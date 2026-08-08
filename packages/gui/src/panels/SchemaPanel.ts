import { computed, defineComponent, inject, onMounted, ref } from "vue";
import { ENGINE_KEY } from "../engine/key.ts";
import {
  emptySelection,
  GREEDY_SAMPLING,
  type ContextSelection,
  type TokenSampling,
} from "../engine/useEngine.ts";
import ContextPicker from "../ui/ContextPicker.ts";
import { fmt, TosWindow } from "../ui/tos.ts";

const PRESETS: Array<{ name: string; source: string }> = [
  { name: "literal", source: `type("'ok'")` },
  { name: "string", source: `type("string < 32")` },
  { name: "number", source: `type("number")` },
  { name: "object", source: `type({ id: "number", name: "string < 24" })` },
  { name: "enum", source: `type("'red' | 'green' | 'blue'")` },
  { name: "optional", source: `type({ id: "number", "note?": "string < 16" })` },
  /**
   * Arrays need both bounds, for different reasons.
   *
   * The upper bound is what makes the decode budget finite — it is derived from
   * the schema's worst case, and a wide row with atMostLength(10) still needs
   * ~1050 tokens, just over the 1024 budget.
   *
   * The lower bound is what makes the answer interesting: without it the empty
   * array satisfies the schema, and that is what the model will emit. Note that
   * it costs nothing — the budget is the same 194 either way. Length predicates
   * are the *Length spellings; `.moreThan(0)` is a numeric comparison and
   * arktype rejects it on an array.
   */
  {
    name: "array",
    source: `type({ id: "number", name: "string < 16" }).array().atLeastLength(1).atMostLength(3)`,
  },
];

/** Uniform u32. Only the seed is random here — the sampler itself is not. */
function randomSeed(): number {
  return Math.floor(Math.random() * 0x1_0000_0000) >>> 0;
}

export default defineComponent({
  name: "SchemaPanel",
  components: { ContextPicker, TosWindow },
  setup() {
    const api = inject(ENGINE_KEY)!;
    const source = ref(PRESETS[3]!.source);
    const maxTokens = ref(32);
    const sampled = ref(false);
    const temperature = ref(0.8);
    const topK = ref(40);
    // A fresh seed per session, not per click: leaving it fixed is what makes
    // two runs comparable, and rerolling is one button away.
    const seed = ref(randomSeed());
    const compileError = ref<string | null>(null);
    // This window's own request context, independent of the other panels.
    const selection = ref<ContextSelection>(emptySelection());
    const showContext = ref(false);

    const ready = computed(() => api.state.phase === "ready");
    // Generation needs at least one context token; an empty context is rejected
    // by the engine, so gate the controls instead of reporting it afterwards.
    const hasContext = computed(() =>
      selection.value.checkpoint !== null || selection.value.blocks.length > 0
    );
    const program = computed(() => api.state.lastProgram);
    /**
     * The decode budget is derived from the schema, and the runtime rejects a
     * request whose budget exceeds its capacity. Bounded arrays overflow it
     * easily, so surface it before the click rather than as a failed run.
     */
    const budgetOverflow = computed(() => {
      const max = api.state.model?.maxNewTokens;
      const needed = program.value?.maxTokens;
      return max !== undefined && needed !== undefined && needed > max;
    });
    const canRun = computed(() => ready.value && hasContext.value && !budgetOverflow.value);
    const contextText = computed(() => api.selectionText(selection.value));

    function compile(): void {
      compileError.value = null;
      try {
        api.compileSchema(source.value);
      } catch (cause) {
        compileError.value = cause instanceof Error ? cause.message : String(cause);
      }
    }

    async function generate(): Promise<void> {
      compileError.value = null;
      try {
        await api.generateStructured(source.value, selection.value);
      } catch {
        // Surfaced through state.error.
      }
    }

    const sampling = computed<TokenSampling>(() =>
      sampled.value
        ? {
          sampler: "topk",
          temperature: Math.max(0.01, temperature.value),
          topK: Math.min(64, Math.max(1, Math.floor(topK.value))),
          seed: Math.max(0, Math.floor(seed.value)) >>> 0,
        }
        : GREEDY_SAMPLING
    );

    async function raw(): Promise<void> {
      try {
        await api.generateTokens(
          Math.max(1, Math.floor(maxTokens.value)),
          selection.value,
          sampling.value,
        );
      } catch {
        // Surfaced through state.error.
      }
    }

    /**
     * Turn the output into a block and append it to *this window's* context, so
     * the next generation sees it. Appending is the point — storing the block
     * without selecting it would leave the context exactly where it was.
     */
    async function advance(): Promise<void> {
      try {
        const row = await api.advanceWithOutput();
        selection.value = {
          ...selection.value,
          blocks: [...selection.value.blocks, row.id],
        };
      } catch {
        // Surfaced through state.error.
      }
    }

    // Constraint compilation is CPU-only, so the panel can show a real program
    // before any model is loaded.
    onMounted(compile);

    return {
      api, source, maxTokens, compileError, selection, showContext,
      sampled, temperature, topK, seed, randomSeed,
      ready, hasContext, canRun, budgetOverflow, program, contextText, PRESETS,
      compile, generate, raw, advance, fmt,
    };
  },
  template: `
    <TosWindow title="GENERATE" icon="▓" :span="8">
      <div class="stack">
        <ContextPicker v-model="selection" />

        <div class="row">
          <button class="btn" type="button" :class="{ 'btn--on': showContext }"
            @click="showContext = !showContext">PREVIEW CONTEXT</button>
          <span class="muted">{{ contextText.length }} chars resolved</span>
        </div>
        <pre v-if="showContext" class="pre content">{{ contextText || '(empty context)' }}</pre>

        <div class="row row--tight">
          <span class="muted">presets</span>
          <button
            v-for="preset in PRESETS"
            :key="preset.name"
            class="btn"
            type="button"
            @click="source = preset.source; compile()"
          >{{ preset.name }}</button>
        </div>

        <textarea v-model="source" rows="3" spellcheck="false"></textarea>

        <div class="row">
          <button class="btn" type="button" @click="compile">COMPILE</button>
          <button
            class="btn btn--default"
            type="button"
            :disabled="!canRun"
            :title="hasContext ? 'Generate a value of this type' : 'Select a context above first'"
            @click="generate"
          >GENERATE T</button>
          <span class="menubar__spacer"></span>
          <label for="max-tokens" class="muted">maxTokens</label>
          <input id="max-tokens" type="number" min="1" max="1024" v-model.number="maxTokens" style="width:8ch" />
          <button
            class="btn"
            type="button"
            :disabled="!canRun"
            :title="hasContext ? 'Decode without a schema' : 'Select a context above first'"
            @click="raw"
          >GENERATE TOKENS</button>
        </div>

        <div class="row row--tight">
          <button
            class="btn"
            type="button"
            :class="{ 'btn--on': sampled }"
            title="Greedy argmax, or seeded top-k sampling. Structured generation is always greedy."
            @click="sampled = !sampled"
          >{{ sampled ? 'TOP-K' : 'GREEDY' }}</button>
          <template v-if="sampled">
            <label for="sample-temp" class="muted">temp</label>
            <input id="sample-temp" type="number" min="0.01" max="5" step="0.05"
              v-model.number="temperature" style="width:7ch" />
            <label for="sample-topk" class="muted">top-k</label>
            <input id="sample-topk" type="number" min="1" max="64" step="1"
              v-model.number="topK" style="width:6ch" />
            <label for="sample-seed" class="muted">seed</label>
            <input id="sample-seed" type="number" min="0" max="4294967295" step="1"
              v-model.number="seed" style="width:12ch" />
            <button class="btn" type="button" title="Roll a new seed" @click="seed = randomSeed()">↻</button>
          </template>
          <span v-else class="muted">deterministic; same context always decodes the same tokens</span>
        </div>
        <div v-if="ready && !hasContext" class="muted">
          select a base checkpoint or a block above to enable generation
        </div>
        <div v-else-if="budgetOverflow" class="picker__issues">
          ! schema needs {{ fmt.int(program.maxTokens) }} decode tokens, runtime budget is
          {{ fmt.int(api.state.model.maxNewTokens) }} — tighten a length bound
        </div>

        <div v-if="compileError" class="pre">compile error: {{ compileError }}</div>

        <div class="row" style="align-items:stretch;gap:0" v-if="program">
          <dl class="kv grow" style="padding-right:var(--cell-x)">
            <dt>blob</dt><dd>{{ fmt.bytes(program.blobBytes) }} / {{ fmt.int(program.blobWords) }} w</dd>
            <dt>nodes</dt><dd>{{ fmt.int(program.nodes) }} <span class="muted">of {{ fmt.int(program.sourceNodes) }} src</span></dd>
            <dt>edges</dt><dd>{{ fmt.int(program.edges) }}</dd>
            <dt>byte table</dt><dd>{{ fmt.int(program.byteTableLength) }} B</dd>
            <dt>max json</dt><dd>{{ fmt.int(program.maxJsonBytes) }} B</dd>
            <dt>max tokens</dt><dd>{{ fmt.int(program.maxTokens) }}</dd>
          </dl>
          <dl class="kv grow" style="padding-left:var(--cell-x);border-left:1px solid var(--fg)">
            <dt>literal</dt><dd>{{ program.literalNodes }}</dd>
            <dt>switch</dt><dd>{{ program.switchNodes }}</dd>
            <dt>string</dt><dd>{{ program.stringNodes }}</dd>
            <dt>number</dt><dd>{{ program.numberNodes }}</dd>
            <dt>jump</dt><dd>{{ program.jumpNodes }}</dd>
            <dt>accept</dt><dd>{{ program.acceptNodes }}</dd>
          </dl>
        </div>
        <div v-else class="muted">compile a schema to inspect its constraint program</div>

        <div>
          <div class="row">
            <span class="muted grow">output</span>
            <button
              class="btn"
              type="button"
              :disabled="!ready || !api.state.lastOutput"
              title="Store the output as a block and append it to this context"
              @click="advance"
            >ADVANCE</button>
          </div>
          <pre class="pre" style="border:1px solid var(--fg);padding:4px;min-height:var(--cell-y)">{{ api.state.lastOutput || ' ' }}</pre>
        </div>
      </div>
    </TosWindow>
  `,
});
