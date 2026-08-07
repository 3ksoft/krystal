import { computed, defineComponent, inject, onMounted, ref } from "vue";
import { ENGINE_KEY } from "../engine/key.ts";
import { emptySelection, type ContextSelection } from "../engine/useEngine.ts";
import ContextPicker from "../ui/ContextPicker.ts";
import { fmt, TosWindow } from "../ui/tos.ts";

const PRESETS: Array<{ name: string; source: string }> = [
  { name: "literal", source: `type("'ok'")` },
  { name: "string", source: `type("string < 32")` },
  { name: "number", source: `type("number")` },
  { name: "object", source: `type({ id: "number", name: "string < 24" })` },
  { name: "enum", source: `type("'red' | 'green' | 'blue'")` },
  { name: "optional", source: `type({ id: "number", "note?": "string < 16" })` },
];

export default defineComponent({
  name: "SchemaPanel",
  components: { ContextPicker, TosWindow },
  setup() {
    const api = inject(ENGINE_KEY)!;
    const source = ref(PRESETS[3]!.source);
    const maxTokens = ref(32);
    const compileError = ref<string | null>(null);
    // This window's own request context, independent of the other panels.
    const selection = ref<ContextSelection>(emptySelection());
    const showContext = ref(false);

    const ready = computed(() => api.state.phase === "ready");
    const program = computed(() => api.state.lastProgram);
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

    async function raw(): Promise<void> {
      try {
        await api.generateTokens(Math.max(1, Math.floor(maxTokens.value)), selection.value);
      } catch {
        // Surfaced through state.error.
      }
    }

    // Constraint compilation is CPU-only, so the panel can show a real program
    // before any model is loaded.
    onMounted(compile);

    return {
      api, source, maxTokens, compileError, selection, showContext,
      ready, program, contextText, PRESETS, compile, generate, raw, fmt,
    };
  },
  template: `
    <TosWindow title="GENERATE" :span="8">
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
          <button class="btn btn--default" type="button" :disabled="!ready" @click="generate">GENERATE T</button>
          <span class="menubar__spacer"></span>
          <label for="max-tokens" class="muted">maxTokens</label>
          <input id="max-tokens" type="number" min="1" max="1024" v-model.number="maxTokens" style="width:8ch" />
          <button class="btn" type="button" :disabled="!ready" @click="raw">GENERATE TOKENS</button>
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
          <div class="muted">output</div>
          <pre class="pre" style="border:1px solid var(--fg);padding:4px;min-height:var(--cell-y)">{{ api.state.lastOutput || ' ' }}</pre>
        </div>
      </div>
    </TosWindow>
  `,
});
