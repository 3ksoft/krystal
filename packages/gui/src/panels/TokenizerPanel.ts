/**
 * Live tokenizer scratchpad.
 *
 * Text in, tokens out, with no engine round-trip: tokenization is CPU-only, so
 * this updates on every keystroke once a model is loaded. It doubles as the
 * reference for the reserved vocabulary, since `[-token-K-]` is only useful if
 * you can see which token K actually resolves to.
 */
import { computed, defineComponent, inject, ref } from "vue";
import { ENGINE_KEY } from "../engine/key.ts";
import { fmt, TosWindow } from "../ui/tos.ts";

export default defineComponent({
  name: "TokenizerPanel",
  components: { TosWindow },
  setup() {
    const api = inject(ENGINE_KEY)!;
    const text = ref("<|im_start|>user\nprovide value for alfa[-token-1-]");
    const addBos = ref(true);
    const lookup = ref(0);
    const filter = ref("");

    const ready = computed(() => api.state.phase === "ready");

    const result = computed(() => {
      if (!ready.value) return { tokens: [] as number[], expanded: text.value, unknownAliases: [] as number[] };
      return api.inspectText(text.value, addBos.value);
    });

    const cells = computed(() =>
      result.value.tokens.map((id) => {
        const literal = api.tokenLiteral(id);
        const special = api.isSpecial(id);
        const piece = api.tokenPieces([id])[0] ?? "";
        return {
          id,
          special,
          // Specials decode to nothing, so show the vocabulary literal instead.
          shown: special ? literal ?? `<${id}>` : piece.replace(/\n/g, "\\n"),
        };
      })
    );

    const bytes = computed(() => new TextEncoder().encode(result.value.expanded).length);

    /** Specials decode to an empty string, so a round-trip only holds without them. */
    const roundTrip = computed(() => {
      if (!ready.value || !result.value.tokens.length) return null;
      const decoded = api.decode(result.value.tokens);
      return { decoded, exact: decoded === result.value.expanded };
    });

    const lookedUp = computed(() => {
      if (!ready.value) return null;
      const literal = api.tokenLiteral(lookup.value);
      if (literal === null) return null;
      return { id: lookup.value, literal, special: api.isSpecial(lookup.value) };
    });

    /** Reserved reference, filtered so 377 rows stay navigable. */
    const reservedRows = computed(() => {
      const rows: Array<{ index: number; alias: string; literal: string; id: number }> = [];
      for (let index = 1; index <= api.state.reservedCount; index++) {
        const entry = api.reservedEntry(index);
        if (entry.literal === null || entry.id === null) continue;
        const alias = `[-token-${index}-]`;
        if (filter.value && !alias.includes(filter.value) && !entry.literal.includes(filter.value)
          && String(entry.id) !== filter.value) continue;
        rows.push({ index, alias, literal: entry.literal, id: entry.id });
        if (rows.length >= 60) break;
      }
      return rows;
    });

    function insert(alias: string): void {
      text.value += alias;
    }

    return {
      api, text, addBos, lookup, filter, ready, result, cells, bytes,
      roundTrip, lookedUp, reservedRows, insert, fmt,
    };
  },
  template: `
    <TosWindow title="TOKENIZER" icon="┼" :span="6">
      <div class="stack">
        <textarea v-model="text" rows="4" spellcheck="false"></textarea>

        <div class="row">
          <button class="btn" type="button" :class="{ 'btn--on': addBos }" @click="addBos = !addBos">
            {{ addBos ? '[x]' : '[ ]' }} BOS
          </button>
          <span class="menubar__spacer"></span>
          <span class="num">{{ fmt.int(result.tokens.length) }} tok</span>
          <span class="statusbar__sep">│</span>
          <span class="num">{{ fmt.int(text.length) }} chars</span>
          <span class="statusbar__sep">│</span>
          <span class="num">{{ fmt.int(bytes) }} B</span>
        </div>

        <div v-if="!ready" class="muted">load a model to tokenize</div>

        <template v-else>
          <div v-if="result.unknownAliases.length" class="picker__issues">
            ! no reserved token for {{ result.unknownAliases.map(n => '[-token-' + n + '-]').join(', ') }}
            — this model exposes {{ api.state.reservedCount }}
          </div>

          <div class="pieces" style="max-height:170px">
            <span
              v-for="(cell, index) in cells"
              :key="index"
              class="pieces__cell"
              :class="{ 'pieces__cell--special': cell.special }"
              :title="'id ' + cell.id"
            >{{ cell.shown }}</span>
          </div>

          <div v-if="roundTrip" class="muted">
            round-trip {{ roundTrip.exact ? 'exact' : 'differs (special tokens decode to nothing)' }}
          </div>

          <div class="field">
            <label for="tok-lookup">id</label>
            <input id="tok-lookup" type="number" min="0" v-model.number="lookup" style="width:10ch" />
            <span v-if="lookedUp" class="grow">
              <span class="pieces__cell" :class="{ 'pieces__cell--special': lookedUp.special }">{{ lookedUp.literal }}</span>
            </span>
            <span v-else class="muted grow">out of range</span>
          </div>

          <div class="field">
            <label for="tok-filter">reserved</label>
            <input id="tok-filter" class="grow" type="text" v-model="filter" placeholder="filter by alias, literal or id" />
            <span class="muted">{{ fmt.int(api.state.reservedCount) }} free</span>
          </div>
          <div class="list" style="max-height:150px">
            <div v-if="!reservedRows.length" class="list__empty">no match</div>
            <div v-for="row in reservedRows" :key="row.index" class="list__row" @click="insert(row.alias)">
              <span style="width:16ch">{{ row.alias }}</span>
              <span class="grow">{{ row.literal }}</span>
              <span class="num muted">id {{ row.id }}</span>
            </div>
          </div>
        </template>
      </div>
    </TosWindow>
  `,
});
