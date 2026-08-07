/**
 * Assembles one ContextSelection: an optional base checkpoint plus an ordered
 * list of blocks appended on top of it.
 *
 * Each panel embeds its own instance. Nothing here reads or writes a shared
 * selection, so hiding or re-selecting in one window never moves another.
 */
import { computed, defineComponent, inject } from "vue";
import { ENGINE_KEY } from "../engine/key.ts";
import type { ContextSelection } from "../engine/useEngine.ts";
import { fmt } from "./tos.ts";

export default defineComponent({
  name: "ContextPicker",
  props: {
    modelValue: { type: Object as () => ContextSelection, required: true },
    /** Checkpoint excluded from the base list, so it cannot be its own base. */
    excludeCheckpoint: { type: Number, default: null },
    compact: { type: Boolean, default: false },
  },
  emits: ["update:modelValue"],
  setup(props, { emit }) {
    const api = inject(ENGINE_KEY)!;

    const bases = computed(() =>
      api.state.checkpoints.filter((row) => row.id !== props.excludeCheckpoint)
    );

    const tokens = computed(() => api.selectionTokens(props.modelValue));
    const overCapacity = computed(() => {
      const cap = api.state.model?.contextCapacity;
      return cap !== undefined && tokens.value > cap;
    });
    // Malformed compositions still run; the engine has no opinion about BOS.
    // Surfacing them here is the only place the operator can see why a
    // structurally valid request would come back ignoring its own context.
    const issues = computed(() => api.selectionIssues(props.modelValue));

    function update(next: Partial<ContextSelection>): void {
      emit("update:modelValue", { ...props.modelValue, ...next });
    }

    function orderOf(id: number): number {
      return props.modelValue.blocks.indexOf(id);
    }

    function toggle(id: number): void {
      const at = orderOf(id);
      const blocks = [...props.modelValue.blocks];
      if (at >= 0) blocks.splice(at, 1);
      else blocks.push(id);
      update({ blocks });
    }

    function move(id: number, by: number): void {
      const at = orderOf(id);
      const to = at + by;
      if (at < 0 || to < 0 || to >= props.modelValue.blocks.length) return;
      const blocks = [...props.modelValue.blocks];
      const [moved] = blocks.splice(at, 1);
      blocks.splice(to, 0, moved!);
      update({ blocks });
    }

    function setBase(event: Event): void {
      const value = (event.target as HTMLSelectElement).value;
      update({ checkpoint: value === "" ? null : Number(value) });
    }

    function clear(): void {
      update({ checkpoint: null, blocks: [] });
    }

    return { api, bases, tokens, overCapacity, issues, orderOf, toggle, move, setBase, clear, fmt };
  },
  template: `
    <div class="picker">
      <div class="field">
        <label>base</label>
        <select class="grow" :value="modelValue.checkpoint ?? ''" @change="setBase">
          <option value="">(none — start from blocks)</option>
          <option v-for="row in bases" :key="row.id" :value="row.id">
            #{{ row.id }} {{ row.label }} — {{ row.position }} tok
          </option>
        </select>
      </div>

      <div class="list picker__blocks" :class="{ 'picker__blocks--compact': compact }">
        <div v-if="!api.state.blocks.length" class="list__empty">no blocks stored</div>
        <div
          v-for="row in api.state.blocks"
          :key="row.id"
          class="list__row"
          :class="{ 'list__row--on': orderOf(row.id) >= 0 }"
          @click="toggle(row.id)"
        >
          <span class="num" style="width:3ch">{{ orderOf(row.id) >= 0 ? orderOf(row.id) + 1 : '·' }}</span>
          <span style="width:5ch">#{{ row.id }}</span>
          <span class="grow" style="overflow:hidden;text-overflow:ellipsis">{{ row.label }}</span>
          <span class="num muted">{{ row.tokens.length }}t</span>
          <button class="btn" type="button" title="Earlier"
            :disabled="orderOf(row.id) <= 0" @click.stop="move(row.id, -1)">&uarr;</button>
          <button class="btn" type="button" title="Later"
            :disabled="orderOf(row.id) < 0 || orderOf(row.id) === modelValue.blocks.length - 1"
            @click.stop="move(row.id, 1)">&darr;</button>
        </div>
      </div>

      <div class="row">
        <span class="muted grow" style="overflow:hidden;text-overflow:ellipsis">
          {{ api.selectionLabel(modelValue) }}
        </span>
        <span class="num" :class="{ 'picker__over': overCapacity }">
          {{ fmt.int(tokens) }}<span class="muted" v-if="api.state.model"> / {{ fmt.int(api.state.model.contextCapacity) }}</span> tok
        </span>
        <button class="btn" type="button"
          :disabled="modelValue.checkpoint === null && !modelValue.blocks.length"
          @click="clear">CLEAR</button>
      </div>

      <div v-if="issues.length" class="picker__issues">
        <div v-for="(issue, index) in issues" :key="index">! {{ issue }}</div>
      </div>
    </div>
  `,
});
