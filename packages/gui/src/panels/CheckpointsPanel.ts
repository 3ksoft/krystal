import { computed, defineComponent, inject, ref } from "vue";
import { ENGINE_KEY } from "../engine/key.ts";
import { emptySelection, type ContextSelection } from "../engine/useEngine.ts";
import ContextPicker from "../ui/ContextPicker.ts";
import { fmt, TosWindow } from "../ui/tos.ts";

interface TreeRow {
  id: number;
  label: string;
  depth: number;
  position: number;
  parts: number;
}

export default defineComponent({
  name: "CheckpointsPanel",
  components: { ContextPicker, TosWindow },
  setup() {
    const api = inject(ENGINE_KEY)!;
    const label = ref("");
    // This window's own context. Nothing else reads it.
    const selection = ref<ContextSelection>(emptySelection());
    const inspecting = ref<number | null>(null);

    const ready = computed(() => api.state.phase === "ready");
    // A checkpoint needs something to materialize. Gate the control on that
    // rather than letting the click fail and reporting it after the fact.
    const hasContext = computed(() =>
      selection.value.checkpoint !== null || selection.value.blocks.length > 0
    );
    const inspected = computed(() =>
      inspecting.value === null ? null : api.checkpoint(inspecting.value) ?? null
    );

    /**
     * Checkpoints branch, so render them as a tree rooted at the ones with no
     * base. Depth is what makes a branch visible at a glance.
     */
    const tree = computed<TreeRow[]>(() => {
      const rows: TreeRow[] = [];
      const byBase = new Map<number | null, typeof api.state.checkpoints>();
      for (const row of api.state.checkpoints) {
        const list = byBase.get(row.base) ?? [];
        list.push(row);
        byBase.set(row.base, list);
      }
      const walk = (base: number | null, depth: number): void => {
        for (const row of byBase.get(base) ?? []) {
          rows.push({
            id: row.id,
            label: row.label,
            depth,
            position: row.position,
            parts: row.contents.length,
          });
          walk(row.id, depth + 1);
        }
      };
      walk(null, 0);
      return rows;
    });

    async function create(): Promise<void> {
      try {
        const row = await api.createCheckpoint(selection.value, label.value);
        label.value = "";
        inspecting.value = row.id;
        // Chain naturally: the new checkpoint becomes the base for the next one.
        selection.value = { checkpoint: row.id, blocks: [] };
      } catch {
        // Surfaced through state.error.
      }
    }

    async function drop(id: number): Promise<void> {
      try {
        await api.dropCheckpoint(id);
        if (inspecting.value === id) inspecting.value = null;
        if (selection.value.checkpoint === id) {
          selection.value = { ...selection.value, checkpoint: null };
        }
      } catch {
        // Surfaced through state.error.
      }
    }

    return { api, label, selection, inspecting, inspected, ready, hasContext, tree, create, drop, fmt };
  },
  template: `
    <TosWindow title="CHECKPOINTS" icon="▞" :span="4">
      <div class="stack">
        <ContextPicker v-model="selection" :exclude-checkpoint="null" compact />

        <div class="row">
          <input class="grow" type="text" v-model="label" placeholder="name (optional)" :disabled="!ready" />
          <button
            class="btn btn--default"
            type="button"
            :disabled="!ready || !hasContext"
            :title="hasContext ? 'Freeze the selected context' : 'Choose a base checkpoint or at least one block'"
            @click="create"
          >CHECKPOINT</button>
        </div>
        <div v-if="ready && !hasContext" class="muted">
          pick a base checkpoint or a block above to enable CHECKPOINT
        </div>

        <div class="list" style="max-height:150px">
          <div v-if="!tree.length" class="list__empty">no checkpoints</div>
          <div
            v-for="row in tree"
            :key="row.id"
            class="list__row"
            :class="{ 'list__row--on': inspecting === row.id }"
            @click="inspecting = inspecting === row.id ? null : row.id"
          >
            <span class="muted" :style="{ paddingLeft: (row.depth * 10) + 'px' }">{{ row.depth ? '└─' : '●' }}</span>
            <span style="width:5ch">#{{ row.id }}</span>
            <span class="grow" style="overflow:hidden;text-overflow:ellipsis">{{ row.label }}</span>
            <span class="num muted" title="materialized context tokens">pos {{ row.position }}</span>
            <button class="btn" type="button" title="DropCheckpoint" :disabled="!ready" @click.stop="drop(row.id)">DEL</button>
          </div>
        </div>

        <template v-if="inspected">
          <div class="muted">#{{ inspected.id }} contains {{ inspected.contents.length }} block(s)</div>
          <div class="content">
            <div v-if="!inspected.contents.length" class="muted">empty</div>
            <div v-for="(part, index) in inspected.contents" :key="index" class="content__part">
              <div class="content__origin">b#{{ part.blockId }} {{ part.label }} — {{ part.tokenCount }} tok</div>
              <pre class="pre">{{ part.text }}</pre>
            </div>
          </div>
          <dl class="kv">
            <dt>position</dt><dd>{{ fmt.int(inspected.position) }} tok</dd>
            <dt>prefilled</dt><dd>{{ fmt.int(inspected.cost.prefillTokens) }} tok</dd>
            <dt>restored</dt><dd>{{ fmt.bytes(inspected.cost.restoredCheckpointBytes) }}</dd>
            <dt>snapshot</dt><dd>{{ fmt.bytes(inspected.cost.checkpointBytes) }}</dd>
            <dt>create</dt><dd>{{ fmt.us(inspected.cost.checkpointCreateUs) }}</dd>
          </dl>
        </template>
        <div v-else class="muted">select a checkpoint to see what it materializes</div>
      </div>
    </TosWindow>
  `,
});
