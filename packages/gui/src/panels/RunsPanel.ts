import { defineComponent, inject } from "vue";
import { ENGINE_KEY } from "../engine/key.ts";
import { fmt, TosWindow } from "../ui/tos.ts";

export default defineComponent({
  name: "RunsPanel",
  components: { TosWindow },
  setup() {
    const api = inject(ENGINE_KEY)!;
    return { api, fmt };
  },
  template: `
    <TosWindow title="RUNS" icon="═" :span="12" flush>
      <div v-if="!api.state.runs.length" class="list__empty" style="padding:var(--cell-x)">
        nothing generated yet
      </div>
      <div v-else class="list" style="border:0;max-height:300px">
        <div v-for="run in api.state.runs" :key="run.id" class="list__row" style="cursor:default;align-items:flex-start;padding:4px">
          <span class="num" style="width:4ch">{{ run.id }}</span>
          <span style="width:11ch">{{ run.kind }}</span>
          <span style="width:14ch;overflow:hidden;text-overflow:ellipsis" :title="run.contextLabel">{{ run.contextLabel }}</span>
          <span class="grow" style="overflow:hidden;text-overflow:ellipsis" :title="run.detail">
            {{ run.ok ? '' : '✗ ' }}{{ run.detail }}
          </span>
          <span class="num muted" style="width:11ch" title="context tokens recomputed">{{ fmt.int(run.cost.prefillTokens) }} pre</span>
          <span class="num muted" style="width:12ch" title="bytes restored from a checkpoint">{{ fmt.bytes(run.cost.restoredCheckpointBytes) }}</span>
          <span class="num muted" style="width:9ch" title="tokens emitted">{{ fmt.int(run.cost.generatedTokens) }} tok</span>
          <span class="num muted" style="width:10ch" title="wall clock">{{ fmt.ms(run.cost.wallMs) }}</span>
        </div>
      </div>
    </TosWindow>
  `,
});
