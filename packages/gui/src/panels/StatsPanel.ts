import { computed, defineComponent, inject } from "vue";
import { ENGINE_KEY } from "../engine/key.ts";
import { fmt, TosWindow } from "../ui/tos.ts";

export default defineComponent({
  name: "StatsPanel",
  components: { TosWindow },
  setup() {
    const api = inject(ENGINE_KEY)!;
    const stats = computed(() => api.state.stats);

    /**
     * The headline number for this engine: context tokens that had to be
     * recomputed versus tokens a checkpoint made free.
     */
    const reuse = computed(() => {
      const s = stats.value;
      if (!s) return null;
      const total = s.contextTokens;
      if (total <= 0) return null;
      return { prefilled: s.prefillTokens, total, ratio: 1 - Math.min(1, s.prefillTokens / total) };
    });

    return { api, stats, reuse, fmt };
  },
  template: `
    <TosWindow title="STATS" icon="▄" :span="4">
      <div class="stack">
        <div class="row">
          <button class="btn" type="button" @click="api.refreshStats()">REFRESH</button>
          <button class="btn" type="button" @click="api.resetStats()">RESET</button>
        </div>

        <template v-if="stats">
          <div v-if="reuse">
            <div class="muted">context reuse &mdash; {{ fmt.int(reuse.prefilled) }} of {{ fmt.int(reuse.total) }} tokens recomputed</div>
            <div class="meter" style="margin-top:4px">
              <div class="meter__fill dither dither--accent" :style="{ width: (reuse.ratio * 100) + '%' }"></div>
              <div class="meter__label">{{ (reuse.ratio * 100).toFixed(0) }}% reused</div>
            </div>
          </div>

          <dl class="kv">
            <dt>commands</dt><dd>{{ fmt.int(stats.commands) }}</dd>
            <dt>blocks put</dt><dd>{{ fmt.int(stats.blocksPut) }}</dd>
            <dt>blocks dropped</dt><dd>{{ fmt.int(stats.blocksDropped) }}</dd>
            <dt>context tokens</dt><dd>{{ fmt.int(stats.contextTokens) }}</dd>
            <dt>prefill tokens</dt><dd>{{ fmt.int(stats.prefillTokens) }}</dd>
            <dt>ckpt created</dt><dd>{{ fmt.int(stats.checkpointsCreated) }}</dd>
            <dt>ckpt dropped</dt><dd>{{ fmt.int(stats.checkpointsDropped) }}</dd>
            <dt>ckpt hits</dt><dd>{{ fmt.int(stats.checkpointHits) }}</dd>
            <dt>ckpt misses</dt><dd>{{ fmt.int(stats.checkpointMisses) }}</dd>
            <dt>restored</dt><dd>{{ fmt.bytes(stats.restoredCheckpointBytes) }}</dd>
            <dt>snapshot</dt><dd>{{ fmt.bytes(stats.checkpointBytes) }}</dd>
            <dt>kv</dt><dd>{{ fmt.bytes(stats.kvBytes) }}</dd>
            <dt>conv</dt><dd>{{ fmt.bytes(stats.convBytes) }}</dd>
            <dt>hidden</dt><dd>{{ fmt.bytes(stats.hiddenBytes) }}</dd>
            <dt>ckpt create</dt><dd>{{ fmt.us(stats.checkpointCreateUs) }}</dd>
            <dt>ckpt restore</dt><dd>{{ fmt.us(stats.checkpointRestoreUs) }}</dd>
            <dt>generations</dt><dd>{{ fmt.int(stats.generations) }}</dd>
            <dt>tokens out</dt><dd>{{ fmt.int(stats.generatedTokens) }}</dd>
            <dt>cancellations</dt><dd>{{ fmt.int(stats.cancellations) }}</dd>
          </dl>
        </template>
        <div v-else class="muted">engine not started</div>

        <div v-if="api.state.shaderCoverage.length">
          <div class="muted">shaders executed ({{ api.state.shaderCoverage.length }})</div>
          <div class="row row--tight" style="margin-top:4px">
            <span
              v-for="name in api.state.shaderCoverage"
              :key="name"
              style="border:1px solid var(--fg);padding:0 4px"
            >{{ name }}</span>
          </div>
        </div>
      </div>
    </TosWindow>
  `,
});
