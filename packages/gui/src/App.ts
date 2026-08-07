/**
 * GEM desktop shell.
 *
 * The desktop is one grid of always-visible windows rather than draggable,
 * overlapping ones: this is a developer instrument where every primitive
 * should be readable at once, and GEM's window chrome is what carries the look.
 *
 * Windows are independent. Each one that needs a context owns its own
 * ContextSelection, so hiding BLOCKS or re-selecting inside it never changes
 * what CHECKPOINTS or GENERATE will run against.
 */
import { computed, defineComponent, onScopeDispose, provide, reactive, ref } from "vue";
import { ENGINE_KEY } from "./engine/key.ts";
import { useEngine } from "./engine/useEngine.ts";
import AboutDialog from "./panels/AboutDialog.ts";
import BlocksPanel from "./panels/BlocksPanel.ts";
import CheckpointsPanel from "./panels/CheckpointsPanel.ts";
import ModelPanel from "./panels/ModelPanel.ts";
import RunsPanel from "./panels/RunsPanel.ts";
import SchemaPanel from "./panels/SchemaPanel.ts";
import StatsPanel from "./panels/StatsPanel.ts";
import TokenizerPanel from "./panels/TokenizerPanel.ts";
import { fmt, TosAlert, TosMenuBar, type MenuEntry } from "./ui/tos.ts";

export default defineComponent({
  name: "App",
  components: {
    AboutDialog,
    BlocksPanel,
    CheckpointsPanel,
    ModelPanel,
    RunsPanel,
    SchemaPanel,
    StatsPanel,
    TokenizerPanel,
    TosAlert,
    TosMenuBar,
  },
  setup() {
    const api = useEngine();
    provide(ENGINE_KEY, api);
    onScopeDispose(() => void api.dispose());

    const invert = ref(false);
    const about = ref(false);
    const visible = reactive({
      model: true,
      blocks: true,
      checkpoints: true,
      generate: true,
      tokenizer: false,
      stats: true,
      runs: true,
    });

    const dismissed = ref<string | null>(null);
    const alert = computed(() => {
      const error = api.state.error;
      return error && error !== dismissed.value ? error : null;
    });

    function toggleInvert(): void {
      invert.value = !invert.value;
      document.documentElement.dataset.invert = invert.value ? "1" : "0";
    }

    const menus = computed<Array<{ label: string; entries: MenuEntry[] }>>(() => [
      {
        label: "Desk",
        entries: [
          { label: "About Chomato…", action: () => (about.value = true) },
          { separator: true, label: "" },
          { label: invert.value ? "Normal video" : "Inverse video", action: toggleInvert },
        ],
      },
      {
        label: "Windows",
        entries: (Object.keys(visible) as Array<keyof typeof visible>).map((key) => ({
          label: `${visible[key] ? "✓" : " "} ${key}`,
          action: () => {
            visible[key] = !visible[key];
          },
        })),
      },
      {
        label: "Debug",
        entries: [
          { label: "Refresh stats", action: () => api.refreshStats() },
          { label: "Reset counters", action: () => api.resetStats() },
        ],
      },
    ]);

    return { api, alert, about, dismissed, menus, visible, fmt };
  },
  template: `
    <div class="desktop">
      <TosMenuBar :menus="menus" brand="CHOMATO" />

      <div class="workspace dither">
        <ModelPanel v-if="visible.model" />
        <BlocksPanel v-if="visible.blocks" />
        <CheckpointsPanel v-if="visible.checkpoints" />
        <SchemaPanel v-if="visible.generate" />
        <TokenizerPanel v-if="visible.tokenizer" />
        <StatsPanel v-if="visible.stats" />
        <RunsPanel v-if="visible.runs" />
      </div>

      <div class="statusbar">
        <span
          class="statusbar__phase"
          :class="{ 'statusbar__phase--live': !['idle', 'error'].includes(api.state.phase) }"
        >{{ api.state.phase.toUpperCase() }}</span>
        <span class="statusbar__sep">│</span>
        <span class="grow" style="overflow:hidden;text-overflow:ellipsis">{{ api.state.status }}</span>
        <span class="statusbar__sep">│</span>
        <span>blk {{ api.state.blocks.length }}</span>
        <span class="statusbar__sep">│</span>
        <span>ckpt {{ api.state.checkpoints.length }}</span>
        <span class="statusbar__sep">│</span>
        <span>runs {{ api.state.runs.length }}</span>
      </div>

      <AboutDialog v-if="about" @dismiss="about = false" />
      <TosAlert v-else-if="alert" title="Engine error" @dismiss="dismissed = alert">{{ alert }}</TosAlert>
    </div>
  `,
});
