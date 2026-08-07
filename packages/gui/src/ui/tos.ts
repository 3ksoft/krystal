/**
 * GEM-flavoured widget set.
 *
 * These components carry the window/dialog chrome only; every panel supplies
 * its own body. Runtime templates are used deliberately — vite.config.ts
 * aliases the Vue runtime compiler so this harness needs no SFC toolchain.
 */
import { computed, defineComponent, onBeforeUnmount, onMounted, ref } from "vue";

export const TosWindow = defineComponent({
  name: "TosWindow",
  props: {
    title: { type: String, required: true },
    span: { type: Number, default: 6 },
    flush: { type: Boolean, default: false },
    /** Rendered right of the title, before the fold gadget. */
    closable: { type: Boolean, default: false },
  },
  emits: ["close"],
  setup(props, { emit }) {
    const folded = ref(false);
    const spanClass = computed(() => `win--span${props.span}`);
    return { folded, spanClass, emit };
  },
  template: `
    <section class="win" :class="[spanClass, { 'win--collapsed': folded }]">
      <div class="win__bar stripes">
        <button
          v-if="closable"
          class="win__gadget"
          type="button"
          title="Close"
          @click="emit('close')"
        ><span class="win__gadget-mark win__gadget-mark--filled"></span></button>
        <div class="win__title"><span>{{ title }}</span></div>
        <button
          class="win__gadget win__gadget--right"
          type="button"
          :title="folded ? 'Unfold' : 'Fold'"
          @click="folded = !folded"
        ><span class="win__gadget-mark"></span></button>
      </div>
      <div class="win__body" :class="{ 'win__body--flush': flush }">
        <slot />
      </div>
    </section>
  `,
});

export const TosMeter = defineComponent({
  name: "TosMeter",
  props: {
    value: { type: Number, required: true },
    label: { type: String, default: "" },
  },
  setup(props) {
    const pct = computed(() => Math.max(0, Math.min(1, props.value)) * 100);
    return { pct };
  },
  template: `
    <div class="meter">
      <div class="meter__fill dither dither--accent" :style="{ width: pct + '%' }"></div>
      <div class="meter__label">{{ label || (pct.toFixed(0) + '%') }}</div>
    </div>
  `,
});

export const TosAlert = defineComponent({
  name: "TosAlert",
  props: {
    icon: { type: String, default: "!" },
    title: { type: String, default: "" },
  },
  emits: ["dismiss"],
  setup(_props, { emit }) {
    function dismiss(): void {
      emit("dismiss");
    }
    // Escape is the GEM reflex for "make this go away".
    function onKey(event: KeyboardEvent): void {
      if (event.key === "Escape" || event.key === "Enter") dismiss();
    }
    onMounted(() => document.addEventListener("keydown", onKey));
    onBeforeUnmount(() => document.removeEventListener("keydown", onKey));
    return { dismiss };
  },
  template: `
    <div class="modal">
      <div class="modal__scrim dither" @click="dismiss"></div>
      <div class="alert" role="alertdialog">
        <div class="alert__head">
          <div class="alert__icon">{{ icon }}</div>
          <div class="alert__body">
            <div v-if="title"><strong>{{ title }}</strong></div>
            <pre class="pre"><slot /></pre>
          </div>
        </div>
        <div class="alert__buttons">
          <button class="btn btn--default" type="button" autofocus @click="dismiss">OK</button>
        </div>
      </div>
    </div>
  `,
});

export interface MenuEntry {
  label: string;
  hint?: string;
  disabled?: boolean;
  separator?: boolean;
  action?: () => void;
}

export const TosMenuBar = defineComponent({
  name: "TosMenuBar",
  props: {
    menus: { type: Array as () => Array<{ label: string; entries: MenuEntry[] }>, required: true },
    brand: { type: String, default: "" },
  },
  setup() {
    const open = ref<string | null>(null);
    const anchor = ref({ x: 0, y: 0 });

    function toggle(label: string, event: MouseEvent): void {
      const target = event.currentTarget as HTMLElement;
      const box = target.getBoundingClientRect();
      anchor.value = { x: box.left, y: box.bottom };
      open.value = open.value === label ? null : label;
    }

    function choose(entry: MenuEntry): void {
      open.value = null;
      if (!entry.disabled) entry.action?.();
    }

    return { open, anchor, toggle, choose };
  },
  template: `
    <div class="menubar" @mouseleave="open = null">
      <button
        v-for="menu in menus"
        :key="menu.label"
        class="menubar__item"
        type="button"
        :aria-expanded="open === menu.label"
        @click="toggle(menu.label, $event)"
        @mouseenter="open && toggle(menu.label, $event)"
      >{{ menu.label }}</button>
      <span class="menubar__spacer"></span>
      <span v-if="brand" class="menubar__brand">{{ brand }}</span>

      <div
        v-if="open"
        class="menu"
        :style="{ left: anchor.x + 'px', top: anchor.y + 'px' }"
      >
        <template v-for="(entry, index) in (menus.find(m => m.label === open)?.entries ?? [])" :key="index">
          <div v-if="entry.separator" class="menu__sep"></div>
          <button
            v-else
            class="menu__item"
            type="button"
            :disabled="entry.disabled"
            @click="choose(entry)"
          ><span>{{ entry.label }}</span><span v-if="entry.hint" class="muted">{{ entry.hint }}</span></button>
        </template>
      </div>
    </div>
  `,
});

/** Byte/duration/number formatting shared by every panel. */
export const fmt = {
  bytes(value?: number): string {
    if (value === undefined || !Number.isFinite(value)) return "—";
    if (value === 0) return "0 B";
    const units = ["B", "KiB", "MiB", "GiB"];
    const order = Math.min(units.length - 1, Math.floor(Math.log2(Math.abs(value)) / 10));
    const scaled = value / 1024 ** order;
    return `${scaled.toFixed(order === 0 ? 0 : 2)} ${units[order]}`;
  },
  ms(value?: number): string {
    if (value === undefined || !Number.isFinite(value)) return "—";
    return value < 1000 ? `${value.toFixed(1)} ms` : `${(value / 1000).toFixed(2)} s`;
  },
  us(value?: number): string {
    if (value === undefined || !Number.isFinite(value)) return "—";
    return value < 1000 ? `${value.toFixed(0)} µs` : `${(value / 1000).toFixed(2)} ms`;
  },
  int(value?: number): string {
    if (value === undefined || !Number.isFinite(value)) return "—";
    return value.toLocaleString("en-US");
  },
};
