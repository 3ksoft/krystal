/**
 * GEM "About" dialog.
 *
 * The mark below is a placeholder drawn from the same 1px monochrome
 * vocabulary as the rest of the chrome; swap the <svg> body for the real icon
 * without touching anything else.
 */
import { defineComponent, onBeforeUnmount, onMounted } from "vue";

export const REPO_URL = "https://github.com/3ksoft/chomato";

export default defineComponent({
  name: "AboutDialog",
  emits: ["dismiss"],
  setup(_props, { emit }) {
    function dismiss(): void {
      emit("dismiss");
    }
    function onKey(event: KeyboardEvent): void {
      if (event.key === "Escape" || event.key === "Enter") dismiss();
    }
    onMounted(() => document.addEventListener("keydown", onKey));
    onBeforeUnmount(() => document.removeEventListener("keydown", onKey));
    return { dismiss, REPO_URL };
  },
  template: `
    <div class="modal">
      <div class="modal__scrim dither" @click="dismiss"></div>
      <div class="alert about" role="dialog" aria-label="About Chomato">
        <div class="alert__head">
          <div class="about__mark">
            <!-- PLACEHOLDER ICON — replace this svg with the real mark. -->
            <svg viewBox="0 0 32 32" width="48" height="48" aria-hidden="true">
              <rect x="1" y="1" width="30" height="30" fill="none" stroke="currentColor" stroke-width="2"/>
              <circle cx="16" cy="18" r="9" fill="none" stroke="currentColor" stroke-width="2"/>
              <path d="M8 9 h16 M16 9 v-4 M16 5 l5 2 M16 5 l-5 2"
                    fill="none" stroke="currentColor" stroke-width="2"/>
            </svg>
          </div>
          <div class="alert__body">
            <div class="about__name">CHOMATO</div>
            <div class="muted">developer harness for the LFM2 WebGPU engine</div>
            <dl class="kv about__kv">
              <dt>license</dt><dd>AGPL-3.0</dd>
              <dt>source</dt>
              <dd><a :href="REPO_URL" target="_blank" rel="noreferrer noopener">3ksoft/chomato</a></dd>
            </dl>
          </div>
        </div>
        <div class="alert__buttons">
          <button class="btn btn--default" type="button" autofocus @click="dismiss">OK</button>
        </div>
      </div>
    </div>
  `,
});
