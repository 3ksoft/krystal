import { createApp } from "vue";
// Bundled, not fetched: the GUI has to work against a local model with no
// network, and a webfont that silently fails would take the 8x16 cell grid with
// it. Intel One Mono is OFL-1.1.
import "@fontsource/intel-one-mono/400.css";
import "@fontsource/intel-one-mono/700.css";
import App from "./App.ts";
import "./style.css";

createApp(App).mount("#app");
