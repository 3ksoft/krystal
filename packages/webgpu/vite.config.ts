import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const packageRoot = fileURLToPath(new URL("./", import.meta.url));
const repoRoot = fileURLToPath(new URL("../../", import.meta.url));

export default defineConfig({
  root: fileURLToPath(new URL("./test/browser/", import.meta.url)),
  server: {
    fs: {
      // lfm2 imports the sibling schema package in the monorepo.
      allow: [packageRoot, repoRoot],
    },
  },
  build: {
    outDir: fileURLToPath(new URL("./dist/browser/", import.meta.url)),
    emptyOutDir: true,
  },
});
