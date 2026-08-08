import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../..");
const modelsDir = path.resolve(repoRoot, "models");

function ggufRangeServer(): Plugin {
  return {
    name: "chomato-gguf-range-server",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url) return next();
        const url = new URL(req.url, "http://localhost");
        if (!url.pathname.startsWith("/models/")) return next();

        const file = path.join(modelsDir, path.basename(decodeURIComponent(url.pathname)));
        let stat: fs.Stats;
        try {
          stat = fs.statSync(file);
        } catch {
          res.statusCode = 404;
          res.end("model not found");
          return;
        }
        if (!stat.isFile()) {
          res.statusCode = 404;
          res.end("model not found");
          return;
        }

        res.setHeader("Accept-Ranges", "bytes");
        res.setHeader("Content-Type", "application/octet-stream");
        res.setHeader("Cache-Control", "no-store");

        if (req.method === "HEAD") {
          res.statusCode = 200;
          res.setHeader("Content-Length", stat.size);
          res.end();
          return;
        }
        if (req.method !== "GET") {
          res.statusCode = 405;
          res.end("method not allowed");
          return;
        }

        const range = req.headers.range;
        if (!range) {
          res.statusCode = 200;
          res.setHeader("Content-Length", stat.size);
          fs.createReadStream(file).pipe(res);
          return;
        }

        const match = /^bytes=(\d+)-(\d*)$/.exec(range);
        if (!match) {
          res.statusCode = 416;
          res.setHeader("Content-Range", `bytes */${stat.size}`);
          res.end();
          return;
        }

        const start = Number(match[1]);
        const requestedEnd = match[2] ? Number(match[2]) : stat.size - 1;
        const end = Math.min(requestedEnd, stat.size - 1);
        if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || start > end || start >= stat.size) {
          res.statusCode = 416;
          res.setHeader("Content-Range", `bytes */${stat.size}`);
          res.end();
          return;
        }

        res.statusCode = 206;
        res.setHeader("Content-Range", `bytes ${start}-${end}/${stat.size}`);
        res.setHeader("Content-Length", end - start + 1);
        fs.createReadStream(file, { start, end }).pipe(res);
      });
    },
  };
}

export default defineConfig({
  root: here,
  // gui2 intentionally uses runtime templates to keep the first harness free of SFC tooling.
  resolve: { alias: { vue: "vue/dist/vue.esm-bundler.js" } },
  server: {
    /**
     * Bound to all interfaces so a device on the LAN can reach it (10.1.1.3).
     *
     * WebGPU is gated on a secure context, and plain http to an IP is not one —
     * `navigator.gpu` is simply undefined there, with no error to catch. A
     * phone therefore needs either `adb reverse tcp:5174 tcp:5174` (which makes
     * it localhost on the device, and is a secure context) or Chrome's
     * unsafely-treat-insecure-origin-as-secure flag for this origin.
     */
    host: "0.0.0.0",
    port: 5174,
    strictPort: true,
    fs: { allow: [repoRoot] },
    /**
     * No HMR, and no auto reload either — a reload throws away the loaded model
     * along with every block and checkpoint in memory, which is most of the
     * session. Reloading is the developer's call, not the file watcher's.
     *
     * Keep the file watcher on. It is what invalidates the module graph, and
     * that is independent of HMR: edits are still served fresh on a manual
     * refresh, they just no longer push themselves into the page. (`watch: null`
     * would disable HMR just as thoroughly and also break the refresh —
     * measured: the transform result stays cached and the edit never arrives.)
     */
    hmr: false,
  },
  plugins: [ggufRangeServer()],
});
