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
    host: "127.0.0.1",
    port: 5174,
    strictPort: true,
    fs: { allow: [repoRoot] },
  },
  plugins: [ggufRangeServer()],
});
