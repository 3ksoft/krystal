import { readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const shaderDir = resolve(root, "src/shaders");
const includeDir = resolve(shaderDir, "includes");
const output = resolve(root, "src/shaders.generated.ts");

const shaderNames = [
  "embedding",
  "embedding_wq4",
  "rms_norm",
  "matmul_f16",
  "matmul_f32",
  "matmul_wq4",
  "residual_add",
  "silu_mul",
  "shortconv_prefill",
  "shortconv_continue",
  "shortconv_decode",
  "qk_norm_rope",
  "kv_store",
  "attention",
  "arena_copy",
  "argmax_candidates",
  "argmax",
] as const;

async function readMap(dir: string, names: readonly string[], prefix = "") {
  const entries = await Promise.all(names.map(async (name) => {
    const source = await readFile(resolve(dir, `${name}.wgsl`), "utf8");
    return [name, `${prefix}${source}`] as const;
  }));
  return Object.fromEntries(entries);
}

const includeNames = (await readdir(includeDir))
  .filter((name) => name.endsWith(".wgsl"))
  .map((name) => name.slice(0, -5))
  .sort();

const shaders = await readMap(shaderDir, shaderNames, "");
const includes = await readMap(includeDir, includeNames);

const generated = `// THIS FILE IS AUTO-GENERATED - DO NOT EDIT\n// Run: bun run shaders:generate\n\nexport const shaderSources = ${JSON.stringify(shaders, null, 2)} as const;\n\nexport const shaderIncludes = ${JSON.stringify(includes, null, 2)} as const;\n`;

await writeFile(output, generated);
console.log(`[webgpu] generated ${output}`);
