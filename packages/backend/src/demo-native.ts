// End-to-end driver for the "second leg":
//   Engine -> BinaryEngineTransport -> SpawnedNativeChannel -> native exe
//
// Usage:
//   bun run src/demo-native.ts            # mock exe (bun script, or the
//                                         # scriptc binary when it exists)
//   bun run src/demo-native.ts --dawn     # real Lfm2Forward on Dawn
import { resolve } from "node:path";
import { BinaryEngineTransport } from "@chomato/engine-ts/binary-transport";
import { SpawnedNativeChannel } from "@chomato/engine-ts/spawn";
import { Engine } from "@chomato/engine-ts/transport";
import { mockGenerate } from "./exe/mock-backend.ts";
import { pickExeCommand } from "./exe/pick-command.ts";

const backendRoot = resolve(import.meta.dir, "..");
const repoRoot = resolve(backendRoot, "..");
const useDawn = process.argv.includes("--dawn");

function pickCommand(): { command: string; args: string[]; env?: Record<string, string> } {
  if (useDawn) {
    return {
      command: "bun",
      args: ["run", resolve(backendRoot, "src/exe/dawn-exe.ts")],
      env: {
        ...(process.env as Record<string, string>),
        CHOMATO_MODEL:
          process.env.CHOMATO_MODEL ?? resolve(repoRoot, "models/LFM2.5-1.2B-Instruct-WQ4.wq4"),
      },
    };
  }
  return pickExeCommand(backendRoot);
}

const started = Date.now();
const { command, args, env } = pickCommand();
console.log(`spawning: ${command} ${args.join(" ")}`);

const engine = new Engine(
  new BinaryEngineTransport(new SpawnedNativeChannel(command, args, { env })),
);

const block = await engine.putBlock(Uint32Array.of(1, 2, 3));
console.log("putBlock ->", block);

const checkpoint = await engine.checkpoint({ blocks: [block] });
console.log("checkpoint ->", checkpoint);

const tokens: number[] = [];
for await (const token of engine.generate({ checkpoint }, { maxTokens: 4 })) {
  tokens.push(token);
}
console.log("generate ->", tokens);

if (!useDawn) {
  const expected = mockGenerate([1, 2, 3], 4);
  const ok = tokens.length === expected.length && tokens.every((value, i) => value === expected[i]);
  console.log(`expected -> ${JSON.stringify(expected)} ${ok ? "MATCH ✓" : "MISMATCH ✗"}`);
  if (!ok) throw new Error(`Native leg produced wrong tokens: ${JSON.stringify(tokens)}`);
}

await engine.close();
console.log(`closed · ${Date.now() - started} ms`);
