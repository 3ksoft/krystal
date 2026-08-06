import { spawn, type ChildProcess } from "node:child_process";
import type { BinaryChannel } from "./binary-transport";

export interface SpawnedNativeChannelOptions {
  cwd?: string;
  env?: Record<string, string | undefined>;
}

/**
 * BinaryChannel over a spawned process's stdin/stdout. This is the transport
 * side of the "binary leg": engine-ts Engine <-> spawned native exe speaking
 * @chomato/bridge frames (scriptc-compiled binary or a bun script).
 */
export class SpawnedNativeChannel implements BinaryChannel {
  private readonly child: ChildProcess;
  private readonly listeners = new Set<(bytes: Uint8Array) => void>();
  private exitCode: number | null = null;

  constructor(
    command: string,
    args: readonly string[] = [],
    options: SpawnedNativeChannelOptions = {},
  ) {
    this.child = spawn(command, [...args], {
      stdio: ["pipe", "pipe", "inherit"],
      ...options,
    });
    this.child.stdout!.on("data", (chunk: Uint8Array) => {
      // Copy: a single chunk may span frame boundaries or contain several.
      for (const listener of this.listeners) listener(chunk.slice());
    });
    this.child.on("exit", (code) => {
      this.exitCode = code;
    });
    this.child.on("error", (error) => {
      console.error("[spawn-native-channel] child error:", error.message);
    });
  }

  send(bytes: Uint8Array): void {
    if (this.exitCode !== null) throw new Error("Native channel is closed");
    this.child.stdin!.write(bytes);
  }

  subscribe(listener: (bytes: Uint8Array) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  async close(): Promise<void> {
    const child = this.child;
    if (child.stdin) child.stdin.end();
    if (this.exitCode === null) child.kill();
    await new Promise<void>((resolve) => {
      if (this.exitCode !== null) return resolve();
      child.once("exit", () => resolve());
    });
    this.listeners.clear();
  }
}
