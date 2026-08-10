/** Node/Bun file source. Requires `node:fs/promises`; see ./source.ts. */
import { open, type FileHandle } from "node:fs/promises";
import type { RandomAccessSource } from "./source.ts";

export class NodeFileSource implements RandomAccessSource {
  readonly size: number;
  private constructor(private readonly handle: FileHandle, size: number) {
    this.size = size;
  }

  static async open(path: string): Promise<NodeFileSource> {
    const handle = await open(path, "r");
    try {
      const stat = await handle.stat();
      if (!stat.isFile) throw new Error(`Not a file: ${path}`);
      return new NodeFileSource(handle, stat.size);
    } catch (error) {
      await handle.close();
      throw error;
    }
  }

  async read(offset: number, length: number): Promise<Uint8Array> {
    if (offset < 0 || length < 0 || offset + length > this.size) {
      throw new RangeError(`Read [${offset}, ${offset + length}) outside file size ${this.size}`);
    }
    const out = new Uint8Array(length);
    let cursor = 0;
    while (cursor < length) {
      const { bytesRead } = await this.handle.read(out, cursor, length - cursor, offset + cursor);
      if (bytesRead === 0) throw new Error(`Unexpected EOF at ${offset + cursor}`);
      cursor += bytesRead;
    }
    return out;
  }

  async close(): Promise<void> {
    await this.handle.close();
  }
}
