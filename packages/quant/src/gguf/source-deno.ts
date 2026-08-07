/** Deno-only file source. Requires the `Deno` global; see ./source.ts. */
import type { RandomAccessSource } from "./source.ts";

export class DenoFileSource implements RandomAccessSource {
  readonly size: number;
  private constructor(private readonly file: Deno.FsFile, size: number) {
    this.size = size;
  }

  static async open(path: string): Promise<DenoFileSource> {
    const stat = await Deno.stat(path);
    if (!stat.isFile) throw new Error(`Not a file: ${path}`);
    const file = await Deno.open(path, { read: true });
    return new DenoFileSource(file, stat.size);
  }

  async read(offset: number, length: number): Promise<Uint8Array> {
    if (offset < 0 || length < 0 || offset + length > this.size) {
      throw new RangeError(`Read [${offset}, ${offset + length}) outside file size ${this.size}`);
    }
    await this.file.seek(offset, Deno.SeekMode.Start);
    const out = new Uint8Array(length);
    let cursor = 0;
    while (cursor < length) {
      const n = await this.file.read(out.subarray(cursor));
      if (n === null) throw new Error(`Unexpected EOF at ${offset + cursor}`);
      cursor += n;
    }
    return out;
  }

  close(): void {
    this.file.close();
  }
}
