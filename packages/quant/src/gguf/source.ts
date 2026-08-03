export interface RandomAccessSource {
  readonly size: number;
  read(offset: number, length: number): Promise<Uint8Array>;
  close?(): void | Promise<void>;
}

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

export class BlobSource implements RandomAccessSource {
  readonly size: number;
  constructor(private readonly blob: Blob) {
    this.size = blob.size;
  }

  async read(offset: number, length: number): Promise<Uint8Array> {
    return new Uint8Array(await this.blob.slice(offset, offset + length).arrayBuffer());
  }
}

/** Browser path. The server must support HTTP Range requests. */
export class HttpRangeSource implements RandomAccessSource {
  constructor(
    readonly url: string,
    readonly size: number,
    private readonly headers: HeadersInit = {},
  ) {}

  /** Discover file size without downloading the GGUF. */
  static async open(url: string, headers: HeadersInit = {}): Promise<HttpRangeSource> {
    const requestHeaders = new Headers(headers);
    const head = await fetch(url, { method: "HEAD", headers: requestHeaders });
    if (head.ok) {
      const rawLength = head.headers.get("Content-Length");
      const size = rawLength ? Number(rawLength) : NaN;
      if (Number.isFinite(size) && size > 0) {
        return new HttpRangeSource(url, size, headers);
      }
    }

    // Some static servers do not implement HEAD correctly. A one-byte range
    // still gives us the total length in Content-Range without fetching the file.
    const probeHeaders = new Headers(headers);
    probeHeaders.set("Range", "bytes=0-0");
    const probe = await fetch(url, { headers: probeHeaders });
    if (probe.status !== 206) {
      throw new Error(
        `Could not discover GGUF size for ${url}. HEAD=${head.status}, range probe=${probe.status}. ` +
        `The server must support HTTP Range requests.`,
      );
    }
    const contentRange = probe.headers.get("Content-Range");
    const match = contentRange?.match(/\/(\d+)$/);
    if (!match) throw new Error(`Missing total size in Content-Range: ${contentRange ?? "<none>"}`);
    return new HttpRangeSource(url, Number(match[1]), headers);
  }

  async read(offset: number, length: number): Promise<Uint8Array> {
    if (offset < 0 || length < 0 || offset + length > this.size) {
      throw new RangeError(`Read [${offset}, ${offset + length}) outside source size ${this.size}`);
    }
    if (length === 0) return new Uint8Array();

    const end = offset + length - 1;
    const headers = new Headers(this.headers);
    headers.set("Range", `bytes=${offset}-${end}`);
    const response = await fetch(this.url, { headers });
    if (response.status !== 206) {
      throw new Error(
        `Range request failed (${response.status}) for ${offset}-${end}. ` +
        `Refusing a full-file fallback for a ${this.size}-byte GGUF.`,
      );
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.byteLength !== length) {
      throw new Error(`Range ${offset}-${end} returned ${bytes.byteLength} bytes, expected ${length}`);
    }
    return bytes;
  }
}
