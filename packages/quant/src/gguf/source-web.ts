/** Browser sources. Require `Blob`, `Headers` and `fetch`; see ./source.ts. */
import type { RandomAccessSource } from "./source.ts";

export class BlobSource implements RandomAccessSource {
  readonly size: number;
  constructor(private readonly blob: Blob) {
    this.size = blob.size;
  }

  async read(offset: number, length: number): Promise<Uint8Array> {
    return new Uint8Array(await this.blob.slice(offset, offset + length).arrayBuffer());
  }
}

/** The server must support HTTP Range requests. */
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
