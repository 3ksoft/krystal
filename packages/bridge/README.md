# @chomato/bridge

Portable control ABI shared by `engine-ts`, the in-process WebGPU backend, and
future native engines.

The schema contains only fixed-size records. Variable data lives in the frame
payload:

| message | payload |
| --- | --- |
| `PutBlock` | `tokenCount` little-endian `u32` token IDs |
| `CreateCheckpoint` | `context.blockCount` little-endian `u32` block IDs |
| `Generate` | `context.blockCount` little-endian `u32` block IDs |
| `Failed` event | `messageBytes` bytes of UTF-8 text |

`ContextRef.checkpoint = 0` means no checkpoint. Block IDs are ordered exactly
as supplied by the caller.

A byte-oriented transport frames data as:

```
FrameHeader
EngineCommand | EngineEvent
payload
```

Run `bun run build` to regenerate standalone C++, TS types, and the
analyzed plan under `generated/`.
