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

## Execution telemetry

`ExecutionStats` is emitted by the backend for a `Generate` operation and is
execution truth, not a client-side estimate. In particular:

- `prefillTokens` counts tokens that actually traversed fresh prefill/continuation work,
- `checkpointHits` means a materialized checkpoint state was actually restored,
- `checkpointMisses` means a requested checkpoint could not be reused,
- `restoredBytes` is the number of checkpoint bytes physically restored.

`engine-ts` only aggregates these values; it must never infer them from the
requested `ContextRef`.

Wire version `2` adds `ExecutionStats` and changes the fixed `EngineEvent` body from 9 to 21 bytes; native peers must be regenerated/rebuilt together with this bridge.

### Checkpoint metrics (wire v3)

`ExecutionStats` also carries physical checkpoint accounting:

- `checkpointBytes`: snapshot bytes materialized by `CreateCheckpoint`,
- `kvBytes`: KV bytes stored in the snapshot,
- `kvCapacityBytes`: live KV capacity represented by that snapshot,
- `convBytes`: recurrent convolution state bytes,
- `hiddenBytes`: last-hidden snapshot bytes,
- `checkpointCreateUs`: backend wall time for checkpoint materialization,
- `checkpointRestoreUs`: restore-only duration when measurable without perturbing execution (`0` means unavailable).

These values are backend facts. For a compact checkpoint implementation `kvBytes`
should scale with the populated prefix while `kvCapacityBytes` remains the live
capacity, making `kvBytes / kvCapacityBytes` a useful utilization metric.

Wire version `3` expands the fixed `ExecutionStats` body to 48 bytes and
`EngineEvent` to 49 bytes; regenerate/rebuild native peers together with the
bridge.
