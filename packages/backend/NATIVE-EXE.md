# Native inference exe — state and open work

Goal: run inference in a scriptc-compiled native binary, with no JS runtime in
the process. The motivation is not startup time — it is that Dawn under bun is
flaky, and moving the GPU out of bun removes the whole class rather than
patching around it.

Status as of 2026-08-08. Measured on RTX 3060, scriptc 0.0.22, Linux.

## Two legs, one protocol

Both speak the `@chomato/bridge` frame protocol over stdio and share
`BridgeEngineBackend` + `runStdioBridge`. They differ only in IO and generation:

| entry | runtime | IO | generation | scriptc coverage |
| --- | --- | --- | --- | --- |
| `src/exe/native-exe.ts` | native ELF | `ffiIo` (C shim) | mock | **270 stmts, 100% static** |
| `src/exe/dawn-exe.ts` | bun | `nodeIo` | real Dawn | n/a (not a target) |
| `src/exe/dawn-native-exe.ts` | native (goal) | `ffiIo` | real Dawn | **1446 stmts, 94.9% static, 14 dynamic sites** |

`NativeIo` (`src/exe/native-io.ts`) is the seam that lets one bridge loop run in
both worlds. Keep it that way.

Reproduce any number with:

```bash
cd packages/backend
./node_modules/.bin/scriptc coverage src/exe/dawn-native-exe.ts
```

## What is already done

- **Portability blockers cleared.** The Dawn path used to fail type resolution
  on 17 errors before coverage could even run: `Deno`/`Blob`/`HeadersInit` from
  `quant/src/gguf/source.ts`, `navigator` in `webgpu/src/device.ts`, plus
  `Uint8Array` variance and Map destructuring in `engine-ts`. The GGUF sources
  are now split by runtime (`source.ts` = interface, `source-deno.ts`,
  `source-web.ts`).
- **ABI codec generated statically.** `bun run build:schema` emits
  `packages/schema/generated/schema.codec.ts` (77 exports, plain `DataView`,
  one type-only import) and `schema.types.ts`. Import them as
  `@sandblaster/schema/codec` and `/types`. This is what lets the native exe
  serialize the engine ABI without arktype or @schema-pop in the process — the
  default codec is built at runtime with `new Function`, which can never
  compile statically.
  `src/exe/codec-probe.ts` exercises the codec alone: **587 stmts, 92% static**,
  the remainder being scriptc stdlib gaps (`ArrayBuffer`, `DataView`,
  `TextDecoder`), not our code.

## The remaining 14 dynamic sites (was 26)

The startup-graph rebuild is gone. `packages/webgpu/src/lfm2.ts` now builds the
runtime definition with `Sandblaster.fromArtifact(lfm2.artifact.generated.ts)`
via `lfm2-artifact.ts` (artifact handle creation) + `lfm2-layout.ts` (pure-TS
constants/arena/pass geometry). The arktype DSL (`lfm2-definition.ts`), `$`,
@schema-pop and the generated schema scope are out of the runtime graph — the
AOT scripts still use them. 2026-08-08 measured 26 -> 14 dynamic sites.

| sites | source | owner |
| --- | --- | --- |
| ~4 | the WebGPU binding: `import("webgpu")`, `Promise<WebGpuContext>`, `@webgpu/types`, `createWebGpuDevice`/`device`/`adapter` | the shim |
| ~4 | `@sandblaster/core` package boundary (SC2013) | resolution wrinkle, below |
| ~6 | the `lfm2` definition/resources/programs/passes objects, downstream of the package boundary | the shim |

**Package resolution wrinkle (new finding).** scriptc puts bare-specifier npm
packages on the dynamic island by policy; `--npm-static @sandblaster/core` and
tsconfig `paths` do not change that. The stub in `packages/backend/stubs/core`
is unreachable by default because scriptc resolves `@sandblaster/core` through
`packages/webgpu/node_modules` (the real bun-linked package). Pointing that
symlink at the stub makes the graph typecheck against the stub's surface —
`stubs/core` now declares `fromArtifact`/`resource`/`computeProgram`/`compile`
plus the device surface (`createBuffer`, `createCommandEncoder`, `queue.submit`,
`compiledInfo.byteSize`, `AnyComputeHandle.manifest`) — but the swap breaks
bun/browser, which need the real core. Deciding how a scriptc build selects the
stub without disturbing browser resolution belongs to the shim step.

## The shim: scriptc FFI has a trap, but a cheap one

Dawn's C API is handle-based: almost every call returns an opaque pointer that
has to be captured. scriptc 0.0.22 miscompiles exactly that shape.

Upstream: [#21](https://github.com/vercel-labs/scriptc/issues/21) (open) and
[#71](https://github.com/vercel-labs/scriptc/issues/71) (closed as *completed*
on 2026-08-01, before v0.0.22 shipped on 2026-08-03). **It still reproduces on
0.0.22** — verified, do not assume the closed issue means fixed. Re-check on
each upgrade.

The failure is silent: the build succeeds and the binary dies at load with
`Uncaught ReferenceError: <name> is not defined`.

### What breaks, exactly

The trigger is a **bare single-assignment binding**, not `const`:

| call position | result |
| --- | --- |
| `ffi(1);` statement | ok |
| `f(ffi(21))` argument | ok |
| `ffi(21) + 1` expression | ok |
| `let y = ffi(21); y = y + 1;` later reassigned | ok |
| `const x = ffi(21);` | **ReferenceError** |
| `let c = 0; c = ffi(21);` | **ReferenceError** |
| `const x = ffi(21);` inside a function | **ReferenceError** |

### The cheap workaround

Any wrapping at all restores correct binding — no C-side contortions needed:

| form | result |
| --- | --- |
| `const h = ffi(21) + 0;` | ok |
| `const h = Number(ffi(21));` | ok |
| `const a = [ffi(21)];` | ok |
| `const o = { v: ffi(21) };` | ok |
| `function g() { return ffi(21); }` | ok |
| `let h = ffi(21); h = h;` | ok |

**Recommended idiom.** Declare the raw binding private and expose it through a
one-line wrapper. `return ffi(...)` is a safe position, so every call site
afterwards is ordinary TypeScript:

```ts
declare function _wgpuDeviceCreateBuffer(device: number, desc: number): number;

function wgpuDeviceCreateBuffer(device: number, desc: number): number {
  return _wgpuDeviceCreateBuffer(device, desc);
}
```

This matters for scale: it means the shim can bind `webgpu.h` directly and
return handles normally. The out-buffer trick in `src/exe/shim.c`
(`posix_read_fill` returns its byte count through `buf[0..8)`) was written
before this was mapped and is **not** required for the Dawn surface — one
wrapper per symbol is enough.

## Surface to bind

Sandblaster touches **37 distinct WebGPU calls**; chomato needs ~25 of them
(the render-pipeline ones are unused). Enumerate the current list with:

```bash
grep -ohE "device\.[a-zA-Z]+\(|queue\.[a-zA-Z]+\(|pass\.[a-zA-Z]+\(|encoder\.[a-zA-Z]+\(" \
  /home/kr/Projects/sandblaster-v2/src/*.ts | sort -u
```

Grouping: adapter/device acquisition, buffer + query-set creation, shader
module and pipeline creation, bind group + layout creation, command encoding
(begin/end compute pass, copy, resolve, finish), dispatch (direct and
indirect), queue submit/writeBuffer/onSubmittedWorkDone, and buffer
map/getMappedRange/unmap for readback.

The surface is bounded because **Sandblaster is the abstraction boundary**. Code
calling WebGPU directly would have an open-ended surface; this one is
enumerable, which is what makes a hand-written shim tractable.

## Suggested order

1. ✅ `Sandblaster.fromArtifact()` — done on the sandblaster side; wired into
   the chomato runtime 2026-08-08 (`lfm2-artifact.ts`, `lfm2-layout.ts`, stub
   surface). The graph went from 26 dynamic sites to 14; the remainder is the
   WebGPU binding + the package-resolution wrinkle above.
2. The Dawn shim, using the wrapper idiom above. It also owns the decision of
   how a scriptc build resolves `@sandblaster/core` to the stub.
3. Re-run `scriptc coverage src/exe/dawn-native-exe.ts` after each step.

## Known scriptc stdlib gaps hit so far

Not blockers to plan around yet, just things that will surface:
`ArrayBuffer`, `new DataView` over a buffer expression, `TextEncoder`/
`TextDecoder`, `Set` from an iterable, `Uint32Array.from`, `new Array(n)`,
`Array.fill`, `Object.hasOwn`, `String.fromCodePoint`, enum reverse lookup with
a runtime index.

`TextDecoder`/`TextEncoder` appear in the generated codec only for the
host-side string fields (`GpuTensor.name`, `MatmulKernelSpec.entryPoint`); the
GPU ABI structs the exe actually needs — `LlmRuntime`, `OpParams`,
`DecodeTelemetryEntry` — have no string fields.
