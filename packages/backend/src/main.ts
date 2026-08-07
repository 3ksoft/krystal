// scriptc experiment harness.
//
// Imports the REAL chomato webgpu host-side modules and runs their pure logic
// as a native binary:
//   - lfm2.ts / lfm2-definition.ts  -> arena layout, capacities, pass geometry
//   - pass.ts                       -> lfm2Pass dispatch resolution + Lfm2Executor
//   - model.ts + quant Wq4Reader    -> real WQ4 v3 container parsing + config read
//
// The GPU surface (navigator.gpu, device.createBuffer, ...) is not exercised:
// sandblaster is the local experimental stub (packages/backend/stubs/core)
// and the model is opened with preload:false against a fake GPUDevice.

/// <reference types="@webgpu/types" />
/// <reference types="deno" />
/// <reference path="./shims.d.ts" />

import {
  LFM2_ARENA,
  KV_ELEMENTS,
  CONV_ELEMENTS,
  TOKEN_CAPACITY,
  OP_PARAM_BUFFER_BYTES,
  lfm2,
} from "../../webgpu/src/lfm2.ts";
import { lfm2Pass, Lfm2Executor } from "../../webgpu/src/pass.ts";
import { Lfm2GpuModel } from "../../webgpu/src/model.ts";
import {
  Wq4Reader,
  WQ4_MAGIC,
  WQ4_VERSION,
  WQ4_BLOCK_SIZE,
  WQ4_HEADER_BYTES,
} from "../../quant/src/wq4/reader.ts";
import type { RandomAccessSource } from "../../quant/src/gguf/source.ts";

function fmt(n: number): string {
  return n.toLocaleString("en-US");
}

function demoArenaAndPasses(): void {
  console.log("--- lfm2 definition (native) ---");
  console.log("capacities:", JSON.stringify(lfm2.capacities));
  console.log("arena elements:", fmt(LFM2_ARENA.elements));
  console.log(
    "arena.work:",
    JSON.stringify({
      hiddenA: LFM2_ARENA.hiddenA,
      hiddenB: LFM2_ARENA.hiddenB,
      tmpH: LFM2_ARENA.tmpH,
      tmpA: LFM2_ARENA.tmpA,
      tmpB: LFM2_ARENA.tmpB,
    }),
  );
  console.log("arena.repair:", JSON.stringify(LFM2_ARENA.repair));
  console.log("arena.logits offset:", LFM2_ARENA.logits);
  console.log("kv elements:", fmt(KV_ELEMENTS));
  console.log("conv elements:", fmt(CONV_ELEMENTS));
  console.log("token capacity:", TOKEN_CAPACITY);
  console.log("op param buffer bytes:", OP_PARAM_BUFFER_BYTES);

  const wq4 = lfm2Pass("matmul_wq4", {
    rowCount: 512,
    tokenCount: 7,
    inputDim: 2048,
    outputDim: 8192,
  });
  console.log("matmul_wq4 ->", JSON.stringify(wq4.workgroups), "weight:", wq4.weight);

  const attn = lfm2Pass("attention", { tokenCount: 7 });
  console.log("attention  ->", JSON.stringify(attn.workgroups), "weight:", attn.weight);

  const qknorm = lfm2Pass("qk_norm_rope", { tokenCount: 7, u0: 0 });
  console.log("qk_norm_rope ->", JSON.stringify(qknorm.workgroups), "weight:", qknorm.weight);
}

const LFM2_KV_HEADS = [0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0];

/** Minimal self-contained WQ4 v3 file that satisfies readConfig(). */
function buildWq4File(): Uint8Array {
  const metadata: Record<string, unknown> = {
    "general.architecture": "lfm2",
    "lfm2.block_count": 16,
    "lfm2.embedding_length": 2048,
    "lfm2.attention.head_count": 32,
    "lfm2.attention.head_count_kv": LFM2_KV_HEADS,
    "lfm2.context_length": 1024,
    "lfm2.feed_forward_length": 8192,
    "lfm2.rope.freq_base": 1000000,
    "lfm2.vocab_size": 65536,
    "lfm2.shortconv.l_cache": 3,
    "lfm2.attention.layer_norm_rms_epsilon": 0.00001,
    "tokenizer.ggml.bos_token_id": 1,
    "tokenizer.ggml.eos_token_id": 7,
    "tokenizer.ggml.add_bos_token": true,
    "tokenizer.ggml.add_eos_token": false,
  };

  const tensorSize = 4096; // raw F16, dimensions [2048]
  const index = {
    version: WQ4_VERSION,
    metadata,
    tensors: [
      {
        name: "blk.0.attn_norm.weight",
        dimensions: [2048],
        offset: WQ4_HEADER_BYTES,
        size: tensorSize,
        sourceBytes: tensorSize,
        encoding: "raw",
        sourceType: 1, // GgmlType.F16
      },
    ],
  };
  const indexBytes = new TextEncoder().encode(JSON.stringify(index));
  const total = WQ4_HEADER_BYTES + tensorSize + indexBytes.length;
  const file = new Uint8Array(total);
  const dv = new DataView(file.buffer);
  dv.setUint32(0, WQ4_MAGIC, true);
  dv.setUint32(4, WQ4_VERSION, true);
  dv.setUint32(8, 1, true); // tensorCount
  dv.setUint32(12, WQ4_BLOCK_SIZE, true);
  dv.setBigUint64(16, BigInt(WQ4_HEADER_BYTES + tensorSize), true); // indexOffset
  dv.setBigUint64(24, BigInt(indexBytes.length), true); // indexSize
  file.set(indexBytes, WQ4_HEADER_BYTES + tensorSize);
  return file;
}

class MemorySource implements RandomAccessSource {
  readonly size: number;
  constructor(private readonly bytes: Uint8Array) {
    this.size = bytes.length;
  }
  async read(offset: number, length: number): Promise<Uint8Array> {
    return this.bytes.slice(offset, offset + length);
  }
}

async function demoModel(): Promise<void> {
  console.log("--- WQ4 container + model config (native) ---");
  const file = buildWq4File();
  const source = new MemorySource(file);
  const reader = await Wq4Reader.open(source);
  console.log(
    "WQ4 selfContained:",
    reader.selfContained,
    "| tensors:",
    reader.tensors.size,
    "| block_count:",
    reader.metadataValue("lfm2.block_count"),
  );

  const fakeDevice = {
    limits: {
      maxStorageBufferBindingSize: 1 << 30,
      maxBufferSize: 1 << 31,
      maxComputeWorkgroupsPerDimension: 65535,
      minUniformBufferOffsetAlignment: 256,
    },
    queue: {
      writeBuffer() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    pushErrorScope() {},
    popErrorScope() {
      return Promise.resolve(undefined);
    },
    createBuffer() {
      return { destroy() {} };
    },
  } as unknown as GPUDevice;

  const model = await Lfm2GpuModel.open(fakeDevice, source, { preload: false });
  console.log(
    "config:",
    JSON.stringify({
      blockCount: model.config.blockCount,
      hiddenSize: model.config.hiddenSize,
      feedForwardSize: model.config.feedForwardSize,
      vocabSize: model.config.vocabSize,
      contextLength: model.config.contextLength,
      layers: model.config.layers,
      attentionLayerSlots: model.config.attentionLayerSlots,
    }),
  );
  console.log("metadata eos_token:", model.metadata<number>("tokenizer.ggml.eos_token_id"));
}

/**
 * Exercises Lfm2Executor: OpParams alloc + the real 256-byte ABI encode.
 * With the REAL sandblaster this needs engine.compile() (GPU), so a failure
 * here is expected in a headless native build and is reported, not fatal.
 */
function demoExecutor(): void {
  console.log("--- Lfm2Executor / OpParams ABI (native) ---");
  try {
    const executor = new Lfm2Executor(lfm2);
    executor.submit((encoder) => {
      encoder.compute((pass) => {
        pass.run(
          "embedding_wq4",
          {
            tokenCount: 4,
            outputDim: 2048,
            rowStart: 0,
            rowCount: 65536,
            mode: "prefill",
            u0: 0,
          },
          { buffer: {} } as any,
        );
        pass.run(
          "rms_norm",
          {
            tokenCount: 4,
            inputDim: 2048,
            inputOffset: 0,
            outputOffset: 0,
            f0: 0.00001,
          },
          { buffer: {} } as any,
        );
        pass.run("attention", { tokenCount: 7 });
      }, { label: "native-demo" });
    });
    console.log("executor shader coverage:", executor.shaderCoverage.join(", "));
  } catch (error) {
    console.log(
      "Lfm2Executor skipped (needs compiled GPU engine):",
      error instanceof Error ? error.message : String(error),
    );
  }
}

async function main(): Promise<void> {
  console.log("=== chomato webgpu -> native (scriptc experiment) ===");
  demoArenaAndPasses();
  await demoModel();
  demoExecutor();
  console.log("=== OK ===");
}

main().catch((error) => {
  console.error("FAILED:", error);
  throw error;
});
