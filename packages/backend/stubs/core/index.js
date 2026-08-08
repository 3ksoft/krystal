// EXPERIMENTAL LOCAL STUB — see package.json.
// Minimal runtime for @sandblaster/core-next. Enough to let
// defineLfm2()/Lfm2Executor/lfm2Pass run natively without a GPU:
//   - type()/buffer()/compute()/submit()/deserialize() no-op plumbing
//   - encodeInto() writes the REAL OpParams (256 B) / LlmRuntime (48 B) binary
//     layouts taken from packages/webgpu/src/lfm2.artifact.generated.ts

var MODE = { prefill: 0, decode: 1, continuation: 2 };
var STATUS = { idle: 0, running: 1, eos: 2, done: 3, error: 4 };

export class TypeHandle {
  constructor(name) {
    this.name = name;
  }
  assert(value) {
    return value;
  }
}

class StubBuffer {
  constructor(name, options) {
    this.name = name;
    this.options = options || {};
    this.gpu = { label: "stub:" + name, destroy: function () {} };
    // Latent mismatch, unreachable in mock builds: Lfm2Executor asserts the op
    // record stride is 64 B and never runs against the stub (step 2's shim
    // supplies the real value). 256 here predates the fromArtifact path.
    this.compiledInfo = {
      physicalStride: name === "OpParams" ? 256 : 0,
      byteSize: 0,
    };
    this._written = null;
  }

  write(value) {
    this._written = value;
  }

  encodeInto(value, view, offset) {
    var dv = view;
    var o = offset;
    function u32(v) {
      dv.setUint32(o, typeof v === "number" ? v : 0, true);
      o += 4;
    }
    function f32(v) {
      dv.setFloat32(o, typeof v === "number" ? v : 0, true);
      o += 4;
    }

    if (this.name === "OpParams") {
      u32(value.inputOffset);
      u32(value.outputOffset);
      u32(value.auxOffset);
      u32(value.aux2Offset);
      u32(value.tokenCount);
      u32(value.inputDim);
      u32(value.outputDim);
      u32(value.rowStart);
      u32(value.rowCount);
      u32(value.layerIndex);
      u32(value.attentionSlot);
      f32(value.f0);
      f32(value.f1);
      u32(value.u0);
      u32(value.u1);
      var reserved = value.reserved || [];
      for (var i = 0; i < 48; i++) u32(reserved[i] || 0);
      dv.setUint8(offset + 252, MODE[value.mode] !== undefined ? MODE[value.mode] : 0);
      return 256;
    }

    if (this.name === "LlmRuntime") {
      u32(value.contextCapacity);
      u32(value.maxNewTokens);
      u32(value.eosToken);
      u32(value.promptTokenCount);
      u32(value.position);
      u32(value.generatedCount);
      u32(value.currentToken);
      u32(value.telemetryRevision);
      u32(value.lastToken);
      u32(value.errorCode);
      u32(value.pad0);
      dv.setUint8(offset + 44, STATUS[value.status] !== undefined ? STATUS[value.status] : 0);
      return 48;
    }

    return 0;
  }

  readback() {
    return Promise.resolve(this._written !== null ? this._written : {});
  }
}

export class Sandblaster {
  static create(_schema, _options) {
    return new Sandblaster();
  }

  // The serialized artifact is intentionally not parsed here: under scriptc
  // JSON.parse of the multi-MB artifact would be a dynamic island, and the stub
  // exists to prove the graph compiles statically. The real engine (browser/bun
  // via the linked core, native via the shim) parses it.
  static fromArtifact(_serialized, _options) {
    return new Sandblaster();
  }

  constructor() {
    this.device = {
      queue: {
        writeBuffer: function () {},
        onSubmittedWorkDone: function () {
          return Promise.resolve();
        },
        submit: function () {},
      },
      // Real WebGPU objects; the native shim (step 2) supplies them over FFI.
      createBuffer: function () {
        return { destroy: function () {}, size: 0 };
      },
      createCommandEncoder: function () {
        return { finish: function () { return {}; } };
      },
      limits: {
        minUniformBufferOffsetAlignment: 256,
        maxStorageBufferBindingSize: 1073741824,
        maxBufferSize: 2147483648,
        maxComputeWorkgroupsPerDimension: 65535,
      },
    };
  }

  resource(key) {
    return new StubBuffer(String(key), {});
  }

  computeProgram(label) {
    return {
      kind: "compute",
      label: label,
      manifest: { bindings: [] },
    };
  }

  async compile(_options) {
    return { status: "ok", failed: 0, total: 0 };
  }

  type(nameOrShape) {
    return new TypeHandle(typeof nameOrShape === "string" ? nameOrShape : "anonymous");
  }

  buffer(type, options) {
    return new StubBuffer(type.name, options);
  }

  compute(options) {
    var label = options && options.label ? options.label : "compute";
    return { kind: "compute", label: label };
  }

  deserialize(json) {
    // The serialized Sandblaster artifact is intentionally ignored by the stub.
  }

  submit(callback) {
    if (typeof callback !== "function") return;
    var encoder = {
      gpu: {},
      compute: function (_descriptor, passCallback) {
        if (typeof passCallback === "function") passCallback({ run: function () {} });
      },
    };
    callback(encoder);
  }
}
