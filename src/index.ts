// Root entry for the local engine tests.
//
// The tests in tests/ import `loadModel` from "../src". With a model file and
// the `webgpu` Dawn bindings available this adapter returns an engine-ts
// Engine backed by the real Lfm2Forward on the GPU (exact physical
// checkpoints). It falls back to the spawned native mock exe (no GPU, no
// model file) with a warning when the real engine is unavailable.
export { loadModel } from "../packages/backend/src/local-model.ts";
export type { LocalModel } from "../packages/backend/src/local-model.ts";
