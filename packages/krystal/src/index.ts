/**
 * @krystal/krystal — BrainFrame packing, catalog fixtures and the frozen SoA
 * BinaryLayoutPlan (concerns answer 30: packages/krystal owns BrainFrame
 * packing, model graph, catalog and trainStep orchestration; generic kernels
 * stay in packages/webgpu; contracts stay in packages/schema).
 */
export * from "./hash.ts";
export * from "./binary-layout-plan.ts";
export * from "./frame/packer.ts";
export * from "./fixtures/vocabulary.ts";
export * from "./fixtures/record-schemas.ts";
export * from "./fixtures/action-intents.ts";
export * from "./fixtures/frame.ts";
export * from "./forward/model.ts";
export * from "./forward/masks.ts";
export * from "./forward/oracle.ts";
export * from "./forward/backward.ts";
export * from "./forward/intentset.ts";
// The versioned simulation API. Everything a simulation talks to is here.
export * from "./bridge/index.ts";

// Training harness: the S1-S10 curriculum, its episode generator and its
// transition oracle. Deliberately NOT part of the simulation boundary — these
// synthesize training data and know the fixture vocabulary, which is exactly
// what `bridge/` must not.
export * from "./training/comfort.ts";
export * from "./training/policy.ts";
export * from "./training/transition.ts";
export * from "./training/curriculum.ts";
export * from "./fixtures/capabilities.ts";
