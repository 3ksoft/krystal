/**
 * @krystal/krystal — the model graph, its CPU oracle and the frame it reads.
 *
 * The entrypoint a SIMULATION uses is `@krystal/krystal/host`: records of
 * tokens in, chosen records out, and one update to live with what followed. It
 * is deliberately not re-exported here, because importing a brain must not drag
 * in a world — and that coupling is the whole reason the host entrypoint exists.
 *
 * What is left is the model and nothing else: dimensions, weights, the CPU
 * oracle, its backward, and the masks the forward reads. No vocabulary, no
 * relation catalog, no world contract — those belong to whoever has a world.
 */
export * from "./hash.ts";
export * from "./binary-layout-plan.ts";
export * from "./forward/model.ts";
export * from "./forward/masks.ts";
export * from "./forward/oracle.ts";
export * from "./forward/backward.ts";
