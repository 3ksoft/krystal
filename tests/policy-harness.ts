// Test adapter around the production policy runner. Keeping the GPU device
// bootstrap test-local avoids coupling runtime code to `bun:test`.
import { getTrainingHarness } from "./training-harness.ts";
import type { WordBias } from "../packages/krystal/src/forward/masks.ts";
import {
  productionSelection as runProductionSelection,
  type ProductionSelection,
} from "../packages/webgpu/src/policy-runtime.ts";
import type { CompiledActionCatalog } from "../packages/krystal/src/fixtures/action-intents.ts";
import type { KrystalForward } from "../packages/webgpu/src/krystal-forward.ts";
import type { v1_0_0 } from "../packages/schema/generated/krystal.types.ts";

export * from "../packages/webgpu/src/policy-runtime.ts";
export { getTrainingHarness };

export async function productionSelection(
  harness: Awaited<ReturnType<typeof getTrainingHarness>>,
  runner: KrystalForward,
  frame: v1_0_0.BrainFrameGpu,
  catalog: CompiledActionCatalog,
  wordBias?: WordBias,
): Promise<ProductionSelection | null> {
  return runProductionSelection(harness.device, runner, frame, catalog, wordBias);
}
