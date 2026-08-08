// The per-shape tiling rule has to actually reach both programs.
//
// matmulWq4Program() picks by output width, and every width in the model is
// decided by one comparison — so a wrong threshold does not fail, it silently
// runs everything on one tiling and the win disappears with no error anywhere.
import { expect, test } from "bun:test";
import { loadModel } from "../src";
import { lfm2 } from "../packages/webgpu/src/lfm2.ts";

test("a decode step uses both matmul_wq4 tilings", async () => {
  const model = await loadModel("./models/LFM2.5-1.2B-Instruct-WQ4.wq4");
  const forward = model.forward!;
  try {
    await forward.prepareAll();
    forward.executor.clearShaderCoverage();
    await forward.generateGreedy(new Uint32Array([1, 2, 3]), { maxNewTokens: 2, resetState: true });
    const coverage = forward.executor.shaderCoverage;
    console.log("[coverage]", JSON.stringify([...coverage].sort()));
    expect(coverage).toContain("matmul_wq4");
    expect(coverage).toContain("matmul_wq4_wide");
  } finally {
    await model.dispose();
  }
}, 60_000);
