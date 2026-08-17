import { getTrainingHarness, readArenaRegion } from "./tests/training-harness.ts";
import { KrystalForward } from "./packages/webgpu/src/krystal-forward.ts";
import { KRYSTAL_FORWARD_ARENA, KRYSTAL_FORWARD_ARENA_BASE } from "./packages/webgpu/src/krystal-layout.ts";
import { packBrainFrame } from "./packages/krystal/src/frame/packer.ts";
import { compileActiveFrame, compileMixerMask } from "./packages/krystal/src/forward/masks.ts";
import { BRAIN_FORWARD_CONFIG, createBrainForwardWeights } from "./packages/krystal/src/forward/model.ts";
import { generatePolicyEpisode, lowerPolicyFrame } from "./packages/krystal/src/bridge/policy.ts";

const POLICY_CONFIG = { ...BRAIN_FORWARD_CONFIG, routeKindCount: 6 };
const maskOffset = KRYSTAL_FORWARD_ARENA_BASE + KRYSTAL_FORWARD_ARENA.mixerMask;

async function main() {
  const h = await getTrainingHarness();
  const episode = generatePolicyEpisode("S2", 6);
  const frame = packBrainFrame(lowerPolicyFrame(episode.frames[0]!, episode)).frame;
  const active = compileActiveFrame(frame);
  const q = active.queryRecords.length;
  const r = active.bankRecords.length;
  const t = active.activeTokens.length;
  const weights = createBrainForwardWeights(POLICY_CONFIG, 1337);
  const runner = new KrystalForward(weights, POLICY_CONFIG);
  const prepared = runner.prepare(frame, undefined);
  await h.device.queue.onSubmittedWorkDone();

  const readMask = async () => {
    const m = await readArenaRegion(h, maskOffset, q * r);
    return { open: Array.from(m).filter((v) => v > -1e29).length, first: Array.from(m.slice(0, 3)).map((v) => v.toFixed(3)).join(",") };
  };
  console.log("after prepare:", JSON.stringify(await readMask()));

  // Stage 1: field embed only
  const A = KRYSTAL_FORWARD_ARENA;
  const { hiddenSize: hd } = POLICY_CONFIG;
  runner.executor.submit((enc) => {
    const bases = (require("./packages/krystal/src/forward/model.ts") as any).embeddingTableBases(POLICY_CONFIG);
    enc.compute((pass) => pass.run("krystal_field_embed", {
      inputOffset: runner.region(A.tokenIds, 1024),
      auxOffset: runner.region(A.fieldRoles, 1024),
      aux2Offset: runner.region(A.schemaIds, 128),
      aux3Offset: runner.region(A.bandIds, 128),
      aux4Offset: runner.region(A.activeTokens, t),
      aux5Offset: runner.region(A.streamIds, 128),
      outputOffset: runner.region(A.fieldStates, t * hd),
      tokenCount: t, inputDim: hd,
      u0: bases.token, u1: bases.field, u2: bases.schema, u3: bases.band, u4: bases.stream, u5: bases.pos,
    }, runner.embeddingsPage));
  });
  await h.device.queue.onSubmittedWorkDone();
  console.log("after field embed:", JSON.stringify(await readMask()));
  runner.destroy();
}
main().catch((e) => { console.error(e); process.exit(1); });
