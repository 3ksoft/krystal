// Same-word attention-bias assay runner (docs/word_attention_bias.md).
//   P0 wordAlpha=0            expected: chance (inputs are identical by construction)
//   P1 wordAlpha=4            expected: near-perfect
//   P2 wordAlpha=4, scrambled expected: chance (bias present, binding destroyed)
//   P3 P1 checkpoint, word ids re-bijected: predictions must be identical
import { buildFixtureActionCatalog } from "./packages/krystal/src/fixtures/action-intents.ts";
import { KrystalBackward } from "./packages/webgpu/src/krystal-backward.ts";
import { KrystalForward } from "./packages/webgpu/src/krystal-forward.ts";
import {
  POLICY_CONFIG, createBrainForwardWeights, emitPrediction, getTrainingHarness,
  packBrainFrame, prepareTrainFrame, productionSelection,
} from "./tests/policy-harness.ts";
import { compileActiveFrame, compileIntentMask } from "./packages/krystal/src/forward/masks.ts";
import { ACTION_INTENT_SCHEMA_ID } from "./packages/krystal/src/fixtures/frame.ts";
import { fixtureTokenId } from "./packages/krystal/src/fixtures/vocabulary.ts";
import { BRAIN_LIMITS } from "./packages/schema/src/krystal-engine-schema.ts";
import { wordBindPairs, type WordBindExample, type WordBindOptions } from "./tests/wordbind-fixture.ts";

const TRAIN = Array.from({ length: 256 }, (_, i) => i);
const EVAL = Array.from({ length: 64 }, (_, i) => 1000 + i);
const EPOCHS = Number(Bun.env.WB_EPOCHS ?? 3);
const LR = Number(Bun.env.WB_LR ?? 0.01);
const SEEDS = (Bun.env.WB_SEEDS ?? "42,7,1337").split(",").map(Number);
const ALPHA = Number(Bun.env.WB_ALPHA ?? 4);

const h = await getTrainingHarness();
const catalog = buildFixtureActionCatalog();

interface Profile { name: string; alpha: number; data: WordBindOptions }
const PROFILES: Profile[] = [
  { name: "P0 alpha=0", alpha: 0, data: {} },
  { name: `P1 alpha=${ALPHA}`, alpha: ALPHA, data: {} },
  { name: `P2 alpha=${ALPHA} scr`, alpha: ALPHA, data: { scrambleWords: true } },
];

const bias = (e: WordBindExample, alpha: number) => (alpha === 0 ? undefined : { wordIds: e.wordIds, alpha });

async function evaluate(runner: KrystalForward, set: WordBindExample[], alpha: number) {
  let intentOK = 0, pointerOK = 0, joint = 0, invalid = 0;
  const p = { EAT: [] as number[], LOOK: [] as number[] };
  const EAT_TOKEN = fixtureTokenId("EAT"), LOOK_TOKEN = fixtureTokenId("LOOK");
  for (const e of set) {
    const frame = packBrainFrame(e.frame).frame;
    const wb = bias(e, alpha);
    const sel = await productionSelection(h, runner, frame, catalog, wb);
    const pred = sel ? emitPrediction(sel, catalog) : null;
    if (!pred) invalid++;
    if (pred?.action === e.gold.action) intentOK++;
    if (pred?.refToken === e.gold.refToken) pointerOK++;
    if (pred?.action === e.gold.action && pred.refToken === e.gold.refToken) joint++;

    const active = compileActiveFrame(frame);
    const im = compileIntentMask(frame, active, ACTION_INTENT_SCHEMA_ID);
    runner.forward(frame, { intentMask: im, argMask: im }, wb);
    await h.device.queue.onSubmittedWorkDone();
    const r = active.bankRecords.length;
    const s = await runner.readSelection(active.queryRecords.length, r, POLICY_CONFIG.hiddenSize);
    const idx = (token: number) => active.bankRecords.findIndex((slot) =>
      frame.schemaIds[slot] === ACTION_INTENT_SCHEMA_ID &&
      frame.tokenIds[slot * BRAIN_LIMITS.recordWidth] === token);
    p[e.gold.action].push(s.intent.p[idx(e.gold.action === "EAT" ? EAT_TOKEN : LOOK_TOKEN)]!);
  }
  const n = set.length;
  const mean = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);
  return {
    intent: intentOK / n, pointer: pointerOK / n, joint: joint / n, invalid: invalid / n,
    pGoldEat: mean(p.EAT), pGoldLook: mean(p.LOOK),
  };
}

for (const seed of SEEDS) {
  console.log(`\n=== model seed ${seed} ===`);
  for (const profile of PROFILES) {
    const train = wordBindPairs(TRAIN, profile.data);
    const evalSet = wordBindPairs(EVAL, profile.data);
    const runner = new KrystalForward(createBrainForwardWeights(POLICY_CONFIG, seed), POLICY_CONFIG);
    const trainer = new KrystalBackward(runner);
    let last = 0;
    for (let epoch = 0; epoch < EPOCHS; epoch++) {
      for (let i = 0; i < train.length; i++) {
        const e = train[i]!;
        const res = await trainer.trainStep({
          ...prepareTrainFrame(packBrainFrame(e.frame).frame, e.gold, catalog),
          wordBias: bias(e, profile.alpha),
          learningRate: LR,
          telemetry: epoch === EPOCHS - 1 && i >= train.length - 20,
        });
        if (res.loss !== undefined) last = res.loss;
      }
    }
    const m = await evaluate(runner, evalSet, profile.alpha);
    const alive = Number.isFinite(last);
    console.log(
      `  ${profile.name.padEnd(16)} joint=${m.joint.toFixed(3)} intent=${m.intent.toFixed(3)} ` +
      `ptr=${m.pointer.toFixed(3)} invalid=${m.invalid.toFixed(3)} ` +
      `P(gold|EAT)=${m.pGoldEat.toFixed(3)} P(gold|LOOK)=${m.pGoldLook.toFixed(3)} ` +
      `loss=${last.toFixed(4)}${alive ? "" : "  ⚠ MARTWY"}`);
    runner.destroy();
  }
}
