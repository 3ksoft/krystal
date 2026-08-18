// Grammatical-case attention-bias assay runner (W2).
//   P0 wordAlpha=0            expected: chance (~50% pointer accuracy)
//   P1 wordAlpha=4            expected: near-perfect (100% pointer to ACCUSATIVE)
//   P2 wordAlpha=4, scrambled expected: chance (~50% binding broken)
import { buildFixtureActionCatalog } from "./packages/krystal/src/fixtures/action-intents.ts";
import { KrystalBackward } from "./packages/webgpu/src/krystal-backward.ts";
import { KrystalForward } from "./packages/webgpu/src/krystal-forward.ts";
import {
  POLICY_CONFIG, createBrainForwardWeights, emitPrediction, getTrainingHarness,
  packBrainFrame, prepareTrainFrame, productionSelection,
} from "./tests/policy-harness.ts";
import { caseBindPairs, type CaseBindExample, type CaseBindOptions } from "./tests/accusative-fixture.ts";

const TRAIN = Array.from({ length: 16 }, (_, i) => i);
const EVAL = Array.from({ length: 8 }, (_, i) => 2000 + i);
const EPOCHS = Number(Bun.env.WB_EPOCHS ?? 1);
const LR = Number(Bun.env.WB_LR ?? 0.01);
const SEEDS = (Bun.env.WB_SEEDS ?? "42,7,1337").split(",").map(Number);
const ALPHA = Number(Bun.env.WB_ALPHA ?? 4);

const h = await getTrainingHarness();
const catalog = buildFixtureActionCatalog();

interface Profile { name: string; alpha: number; data: CaseBindOptions }
const PROFILES: Profile[] = [
  { name: "P0 alpha=0", alpha: 0, data: {} },
  { name: `P1 alpha=${ALPHA}`, alpha: ALPHA, data: {} },
  { name: `P2 alpha=${ALPHA} scr`, alpha: ALPHA, data: { scrambleBinding: true } },
];

const bias = (e: CaseBindExample, alpha: number) => (alpha === 0 ? undefined : { wordIds: e.wordIds, alpha });

async function evaluate(runner: KrystalForward, set: CaseBindExample[], alpha: number) {
  let intentOK = 0, pointerOK = 0, joint = 0, invalid = 0;
  let dogAsTarget = 0, catAsTarget = 0;

  for (const e of set) {
    const frame = packBrainFrame(e.frame).frame;
    const wb = bias(e, alpha);
    const sel = await productionSelection(h, runner, frame, catalog, wb);
    const pred = sel ? emitPrediction(sel, catalog) : null;

    if (!pred) invalid++;
    if (pred?.action === e.gold.action) intentOK++;
    if (pred?.refToken === e.gold.refToken) {
      pointerOK++;
      if (e.patientNoun === "DOG") dogAsTarget++;
      if (e.patientNoun === "CAT") catAsTarget++;
    }
    if (pred?.action === e.gold.action && pred?.refToken === e.gold.refToken) joint++;
  }

  const n = set.length;
  return {
    intent: intentOK / n,
    pointer: pointerOK / n,
    joint: joint / n,
    invalid: invalid / n,
    dogAccuracy: dogAsTarget / (n / 2),
    catAccuracy: catAsTarget / (n / 2),
  };
}

for (const seed of SEEDS) {
  console.log(`\n=== W2 Case-Binding | model seed ${seed} ===`);
  for (const profile of PROFILES) {
    const train = caseBindPairs(TRAIN, profile.data);
    const evalSet = caseBindPairs(EVAL, profile.data);
    const runner = new KrystalForward(createBrainForwardWeights(POLICY_CONFIG, seed), POLICY_CONFIG);
    const trainer = new KrystalBackward(runner);
    let lastLoss = 0;

    for (let epoch = 0; epoch < EPOCHS; epoch++) {
      for (let i = 0; i < train.length; i++) {
        const e = train[i]!;
        const res = await trainer.trainStep({
          ...prepareTrainFrame(packBrainFrame(e.frame).frame, e.gold, catalog),
          wordBias: bias(e, profile.alpha),
          learningRate: LR,
          telemetry: epoch === EPOCHS - 1 && i >= train.length - 20,
        });
        if (res.loss !== undefined) lastLoss = res.loss;
      }
    }

    const m = await evaluate(runner, evalSet, profile.alpha);
    const alive = Number.isFinite(lastLoss);

    console.log(
      `  ${profile.name.padEnd(16)} joint=${m.joint.toFixed(3)} ptr=${m.pointer.toFixed(3)} ` +
      `[DOG_acc=${m.dogAccuracy.toFixed(2)} CAT_acc=${m.catAccuracy.toFixed(2)}] ` +
      `loss=${lastLoss.toFixed(4)}${alive ? "" : "  ⚠ MARTWY"}`
    );
    runner.destroy();
  }
}