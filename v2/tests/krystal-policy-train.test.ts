// M-A (S2-S4) end-to-end training proof + generalization metrics
// (docs/S2_S10_CURRICULUM_TASK.md, "Test requirements").
//
// Trains the composed runner (decision-head CE + intent-slot pointer loss +
// intent-conditional argument pointer loss) on the deterministic 60/30/10
// curriculum mixture over S2/S3/S4 with S1 replay, then evaluates on held-out
// seeds/layouts/resource ids through the production emission path
// (emitIntentSet) and reports the milestone metrics:
//
//   intent accuracy, pointer accuracy (given correct pointer-bearing intent),
//   joint exact-match, invalid-pointer rate, per-stage confusion matrix,
//   and the full stage x variant x goldIntent coverage table.
//
// The joint result is the success metric: EAT with the wrong resource is
// wrong, and the right resource with the wrong intent is wrong.
//
// Status: this test is intentionally RED until the milestone contract (100%
// intent/pointer/joint, 0 invalid pointers on unseen seeds) is genuinely
// reached — see docs/M-A_TRAINING_STATUS.md and docs/FOLLOW_UP.md.
import { expect, test } from "bun:test";
import { buildFixtureActionCatalog, type CompiledActionCatalog } from "../../packages/krystal/src/fixtures/action-intents.ts";
import { lowerPolicyFrame, type PolicyEpisode } from "../../packages/krystal/src/bridge/policy.ts";
import { buildCurriculum } from "../../packages/krystal/src/bridge/curriculum.ts";
import { KrystalForward } from "../../packages/webgpu/src/krystal-forward.ts";
import { KrystalBackward } from "../../packages/webgpu/src/krystal-backward.ts";
import {
  POLICY_CONFIG,
  prepareTrainFrame,
  productionSelection,
  emitPrediction,
  getTrainingHarness,
  packBrainFrame,
  createBrainForwardWeights,
} from "./policy-harness.ts";

/**
 * Coverage table: stage x variant x goldIntent frame counts for a split.
 * The variant is the gold action + whether it carries a reference (for S2 the
 * three variants are EAT+ref / CRY / LAUGH; other stages have a single form).
 */
function logCoverageTable(title: string, episodes: readonly PolicyEpisode[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const episode of episodes) {
    for (const frame of episode.frames) {
      const key = `${episode.stage} | ${frame.gold.action}${frame.gold.refToken !== undefined ? "+ref" : ""} | ${episode.seed}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  console.log(`coverage ${title} (${episodes.length} episodes):`);
  const agg = new Map<string, number>();
  for (const episode of episodes) {
    for (const frame of episode.frames) {
      const key = `${episode.stage} | ${frame.gold.action}${frame.gold.refToken !== undefined ? "+ref" : ""}`;
      agg.set(key, (agg.get(key) ?? 0) + 1);
    }
  }
  for (const [key, count] of [...agg.entries()].sort()) {
    console.log(`  ${key}: ${count}`);
  }
  return agg;
}

test("M-A (S2-S4): intent + exact pointer generalize on unseen seeds/layouts/ids", async () => {
  const h = await getTrainingHarness();
  const catalog = buildFixtureActionCatalog();

  const split = buildCurriculum({
    stages: ["S2", "S3", "S4"],
    replayStages: ["S1"],
    trainSeeds: [0, 256],
    evalSeeds: [256, 288],
  });
  console.log(split.log.join("\n"));

  // Coverage table (S2 variant coverage is guaranteed by the decoupled
  // variant hash — FOLLOW_UP.md §1; assert the intended counts).
  const trainCoverage = logCoverageTable("train", split.train);
  const evalCoverage = logCoverageTable("eval", split.eval);
  // All three S2 variants (EAT+ref / CRY / LAUGH) appear in train AND eval.
  for (const table of [trainCoverage, evalCoverage]) {
    expect(table.get("S2 | EAT+ref") ?? 0).toBeGreaterThan(0);
    expect(table.get("S2 | CRY") ?? 0).toBeGreaterThan(0);
    expect(table.get("S2 | LAUGH") ?? 0).toBeGreaterThan(0);
  }
  // Intended deterministic counts for these seed ranges.
  expect(trainCoverage.get("S2 | EAT+ref")).toBe(16);
  expect(trainCoverage.get("S2 | CRY")).toBe(16);
  expect(trainCoverage.get("S2 | LAUGH")).toBe(17);
  expect(evalCoverage.get("S2 | EAT+ref")).toBe(3);
  expect(evalCoverage.get("S2 | CRY")).toBe(3);
  expect(evalCoverage.get("S2 | LAUGH")).toBe(4);
  // 60/30/10 stage mixture preserved (deterministic rng draws).
  const trainEpisodes = split.train;
  const current = trainEpisodes.filter((e) => e.stage === "S2" || e.stage === "S3" || e.stage === "S4").length;
  const replay = trainEpisodes.filter((e) => e.stage === "S1").length;
  const adversarial = trainEpisodes.filter((e) => e.stage !== "S1" && e.stage !== "S2" && e.stage !== "S3" && e.stage !== "S4").length;
  expect(current / trainEpisodes.length).toBeGreaterThan(0.55);
  expect(current / trainEpisodes.length).toBeLessThan(0.75);
  expect(replay / trainEpisodes.length).toBeGreaterThan(0.15);
  expect(replay / trainEpisodes.length).toBeLessThan(0.35);
  expect(adversarial / trainEpisodes.length).toBeGreaterThan(0.02);
  expect(adversarial / trainEpisodes.length).toBeLessThan(0.15);

  // Train/eval disjoint seeds => disjoint resource ids and physical layouts.
  for (const train of split.train) {
    for (const evalEp of split.eval) {
      expect(train.seed).not.toBe(evalEp.seed);
    }
  }

  // --- Train (all frames of every mixture episode; per-frame trainStep) ---
  const trainFrames = split.train.flatMap((episode) =>
    episode.frames.map((frame) => ({ episode, frame })),
  );
  const weights = createBrainForwardWeights(POLICY_CONFIG, 42);
  const runner = new KrystalForward(weights, POLICY_CONFIG);
  const trainer = new KrystalBackward(runner);

  const EPOCHS = 2;
  let firstLoss: number | undefined;
  let lastLoss: number | undefined;
  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    for (let i = 0; i < trainFrames.length; i++) {
      const { episode, frame: rawFrame } = trainFrames[i]!;
      const step = epoch * trainFrames.length + i;
      const prepared = prepareTrainFrame(
        packBrainFrame(lowerPolicyFrame(rawFrame, episode)).frame,
        rawFrame.gold,
        catalog,
      );
      const telemetry = step === 0 || step === EPOCHS * trainFrames.length - 1;
      const result = await trainer.trainStep({
        ...prepared,
        learningRate: 0.01,
        telemetry,
      });
      if (step === 0) firstLoss = result.loss;
      if (step === EPOCHS * trainFrames.length - 1) lastLoss = result.loss;
    }
  }
  expect(firstLoss).toBeDefined();
  expect(lastLoss).toBeDefined();
  expect(firstLoss!).toBeGreaterThan(lastLoss!); // loss descends

  // --- Evaluate on held-out seeds through the production emission path ---
  const evalFrames = split.eval.flatMap((episode) =>
    episode.frames.map((frame) => ({ episode, frame })),
  );
  const confusion = new Map<string, Map<string, number>>();
  let intentCorrect = 0;
  let pointerFrames = 0;
  let pointerCorrect = 0;
  let pointerGivenIntent = 0;
  let pointerGivenIntentCorrect = 0;
  let jointCorrect = 0;
  let invalidPointer = 0;
  for (const { episode, frame } of evalFrames) {
    const gold = frame.gold;
    const needsPointer = gold.refToken !== undefined;
    const sel = await productionSelection(
      h, runner,
      packBrainFrame(lowerPolicyFrame(frame, episode)).frame,
      catalog,
    );
    const pred = sel ? emitPrediction(sel, catalog) : null;

    const predAction = pred?.action ?? "(none)";
    const intentOK = pred !== null && pred.action === gold.action;
    if (intentOK) intentCorrect++;

    if (needsPointer) {
      pointerFrames++;
      const pointerOK = pred !== null && pred.refToken === gold.refToken;
      if (pointerOK) pointerCorrect++;
      if (intentOK) {
        pointerGivenIntent++;
        if (pointerOK) pointerGivenIntentCorrect++;
      }
      // Invalid pointer: a pointer-bearing action whose resolved ref is not
      // the gold (or could not resolve -> no executable proposal).
      if (pred !== null && !pointerOK) invalidPointer++;
    }

    if (intentOK && (!needsPointer || pred!.refToken === gold.refToken)) jointCorrect++;

    const row = confusion.get(gold.action) ?? new Map<string, number>();
    row.set(predAction, (row.get(predAction) ?? 0) + 1);
    confusion.set(gold.action, row);
  }

  const n = evalFrames.length;
  const intentAcc = intentCorrect / n;
  const pointerAcc = pointerFrames === 0 ? 1 : pointerCorrect / pointerFrames;
  const pointerGivenIntentAcc = pointerGivenIntent === 0 ? 1 : pointerGivenIntentCorrect / pointerGivenIntent;
  const jointAcc = jointCorrect / n;
  const invalidRate = invalidPointer / n;

  console.log(`M-A eval: ${n} held-out frames (seeds 256..287)`);
  console.log(`  intent accuracy            ${intentAcc}`);
  console.log(`  pointer accuracy           ${pointerAcc} (${pointerCorrect}/${pointerFrames} ref-bearing frames)`);
  console.log(`  pointer | correct intent   ${pointerGivenIntentAcc} (${pointerGivenIntentCorrect}/${pointerGivenIntent})`);
  console.log(`  joint exact-match          ${jointAcc}`);
  console.log(`  invalid-pointer rate       ${invalidRate} (${invalidPointer}/${n})`);
  console.log("  confusion (gold -> pred):");
  for (const [gold, row] of confusion) {
    console.log(`    ${gold.padEnd(12)} -> ${[...row.entries()].map(([a, c]) => `${a}:${c}`).join(", ")}`);
  }

  // The milestone contract (FOLLOW_UP.md: do not weaken these).
  expect(intentAcc).toBe(1);
  expect(pointerAcc).toBe(1);
  expect(pointerGivenIntentAcc).toBe(1);
  expect(jointAcc).toBe(1);
  expect(invalidRate).toBe(0);
  runner.destroy();
}, 180_000);
