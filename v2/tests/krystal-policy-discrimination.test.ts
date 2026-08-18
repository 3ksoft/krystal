// Controlled balanced discrimination test (docs/FOLLOW_UP.md §2).
//
// The minimal EAT-vs-CRY problem, isolated from LAUGH and unrelated stages:
//   50% bad comfort + visible edible -> EAT(appleRef)
//   50% identical bad comfort + no edible -> CRY
// with paired frames that differ only by edible presence (same seed => the
// same kaleidoscope noise; the apple record occupies exactly the slot that
// carries a noise record in the CRY frame).
//
// The current architecture and the production evaluation path are used
// unchanged; lr 0.01, 2 epochs (no compensation with longer training). The
// observed failure is bistable, so every metric is reported across at least
// three initialization seeds. Expected: 100% discrimination on this
// deterministic fixture.
import { expect, test } from "bun:test";
import type { PolicyEpisode } from "../../packages/krystal/src/bridge/policy.ts";
import { policyRefToken } from "../../packages/krystal/src/bridge/policy.ts";
import { compileActiveFrame } from "../../packages/krystal/src/forward/masks.ts";
import { buildFixtureActionCatalog } from "../../packages/krystal/src/fixtures/action-intents.ts";
import { ACTION_INTENT_SCHEMA_ID } from "../../packages/krystal/src/fixtures/frame.ts";
import { KrystalForward } from "../../packages/webgpu/src/krystal-forward.ts";
import { KrystalBackward } from "../../packages/webgpu/src/krystal-backward.ts";
import {
  POLICY_CONFIG,
  ACTION_TOKEN,
  prepareTrainFrame,
  productionSelection,
  emitPrediction,
  getTrainingHarness,
  packBrainFrame,
  createBrainForwardWeights,
  catalogBankIndex,
} from "./policy-harness.ts";
import { lowerPolicyFrame } from "../../packages/krystal/src/bridge/policy.ts";

/** Paired EAT/CRY episodes from one seed (identical noise, apple vs none). */
function pairedEpisodes(seed: number): { eat: PolicyEpisode; cry: PolicyEpisode } {
  const appleRef = policyRefToken(seed, 0);
  return {
    eat: {
      stage: "S2",
      seed,
      frames: [{
        tick: 10, comfort: -1,
        resources: [{ kind: "apple", refToken: appleRef, generation: 1, band: "vision", properties: ["RED", "SMALL"] }],
        gold: { action: "EAT", refToken: appleRef },
      }],
    },
    cry: {
      stage: "S2",
      seed,
      frames: [{
        tick: 10, comfort: -1,
        resources: [],
        gold: { action: "CRY" },
      }],
    },
  };
}

interface SeedReport {
  readonly seed: number;
  readonly trainAcc: number;
  readonly evalAcc: number;
  readonly evalEatAcc: number;
  readonly evalCryAcc: number;
  readonly confusion: Map<string, Map<string, number>>;
  readonly pEatOnEat: number;
  readonly pEatOnCry: number;
  readonly pCryOnEat: number;
  readonly pCryOnCry: number;
  readonly qDiffInit: number;
  readonly qDiffTrained: number;
  readonly qRelTrained: number;
  readonly pointerAcc: number;
  readonly invalidRate: number;
}

async function runSeed(
  h: Awaited<ReturnType<typeof getTrainingHarness>>,
  catalog: ReturnType<typeof buildFixtureActionCatalog>,
  initSeed: number,
): Promise<SeedReport> {
  const TRAIN_SEEDS = 16; // 0..15
  const EVAL_SEEDS = 8; // 16..23
  const pairs = Array.from({ length: TRAIN_SEEDS }, (_, s) => pairedEpisodes(s));
  const evalPairs = Array.from({ length: EVAL_SEEDS }, (_, s) => pairedEpisodes(TRAIN_SEEDS + s));
  const trainFrames = pairs.flatMap((p) => [
    { episode: p.eat, frame: p.eat.frames[0]! },
    { episode: p.cry, frame: p.cry.frames[0]! },
  ]);

  const weights = createBrainForwardWeights(POLICY_CONFIG, initSeed);

  const runner = new KrystalForward(weights, POLICY_CONFIG);

  // Query-state EAT/CRY separation on the actual GPU-resident checkpoint.
  // KrystalBackward updates GPU pages in place; the host-side `weights`
  // arrays remain the initialization and therefore cannot measure the trained
  // state.
  const qDiff = async (): Promise<{ diff: number; mag: number }> => {
    const pair = pairedEpisodes(TRAIN_SEEDS);
    const measure = async (episode: PolicyEpisode) => {
      const frame = packBrainFrame(lowerPolicyFrame(episode.frames[0]!, episode)).frame;
      const active = compileActiveFrame(frame);
      runner.forward(frame);
      await h.device.queue.onSubmittedWorkDone();
      return runner.readQueryOutput(active.queryRecords.length, POLICY_CONFIG.hiddenSize);
    };
    const eat = await measure(pair.eat);
    const cry = await measure(pair.cry);
    let diff = 0;
    let mag = 0;
    for (let i = 0; i < eat.length; i++) {
      diff = Math.max(diff, Math.abs(eat[i]! - cry[i]!));
      mag = Math.max(mag, Math.abs(eat[i]!));
    }
    return { diff, mag };
  };

  const initQ = await qDiff();
  const trainer = new KrystalBackward(runner);

  // lr 0.01, 2 epochs — the existing settings, no compensation.
  for (let epoch = 0; epoch < 2; epoch++) {
    for (const { episode, frame } of trainFrames) {
      const prepared = prepareTrainFrame(
        packBrainFrame(lowerPolicyFrame(frame, episode)).frame,
        frame.gold,
        catalog,
      );
      await trainer.trainStep({ ...prepared, learningRate: 0.01 });
    }
  }

  const trainedQ = await qDiff();

  // Eval metrics (production path).
  let evalAcc = 0;
  let eatAcc = 0;
  let cryAcc = 0;
  let pointerFrames = 0;
  let pointerCorrect = 0;
  let invalid = 0;
  const confusion = new Map<string, Map<string, number>>();
  const pEatOnEat: number[] = [];
  const pEatOnCry: number[] = [];
  const pCryOnEat: number[] = [];
  const pCryOnCry: number[] = [];
  const eatRecordIdx = catalog.descriptors.find((d) => d.actionToken === ACTION_TOKEN.EAT)!.intentId;
  const cryRecordIdx = catalog.descriptors.find((d) => d.actionToken === ACTION_TOKEN.CRY)!.intentId;

  for (const pair of evalPairs) {
    for (const [episode, isEat] of [[pair.eat, true], [pair.cry, false]] as const) {
      const frame = packBrainFrame(lowerPolicyFrame(episode.frames[0]!, episode)).frame;
      const sel = await productionSelection(h, runner, frame, catalog);
      const pred = sel ? emitPrediction(sel, catalog) : null;
      const gold = episode.frames[0]!.gold;
      const ok = pred !== null && pred.action === gold.action;
      if (ok) evalAcc++;
      if (isEat && ok) eatAcc++;
      if (!isEat && ok) cryAcc++;

      if (sel) {
        const active = compileActiveFrame(frame);
        const pEat = sel.intent.p[catalogBankIndex(frame, active, ACTION_TOKEN.EAT)]!;
        const pCry = sel.intent.p[catalogBankIndex(frame, active, ACTION_TOKEN.CRY)]!;
        if (isEat) { pEatOnEat.push(pEat); pCryOnEat.push(pCry); }
        else { pEatOnCry.push(pEat); pCryOnCry.push(pCry); }
      }

      if (gold.refToken !== undefined) {
        pointerFrames++;
        if (pred !== null && pred.refToken === gold.refToken) pointerCorrect++;
        if (pred !== null && pred.refToken !== gold.refToken) invalid++;
      }

      const row = confusion.get(gold.action) ?? new Map<string, number>();
      row.set(pred?.action ?? "(none)", (row.get(pred?.action ?? "(none)") ?? 0) + 1);
      confusion.set(gold.action, row);
    }
  }

  // Train accuracy (production path) for reference.
  let trainAcc = 0;
  for (const { episode, frame } of trainFrames) {
    const sel = await productionSelection(
      h, runner,
      packBrainFrame(lowerPolicyFrame(frame, episode)).frame,
      catalog,
    );
    const pred = sel ? emitPrediction(sel, catalog) : null;
    if (pred !== null && pred.action === frame.gold.action) trainAcc++;
  }

  const n = evalPairs.length * 2;
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
  runner.destroy();
  return {
    seed: initSeed,
    trainAcc: trainAcc / trainFrames.length,
    evalAcc: evalAcc / n,
    evalEatAcc: eatAcc / evalPairs.length,
    evalCryAcc: cryAcc / evalPairs.length,
    confusion,
    pEatOnEat: mean(pEatOnEat),
    pEatOnCry: mean(pEatOnCry),
    pCryOnEat: mean(pCryOnEat),
    pCryOnCry: mean(pCryOnCry),
    qDiffInit: initQ.diff / initQ.mag,
    qDiffTrained: trainedQ.diff / trainedQ.mag,
    qRelTrained: trainedQ.diff / trainedQ.mag,
    pointerAcc: pointerCorrect / pointerFrames,
    invalidRate: invalid / pointerFrames,
  };
}

test("balanced EAT-vs-CRY discrimination: 100% across initialization seeds", async () => {
  const h = await getTrainingHarness();
  const catalog = buildFixtureActionCatalog();
  const initSeeds = [42, 7, 2026];
  const reports: SeedReport[] = [];
  for (const initSeed of initSeeds) {
    reports.push(await runSeed(h, catalog, initSeed));
  }

  for (const r of reports) {
    console.log(`--- init seed ${r.seed} ---`);
    console.log(`  train acc ${(r.trainAcc * 100).toFixed(1)}%  eval acc ${(r.evalAcc * 100).toFixed(1)}%  (EAT ${(r.evalEatAcc * 100).toFixed(1)}%, CRY ${(r.evalCryAcc * 100).toFixed(1)}%)`);
    console.log(`  mean P(EAT): on EAT frames ${r.pEatOnEat.toFixed(3)}, on CRY frames ${r.pEatOnCry.toFixed(3)}`);
    console.log(`  mean P(CRY): on EAT frames ${r.pCryOnEat.toFixed(3)}, on CRY frames ${r.pCryOnCry.toFixed(3)}`);
    console.log(`  qOut EAT/CRY separation: init rel ${r.qDiffInit.toExponential(2)}, trained rel ${r.qDiffTrained.toExponential(2)}`);
    console.log(`  pointer acc ${(r.pointerAcc * 100).toFixed(1)}%  invalid-pointer rate ${(r.invalidRate * 100).toFixed(1)}%`);
    console.log("  confusion (gold -> pred):");
    for (const [gold, row] of r.confusion) {
      console.log(`    ${gold.padEnd(4)} -> ${[...row.entries()].map(([a, c]) => `${a}:${c}`).join(", ")}`);
    }
  }

  for (const r of reports) {
    expect(r.evalAcc).toBe(1);
    expect(r.evalEatAcc).toBe(1);
    expect(r.evalCryAcc).toBe(1);
    expect(r.pointerAcc).toBe(1);
    expect(r.invalidRate).toBe(0);
  }
}, 300_000);
