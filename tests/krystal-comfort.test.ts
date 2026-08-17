// TRAINING.md Step 1 (Comfort) — the first curriculum slice on the composed
// runner.
//
//   pira `comfort-episodes@1` JSON artifact (docs/krystal-sensory-bridge.md)
//   -> Krystal lowerer (packages/krystal/src/bridge/comfort.ts) -> BrainFrameGpu
//   -> KrystalForward/KrystalBackward trainStep (route-kind CE + intent
//      pointer loss over the CRY/LAUGH catalog records)
//
// Pass criteria (TRAINING.md Step 1):
//   1. 100% accuracy on unseen noise seeds;
//   2. comfort ablation -> chance accuracy (noise alone carries no signal);
//   3. noise ablation -> no accuracy change (signal alone is sufficient).
//
// Both stages (1A extremes, 1B scale) are trained in curriculum order and
// evaluated separately on held-out seeds.
import { expect, test } from "bun:test";
import { getTrainingHarness, type TrainingHarness } from "./training-harness.ts";
import { KrystalForward, type SelectionMasks } from "../packages/webgpu/src/krystal-forward.ts";
import { KrystalBackward } from "../packages/webgpu/src/krystal-backward.ts";
import { packBrainFrame } from "../packages/krystal/src/frame/packer.ts";
import {
  compileActiveFrame,
  compileIntentMask,
} from "../packages/krystal/src/forward/masks.ts";
import {
  BRAIN_FORWARD_CONFIG,
  createBrainForwardWeights,
  type BrainForwardConfig,
} from "../packages/krystal/src/forward/model.ts";
import {
  catalogBankIndex,
  comfortRouteKind,
  lowerComfortEpisode,
  type ComfortEpisode,
  type ComfortEpisodesArtifact,
} from "../packages/krystal/src/bridge/comfort.ts";
import { ACTION_INTENT_SCHEMA_ID } from "../packages/krystal/src/fixtures/frame.ts";
import type { v1_0_0 } from "../packages/schema/generated/krystal.types.ts";

const ARTIFACT = JSON.parse(
  await Bun.file(new URL("./fixtures/comfort-episodes-v1.json", import.meta.url)).text(),
) as ComfortEpisodesArtifact;

const COMFORT_CONFIG: BrainForwardConfig = {
  ...BRAIN_FORWARD_CONFIG,
  routeKindCount: 2, // CRY / LAUGH
};

const TRAIN_SEEDS = 32; // 0..31
const EVAL_SEEDS = 8; // 32..39

const episodes = ARTIFACT.episodes;
const byStage = (stage: ComfortEpisode["stage"]): ComfortEpisode[] =>
  episodes.filter((episode) => episode.stage === stage);
const stageEpisodes = (stage: ComfortEpisode["stage"], seedRange: readonly [number, number]): ComfortEpisode[] =>
  byStage(stage).filter((e) => e.noiseSeed >= seedRange[0] && e.noiseSeed < seedRange[1]);

interface PreparedComfort {
  readonly frame: v1_0_0.BrainFrameGpu;
  readonly selection: SelectionMasks;
  readonly routeKinds: Uint32Array;
  readonly intentGold: Uint32Array;
}

function prepareComfort(
  episode: ComfortEpisode,
  options?: { ablateComfort?: boolean; ablateNoise?: boolean },
): PreparedComfort {
  const frame = packBrainFrame(lowerComfortEpisode(episode, options)).frame;
  const active = compileActiveFrame(frame);
  const intentMask = compileIntentMask(frame, active, ACTION_INTENT_SCHEMA_ID);
  const routeKinds = Uint32Array.of(comfortRouteKind(episode));
  const actionToken = episode.target.action === "CRY" ? 0x605 : 0x606;
  return {
    frame,
    selection: { intentMask, argMask: intentMask }, // arity-0: neutral arg slot
    routeKinds,
    intentGold: Uint32Array.of(catalogBankIndex(active, frame, actionToken)),
  };
}

async function predict(
  h: TrainingHarness,
  runner: KrystalForward,
  episode: ComfortEpisode,
  options?: { ablateComfort?: boolean; ablateNoise?: boolean },
): Promise<{ route: number; intentBankIndex: number }> {
  const p = prepareComfort(episode, options);
  runner.forward(p.frame, p.selection);
  await h.device.queue.onSubmittedWorkDone();
  // Readbacks share one staging buffer: sequential awaits only.
  const logits = await runner.readDecisionLogits(1, COMFORT_CONFIG.routeKindCount);
  const intentIdx = await runner.readIntentIndices(1);
  const route = logits[0]! >= logits[1]! ? 0 : 1;
  return { route, intentBankIndex: intentIdx[0]! };
}

function accuracy(results: readonly boolean[]): number {
  if (results.length === 0) return NaN;
  return results.filter(Boolean).length / results.length;
}

interface StageOutcome {
  readonly evalAccuracy: number;
  readonly intentAccuracy: number;
  readonly ablationComfortAccuracy: number;
  readonly ablationNoiseAccuracy: number;
  readonly firstLoss: number;
  readonly lastLoss: number;
}

async function trainStage(
  h: TrainingHarness,
  stage: ComfortEpisode["stage"],
  epochs: number,
  learningRate: number,
): Promise<StageOutcome> {
  const weights = createBrainForwardWeights(COMFORT_CONFIG, 1337);
  const runner = new KrystalForward(weights, COMFORT_CONFIG);
  const trainer = new KrystalBackward(runner);
  const train = stageEpisodes(stage, [0, TRAIN_SEEDS]);
  const evalEpisodes = stageEpisodes(stage, [TRAIN_SEEDS, TRAIN_SEEDS + EVAL_SEEDS]);
  expect(train.length).toBeGreaterThan(0);
  expect(evalEpisodes.length).toBeGreaterThan(0);

  let firstLoss: number | undefined;
  let lastLoss = 0;
  for (let epoch = 0; epoch < epochs; epoch++) {
    for (const episode of train) {
      const p = prepareComfort(episode);
      const { loss } = await trainer.trainStep({
        frame: p.frame,
        selection: p.selection,
        routeKinds: p.routeKinds,
        intentGold: p.intentGold,
        learningRate,
        telemetry: true,
      });
      await h.device.queue.onSubmittedWorkDone();
      firstLoss ??= loss;
      lastLoss = loss ?? lastLoss;
    }
  }

  // Criterion 1: unseen noise seeds -> 100% (decision head route) and the
  // intent selector picks the correct catalog record.
  const routeResults: boolean[] = [];
  const intentResults: boolean[] = [];
  for (const episode of evalEpisodes) {
    const { route, intentBankIndex } = await predict(h, runner, episode);
    routeResults.push(route === comfortRouteKind(episode));
    intentResults.push(intentBankIndex === prepareComfort(episode).intentGold[0]!);
  }
  const evalAccuracy = accuracy(routeResults);

  // Criterion 2: comfort ablation (signal removed, noise kept) -> chance.
  // Measured over all seeds so the estimate is stable.
  const ablationComfortResults: boolean[] = [];
  for (const episode of byStage(stage)) {
    const { route } = await predict(h, runner, episode, { ablateComfort: true });
    ablationComfortResults.push(route === comfortRouteKind(episode));
  }
  const ablationComfortAccuracy = accuracy(ablationComfortResults);

  // Criterion 3: noise ablation (noise removed, signal kept) -> unchanged.
  const ablationNoiseResults: boolean[] = [];
  for (const episode of evalEpisodes) {
    const { route } = await predict(h, runner, episode, { ablateNoise: true });
    ablationNoiseResults.push(route === comfortRouteKind(episode));
  }
  const ablationNoiseAccuracy = accuracy(ablationNoiseResults);

  runner.destroy();
  return {
    evalAccuracy,
    intentAccuracy: accuracy(intentResults),
    ablationComfortAccuracy,
    ablationNoiseAccuracy,
    firstLoss: firstLoss ?? NaN,
    lastLoss,
  };
}

test("comfort episodes artifact is a balanced CRY/LAUGH counterfactual set", () => {
  expect(ARTIFACT.contract).toBe("comfort-episodes@1");
  expect(episodes.length).toBe(160);
  for (const stage of ["1A-extremes", "1B-scale"] as const) {
    const list = byStage(stage);
    expect(list.length).toBe(80);
    expect(list.filter((e) => e.target.action === "CRY").length).toBe(40);
    expect(list.filter((e) => e.target.action === "LAUGH").length).toBe(40);
    // Counterfactual identity: both members of a pair share the noise seed
    // (and therefore the lowerer's noise records), only comfort differs.
    for (let seed = 0; seed < 40; seed++) {
      const pair = list.filter((e) => e.noiseSeed === seed);
      expect(pair.length).toBe(2);
      expect(pair[0]!.comfort).toBe(-pair[1]!.comfort);
    }
  }
});

test("lowerer: counterfactual pair frames differ only in the homeostasis signal", () => {
  const stage = stageEpisodes("1B-scale", [0, 1]);
  const [negative, positive] = stage;
  expect(negative!.target.action).toBe("CRY");
  expect(positive!.target.action).toBe("LAUGH");
  const a = packBrainFrame(lowerComfortEpisode(negative!)).frame;
  const b = packBrainFrame(lowerComfortEpisode(positive!)).frame;
  let signalTokens = 0;
  let noiseTokens = 0;
  for (let i = 0; i < a.tokenIds.length; i++) {
    if (a.tokenIds[i] !== b.tokenIds[i]) {
      signalTokens++;
      // The only differing positions are the FEEL_BAD/FEEL_GOOD sign tokens
      // inside the two signal records (homeostasis summary slot 4 + query
      // slot 122); every noise coordinate is byte-identical.
      const slot = i >> 3;
      expect(slot === 4 || slot === 122).toBe(true);
    }
    if (a.tokenIds[i] !== 0) noiseTokens++;
  }
  expect(signalTokens).toBe(2); // sign token differs in both signal records
  expect(noiseTokens).toBeGreaterThan(800); // full sensory noise background, not PAD
});

test("Step 1A (extremes): unseen seeds 100%, comfort ablation -> chance, noise ablation unchanged", async () => {
  const h = await getTrainingHarness();
  const outcome = await trainStage(h, "1A-extremes", 2, 0.01);
  expect(outcome.firstLoss).toBeGreaterThan(outcome.lastLoss); // loss descends
  expect(outcome.evalAccuracy).toBe(1);
  expect(outcome.intentAccuracy).toBe(1);
  expect(outcome.ablationNoiseAccuracy).toBe(1);
  expect(Math.abs(outcome.ablationComfortAccuracy - 0.5)).toBeLessThanOrEqual(0.2);
  // 64 GPU trainSteps over ~840-token full-noise frames plus eval/ablation
  // forwards; the shared harness needs seconds, not the 5s default.
}, 120_000);

test("Step 1B (scale): unseen seeds 100%, comfort ablation -> chance, noise ablation unchanged", async () => {
  const h = await getTrainingHarness();
  const outcome = await trainStage(h, "1B-scale", 2, 0.01);
  expect(outcome.firstLoss).toBeGreaterThan(outcome.lastLoss);
  expect(outcome.evalAccuracy).toBe(1);
  expect(outcome.intentAccuracy).toBe(1);
  expect(outcome.ablationNoiseAccuracy).toBe(1);
  expect(Math.abs(outcome.ablationComfortAccuracy - 0.5)).toBeLessThanOrEqual(0.2);
}, 120_000);
