// Decisive forgetting probe: train S1-S8, measure, then train S9 on the SAME
// checkpoint and measure the identical prior-stage frames again. Repeated over
// seeds so the spread and the forgetting delta come from one set of runs.
import { buildFixtureActionCatalog } from "./packages/krystal/src/fixtures/action-intents.ts";
import { buildCurriculum } from "./packages/krystal/src/bridge/curriculum.ts";
import { mix32 } from "./packages/krystal/src/bridge/comfort.ts";
import {
  generatePolicyEpisode, lowerPolicyFrame,
  type PolicyEpisode, type PolicyStage,
} from "./packages/krystal/src/bridge/policy.ts";
import { KrystalBackward } from "./packages/webgpu/src/krystal-backward.ts";
import { KrystalForward } from "./packages/webgpu/src/krystal-forward.ts";
import {
  POLICY_CONFIG, createBrainForwardWeights, emitPrediction, getTrainingHarness,
  packBrainFrame, prepareTrainFrame, productionSelection,
} from "./tests/policy-harness.ts";

const PRIOR: readonly PolicyStage[] = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"];
// The exact frames the S9 slice audits, so numbers are directly comparable.
const priorEpisodes = PRIOR.flatMap((stage, i) =>
  Array.from({ length: 8 }, (_, k) => generatePolicyEpisode(stage, 448 + i * 8 + k, "eval")),
);

function balanceS1(episodes: readonly PolicyEpisode[]): PolicyEpisode[] {
  return episodes.map((e) => {
    if (e.stage !== "S1") return e;
    const bad = (mix32((e.seed >>> 0) ^ 0x51) & 1) === 0;
    const f = e.frames[0]!;
    return { ...e, frames: [{ ...f, comfort: bad ? -1 : 1, gold: { action: bad ? "CRY" : "LAUGH" } as const }] };
  });
}

const h = await getTrainingHarness();
const catalog = buildFixtureActionCatalog();

async function train(
  trainer: KrystalBackward, label: string,
  episodes: readonly PolicyEpisode[], epochs: number,
) {
  const frames = episodes.flatMap((episode) => episode.frames.map((frame) => ({ episode, frame })));
  const trail: string[] = [];
  let step = 0;
  for (let epoch = 0; epoch < epochs; epoch++) {
    for (const { episode, frame } of frames) {
      const probe = step % 150 === 0;
      const res = await trainer.trainStep({
        ...prepareTrainFrame(packBrainFrame(lowerPolicyFrame(frame, episode)).frame, frame.gold, catalog),
        learningRate: 0.01, telemetry: probe,
      });
      if (probe) trail.push(Number.isFinite(res.loss!) ? res.loss!.toFixed(3) : String(res.loss));
      step++;
    }
  }
  console.log(`  ${label} loss:`, trail.join(" "));
}

async function score(runner: KrystalForward, episodes: readonly PolicyEpisode[]) {
  const per = new Map<string, { c: number; t: number }>();
  let c = 0, t = 0;
  for (const episode of episodes) {
    for (const frame of episode.frames) {
      const sel = await productionSelection(h, runner, packBrainFrame(lowerPolicyFrame(frame, episode)).frame, catalog);
      const pred = sel ? emitPrediction(sel, catalog) : null;
      const ok = pred?.action === frame.gold.action &&
        (frame.gold.refToken === undefined || pred.refToken === frame.gold.refToken);
      const s = per.get(episode.stage) ?? { c: 0, t: 0 };
      s.t++; t++;
      if (ok) { s.c++; c++; }
      per.set(episode.stage, s);
    }
  }
  return { c, t, per };
}

const s1s8 = buildCurriculum({
  stages: ["S2", "S3", "S4", "S5", "S6", "S7", "S8"], replayStages: ["S1"],
  trainSeeds: [0, 256], evalSeeds: [384, 392],
});
const s9 = buildCurriculum({
  stages: ["S9"], replayStages: [...PRIOR],
  trainSeeds: [0, 256], evalSeeds: [384, 448],
});

for (const seed of [42, 7, 1337]) {
  const runner = new KrystalForward(createBrainForwardWeights(POLICY_CONFIG, seed), POLICY_CONFIG);
  const trainer = new KrystalBackward(runner);
  console.log(`\n=== seed ${seed} ===`);
  await train(trainer, "faza1 S1-S8", balanceS1(s1s8.train), 3);
  const before = await score(runner, priorEpisodes);
  await train(trainer, "faza2 S9   ", balanceS1(s9.train), 3);
  const after = await score(runner, priorEpisodes);
  const s9after = await score(runner, s9.eval);
  const line = (l: string, r: { c: number; t: number; per: Map<string, { c: number; t: number }> }) =>
    `${l} ${r.c}/${r.t} (${(r.c / r.t).toFixed(3)})  ` +
    PRIOR.map((s) => `${s}:${r.per.get(s)?.c ?? 0}/${r.per.get(s)?.t ?? 0}`).join(" ");
  console.log(line("S1-S8 przed S9:", before));
  console.log(line("S1-S8 po   S9:", after));
  console.log(`S9 eval po S9:  ${s9after.c}/${s9after.t} (${(s9after.c / s9after.t).toFixed(3)})`);
  runner.destroy();
}
