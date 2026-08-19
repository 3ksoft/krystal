import { BRAIN_LIMITS, INVALID_U32 } from "../packages/schema/src/krystal-engine-schema.ts";
import { getTrainingHarness } from "./tests/training-harness.ts";
import { KrystalForward } from "../packages/webgpu/src/krystal-forward.ts";
import { KrystalBackward } from "../packages/webgpu/src/krystal-backward.ts";
import { packBrainFrame } from "../packages/krystal/src/frame/packer.ts";
import { buildFixtureActionCatalog } from "../packages/krystal/src/fixtures/action-intents.ts";
import { FIXTURE_ACTION_INTENTS } from "../packages/krystal/src/fixtures/action-intents.ts";
import { ACTION_INTENT_SCHEMA_ID } from "../packages/krystal/src/fixtures/frame.ts";
import { emitIntentSet } from "../packages/krystal/src/forward/intentset.ts";
import {
  argMaskFor,
  compileActiveFrame,
  compileIntentMask,
  compileMixerMask,
  compileRecordMask,
} from "../packages/krystal/src/forward/masks.ts";
import {
  BRAIN_FORWARD_CONFIG,
  createBrainForwardWeights,
} from "../packages/krystal/src/forward/model.ts";
import { brainForwardOracle, matmulOracle } from "../packages/krystal/src/forward/oracle.ts";
import { attentionWithP } from "../packages/krystal/src/forward/backward.ts";
import { generatePolicyEpisode, lowerPolicyFrame, policyRefToken, type PolicyAction, type PolicyEpisode } from "../packages/krystal/src/training/policy.ts";

const POLICY_CONFIG = { ...BRAIN_FORWARD_CONFIG, routeKindCount: 6 };
const ROUTE: Record<PolicyAction, number> = { CRY: 0, LAUGH: 1, EAT: 2, MOVE_TOWARDS: 3, LOOK: 4, WAIT: 5 };
const TOKEN: Record<PolicyAction, number> = { LOOK: 0x600, EAT: 0x601, MOVE_TOWARDS: 0x602, WAIT: 0x603, CRY: 0x605, LAUGH: 0x606 };

function pairedEpisodes(seed: number): { eat: PolicyEpisode; cry: PolicyEpisode } {
  const appleRef = policyRefToken(seed, 0);
  return {
    eat: { stage: "S2", seed, frames: [{ tick: 10, comfort: -1, resources: [{ kind: "apple", refToken: appleRef, generation: 1, band: "vision", properties: ["RED", "SMALL"] }], gold: { action: "EAT", refToken: appleRef } }] },
    cry: { stage: "S2", seed, frames: [{ tick: 10, comfort: -1, resources: [], gold: { action: "CRY" } }] },
  };
}

function lower(episode: PolicyEpisode, noisePerBand: number) {
  return packBrainFrame(lowerPolicyFrame(episode.frames[0]!, episode, { noisePerBand })).frame;
}

function prepare(frame: any, gold: any, catalog: any) {
  const active = compileActiveFrame(frame);
  const goldId = catalog.descriptors.find((d: any) => d.actionToken === TOKEN[gold.action as PolicyAction])!.intentId;
  const intentMask = compileIntentMask(frame, active, ACTION_INTENT_SCHEMA_ID);
  const argMask = argMaskFor(frame, active, catalog, goldId, 0);
  const intentGold = active.bankRecords.findIndex((slot: number) => frame.schemaIds[slot] === ACTION_INTENT_SCHEMA_ID && frame.tokenIds[slot * BRAIN_LIMITS.recordWidth] === TOKEN[gold.action as PolicyAction]);
  let argTarget = INVALID_U32;
  if (gold.refToken !== undefined) {
    for (let j = 0; j < active.bankRecords.length; j++) {
      const slot = active.bankRecords[j]!;
      if ((frame.runtimeRefs[slot * BRAIN_LIMITS.maxReferencesPerRecord]! & 0xffff) === gold.refToken) { argTarget = j; break; }
    }
  }
  return { frame, selection: { intentMask, argMask }, routeKinds: Uint32Array.of(ROUTE[gold.action as PolicyAction]), intentGold: Uint32Array.of(intentGold), argumentTargets: [Uint32Array.of(argTarget)] };
}

async function predict(h: any, runner: KrystalForward, frame: any, catalog: any) {
  const active = compileActiveFrame(frame);
  const q = active.queryRecords.length;
  const r = active.bankRecords.length;
  const hDim = POLICY_CONFIG.hiddenSize;
  const intentMask = compileIntentMask(frame, active, ACTION_INTENT_SCHEMA_ID);
  runner.forward(frame, { intentMask, argMask: intentMask });
  await h.device.queue.onSubmittedWorkDone();
  const sel1 = await runner.readSelection(q, r, hDim);
  const bankIdx = sel1.intent.index[0]!;
  if (bankIdx >= r) return null;
  const slot = active.bankRecords[bankIdx]!;
  if (frame.schemaIds[slot] !== ACTION_INTENT_SCHEMA_ID) return null;
  const actionToken = frame.tokenIds[slot * BRAIN_LIMITS.recordWidth]!;
  const descriptor = catalog.descriptors.find((d: any) => d.actionToken === actionToken);
  if (!descriptor) return null;
  runner.forward(frame, { intentMask, argMask: argMaskFor(frame, active, catalog, descriptor.intentId, 0) });
  await h.device.queue.onSubmittedWorkDone();
  const sel2 = await runner.readSelection(q, r, hDim);
  const { intentSet } = emitIntentSet({ frame, active, catalog, intentSchemaId: ACTION_INTENT_SCHEMA_ID, intent: sel1.intent, argument: sel2.argument, tick: 10 });
  if (intentSet.count === 0) return null;
  const proposal = intentSet.proposals[0]!;
  return { action: FIXTURE_ACTION_INTENTS[proposal.intentId]!.name, refToken: proposal.arguments[0]!.handle.tokenId };
}

/**
 * First-mixer-block attention mass on record categories (trained weights):
 * the apple record vs the catalog records vs noise/other records, mean over
 * heads and the single query row. Mirrors the oracle's mixer step.
 */
function attentionMass(episode: PolicyEpisode, noisePerBand: number, weights: any) {
  const frame = lower(episode, noisePerBand);
  const active = compileActiveFrame(frame);
  const { mask: recordMask } = compileRecordMask(active.activeTokens);
  const r = active.bankRecords.length;
  const q = active.queryRecords.length;
  const h = POLICY_CONFIG.hiddenSize;
  const heads = POLICY_CONFIG.headCount;
  const headDim = POLICY_CONFIG.headDim;
  const mixerMask = compileMixerMask(frame, active);
  const fwd = brainForwardOracle(frame, active, weights, POLICY_CONFIG, recordMask, mixerMask);
  const block = weights.mixer[0]!;
  const qProj = matmulOracle(fwd.queryValues, block.wq, h, h);
  const kProj = matmulOracle(fwd.bankKeys, block.wk, h, h);
  const vProj = matmulOracle(fwd.bankValues, block.wv, h, h);
  const attn = attentionWithP(qProj, kProj, vProj, mixerMask, q, r, h, heads, headDim);
  const p = attn.p; // [heads, Q, R]
  const perRecord = new Array<number>(r).fill(0);
  for (let hd = 0; hd < heads; hd++) {
    for (let j = 0; j < r; j++) perRecord[j]! += p[hd * q * r + j]! / heads;
  }
  let apple = 0;
  let catalog = 0;
  let noise = 0;
  for (let j = 0; j < r; j++) {
    const slot = active.bankRecords[j]!;
    const schema = frame.schemaIds[slot]!;
    if (schema === 2) apple += perRecord[j]!; // Apple
    else if (schema === ACTION_INTENT_SCHEMA_ID) catalog += perRecord[j]!;
    else noise += perRecord[j]!;
  }
  return { r, apple, catalog, noise };
}

function qSeparation(episodePair: { eat: PolicyEpisode; cry: PolicyEpisode }, noisePerBand: number, weights: any): number {
  const measure = (episode: PolicyEpisode) => {
    const frame = lower(episode, noisePerBand);
    const active = compileActiveFrame(frame);
    const { mask: recordMask } = compileRecordMask(active.activeTokens);
    const mixerMask = compileMixerMask(frame, active);
    return brainForwardOracle(frame, active, weights, POLICY_CONFIG, recordMask, mixerMask).queryOutput;
  };
  const a = measure(episodePair.eat);
  const b = measure(episodePair.cry);
  let diff = 0, mag = 0;
  for (let i = 0; i < a.length; i++) {
    diff = Math.max(diff, Math.abs(a[i]! - b[i]!));
    mag = Math.max(mag, Math.abs(a[i]!));
  }
  return diff / mag;
}

async function runR(h: any, catalog: any, noisePerBand: number, initSeed: number) {
  const pairs = Array.from({ length: 16 }, (_, s) => pairedEpisodes(s));
  const evalPairs = Array.from({ length: 8 }, (_, s) => pairedEpisodes(16 + s));
  const trainFrames = pairs.flatMap((p) => [
    { episode: p.eat, frame: p.eat.frames[0]! },
    { episode: p.cry, frame: p.cry.frames[0]! },
  ]);
  const weights = createBrainForwardWeights(POLICY_CONFIG, initSeed);
  const pair0 = pairedEpisodes(16);
  const initSep = qSeparation(pair0, noisePerBand, weights);

  const runner = new KrystalForward(weights, POLICY_CONFIG);
  const trainer = new KrystalBackward(runner);
  for (let epoch = 0; epoch < 8; epoch++) {
    for (const { episode, frame } of trainFrames) {
      const prepared = prepare(lower(episode, noisePerBand), frame.gold, catalog);
      await trainer.trainStep({ ...prepared, learningRate: 0.01 });
    }
  }
  const trainedSep = qSeparation(pair0, noisePerBand, weights);
  const mass = attentionMass(pair0.eat, noisePerBand, weights);

  let eatAcc = 0, cryAcc = 0, R = 0;
  let pEatOnEat = 0, pEatOnCry = 0, nEat = 0, nCry = 0;
  for (const pair of evalPairs) {
    for (const [episode, isEat] of [[pair.eat, true], [pair.cry, false]] as const) {
      const frame = lower(episode, noisePerBand);
      const active = compileActiveFrame(frame);
      R = active.bankRecords.length;
      const pred = await predict(h, runner, frame, catalog);
      const gold = episode.frames[0]!.gold;
      const ok = pred !== null && pred.action === gold.action;
      if (isEat && ok) eatAcc++;
      if (!isEat && ok) cryAcc++;
      // gold-intent probability via a read of the intent head
      const q = active.queryRecords.length;
      const r = active.bankRecords.length;
      const hDim = POLICY_CONFIG.hiddenSize;
      const intentMask = compileIntentMask(frame, active, ACTION_INTENT_SCHEMA_ID);
      runner.forward(frame, { intentMask, argMask: intentMask });
      await h.device.queue.onSubmittedWorkDone();
      const sel = await runner.readSelection(q, r, hDim);
      const eatIdx = active.bankRecords.findIndex((slot: number) => frame.schemaIds[slot] === ACTION_INTENT_SCHEMA_ID && frame.tokenIds[slot * BRAIN_LIMITS.recordWidth] === TOKEN.EAT);
      if (isEat) { pEatOnEat += sel.intent.p[eatIdx]!; nEat++; }
      else { pEatOnCry += sel.intent.p[eatIdx]!; nCry++; }
    }
  }
  runner.destroy();
  return {
    R, eatAcc: eatAcc / 8, cryAcc: cryAcc / 8,
    pEatOnEat: pEatOnEat / nEat, pEatOnCry: pEatOnCry / nCry,
    initSep, trainedSep, mass,
  };
}

async function main() {
  const h = await getTrainingHarness();
  const catalog = buildFixtureActionCatalog();
  const settings: Array<[string, number]> = [
    ["R=0", 0], ["R~8", 1], ["R~32", 4], ["R~64", 9], ["R~100", 15],
  ];
  for (const [label, noisePerBand] of settings) {
    const res = await runR(h, catalog, noisePerBand, 42);
    console.log(`${label} (bank=${res.R}): EAT ${(res.eatAcc * 100).toFixed(0)}% CRY ${(res.cryAcc * 100).toFixed(0)}% | P(EAT) on EAT ${res.pEatOnEat.toFixed(3)} on CRY ${res.pEatOnCry.toFixed(3)} | qSep init ${res.initSep.toExponential(2)} trained ${res.trainedSep.toExponential(2)} | attn apple ${res.mass.apple.toFixed(3)} catalog ${res.mass.catalog.toFixed(3)} noise ${res.mass.noise.toFixed(3)}`);
  }
  console.log("done");
}
main().catch((e) => { console.error(e); process.exit(1); });
