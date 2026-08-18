import { BRAIN_LIMITS, INVALID_U32 } from "../packages/schema/src/krystal-engine-schema.ts";
import { getTrainingHarness, readArenaRegion } from "./tests/training-harness.ts";
import {
  KRYSTAL_BACKWARD_ARENA,
  KRYSTAL_BACKWARD_ARENA_BASE,
} from "../packages/webgpu/src/krystal-layout.ts";
import { KrystalForward } from "../packages/webgpu/src/krystal-forward.ts";
import { KrystalBackward } from "../packages/webgpu/src/krystal-backward.ts";
import { packBrainFrame } from "../packages/krystal/src/frame/packer.ts";
import { buildFixtureActionCatalog } from "../packages/krystal/src/fixtures/action-intents.ts";
import { ACTION_INTENT_SCHEMA_ID } from "../packages/krystal/src/fixtures/frame.ts";
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
  type BrainForwardWeights,
} from "../packages/krystal/src/forward/model.ts";
import { brainForwardOracle, matmulOracle } from "../packages/krystal/src/forward/oracle.ts";
import { lowerPolicyFrame, policyRefToken, type PolicyAction, type PolicyEpisode } from "../packages/krystal/src/bridge/policy.ts";

const POLICY_CONFIG = { ...BRAIN_FORWARD_CONFIG, routeKindCount: 6 };
const ROUTE: Record<PolicyAction, number> = { CRY: 0, LAUGH: 1, EAT: 2, MOVE_TOWARDS: 3, LOOK: 4, WAIT: 5 };
const TOKEN: Record<PolicyAction, number> = { LOOK: 0x600, EAT: 0x601, MOVE_TOWARDS: 0x602, WAIT: 0x603, CRY: 0x605, LAUGH: 0x606 };

function pairedEat(seed: number): PolicyEpisode {
  const appleRef = policyRefToken(seed, 0);
  return {
    stage: "S2", seed,
    frames: [{ tick: 10, comfort: -1, resources: [{ kind: "apple", refToken: appleRef, generation: 1, band: "vision", properties: ["RED", "SMALL"] }], gold: { action: "EAT", refToken: appleRef } }],
  };
}

function weightsChecksum(w: BrainForwardWeights): number {
  let sum = 0;
  let n = 0;
  const visit = (a: Float32Array) => { for (let i = 0; i < a.length; i++) sum += a[i]! * (i % 7 + 1); n += a.length; };
  visit(w.embeddings);
  for (const b of w.enc) { visit(b.wq); visit(b.wk); visit(b.wv); visit(b.w1); visit(b.w2); }
  for (const b of w.mixer) { visit(b.wq); visit(b.wk); visit(b.wv); visit(b.w1); visit(b.w2); }
  visit(w.pool);
  visit(w.selector.wq);
  visit(w.selector.wk);
  visit(w.decisionHeadWh);
  return sum;
}

function prepareFor(frame: any, gold: any, catalog: any) {
  const active = compileActiveFrame(frame);
  const goldId = catalog.descriptors.find((d: any) => d.actionToken === TOKEN[gold.action as PolicyAction])!.intentId;
  const intentMask = compileIntentMask(frame, active, ACTION_INTENT_SCHEMA_ID);
  const argMask = argMaskFor(frame, active, catalog, goldId, 0);
  const intentGold = active.bankRecords.findIndex((slot: number) => frame.schemaIds[slot] === ACTION_INTENT_SCHEMA_ID && frame.tokenIds[slot * BRAIN_LIMITS.recordWidth] === TOKEN[gold.action as PolicyAction]);
  let argTarget = INVALID_U32;
  if (gold.refToken !== undefined) {
    for (let j = 0; j < active.bankRecords.length; j++) {
      const slot = active.bankRecords[j]!;
      if ((frame.runtimeRefs[slot * BRAIN_LIMITS.maxReferencesPerRecord]! & 0xfff) === gold.refToken) { argTarget = j; break; }
    }
  }
  return { frame, selection: { intentMask, argMask }, routeKinds: Uint32Array.of(ROUTE[gold.action as PolicyAction]), intentGold: Uint32Array.of(intentGold), argumentTargets: [Uint32Array.of(argTarget)] };
}

function bwdRegion(name: keyof typeof KRYSTAL_BACKWARD_ARENA, elements: number): number {
  return KRYSTAL_BACKWARD_ARENA_BASE + KRYSTAL_BACKWARD_ARENA[name];
}

function maxAbs(a: Float32Array, b: Float32Array): number {
  let max = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) max = Math.max(max, Math.abs(a[i]! - b[i]!));
  return max;
}

async function main() {
  const h = await getTrainingHarness();
  const catalog = buildFixtureActionCatalog();

  // Run the metamorphic comparison for both the EAT frame and its paired CRY frame.
  const eatEpisode = pairedEat(16);
  const cryEpisode: PolicyEpisode = {
    stage: "S2", seed: 16,
    frames: [{ tick: 10, comfort: -1, resources: [], gold: { action: "CRY" } }],
  };
  for (const [tag, episode] of [["EAT", eatEpisode], ["CRY", cryEpisode]] as const) {
    await runMetamorphic(h, catalog, tag, episode);
  }
  console.log("done");
}

async function runMetamorphic(h: Awaited<ReturnType<typeof getTrainingHarness>>, catalog: any, tag: string, episode: PolicyEpisode) {
  console.log(`=== ${tag} frame ===`);
  const lower = (noisePerBand: number) => packBrainFrame(lowerPolicyFrame(episode.frames[0]!, episode, { noisePerBand })).frame;
  const frameA = lower(4); // bank ~35
  const frameB = lower(15); // bank ~83

  const activeA = compileActiveFrame(frameA);
  const activeB = compileActiveFrame(frameB);
  console.log(`bank A=${activeA.bankRecords.length} B=${activeB.bankRecords.length}`);

  // --- Forward comparison (CPU oracle, same init weights) ---
  const w0 = createBrainForwardWeights(POLICY_CONFIG, 42);
  console.log(`init checksum (same seed, two constructions): ${weightsChecksum(w0).toFixed(6)} vs ${weightsChecksum(createBrainForwardWeights(POLICY_CONFIG, 42)).toFixed(6)}`);

  const { hiddenSize: hDim } = POLICY_CONFIG;
  const fwdFor = (frame: any, active: any) => {
    const { mask: recordMask } = compileRecordMask(active.activeTokens);
    const mixerMask = compileMixerMask(frame, active);
    return brainForwardOracle(frame, active, w0, POLICY_CONFIG, recordMask, mixerMask);
  };
  const fwdA = fwdFor(frameA, activeA);
  const fwdB = fwdFor(frameB, activeB);
  const qDiff = maxAbs(fwdA.queryOutput, fwdB.queryOutput);
  console.log(`[1] mixed query state  maxAbsDiff = ${qDiff.toExponential(3)} ${qDiff === 0 ? "(IDENTICAL)" : ""}`);

  // --- Raw intent logits (pre-softmax) via the selector projection ---
  const rawIntentLogits = (fwd: any, frame: any, active: any) => {
    const q = active.queryRecords.length;
    const r = active.bankRecords.length;
    const intentMask = compileIntentMask(frame, active, ACTION_INTENT_SCHEMA_ID);
    const qProj = matmulOracle(fwd.queryOutput, w0.selector.wq, hDim, hDim);
    const kProj = matmulOracle(fwd.bankKeys, w0.selector.wk, hDim, hDim);
    const scale = 1 / Math.sqrt(hDim);
    const out = new Float32Array(r);
    for (let j = 0; j < r; j++) {
      let s = 0;
      for (let d = 0; d < hDim; d++) s += qProj[d]! * kProj[j * hDim + d]!;
      out[j] = s * scale + intentMask[j]!;
    }
    return out;
  };
  const logitsA = rawIntentLogits(fwdA, frameA, activeA);
  const logitsB = rawIntentLogits(fwdB, frameB, activeB);
  const eatIdxA = activeA.bankRecords.findIndex((slot: number) => frameA.schemaIds[slot] === ACTION_INTENT_SCHEMA_ID && frameA.tokenIds[slot * BRAIN_LIMITS.recordWidth] === TOKEN.EAT);
  const eatIdxB = activeB.bankRecords.findIndex((slot: number) => frameB.schemaIds[slot] === ACTION_INTENT_SCHEMA_ID && frameB.tokenIds[slot * BRAIN_LIMITS.recordWidth] === TOKEN.EAT);
  const cryIdxA = activeA.bankRecords.findIndex((slot: number) => frameA.schemaIds[slot] === ACTION_INTENT_SCHEMA_ID && frameA.tokenIds[slot * BRAIN_LIMITS.recordWidth] === TOKEN.CRY);
  const cryIdxB = activeB.bankRecords.findIndex((slot: number) => frameB.schemaIds[slot] === ACTION_INTENT_SCHEMA_ID && frameB.tokenIds[slot * BRAIN_LIMITS.recordWidth] === TOKEN.CRY);
  console.log(`[2] raw intent logits  A: EAT=${logitsA[eatIdxA]!.toFixed(6)} CRY=${logitsA[cryIdxA]!.toFixed(6)}  B: EAT=${logitsB[eatIdxB]!.toFixed(6)} CRY=${logitsB[cryIdxB]!.toFixed(6)}`);

  // --- One GPU trainStep for A and B from identical init ---
  const run = async (frame: any, tag: string) => {
    const weights = createBrainForwardWeights(POLICY_CONFIG, 42);
    const runner = new KrystalForward(weights, POLICY_CONFIG);
    const trainer = new KrystalBackward(runner);
    const prepared = prepareFor(frame, episode.frames[0]!.gold, catalog);
    const result = await trainer.trainStep({ ...prepared, learningRate: 0.01, telemetry: true });
    await h.device.queue.onSubmittedWorkDone();
    const q = prepared.frame.activeQueryRecord !== undefined ? 0 : 0;
    const active = compileActiveFrame(frame);
    const t = active.activeTokens.length;
    const r = active.bankRecords.length;
    const qr = active.queryRecords.length;
    void q;
    const regions = {
      dSelectorWq: [KRYSTAL_BACKWARD_ARENA.dSelectorWq, hDim * hDim],
      dSelectorWk: [KRYSTAL_BACKWARD_ARENA.dSelectorWk, hDim * hDim],
      dDecisionWh: [KRYSTAL_BACKWARD_ARENA.dDecisionWh, POLICY_CONFIG.routeKindCount * 3 * hDim],
      dFieldStates: [KRYSTAL_BACKWARD_ARENA.dFieldStates, t * hDim],
      dPool: [KRYSTAL_BACKWARD_ARENA.dPool, 2 * hDim],
      dBankKeys: [KRYSTAL_BACKWARD_ARENA.dBankKeys, r * hDim],
      dBankValues: [KRYSTAL_BACKWARD_ARENA.dBankValues, r * hDim],
    } as const;
    const grads: Record<string, Float32Array> = {};
    for (const [name, [off, elems]] of Object.entries(regions)) {
      grads[name] = await readArenaRegion(h, bwdRegion(name as keyof typeof KRYSTAL_BACKWARD_ARENA, elems as number), elems as number);
    }
    void qr;
    const post = weightsChecksum(weights);
    runner.destroy();
    return { loss: result.loss!, grads, post, t, r };
  };

  const resA = await run(frameA, "A");
  const resB = await run(frameB, "B");
  console.log(`[3/4/5] CE loss (telemetry): A=${resA.loss.toExponential(4)} B=${resB.loss.toExponential(4)}`);
  for (const name of ["dSelectorWq", "dSelectorWk", "dDecisionWh", "dPool"] as const) {
    const diff = maxAbs(resA.grads[name]!, resB.grads[name]!);
    console.log(`[6/7] ${name.padEnd(14)} maxAbsDiff=${diff.toExponential(3)}`);
  }
  console.log(`[8] post-update param checksum: A=${resA.post.toFixed(6)} B=${resB.post.toFixed(6)}`);

  // --- Per-record comparison of the common slots + zero-check on extras ---
  const { hiddenSize: hh } = POLICY_CONFIG;
  const bankGradBySlot = (frame: any, active: any, grads: Record<string, Float32Array>, name: "dBankKeys" | "dBankValues") => {
    const map = new Map<number, Float32Array>();
    for (let j = 0; j < active.bankRecords.length; j++) {
      const slot = active.bankRecords[j]!;
      map.set(slot, grads[name]!.subarray(j * hh, (j + 1) * hh));
    }
    return map;
  };
  for (const name of ["dBankKeys", "dBankValues"] as const) {
    const a = bankGradBySlot(frameA, activeA, resA.grads, name);
    const b = bankGradBySlot(frameB, activeB, resB.grads, name);
    let commonMaxDiff = 0;
    let commonSlots = 0;
    for (const [slot, va] of a) {
      const vb = b.get(slot);
      if (!vb) continue;
      commonSlots++;
      commonMaxDiff = Math.max(commonMaxDiff, maxAbs(va, vb));
    }
    // Zero-check on the records that only exist in B (the appended distractors).
    let extraMaxAbs = 0;
    for (const [slot, vb] of b) {
      if (a.has(slot)) continue;
      for (let i = 0; i < vb.length; i++) extraMaxAbs = Math.max(extraMaxAbs, Math.abs(vb[i]!));
    }
    console.log(`${name}: common-slots=${commonSlots} commonMaxDiff=${commonMaxDiff.toExponential(3)} extra-records-maxAbs=${extraMaxAbs.toExponential(3)}`);
  }

  // dFieldStates per record (tokens are record-major by active slot order).
  {
    const perRecord = (frame: any, active: any, grads: Record<string, Float32Array>) => {
      const map = new Map<number, Float32Array>();
      for (let i = 0; i < active.activeTokens.length; i++) {
        const slot = active.activeTokens[i]! >> 3;
        const existing = map.get(slot);
        const chunk = grads.dFieldStates!.subarray(i * hh, (i + 1) * hh);
        if (existing) {
          const merged = Float32Array.from(existing);
          for (let d = 0; d < hh; d++) merged[d]! += chunk[d]!;
          map.set(slot, merged);
        } else {
          map.set(slot, Float32Array.from(chunk));
        }
      }
      return map;
    };
    const a = perRecord(frameA, activeA, resA.grads);
    const b = perRecord(frameB, activeB, resB.grads);
    let commonMaxDiff = 0;
    let commonSlots = 0;
    for (const [slot, va] of a) {
      const vb = b.get(slot);
      if (!vb) continue;
      commonSlots++;
      commonMaxDiff = Math.max(commonMaxDiff, maxAbs(va, vb));
    }
    let extraMaxAbs = 0;
    for (const [slot, vb] of b) {
      if (a.has(slot)) continue;
      for (let i = 0; i < vb.length; i++) extraMaxAbs = Math.max(extraMaxAbs, Math.abs(vb[i]!));
    }
    console.log(`dFieldStates: common-slots=${commonSlots} commonMaxDiff=${commonMaxDiff.toExponential(3)} extra-records-maxAbs=${extraMaxAbs.toExponential(3)}`);
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
