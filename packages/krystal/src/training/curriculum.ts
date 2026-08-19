/**
 * S2-S10 curriculum runner (docs/S2_S10_CURRICULUM_TASK.md, "Milestones and
 * mixture"): samples the stated 60/30/10 episode mixture deterministically,
 * emits train/eval splits with disjoint seeds/layouts/resource ids, and logs
 * the mixture + root seeds so failures replay exactly.
 *
 *   buildCurriculum({ stages, replayStages, ... })
 *     -> { train, eval, log }
 *
 * The adversarial tenth covers the required negatives: no target, duplicate-
 * looking resources, reordered records, a salient Mother/distractor,
 * inaccessible target, incompatible target and poisoned/negative consumables.
 */
import { mulberry32 } from "../forward/model.ts";
import {
  generatePolicyEpisode,
  type PolicyEpisode,
  type PolicyStage,
} from "./policy.ts";

export interface CurriculumOptions {
  /** Current milestone stages (60% of the mixture), in order. */
  readonly stages: readonly PolicyStage[];
  /** Completed-stage replay pool (30%). */
  readonly replayStages: readonly PolicyStage[];
  /** Train seeds [start, end). */
  readonly trainSeeds: readonly [number, number];
  /** Held-out eval seeds [start, end) — disjoint from train. */
  readonly evalSeeds: readonly [number, number];
  /** Mixture root seed (defaults to the first train seed). */
  readonly rootSeed?: number;
  /** Log lines appended (mixture + seeds); exposed for the failure report. */
  readonly log?: string[];
}

export interface CurriculumSplit {
  readonly train: readonly PolicyEpisode[];
  readonly eval: readonly PolicyEpisode[];
  /** Deterministic human-readable mixture report. */
  readonly log: readonly string[];
}

/** Adversarial episode kinds (the required negative tenth). */
export type AdversarialKind =
  | "noTarget"
  | "duplicateResources"
  | "reorderedRecords"
  | "salientDistractor"
  | "inaccessibleTarget"
  | "incompatibleTarget"
  | "poisonedConsumable";

export const ADVERSARIAL_KINDS: readonly AdversarialKind[] = [
  "noTarget",
  "duplicateResources",
  "reorderedRecords",
  "salientDistractor",
  "inaccessibleTarget",
  "incompatibleTarget",
  "poisonedConsumable",
];

/**
 * One adversarial episode. The generated frames reuse the stage machinery but
 * carry the adversarial resource layout; the gold decision is the correct
 * response under that layout (e.g. no edible candidate -> CRY).
 */
export interface AdversarialEpisode extends PolicyEpisode {
  readonly adversarialKind: AdversarialKind;
}

/** Build the adversarial episode for a kind from a seed (deterministic). */
export function generateAdversarialEpisode(kind: AdversarialKind, seed: number): AdversarialEpisode {
  switch (kind) {
    case "noTarget":
      // Bad comfort, nothing edible visible -> the model must not fabricate EAT.
      return {
        stage: "S2", seed, adversarialKind: kind,
        frames: [{ tick: 10, comfort: -1, resources: [], gold: { action: "CRY" } }],
      };
    case "duplicateResources":
      // Two apple-looking records with different refs; the gold is the first
      // specific ref, so a handle must be resolved, not a lookalike guessed.
      return {
        stage: "S4", seed, adversarialKind: kind,
        frames: [{
          tick: 10, comfort: -1,
          resources: [
            { kind: "apple", refToken: 0x8010, generation: 1, band: "vision", properties: ["RED", "SMALL"] },
            { kind: "apple", refToken: 0x8011, generation: 1, band: "vision", properties: ["RED", "SMALL"] },
          ],
          gold: { action: "EAT", refToken: 0x8010 },
        }],
      };
    case "reorderedRecords":
      // The edible is deliberately not first in slot order (permutation is
      // the S4 requirement; this pins a specific adversarial layout).
      return {
        stage: "S4", seed, adversarialKind: kind,
        frames: [{
          tick: 10, comfort: -1,
          resources: [
            { kind: "stone", refToken: 0x8012, generation: 1, band: "vision", properties: [] },
            { kind: "mother", refToken: 0x8013, generation: 1, band: "vision", properties: [] },
            { kind: "apple", refToken: 0x8014, generation: 1, band: "vision", properties: ["RED", "SMALL"] },
          ],
          gold: { action: "EAT", refToken: 0x8014 },
        }],
      };
    case "salientDistractor":
      // A salient Mother record present alongside the Apple; EAT must still
      // point at the Apple, never the agent.
      return {
        stage: "S4", seed, adversarialKind: kind,
        frames: [{
          tick: 10, comfort: -1,
          resources: [
            { kind: "mother", refToken: 0x8015, generation: 1, band: "vision", properties: [] },
            { kind: "apple", refToken: 0x8016, generation: 1, band: "vision", properties: ["RED", "SMALL"] },
          ],
          gold: { action: "EAT", refToken: 0x8016 },
        }],
      };
    case "inaccessibleTarget":
      // Apple is far: EAT is not valid yet; the correct action is to approach.
      return {
        stage: "S5", seed, adversarialKind: kind,
        frames: [{
          tick: 10, comfort: -1,
          resources: [{ kind: "apple", refToken: 0x8017, generation: 1, band: "vision", properties: ["RED", "SMALL", "FAR"] }],
          gold: { action: "MOVE_TOWARDS", refToken: 0x8017 },
        }],
      };
    case "incompatibleTarget":
      // Only a negative consumable (Stone) and an agent are visible: EAT has
      // no legal argument, so the executable response is CRY.
      return {
        stage: "S7", seed, adversarialKind: kind,
        frames: [{
          tick: 10, comfort: -1,
          resources: [
            { kind: "stone", refToken: 0x8018, generation: 1, band: "vision", properties: [] },
            { kind: "mother", refToken: 0x8019, generation: 1, band: "vision", properties: [] },
          ],
          gold: { action: "CRY" },
        }],
      };
    case "poisonedConsumable":
      // The apple carries the poison trait; the model must not EAT it.
      return {
        stage: "S8", seed, adversarialKind: kind,
        frames: [{
          tick: 10, comfort: -1,
          resources: [{ kind: "apple", refToken: 0x801a, generation: 1, band: "vision", properties: ["RED", "SMALL", "POISONED"] }],
          gold: { action: "CRY" },
        }],
      };
  }
}

function stageLabel(stages: readonly PolicyStage[]): string {
  return stages.length === 0 ? "—" : stages.join("+");
}

/**
 * Build the deterministic curriculum: 60% current-stage episodes, 30% replay
 * from completed stages, 10% adversarial negatives. Train/eval splits use
 * disjoint seed ranges; resource ids, slot permutations and noise are pure
 * functions of the seed, so held-out eval is genuinely unseen.
 */
export function buildCurriculum(options: CurriculumOptions): CurriculumSplit {
  const { stages, replayStages, trainSeeds, evalSeeds } = options;
  const rootSeed = options.rootSeed ?? trainSeeds[0];
  const log = options.log ?? [];

  const sample = (pool: readonly PolicyStage[], seed: number): PolicyEpisode => {
    const stage = pool[seed % pool.length]!;
    return generatePolicyEpisode(stage, seed);
  };

  const train: PolicyEpisode[] = [];
  const rng = mulberry32(rootSeed);
  const trainCount = trainSeeds[1] - trainSeeds[0];
  const mixture: { current: number; replay: number; adversarial: number } = { current: 0, replay: 0, adversarial: 0 };
  for (let i = 0; i < trainCount; i++) {
    const seed = trainSeeds[0] + i;
    const draw = rng();
    if (draw < 0.6 && stages.length > 0) {
      train.push(sample(stages, seed));
      mixture.current++;
    } else if (draw < 0.9 && replayStages.length > 0) {
      train.push(sample(replayStages, seed));
      mixture.replay++;
    } else {
      const kind = ADVERSARIAL_KINDS[seed % ADVERSARIAL_KINDS.length]!;
      train.push(generateAdversarialEpisode(kind, seed));
      mixture.adversarial++;
    }
  }

  const evalEpisodes: PolicyEpisode[] = [];
  for (let seed = evalSeeds[0]; seed < evalSeeds[1]; seed++) {
    // Eval covers the current stages across the held-out seeds, so every stage
    // has an unseen-layout measurement. The eval band gives held-out episodes
    // a disjoint resource-id sub-range (policyRefToken band), so train/eval
    // resource ids never collide by construction.
    const stage = stages[seed % stages.length] ?? replayStages[0] ?? "S2";
    evalEpisodes.push(generatePolicyEpisode(stage, seed, "eval"));
  }

  log.push(`curriculum: stages=${stageLabel(stages)} replay=${stageLabel(replayStages)}`);
  log.push(`curriculum: trainSeeds=[${trainSeeds[0]}..${trainSeeds[1]}) evalSeeds=[${evalSeeds[0]}..${evalSeeds[1]}) rootSeed=${rootSeed}`);
  log.push(`curriculum: mixture current=${mixture.current} replay=${mixture.replay} adversarial=${mixture.adversarial} (of ${train.length})`);
  log.push(`curriculum: adversarial kinds=${ADVERSARIAL_KINDS.join(",")}`);

  return { train, eval: evalEpisodes, log };
}
