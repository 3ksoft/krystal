/**
 * Run the first world.
 *
 * A child, a mother, an apple and a stone, and no label anywhere saying which
 * of the two is food. The mother demonstrates eating the apple for the first
 * few ticks; after that the creature is on its own with random weights and
 * whatever it made of what it saw.
 *
 * This is a harness, not a benchmark. What it shows is that the whole v3 chain
 * runs end to end — vocabulary, percept, frame, role selection, intent, verdict
 * — and what the creature actually reached for while doing it.
 *
 *   bun run tools/first-world.ts [ticks]
 */
import {
  CATALOG_SCHEMA_ID,
  ExperienceBuffer,
  ReferenceTable,
  compileRelationCatalog,
  createAgent,
  lowerPercept,
  toAgentIntents,
  validatePercept,
} from "../packages/krystal/src/bridge/index.ts";
import {
  FIRST_WORLD,
  firstWorldApply,
  firstWorldPercept,
  firstWorldState,
  firstWorldStep,
} from "../packages/krystal/src/fixtures/first-world.ts";
import { decide } from "../packages/krystal/src/forward/decide.ts";
import { trainValueHead } from "../packages/krystal/src/forward/value-training.ts";
import { compilePerRowArgumentMask } from "../packages/krystal/src/forward/masks.ts";
import { hashString, mix32 } from "../packages/krystal/src/forward/sampling.ts";
import {
  BRAIN_FORWARD_CONFIG,
  createBrainForwardWeights,
} from "../packages/krystal/src/forward/model.ts";

const TICKS = Number(process.argv[2] ?? 40);
/** Ticks the mother spends showing what to do with an apple. */
const DEMONSTRATE_UNTIL = 6;
const TRAIN_EVERY = 8;

const agent = createAgent({ vocabulary: FIRST_WORLD });
const catalog = compileRelationCatalog(agent.vocabulary);
const config = { ...BRAIN_FORWARD_CONFIG, tokenRows: agent.vocabulary.tokenRows };
const weights = createBrainForwardWeights(config, 1337);
const references = new ReferenceTable();
const buffer = new ExperienceBuffer();
const state = firstWorldState();

console.log(
  `[first-world] ${agent.vocabulary.header.activeTokenCount} symbols, ` +
    `${agent.vocabulary.channels.size} channels, ${catalog.header.intentCount} relations`,
);

let previousValence: number | undefined;
let loss = 0;
const reached = new Map<string, number>();

for (let step = 0; step < TICKS; step++) {
  const raw = firstWorldPercept(state, { demonstrate: state.tick < DEMONSTRATE_UNTIL });
  const { percept } = validatePercept(raw, agent.vocabulary);

  const lowered = lowerPercept(percept, agent.vocabulary, references, undefined, undefined, previousValence);
  previousValence = percept.valence;
  buffer.record(lowered.frame, percept.tick, lowered.valenceDelta);

  const result = decide({
    frame: lowered.frame,
    weights,
    config,
    catalog,
    intentSchemaId: CATALOG_SCHEMA_ID,
    tick: percept.tick,
    explore: { seed: mix32(hashString("child") ^ mix32(percept.tick)) },
  });

  const intents = toAgentIntents(
    result.intentSet,
    (intentId) => agent.vocabulary.relations[intentId]?.relation,
    (refToken) => references.instanceFor(refToken),
  );

  let verdict = "nothing proposed";
  if (intents[0]) {
    const intent = intents[0];
    const patient = intent.roles.find((binding) => binding.role === "patient")?.operand;
    const patientId = patient?.kind === "instance" ? patient.instanceId : undefined;
    const key = `${intent.relation}(${patientId ?? "-"})`;
    reached.set(key, (reached.get(key) ?? 0) + 1);
    verdict = firstWorldApply(state, {
      relation: intent.relation,
      patient: patientId,
      volitive: intent.volitive,
    });
  }

  if ((step + 1) % TRAIN_EVERY === 0) {
    const entries = buffer.drain();
    if (entries.length > 0) {
      loss = trainValueHead({
        entries,
        weights,
        config,
        intentSchemaId: CATALOG_SCHEMA_ID,
        argMask: (frame, active) =>
          compilePerRowArgumentMask(
            frame,
            active,
            catalog,
            new Array(active.queryRecords.length).fill(0),
            "patient",
          ),
        learningRate: 0.05,
      }).meanValueLoss;
    }
  }

  const label = intents[0]
    ? `${intents[0].relation}(${
        intents[0].roles.map((b) => `${b.role[0]}:${b.operand.kind === "instance" ? b.operand.instanceId : b.operand.kind}`).join(" ")
      })${intents[0].volitive ? " WANT" : ""}`
    : "—";
  console.log(
    `t=${String(state.tick).padStart(3)} ` +
      `val=${percept.valence.toFixed(2)} ` +
      `sat=${state.satiation.toFixed(2)} ` +
      `${state.tick < DEMONSTRATE_UNTIL ? "[shown] " : "        "}` +
      `${label.padEnd(44)} ${verdict.padEnd(22)} loss=${loss.toFixed(4)}`,
  );

  firstWorldStep(state);
}

console.log("\nwhat it reached for:");
for (const [key, count] of [...reached].sort((a, b) => b[1] - a[1])) {
  console.log(`  ${String(count).padStart(3)}x ${key}`);
}
console.log(
  `\nfinal satiation ${state.satiation.toFixed(2)}, ` +
    `apple ${state.entities.find((e) => e.instanceId === "apple")!.present ? "still there" : "eaten"}`,
);
