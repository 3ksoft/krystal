/**
 * Learning from consequences alone.
 *
 * No lessons here. The mother demonstrates briefly and then stops; after that
 * the only thing telling the creature anything is what happens to its valence
 * after it acts. The apple regrows, so a good outcome can be tested rather than
 * merely enjoyed once.
 *
 * Reported per window: how often the creature reached for the apple, how often
 * it bit something that hurt, and where satiation ended up. A policy that is
 * learning shifts the first up and the second down.
 *
 *   bun run tools/first-world-rl.ts [ticks]
 */
import {
  FIRST_WORLD,
  firstWorldApply,
  firstWorldPercept,
  firstWorldState,
  firstWorldStep,
} from "../packages/krystal/src/fixtures/first-world.ts";

const TICKS = Number(process.argv[2] ?? 400);
const WINDOW = Number(process.argv[3] ?? 50);
const PORT = Number(Bun.env.KRYSTAL_PIRA_PORT ?? 8897);
const BASE = `http://127.0.0.1:${PORT}`;
const AGENT = "child";
const DEMONSTRATE_UNTIL = 8;

const service = Bun.spawn(["bun", "run", "tools/pira-service.ts"], {
  env: {
    ...process.env,
    KRYSTAL_PIRA_PORT: String(PORT),
    KRYSTAL_TRAIN_EVERY: Bun.env.KRYSTAL_TRAIN_EVERY ?? "8",
    KRYSTAL_LR: Bun.env.KRYSTAL_LR ?? "0.1",
    KRYSTAL_POLICY_SCALE: Bun.env.KRYSTAL_POLICY_SCALE ?? "1",
  },
  stdout: "ignore",
  stderr: "pipe",
});

async function waitForHealth(attempts = 60): Promise<void> {
  for (let i = 0; i < attempts; i++) {
    try {
      if ((await fetch(`${BASE}/health`)).ok) return;
    } catch {
      // not up yet
    }
    await Bun.sleep(100);
  }
  throw new Error("service did not become healthy");
}

async function post(path: string, body: unknown): Promise<any> {
  const response = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const json = (await response.json()) as any;
  if (!response.ok) throw new Error(`${path}: ${json?.error ?? response.status}`);
  return json;
}

try {
  await waitForHealth();
  await post(`/v1/agents/${AGENT}`, { vocabulary: FIRST_WORLD });

  const state = firstWorldState();
  let ateApple = 0;
  let bitSomething = 0;
  let valenceSum = 0;
  let advantage = 0;
  let choseEat = 0;
  let chosePatientApple = 0;
  let loss = 0;

  console.log(`[rl] ${TICKS} ticks, window ${WINDOW}, no lessons after t=${DEMONSTRATE_UNTIL}\n`);
  console.log(" ticks   chose EAT   patient=apple   ate   bit   mean valence   loss      adv");

  for (let step = 0; step < TICKS; step++) {
    const percept = firstWorldPercept(state, { demonstrate: state.tick < DEMONSTRATE_UNTIL });
    valenceSum += percept.valence;
    const reply = await post(`/v1/agents/${AGENT}/action-intents`, { percept });
    advantage = reply.meanAdvantage ?? advantage;

    loss = reply.valueLoss ?? loss;
    const intent = reply.intents?.[0];
    if (intent) {
      if (intent.relation === "EAT") choseEat++;
      const patient = intent.roles.find((binding: any) => binding.role === "patient")?.operand;
      const verdict = firstWorldApply(state, {
        relation: intent.relation,
        patient: patient?.kind === "instance" ? patient.instanceId : undefined,
        volitive: intent.volitive,
      });
      if (patient?.kind === "instance" && patient.instanceId === "apple") chosePatientApple++;
      if (verdict.startsWith("ate ")) ateApple++;
      if (verdict.startsWith("bit ")) bitSomething++;
    }

    firstWorldStep(state);

    if ((step + 1) % WINDOW === 0) {
      console.log(
        `${String(step + 1).padStart(6)}   ` +
          `${String(choseEat).padStart(9)}   ` +
          `${String(chosePatientApple).padStart(13)}   ` +
          `${String(ateApple).padStart(3)}   ` +
          `${String(bitSomething).padStart(3)}   ` +
          `${(valenceSum / WINDOW).toFixed(4).padStart(12)}   ` +
          `${loss.toFixed(5)}   ${advantage.toFixed(5)}`,
      );
      ateApple = 0;
      bitSomething = 0;
      valenceSum = 0;
      choseEat = 0;
      chosePatientApple = 0;
    }
  }
} finally {
  service.kill();
}
