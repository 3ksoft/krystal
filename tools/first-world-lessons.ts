/**
 * Teach the first world, over the wire.
 *
 * The same scene every time — child, mother, apple, stone — with one lesson:
 * EAT the apple. Nothing about the apple says it is food, so if the creature
 * comes to prefer it, that preference was taught rather than looked up.
 *
 * `teach` reports what the policy believed BEFORE each update, so the hit rate
 * across a window is a measurement of learning rather than of the last nudge.
 *
 *   bun run tools/first-world-lessons.ts [lessons]
 */
import {
  FIRST_WORLD,
  firstWorldPercept,
  firstWorldState,
} from "../packages/krystal/src/fixtures/first-world.ts";
import type { v1_0_0 as contract } from "../packages/schema/generated/world.types.ts";

const LESSONS = Number(process.argv[2] ?? 60);
const WINDOW = 10;
const PORT = Number(Bun.env.KRYSTAL_PIRA_PORT ?? 8898);
const BASE = `http://127.0.0.1:${PORT}`;
const AGENT = "child";

const service = Bun.spawn(["bun", "run", "tools/pira-service.ts"], {
  env: { ...process.env, KRYSTAL_PIRA_PORT: String(PORT) },
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
  const percept = firstWorldPercept(state);
  const lesson: contract.Lesson = {
    contract: "krystal-lesson@3",
    label: "eat the apple",
    percept,
    expect: {
      relation: "EAT",
      roles: [
        { role: "agent", operand: { kind: "instance", instanceId: "child" } },
        { role: "patient", operand: { kind: "instance", instanceId: "apple" } },
      ],
    },
  };

  console.log(`[lessons] teaching "${lesson.label}" ${LESSONS} times\n`);
  console.log("window   relation hit   patient hit   p(relation)   p(patient)");

  let relationHits = 0;
  let patientHits = 0;
  let pRelation = 0;
  let pPatient = 0;
  const firstWindow: number[] = [];
  const lastWindow: number[] = [];

  for (let i = 0; i < LESSONS; i++) {
    const report = await post(`/v1/agents/${AGENT}/lesson`, { lesson });
    if (!report.applied) {
      console.log(`  skipped: ${report.skipped}`);
      break;
    }
    relationHits += report.intentHit ? 1 : 0;
    patientHits += report.patientHit ? 1 : 0;
    pRelation += report.intentProbability;
    pPatient += report.patientProbability;

    if (i < WINDOW) firstWindow.push(report.intentProbability);
    if (i >= LESSONS - WINDOW) lastWindow.push(report.intentProbability);

    if ((i + 1) % WINDOW === 0) {
      console.log(
        `${String(i + 1).padStart(5)}   ` +
          `${String(relationHits).padStart(9)}/${WINDOW}   ` +
          `${String(patientHits).padStart(8)}/${WINDOW}   ` +
          `${(pRelation / WINDOW).toFixed(4).padStart(11)}   ` +
          `${(pPatient / WINDOW).toFixed(4).padStart(10)}`,
      );
      relationHits = 0;
      patientHits = 0;
      pRelation = 0;
      pPatient = 0;
    }
  }

  const mean = (xs: number[]) => (xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length);
  console.log(
    `\np(EAT) over the first ${WINDOW}: ${mean(firstWindow).toFixed(4)}` +
      `  over the last ${WINDOW}: ${mean(lastWindow).toFixed(4)}`,
  );
} finally {
  service.kill();
}
