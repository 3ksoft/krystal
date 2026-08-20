/**
 * The first world, driven through the HTTP boundary.
 *
 * Same fixture as `first-world.ts`, but nothing here touches Krystal's
 * internals: it POSTs a `krystal-world@3` document and then one
 * `krystal-percept@3` per tick, exactly as a simulation would, and reads the
 * intents back off the wire. That is the point — a fixture that calls the
 * engine directly tests the engine, while this one tests the CONTRACT, which is
 * the half that has two authors and can therefore disagree with itself.
 *
 * Starts its own service on a scratch port and shuts it down after.
 *
 *   bun run tools/first-world-client.ts [ticks]
 */
import {
  FIRST_WORLD,
  firstWorldApply,
  firstWorldPercept,
  firstWorldState,
  firstWorldStep,
} from "../packages/krystal/src/fixtures/first-world.ts";

const TICKS = Number(process.argv[2] ?? 24);
const PORT = Number(Bun.env.KRYSTAL_PIRA_PORT ?? 8899);
const BASE = `http://127.0.0.1:${PORT}`;
const AGENT = "child";
const DEMONSTRATE_UNTIL = 6;

const service = Bun.spawn(["bun", "run", "tools/pira-service.ts"], {
  env: { ...process.env, KRYSTAL_PIRA_PORT: String(PORT) },
  stdout: "pipe",
  stderr: "pipe",
});

async function waitForHealth(attempts = 60): Promise<void> {
  for (let i = 0; i < attempts; i++) {
    try {
      const response = await fetch(`${BASE}/health`);
      if (response.ok) return;
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

  const created = await post(`/v1/agents/${AGENT}`, { vocabulary: FIRST_WORLD });
  console.log(
    `[client] agent created: ${created.symbols} symbols, ${created.channels} channels, ` +
      `${created.relations} relations, checkpoint ${created.checkpointId}`,
  );

  const state = firstWorldState();
  const reached = new Map<string, number>();

  for (let step = 0; step < TICKS; step++) {
    const percept = firstWorldPercept(state, { demonstrate: state.tick < DEMONSTRATE_UNTIL });
    const reply = await post(`/v1/agents/${AGENT}/action-intents`, { percept });

    const intent = reply.intents?.[0];
    let verdict = reply.noIntentReason ?? "nothing proposed";
    let label = "—";

    if (intent) {
      const patient = intent.roles.find((binding: any) => binding.role === "patient")?.operand;
      const patientId = patient?.kind === "instance" ? patient.instanceId : undefined;
      const key = `${intent.relation}(${patientId ?? "-"})`;
      reached.set(key, (reached.get(key) ?? 0) + 1);
      verdict = firstWorldApply(state, {
        relation: intent.relation,
        patient: patientId,
        volitive: intent.volitive,
      });
      label =
        `${intent.relation}(` +
        intent.roles
          .map((b: any) => `${b.role[0]}:${b.operand.kind === "instance" ? b.operand.instanceId : b.operand.kind}`)
          .join(" ") +
        `)${intent.volitive ? " WANT" : ""}`;
    }

    console.log(
      `t=${String(state.tick).padStart(3)} val=${percept.valence.toFixed(2)} ` +
        `sat=${state.satiation.toFixed(2)} ` +
        `${state.tick < DEMONSTRATE_UNTIL ? "[shown] " : "        "}` +
        `${label.padEnd(42)} ${verdict.padEnd(22)} loss=${(reply.valueLoss ?? 0).toFixed(4)}`,
    );

    firstWorldStep(state);
  }

  const trained = await post(`/v1/agents/${AGENT}/train`, {});
  console.log(`\n[client] final train pass: valueLoss=${trained.valueLoss.toFixed(5)} buffered=${trained.buffered}`);

  console.log("what it reached for:");
  for (const [key, count] of [...reached].sort((a, b) => b[1] - a[1])) {
    console.log(`  ${String(count).padStart(3)}x ${key}`);
  }
  const apple = state.entities.find((entity) => entity.instanceId === "apple")!;
  console.log(`final satiation ${state.satiation.toFixed(2)}, apple ${apple.present ? "still there" : "eaten"}`);
} finally {
  service.kill();
}
