/**
 * Krystal's HTTP face for a simulation.
 *
 * Four calls, one per thing a world needs:
 *
 *   POST /v1/agents/:id                  declare the world (krystal-world@3)
 *   POST /v1/agents/:id/action-intents   a tick (krystal-percept@3) -> intents
 *   POST /v1/agents/:id/lesson           be taught (krystal-lesson@3)
 *   POST /v1/agents/:id/train            learn from what has settled
 *
 * There is no boot training and no pretrained policy. A previous one was fitted
 * to a fixture vocabulary, which means it was fitted to a different set of
 * embedding rows: for an agent compiled from a simulation's own vocabulary
 * those weights denote nothing. A creature therefore starts from random weights
 * and learns here, in the world it actually lives in.
 *
 * Random does not mean nonsensical. The relation mask admits only catalog
 * records and each role mask only type-compatible candidates, so an untrained
 * creature proposes structurally valid relations from the first tick — which is
 * what makes the experience it generates worth learning from.
 *
 * Runs on the CPU. The forward and the value-head backward are both CPU paths,
 * so the service needs no GPU device at all.
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
  RELATION_ROLES,
  type Agent,
  type CompiledCatalog,
  type contract,
} from "../packages/krystal/src/bridge/index.ts";
import { decide, type DecideResult } from "../packages/krystal/src/forward/decide.ts";
import { teach } from "../packages/krystal/src/forward/teaching.ts";
import { Lesson } from "../packages/schema/src/world.ts";
import { type } from "arktype";
import { trainPolicy } from "../packages/krystal/src/forward/policy-training.ts";
import { compilePerRowArgumentMask } from "../packages/krystal/src/forward/masks.ts";
import { hashString, mix32 } from "../packages/krystal/src/forward/sampling.ts";
import {
  BRAIN_FORWARD_CONFIG,
  createBrainForwardWeights,
  type BrainForwardConfig,
  type BrainForwardWeights,
} from "../packages/krystal/src/forward/model.ts";

const PORT = Number(Bun.env.KRYSTAL_PIRA_PORT ?? 8801);
const LEARNING_RATE = Number(Bun.env.KRYSTAL_LR ?? 0.05);
/** Ticks between value-head updates. */
const TRAIN_EVERY = Number(Bun.env.KRYSTAL_TRAIN_EVERY ?? 16);
/**
 * How hard the policy step pulls relative to the value step.
 *
 * Below one by default: every actor update changes the distribution the next
 * batch is drawn from, so a rate the critic tolerates easily will make the
 * actor chase noise.
 */
const POLICY_SCALE = Number(Bun.env.KRYSTAL_POLICY_SCALE ?? 0.5);

interface ServiceAgent {
  readonly agent: Agent;
  readonly catalog: CompiledCatalog;
  readonly config: BrainForwardConfig;
  readonly weights: BrainForwardWeights;
  readonly references: ReferenceTable;
  readonly buffer: ExperienceBuffer;
  /** Last tick's valence; its change is derived here, never sent. */
  previousValence: number | undefined;
  ticks: number;
  /** Value-head updates applied so far. */
  updates: number;
  lastValueLoss: number;
  lastAdvantage: number;
}

const POLICY_NAME = "krystal-online";

/**
 * Which brain produced a decision.
 *
 * There is no pretrained policy to name, so identity is the two things that
 * actually distinguish one brain from another: the vocabulary it was compiled
 * against, and how much it has learned since. A creature that has lived longer
 * is a different creature, and the id says so.
 */
function checkpointId(open: ServiceAgent): string {
  const hash = (open.agent.vocabulary.header.manifestHashLo >>> 0).toString(16);
  return `w${hash}-t${open.ticks}-u${open.updates}`;
}

const agents = new Map<string, ServiceAgent>();

function openAgent(agentId: string, vocabulary: contract.WorldVocabulary): ServiceAgent {
  const agent = createAgent({ vocabulary });
  const existing = agents.get(agentId);
  if (existing) {
    // Same id, different vocabulary: refuse rather than hand back the agent
    // that already exists. Silently returning the old one makes iterating on a
    // world deeply confusing — a corrected catalog appears to have no effect —
    // and the alternative, swapping the vocabulary under a live agent, would
    // redefine every embedding row it had learned.
    if (
      existing.agent.vocabulary.header.manifestHashLo !== agent.vocabulary.header.manifestHashLo ||
      existing.agent.vocabulary.header.manifestHashHi !== agent.vocabulary.header.manifestHashHi
    ) {
      throw new Error(
        `agent '${agentId}' already exists with a different vocabulary (${checkpointId(existing)}); ` +
          "use a new agent id, or DELETE this one first — its learned rows follow the vocabulary it was built with",
      );
    }
    return existing;
  }
  // The row table comes from this vocabulary, never from a global: another one
  // assigns different rows, and the same token would otherwise train a
  // different vector with nothing to signal it.
  const config: BrainForwardConfig = {
    ...BRAIN_FORWARD_CONFIG,
    tokenRows: agent.vocabulary.tokenRows,
  };
  const created: ServiceAgent = {
    agent,
    catalog: compileRelationCatalog(agent.vocabulary),
    config,
    weights: createBrainForwardWeights(config, 42),
    references: new ReferenceTable(),
    buffer: new ExperienceBuffer(),
    previousValence: undefined,
    ticks: 0,
    updates: 0,
    lastValueLoss: 0,
    lastAdvantage: 0,
  };
  agents.set(agentId, created);
  return created;
}

/**
 * One pass over experience whose outcome is known.
 *
 * Actor and critic together: the value head learns what a situation was worth,
 * and the selectors are pushed toward or away from what was actually done,
 * scaled by how far the outcome fell from that expectation.
 */
function learn(open: ServiceAgent): number {
  const entries = open.buffer.drain();
  if (entries.length === 0) return open.lastValueLoss;
  const report = trainPolicy({
    entries,
    weights: open.weights,
    config: open.config,
    catalog: open.catalog,
    intentSchemaId: CATALOG_SCHEMA_ID,
    learningRate: LEARNING_RATE,
    policyScale: POLICY_SCALE,
  });
  open.lastValueLoss = report.meanValueLoss;
  open.lastAdvantage = report.meanAdvantage;
  if (report.framesSeen > 0) open.updates++;
  return report.meanValueLoss;
}

/**
 * A small number that stays readable.
 *
 * An untrained policy's commitment sits around 1e-5, and fixed decimals print
 * every one of those as "0.0000" — which reads as "no commitment at all" when
 * the real reading is "very little, and here is how little". The distinction
 * matters precisely while the numbers are small, because that is when one is
 * watching for them to grow.
 */
function fmt(value: number): string {
  if (value === 0) return "0";
  return Math.abs(value) < 1e-3 ? value.toExponential(1) : value.toFixed(4);
}

/** A readable name for one participant. */
function operandLabel(operand: contract.PerceptOperand): string {
  switch (operand.kind) {
    case "instance": return operand.instanceId;
    case "symbol": return operand.symbol;
    default: return operand.kind.toUpperCase();
  }
}

/** `EAT(agent: child, patient: apple)`, short enough for one log line. */
function intentLabel(intent: contract.AgentIntent): string {
  const roles = intent.roles.map((binding) => `${binding.role}: ${operandLabel(binding.operand)}`);
  return `${intent.relation}(${roles.join(", ")})`;
}

interface AttemptReport {
  readonly row: number;
  /** Whether this row's choice survived into an emitted proposal. */
  readonly emitted: boolean;
  readonly relation: string | undefined;
  readonly intentProbability: number;
  readonly bankRecords: number;
  /** Per role: how many candidates it admitted, and what it accepts. */
  readonly roles: Record<string, { candidates: number }>;
  /** One line, already phrased for a log. */
  readonly detail: string;
}

/**
 * What each query row tried, and why it did or did not become a proposal.
 *
 * The point is the failing rows. "No admissible filler" names a category of
 * failure, not an instance of one: it does not say which relation was chosen or
 * which role starved, and without that the next question — is the role too
 * narrow, or was the world simply empty of suitable things — has nowhere to
 * start. A creature reaching for something it cannot reach is doing something,
 * and the log should say what.
 */
function describeAttempts(open: ServiceAgent, result: DecideResult): AttemptReport[] {
  return result.chosen.map((choice, row) => {
    const intentId = result.chosenIntents[row];
    const declared = intentId === undefined ? undefined : open.agent.vocabulary.relations[intentId];
    const relation = declared?.relation;
    const emitted = result.intentSet.proposals.some(
      (proposal) => proposal.proposalSlot === choice.row && proposal.lifecycle !== "empty",
    );
    const bankRecords = result.active.bankRecords.length;

    const roles: Record<string, { candidates: number }> = {};
    const parts: string[] = [];
    for (const role of RELATION_ROLES) {
      const chosenRole = choice.roles[role];
      if (!chosenRole) continue;
      roles[role] = { candidates: chosenRole.candidates };
      parts.push(
        `${role} ${chosenRole.candidates}/${bankRecords}` +
          `${chosenRole.candidates === 0 ? " — nothing in this frame to bind" : ""}`,
      );
    }

    const detail =
      `row ${row}: tried ${relation ?? "?"} p=${fmt(choice.intentProbability)} · ` +
      (parts.length === 0 ? "no roles scored" : parts.join(" · "));

    return {
      row,
      emitted,
      relation,
      intentProbability: choice.intentProbability,
      bankRecords,
      roles,
      detail,
    };
  });
}

function act(agentId: string, body: any) {
  const open = agents.get(agentId);
  if (!open) {
    throw new Error(`agent '${agentId}' was never created; POST the world to /v1/agents/${agentId} first`);
  }

  // Strict: an unknown symbol is refused, never dropped. A boundary that
  // quietly forgets is what makes "why can it not see the apple" cost a day.
  const { percept } = validatePercept(body?.percept, open.agent.vocabulary);
  if (percept.actorId !== agentId) throw new Error("agent path does not match percept.actorId");

  const lowered = lowerPercept(
    percept,
    open.agent.vocabulary,
    open.references,
    undefined,
    undefined,
    open.previousValence,
  );
  open.previousValence = percept.valence;
  open.buffer.record(lowered.frame, percept.tick, lowered.valenceDelta);
  open.ticks++;

  const result = decide({
    frame: lowered.frame,
    weights: open.weights,
    config: open.config,
    catalog: open.catalog,
    intentSchemaId: CATALOG_SCHEMA_ID,
    tick: percept.tick,
    // The creature acts rather than reporting its mode. An argmax policy emits
    // one action per frame and so has nothing to compare against — no reward
    // rule can teach it anything, which is why an untrained creature repeated
    // the same LOOK for as long as anyone watched. The seed is derived from the
    // agent and the tick, so a replayed run draws exactly the same choices and
    // two creatures in one world do not flail in lockstep.
    explore: { seed: mix32(hashString(agentId) ^ mix32(percept.tick)) },
  });

  const intents = toAgentIntents(
    result.intentSet,
    (intentId) => open.agent.vocabulary.relations[intentId]?.relation,
    (refToken) => open.references.instanceFor(refToken),
  );

  // What was chosen, kept against the frame it was chosen in: a policy-gradient
  // update needs the action that was taken, not the one that would win now.
  const choice = result.chosen[0];
  if (choice !== undefined && result.chosenIntents[0] !== undefined) {
    open.buffer.attachChoice({
      intentBank: choice.intentBank,
      intentId: result.chosenIntents[0],
      patientBank: choice.roles.patient?.bank,
    });
  }

  if (open.ticks % TRAIN_EVERY === 0) learn(open);

  const attempts = describeAttempts(open, result);

  // An empty intent set has several causes and only one of them is about
  // learning. Naming which turns "it does nothing" from a mystery into a
  // question: a policy that has not committed looks exactly like a frame with
  // no actor in it, and only the first is something training will fix.
  const d = result.diagnostics;
  const noIntentReason =
    intents.length > 0
      ? undefined
      : d.catalogCandidates === 0
        ? "no catalog records in the frame — the world declared no relations, or they did not reach the catalog band"
        : d.queryRows === 0
          ? "no query row in the frame"
          : d.droppedNoAgent > 0
            ? "no agent could be resolved: send a record for the actor whose instanceId equals the percept's actorId"
            : d.droppedNoPatient > 0
              ? "no admissible patient: nothing in this frame carries a live reference to bind — the creature can only act on what it perceives or remembers"
              : "the policy proposed nothing";

  const overflow = lowered.overflow.map((entry) => `${entry.band}:${entry.offered}>${entry.admitted}`);
  console.log(
    `[krystal] tick=${percept.tick} agent=${agentId} valence=${percept.valence.toFixed(2)}` +
      `${lowered.valenceDelta === undefined ? "" : ` d=${lowered.valenceDelta.toFixed(2)}`}` +
      ` intents=${
        intents.map((i) => `${intentLabel(i)} c=${fmt(i.commitment)}`).join(",") || "none"
      }` +
      `${noIntentReason === undefined ? "" : ` (${noIntentReason})`}` +
      ` loss=${open.lastValueLoss.toFixed(5)} adv=${fmt(open.lastAdvantage)}` +
      `${overflow.length === 0 ? "" : ` overflow=${overflow.join(",")}`}` +
      `${lowered.truncatedRecords === 0 ? "" : ` truncated=${lowered.truncatedRecords}`}`,
  );
  // A dropped proposal used to vanish without saying what it had been. The
  // creature was trying something, and the one thing worth knowing — WHAT it
  // was trying, and how many candidates each role had — was exactly what never
  // reached the log. Print it only when something was lost, so a healthy tick
  // stays one line.
  for (const attempt of attempts) {
    if (attempt.emitted) continue;
    console.log(`           ${attempt.detail}`);
  }

  return {
    intents,
    ...(noIntentReason === undefined
      ? {}
      : {
          noIntentReason,
          diagnostics: result.diagnostics,
          // What it was trying, per query row, so a drop is legible from the
          // simulation side without reading our log.
          attempts: attempts.filter((attempt) => !attempt.emitted),
        }),
    policy: POLICY_NAME,
    checkpointId: checkpointId(open),
    valueLoss: open.lastValueLoss,
    meanAdvantage: open.lastAdvantage,
    // What the frame could not hold, said out loud: a band that drops its tail
    // silently is indistinguishable from a scene that was simply emptier.
    overflow: lowered.overflow,
    truncatedRecords: lowered.truncatedRecords,
    manifestHashLo: open.agent.vocabulary.header.manifestHashLo,
    manifestHashHi: open.agent.vocabulary.header.manifestHashHi,
  };
}

/**
 * One authored lesson: this scene, and what should have been done in it.
 *
 * The percept inside is lowered exactly as a live one would be — a lesson the
 * creature perceives differently from the situation it is meant to generalize
 * to would teach the wrong thing. What the lesson adds is only the target, and
 * the target never enters the frame.
 */
function lesson(agentId: string, body: any) {
  const open = agents.get(agentId);
  if (!open) {
    throw new Error(`agent '${agentId}' was never created; POST the world to /v1/agents/${agentId} first`);
  }

  const shaped = Lesson(body?.lesson);
  if (shaped instanceof type.errors) {
    throw new Error(`lesson does not match krystal-lesson@3: ${shaped.summary}`);
  }
  const doc = shaped as contract.Lesson;

  const { percept } = validatePercept(doc.percept, open.agent.vocabulary);
  if (percept.actorId !== agentId) throw new Error("agent path does not match percept.actorId");

  const intentId = open.agent.vocabulary.relations.findIndex(
    (relation) => relation.relation === doc.expect.relation,
  );
  if (intentId < 0) {
    throw new Error(`lesson expects relation '${doc.expect.relation}', which this world does not declare`);
  }

  const lowered = lowerPercept(percept, open.agent.vocabulary, open.references);

  const patient = doc.expect.roles.find((binding) => binding.role === "patient")?.operand;
  const patientRefToken =
    patient?.kind === "instance" ? open.references.tokenFor(patient.instanceId) : undefined;

  const report = teach({
    frame: lowered.frame,
    weights: open.weights,
    config: open.config,
    catalog: open.catalog,
    intentSchemaId: CATALOG_SCHEMA_ID,
    expect: { intentId, patientRefToken },
    learningRate: LEARNING_RATE,
  });
  if (report.applied) open.updates++;

  console.log(
    `[krystal] lesson${doc.label ? ` '${doc.label}'` : ""} agent=${agentId} ` +
      `expect=${doc.expect.relation}` +
      `${patient?.kind === "instance" ? `(${patient.instanceId})` : ""} ` +
      `${report.applied
        ? `chose=${open.agent.vocabulary.relations[report.chosenIntentId ?? -1]?.relation ?? "?"} ` +
          `hit=${report.intentHit ? "relation" : "-"}${report.patientHit ? "+patient" : ""} ` +
          `p=${fmt(report.intentProbability)}`
        : `skipped: ${report.skipped}`}`,
  );

  return {
    ...report,
    policy: POLICY_NAME,
    checkpointId: checkpointId(open),
  };
}

const cors = {
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "content-type",
  "access-control-allow-methods": "GET,POST,DELETE,OPTIONS",
};

function json(body: unknown, status = 200): Response {
  return Response.json(body, { status, headers: cors });
}

function parseAgentPath(pathname: string): { agentId: string; action: string | undefined } | null {
  const match = pathname.match(/^\/v1\/agents\/([^/]+)(?:\/(action-intents|train|lesson))?$/);
  if (!match) return null;
  return { agentId: decodeURIComponent(match[1]!), action: match[2] };
}

const server = Bun.serve({
  hostname: "127.0.0.1",
  port: PORT,
  async fetch(request) {
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/health") {
      return json({ ready: true, agents: agents.size });
    }
    const path = parseAgentPath(url.pathname);
    // Iterating on a world needs a way to start over: an agent's learned rows
    // follow the vocabulary it was built with, so a changed vocabulary is a new
    // creature rather than an update to this one.
    if (request.method === "DELETE" && path && path.action === undefined) {
      return json({ deleted: agents.delete(path.agentId) });
    }
    if (request.method !== "POST" || !path) return json({ error: "not found" }, 404);
    try {
      const body = (await request.json()) as any;
      if (path.action === undefined) {
        const vocabulary = body?.vocabulary as contract.WorldVocabulary;
        if (!vocabulary || vocabulary.contract !== "krystal-world@3") {
          throw new Error("agent creation requires a krystal-world@3 vocabulary");
        }
        const open = openAgent(path.agentId, vocabulary);
        return json({
          created: true,
          policy: POLICY_NAME,
          checkpointId: checkpointId(open),
          symbols: open.agent.vocabulary.header.activeTokenCount,
          channels: open.agent.vocabulary.channels.size,
          relations: open.catalog.header.intentCount,
          // Bind a checkpoint to this hash: a reordered vocabulary then fails at
          // startup rather than silently redefining every row it trained.
          manifestHashLo: open.agent.vocabulary.header.manifestHashLo,
          manifestHashHi: open.agent.vocabulary.header.manifestHashHi,
        });
      }
      if (path.action === "lesson") return json(lesson(path.agentId, body));
      if (path.action === "train") {
        const open = agents.get(path.agentId);
        if (!open) throw new Error(`agent '${path.agentId}' was never created`);
        return json({
          valueLoss: learn(open),
          buffered: open.buffer.size,
          policy: POLICY_NAME,
          checkpointId: checkpointId(open),
        });
      }
      return json(act(path.agentId, body));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`[krystal] request failed: ${message}`);
      return json({ error: message }, 400);
    }
  },
});

console.log(`[krystal] Krystal service listening on ${server.url}`);
console.log(`[krystal] no pretrained policy; agents start random and learn from valence`);

function shutdown(): void {
  server.stop(true);
}
process.once("SIGINT", shutdown);
process.once("SIGTERM", shutdown);
