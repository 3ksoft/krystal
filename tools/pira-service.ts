/**
 * Krystal's HTTP face for pira.
 *
 * Three calls, one per thing the simulation needs:
 *
 *   POST /v1/agents/:id                  declare the world (pira-grammar@2)
 *   POST /v1/agents/:id/action-intents   a tick (pira-raw-sensory@2) -> intents
 *   POST /v1/agents/:id/train            learn from what has settled
 *
 * There is no boot training and no pretrained policy. The previous one was
 * fitted to a fixture vocabulary, which means it was fitted to a different set
 * of embedding rows: for an agent compiled from a simulation's own grammar
 * those weights denote nothing. A creature therefore starts from random weights
 * and learns here, in the world it actually lives in.
 *
 * Random does not mean nonsensical. The intent mask admits only catalog records
 * and the argument mask only type-compatible candidates, so an untrained
 * creature proposes structurally valid actions from the first tick — which is
 * what makes the experience it generates worth learning from.
 *
 * Runs on the CPU. The forward and the value-head backward are both CPU paths,
 * so the service needs no GPU device at all.
 */
import {
  CATALOG_SCHEMA_ID,
  ExperienceBuffer,
  ReferenceTable,
  compileActionCatalog,
  createAgent,
  lowerSnapshot,
  toAgentIntents,
  validateSnapshot,
  type Agent,
  type CompiledCatalog,
  type ConceptOperandV2,
  type RawSnapshotV2,
  type SimGrammar,
} from "../packages/krystal/src/bridge/index.ts";
import { decide, type DecideResult } from "../packages/krystal/src/forward/decide.ts";
import { trainValueHead } from "../packages/krystal/src/forward/value-training.ts";
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
}

const POLICY_NAME = "krystal-online";

/**
 * Which brain produced a decision.
 *
 * There is no pretrained policy to name any more, so identity is the two things
 * that actually distinguish one brain from another: the grammar it was compiled
 * against, and how much it has learned since. A creature that has lived longer
 * is a different creature, and the id says so.
 */
function checkpointId(open: ServiceAgent): string {
  const hash = (open.agent.grammar.header.manifestHashLo >>> 0).toString(16);
  return `g${hash}-t${open.ticks}-u${open.updates}`;
}

const agents = new Map<string, ServiceAgent>();

function openAgent(agentId: string, grammar: SimGrammar): ServiceAgent {
  const agent = createAgent({ grammar });
  const existing = agents.get(agentId);
  if (existing) {
    // Same id, different grammar: refuse rather than hand back the agent that
    // already exists. Silently returning the old one makes iterating on a
    // grammar deeply confusing — a corrected catalog appears to have no effect
    // — and the alternative, swapping the grammar under a live agent, would
    // redefine every embedding row it had learned.
    if (
      existing.agent.grammar.header.manifestHashLo !== agent.grammar.header.manifestHashLo ||
      existing.agent.grammar.header.manifestHashHi !== agent.grammar.header.manifestHashHi
    ) {
      throw new Error(
        `agent '${agentId}' already exists with a different grammar (${checkpointId(existing)}); ` +
          "use a new agent id, or DELETE this one first — its learned rows follow the grammar it was built with",
      );
    }
    return existing;
  }
  // The row table comes from this grammar, never from a global: another
  // vocabulary assigns different rows, and the same token would otherwise train
  // a different vector with nothing to signal it.
  const config: BrainForwardConfig = { ...BRAIN_FORWARD_CONFIG, tokenRows: agent.grammar.tokenRows };
  const created: ServiceAgent = {
    agent,
    catalog: compileActionCatalog(agent.grammar),
    config,
    weights: createBrainForwardWeights(config, 42),
    references: new ReferenceTable(),
    buffer: new ExperienceBuffer(),
    previousValence: undefined,
    ticks: 0,
    updates: 0,
    lastValueLoss: 0,
  };
  agents.set(agentId, created);
  return created;
}

/** One value-head pass over experience whose outcome is known. */
function learn(open: ServiceAgent): number {
  const entries = open.buffer.drain();
  if (entries.length === 0) return open.lastValueLoss;
  const report = trainValueHead({
    entries,
    weights: open.weights,
    config: open.config,
    intentSchemaId: CATALOG_SCHEMA_ID,
    argMask: (frame, active) =>
      compilePerRowArgumentMask(frame, active, open.catalog, new Array(active.queryRecords.length).fill(0), "object"),
    learningRate: LEARNING_RATE,
  });
  open.lastValueLoss = report.meanValueLoss;
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

/** A readable name for one side of a relation. */
function operandLabel(operand: ConceptOperandV2): string {
  switch (operand.kind) {
    case "instance": return operand.instanceId;
    case "symbol": return operand.symbol;
    default: return operand.kind.toUpperCase();
  }
}

interface AttemptReport {
  readonly row: number;
  /** Whether this row's choice survived into an emitted proposal. */
  readonly emitted: boolean;
  readonly relation: string | undefined;
  readonly intentProbability: number;
  readonly objectCandidates: number;
  readonly bankRecords: number;
  readonly accepts: readonly string[];
  readonly candidateBands: readonly string[];
  /** One line, already phrased for a log. */
  readonly detail: string;
}

/**
 * What each query row tried, and why it did or did not become a proposal.
 *
 * The point is the failing rows. "No admissible object" names a category of
 * failure, not an instance of one: it does not say which action was chosen, and
 * without that the next question — is the role too narrow, or was the world
 * simply empty of suitable things — has nowhere to start. A creature reaching
 * for something it cannot reach is doing something, and the log should say what.
 */
function describeAttempts(open: ServiceAgent, result: DecideResult): AttemptReport[] {
  return result.chosen.map((choice, row) => {
    const intentId = result.chosenIntents[row];
    const action = intentId === undefined ? undefined : open.agent.grammar.actions[intentId];
    const relation = action?.relation;
    const emitted = result.intentSet.proposals.some(
      (proposal) => proposal.proposalSlot === choice.row && proposal.lifecycle !== "empty",
    );
    const accepts = action?.object?.accepts ?? [];
    const candidateBands = action?.object?.candidateBands ?? [];
    const bankRecords = result.active.bankRecords.length;
    const constraint =
      `accepts ${accepts.length === 0 ? "anything" : accepts.join("+")}` +
      `${candidateBands.length === 0 ? "" : ` in ${candidateBands.join("+")}`}`;
    const detail =
      `row ${row}: tried ${relation ?? "?"} p=${fmt(choice.intentProbability)} · ` +
      `object ${choice.objectCandidates}/${bankRecords} admissible (${constraint})` +
      `${choice.objectCandidates === 0 ? " — nothing in this frame fits the role" : ""}`;
    return {
      row,
      emitted,
      relation,
      intentProbability: choice.intentProbability,
      objectCandidates: choice.objectCandidates,
      bankRecords,
      accepts,
      candidateBands,
      detail,
    };
  });
}

function act(agentId: string, body: any) {
  const snapshot = body?.snapshot as RawSnapshotV2;
  if (!snapshot || snapshot.contract !== "pira-raw-sensory@2") {
    throw new Error("request lacks a pira-raw-sensory@2 snapshot");
  }
  if (snapshot.actorId !== agentId) throw new Error("agent path does not match snapshot.actorId");
  const open = agents.get(agentId);
  if (!open) throw new Error(`agent '${agentId}' was never created; POST the grammar to /v1/agents/${agentId} first`);

  // Strict: an unknown symbol is refused, never dropped. A boundary that
  // quietly forgets is what makes "why can it not see the apple" cost a day.
  validateSnapshot(snapshot, open.agent.grammar.tokenBySymbol, {
    quantities: open.agent.grammar.quantities,
  });

  const lowered = lowerSnapshot(
    snapshot,
    open.agent.grammar,
    open.references,
    undefined,
    undefined,
    open.previousValence,
  );
  open.previousValence = snapshot.valence;
  open.buffer.record(lowered.frame, snapshot.tick, lowered.valenceDelta);
  open.ticks++;

  const result = decide({
    frame: lowered.frame,
    weights: open.weights,
    config: open.config,
    catalog: open.catalog,
    intentSchemaId: CATALOG_SCHEMA_ID,
    tick: snapshot.tick,
    // The creature acts rather than reporting its mode. An argmax policy emits
    // one action per frame and so has nothing to compare against — no reward
    // rule can teach it anything, which is why an untrained creature repeated
    // the same LOOK for as long as anyone watched. The seed is derived from the
    // agent and the tick, so a replayed run draws exactly the same choices and
    // two creatures in one world do not flail in lockstep.
    explore: { seed: mix32(hashString(agentId) ^ mix32(snapshot.tick)) },
  });

  const intents = toAgentIntents(
    result.intentSet,
    (intentId) => open.agent.grammar.actions[intentId]?.relation,
    (refToken) => open.references.instanceFor(refToken),
  );

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
        ? "no catalog records in the frame — the grammar declared no actions, or they did not reach the catalog band"
        : d.queryRows === 0
          ? "no query row in the frame"
          : d.droppedNoSubject > 0
            ? "the actor was not found: send a record in the body band whose instanceId equals the snapshot's actorId"
            : d.droppedNoObject > 0
              ? "no admissible object for the chosen action — check that role's accepts and candidateBands against what the frame holds"
              : "the policy proposed nothing";

  const overflow = lowered.overflow.map((entry) => `${entry.band}:${entry.offered}>${entry.admitted}`);
  console.log(
    `[krystal] tick=${snapshot.tick} agent=${agentId} valence=${snapshot.valence.toFixed(2)}` +
      `${lowered.valenceDelta === undefined ? "" : ` d=${lowered.valenceDelta.toFixed(2)}`}` +
      ` intents=${
        intents
          .map((i) => `${i.relation}(${operandLabel(i.object)}) c=${fmt(i.commitment)}`)
          .join(",") || "none"
      }` +
      `${noIntentReason === undefined ? "" : ` (${noIntentReason})`}` +
      ` loss=${open.lastValueLoss.toFixed(5)}` +
      `${overflow.length === 0 ? "" : ` overflow=${overflow.join(",")}`}` +
      `${lowered.truncatedRecords === 0 ? "" : ` truncated=${lowered.truncatedRecords}`}`,
  );
  // A dropped proposal used to vanish without saying what it had been. The
  // creature was trying something, and the one thing worth knowing — WHAT it
  // was trying, and how many candidates its object role had — was exactly what
  // never reached the log. Print it only when something was lost, so a healthy
  // tick stays one line.
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
    // What the frame could not hold, said out loud: a band that drops its tail
    // silently is indistinguishable from a scene that was simply emptier.
    overflow: lowered.overflow,
    truncatedRecords: lowered.truncatedRecords,
    manifestHashLo: open.agent.grammar.header.manifestHashLo,
    manifestHashHi: open.agent.grammar.header.manifestHashHi,
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
  const match = pathname.match(/^\/v1\/agents\/([^/]+)(?:\/(action-intents|train))?$/);
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
    // Iterating on a grammar needs a way to start over: an agent's learned rows
    // follow the grammar it was built with, so a changed grammar is a new
    // creature rather than an update to this one.
    if (request.method === "DELETE" && path && path.action === undefined) {
      return json({ deleted: agents.delete(path.agentId) });
    }
    if (request.method !== "POST" || !path) return json({ error: "not found" }, 404);
    try {
      const body = (await request.json()) as any;
      if (path.action === undefined) {
        const grammar = body?.grammar as SimGrammar;
        if (!grammar || grammar.contract !== "pira-grammar@2") {
          throw new Error("agent creation requires a pira-grammar@2 grammar");
        }
        const open = openAgent(path.agentId, grammar);
        return json({
          created: true,
          policy: POLICY_NAME,
          checkpointId: checkpointId(open),
          symbols: open.agent.grammar.header.activeTokenCount,
          actions: open.catalog.header.intentCount,
          // Bind a checkpoint to this hash: a reordered grammar then fails at
          // startup rather than silently redefining every row it trained.
          manifestHashLo: open.agent.grammar.header.manifestHashLo,
          manifestHashHi: open.agent.grammar.header.manifestHashHi,
        });
      }
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

console.log(`[krystal] Pira service listening on ${server.url}`);
console.log(`[krystal] no pretrained policy; agents start random and learn from valence`);

function shutdown(): void {
  server.stop(true);
}
process.once("SIGINT", shutdown);
process.once("SIGTERM", shutdown);
