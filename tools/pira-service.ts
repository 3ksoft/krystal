import { buildFixtureActionCatalog } from "../packages/krystal/src/fixtures/action-intents.ts";
import { buildCurriculum } from "../packages/krystal/src/bridge/curriculum.ts";
import { mix32 } from "../packages/krystal/src/bridge/comfort.ts";
import { lowerPolicyFrame, type PolicyEpisode } from "../packages/krystal/src/bridge/policy.ts";
import {
  PiraPolicyBridge,
  type PiraRawSnapshotLike,
  type PiraVocabManifestLike,
} from "../packages/krystal/src/bridge/pira.ts";
import { createWebGpuDevice } from "../packages/webgpu/src/device.ts";
import { krystal } from "../packages/webgpu/src/krystal.ts";
import { KrystalBackward } from "../packages/webgpu/src/krystal-backward.ts";
import { KrystalForward } from "../packages/webgpu/src/krystal-forward.ts";
import {
  POLICY_CONFIG,
  createBrainForwardWeights,
  emitPrediction,
  packBrainFrame,
  prepareTrainFrame,
  productionSelection,
} from "../packages/webgpu/src/policy-runtime.ts";

const PORT = Number(Bun.env.KRYSTAL_PIRA_PORT ?? 8801);
const TRAIN_EPISODES = Number(Bun.env.KRYSTAL_BOOT_EPISODES ?? 256);
const TRAIN_EPOCHS = Number(Bun.env.KRYSTAL_BOOT_EPOCHS ?? 3);
const POLICY_NAME = "krystal-s9-village-v1";
const CHECKPOINT_ID = `boot-seed42-e${TRAIN_EPOCHS}-n${TRAIN_EPISODES}`;

function validPositiveInteger(value: number, label: string): void {
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${label} must be a positive integer`);
}

async function installDawn(): Promise<void> {
  if ((globalThis as { navigator?: { gpu?: unknown } }).navigator?.gpu) return;
  const { create, globals } = await import("webgpu");
  Object.assign(globalThis, globals);
  Object.defineProperty(globalThis, "navigator", {
    value: { gpu: create([]) },
    configurable: true,
  });
}

async function createDevice(): Promise<GPUDevice> {
  await installDawn();
  const gpu = (globalThis as { navigator?: { gpu?: GPU } }).navigator!.gpu!;
  const adapter = await gpu.requestAdapter({});
  if (!adapter) throw new Error("Could not acquire a Dawn WebGPU adapter");
  const { device } = await createWebGpuDevice({
    label: "krystal.pira-service",
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 2147483644),
    },
  });
  const compiled = await krystal.engine.compile({ device });
  if (compiled.failed > 0) throw new Error(`Krystal shader compilation failed (${compiled.failed} passes)`);
  return device;
}

function balanceS1Replay(episodes: readonly PolicyEpisode[]): PolicyEpisode[] {
  return episodes.map((episode) => {
    if (episode.stage !== "S1") return episode;
    const bad = (mix32((episode.seed >>> 0) ^ 0x51) & 1) === 0;
    const frame = episode.frames[0]!;
    return {
      ...episode,
      frames: [{ ...frame, comfort: bad ? -1 : 1, gold: { action: bad ? "CRY" : "LAUGH" } }],
    };
  });
}

async function trainBootPolicy(device: GPUDevice): Promise<KrystalForward> {
  validPositiveInteger(TRAIN_EPISODES, "KRYSTAL_BOOT_EPISODES");
  validPositiveInteger(TRAIN_EPOCHS, "KRYSTAL_BOOT_EPOCHS");
  const catalog = buildFixtureActionCatalog();
  const split = buildCurriculum({
    stages: ["S9"],
    replayStages: ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"],
    trainSeeds: [0, TRAIN_EPISODES],
    evalSeeds: [TRAIN_EPISODES + 128, TRAIN_EPISODES + 129],
  });
  const frames = balanceS1Replay(split.train).flatMap((episode) =>
    episode.frames.map((frame) => ({ episode, frame })),
  );
  const runner = new KrystalForward(createBrainForwardWeights(POLICY_CONFIG, 42), POLICY_CONFIG);
  const trainer = new KrystalBackward(runner);
  const started = performance.now();
  console.log(`[krystal] boot training ${frames.length} frames × ${TRAIN_EPOCHS} epochs`);
  for (let epoch = 0; epoch < TRAIN_EPOCHS; epoch++) {
    let finalLoss: number | undefined;
    for (let index = 0; index < frames.length; index++) {
      const { episode, frame } = frames[index]!;
      const result = await trainer.trainStep({
        ...prepareTrainFrame(packBrainFrame(lowerPolicyFrame(frame, episode)).frame, frame.gold, catalog),
        learningRate: 0.01,
        telemetry: index === frames.length - 1,
      });
      finalLoss = result.loss;
    }
    console.log(`[krystal] epoch ${epoch + 1}/${TRAIN_EPOCHS} loss=${finalLoss?.toFixed(5) ?? "—"}`);
  }
  console.log(`[krystal] ready in ${((performance.now() - started) / 1000).toFixed(1)}s`);
  return runner;
}

const cors = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, POST, OPTIONS",
  "access-control-allow-headers": "content-type",
};

function json(body: unknown, status = 200): Response {
  return Response.json(body, { status, headers: cors });
}

function parseAgentPath(pathname: string): { agentId: string; decision: boolean } | null {
  const match = pathname.match(/^\/v1\/agents\/([^/]+)(\/action-intents)?$/);
  if (!match) return null;
  return { agentId: decodeURIComponent(match[1]!), decision: match[2] !== undefined };
}

const device = await createDevice();
const runner = await trainBootPolicy(device);
const catalog = buildFixtureActionCatalog();
const agents = new Map<string, PiraPolicyBridge>();
let decisionQueue: Promise<unknown> = Promise.resolve();

async function decide(agentId: string, body: any) {
  const snapshot = body?.snapshot as PiraRawSnapshotLike;
  const manifest = body?.vocabManifest as PiraVocabManifestLike;
  if (!snapshot || snapshot.contract !== "pira-raw-sensory@1") throw new Error("request lacks pira-raw-sensory@1 snapshot");
  if (snapshot.actorId !== agentId) throw new Error("agent path does not match snapshot.actorId");
  if (!manifest?.symbols || !Array.isArray(manifest.symbols)) throw new Error("request lacks vocabManifest.symbols");
  const bridge = agents.get(agentId) ?? new PiraPolicyBridge();
  agents.set(agentId, bridge);
  const { frame, episode } = bridge.lower(snapshot, manifest);
  const packed = packBrainFrame(lowerPolicyFrame(frame, episode)).frame;
  const selection = await productionSelection(device, runner, packed, catalog);
  const prediction = selection ? emitPrediction(selection, catalog) : null;
  const intent = bridge.intent(prediction, snapshot, manifest);
  console.log(`[krystal] tick=${snapshot.tick} agent=${agentId} prediction=${prediction?.action ?? "none"}` +
    `${prediction?.refToken === undefined ? "" : `#${prediction.refToken.toString(16)}`} intent=${intent?.kind ?? "none"}`);
  return {
    intents: intent ? [intent] : [],
    prediction,
    policy: POLICY_NAME,
    checkpointId: CHECKPOINT_ID,
  };
}

const server = Bun.serve({
  hostname: "127.0.0.1",
  port: PORT,
  async fetch(request) {
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/health") {
      return json({ ready: true, policy: POLICY_NAME, checkpointId: CHECKPOINT_ID, agents: agents.size });
    }
    const path = parseAgentPath(url.pathname);
    if (request.method !== "POST" || !path) return json({ error: "not found" }, 404);
    try {
      const body = await request.json();
      if (!path.decision) {
        agents.set(path.agentId, agents.get(path.agentId) ?? new PiraPolicyBridge());
        return json({ created: true, policy: POLICY_NAME, checkpointId: CHECKPOINT_ID });
      }
      const result = decisionQueue.then(() => decide(path.agentId, body));
      decisionQueue = result.catch(() => undefined);
      return json(await result);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`[krystal] request failed: ${message}`);
      return json({ error: message }, 400);
    }
  },
});

console.log(`[krystal] Pira service listening on ${server.url}`);

function shutdown() {
  server.stop(true);
  runner.destroy();
  device.destroy();
}
process.once("SIGINT", shutdown);
process.once("SIGTERM", shutdown);
