import { BRAIN_FRAME_BANDS } from "../../../schema/src/krystal-engine-schema.ts";
import type { PolicyEpisode, PolicyRawFrame, RawResource, ResourceKind } from "./policy.ts";

export interface PiraRawRecordLike {
  readonly band: "homeostasis" | "body" | "vision" | "audio" | "olfaction" | "taste" | "touch";
  readonly schema: string;
  readonly instanceId?: string;
  readonly values: Readonly<Record<string, unknown>>;
  readonly salience?: number;
  readonly observedAt: number;
}

export interface PiraRawSnapshotLike {
  readonly contract: "pira-raw-sensory@1";
  readonly tick: number;
  readonly actorId: string;
  readonly records: readonly PiraRawRecordLike[];
}

export interface PiraVocabSymbolLike {
  readonly symbol: string;
  readonly displayName: string;
  readonly source: string;
  readonly sourceId: string;
}

export interface PiraVocabManifestLike {
  readonly header: { readonly manifestHashLo?: number; readonly manifestHashHi?: number };
  readonly symbols: readonly PiraVocabSymbolLike[];
}

export interface PiraAgentIntentLike {
  readonly kind: string;
  readonly subject: string;
  readonly instrument?: string;
  readonly target?: string;
  readonly source: "learned";
}

export interface PiraPolicyPredictionLike {
  readonly action: string;
  readonly refToken?: number;
}

const RESOURCE_KIND: Readonly<Record<string, ResourceKind>> = {
  apple: "apple",
  berry: "berry",
  bread: "bread",
  mother: "mother",
  stone: "stone",
  feces: "feces",
};

const VISION_CAPACITY = BRAIN_FRAME_BANDS.find((band) => band.kind === "vision")!.recordCapacity;
const MEMORY_CAPACITY = BRAIN_FRAME_BANDS.find((band) => band.kind === "memory")!.recordCapacity;

function finite(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function schemaNames(manifest: PiraVocabManifestLike): Map<string, string> {
  const names = new Map<string, string>();
  for (const symbol of manifest.symbols) {
    if (symbol.source === "resource" && symbol.symbol.startsWith("resource:")) {
      names.set(symbol.sourceId, symbol.displayName.trim().toLowerCase());
    }
  }
  return names;
}

function actionName(manifest: PiraVocabManifestLike, wanted: readonly string[]): string | undefined {
  const accepted = new Set(wanted.map((value) => value.toLowerCase()));
  return manifest.symbols.find(
    (symbol) => symbol.source === "action" && accepted.has(symbol.displayName.trim().toLowerCase()),
  )?.displayName;
}

function properties(record: PiraRawRecordLike, distance: number): string[] {
  const result: string[] = [distance <= 4 ? "NEAR" : "FAR"];
  for (const key of ["color", "size"] as const) {
    const value = record.values[key];
    const values = Array.isArray(value) ? value : [value];
    for (const item of values) {
      if (typeof item === "string") result.push(item.trim().toUpperCase());
    }
  }
  if (record.values.poisoned === true || record.values.toxic === true) result.push("POISONED");
  return [...new Set(result)];
}

interface RememberedResource {
  resource: RawResource;
  lastSeen: number;
}

/** Stateful host adapter for the current S9 fixture vocabulary. Pira remains
 * exact and unbounded; this class owns the temporary fixture projection,
 * record budget and short visual memory until the general manifest compiler
 * replaces it.
 */
export class PiraPolicyBridge {
  private readonly instanceTokens = new Map<string, number>();
  private readonly tokenInstances = new Map<number, string>();
  private readonly memory = new Map<string, RememberedResource>();
  private nextToken = 0xe00;

  private tokenFor(instanceId: string): number {
    const existing = this.instanceTokens.get(instanceId);
    if (existing !== undefined) return existing;
    if (this.nextToken > 0xeff) throw new Error("Pira runtime reference space exhausted");
    const token = this.nextToken++;
    this.instanceTokens.set(instanceId, token);
    this.tokenInstances.set(token, instanceId);
    return token;
  }

  lower(snapshot: PiraRawSnapshotLike, manifest: PiraVocabManifestLike): { frame: PolicyRawFrame; episode: PolicyEpisode } {
    if (snapshot.contract !== "pira-raw-sensory@1") throw new Error(`Unsupported sensory contract '${snapshot.contract}'`);
    const names = schemaNames(manifest);
    const actor = snapshot.records.find((record) => record.band === "body" && record.instanceId === snapshot.actorId);
    const actorX = finite(actor?.values.x) ?? 0;
    const actorY = finite(actor?.values.y) ?? 0;
    const levels = snapshot.records
      .filter((record) => record.band === "homeostasis")
      .map((record) => finite(record.values.level))
      .filter((value): value is number => value !== undefined);
    const comfort = levels.some((level) => level <= 0) ? -1 : 1;

    const visible = snapshot.records
      .filter((record) => record.band === "vision" && record.instanceId)
      .map((record) => {
        const kind = RESOURCE_KIND[names.get(record.schema) ?? ""];
        if (!kind) return null;
        const x = finite(record.values.x) ?? actorX;
        const y = finite(record.values.y) ?? actorY;
        const distance = Math.hypot(x - actorX, y - actorY);
        const resource: RawResource = {
          kind,
          refToken: this.tokenFor(record.instanceId!),
          generation: 1,
          band: "vision",
          properties: properties(record, distance),
        };
        return { id: record.instanceId!, distance, salience: record.salience ?? 0, resource };
      })
      .filter((entry): entry is NonNullable<typeof entry> => entry !== null)
      .sort((a, b) => b.salience - a.salience || a.distance - b.distance || a.id.localeCompare(b.id))
      .slice(0, VISION_CAPACITY);

    const visibleIds = new Set(visible.map((entry) => entry.id));
    for (const entry of visible) this.memory.set(entry.id, { resource: entry.resource, lastSeen: snapshot.tick });
    for (const [id, remembered] of this.memory) {
      if (snapshot.tick - remembered.lastSeen > 6) this.memory.delete(id);
    }
    const remembered = [...this.memory.entries()]
      .filter(([id]) => !visibleIds.has(id))
      .sort((a, b) => b[1].lastSeen - a[1].lastSeen || a[0].localeCompare(b[0]))
      .slice(0, MEMORY_CAPACITY)
      .map(([, entry]): RawResource => ({ ...entry.resource, band: "memory" }));

    const frame: PolicyRawFrame = {
      tick: snapshot.tick,
      comfort,
      resources: [...visible.map((entry) => entry.resource), ...remembered],
      gold: { action: "WAIT" },
    };
    return { frame, episode: { stage: "S9", seed: snapshot.tick, frames: [frame] } };
  }

  intent(
    prediction: PiraPolicyPredictionLike | null,
    snapshot: PiraRawSnapshotLike,
    manifest: PiraVocabManifestLike,
  ): PiraAgentIntentLike | null {
    if (!prediction) return null;
    const subject = snapshot.actorId;
    const target = prediction.refToken === undefined ? undefined : this.tokenInstances.get(prediction.refToken);
    const learned = { subject, source: "learned" as const };
    switch (prediction.action.toUpperCase()) {
      case "CRY": {
        const kind = actionName(manifest, ["cry"]);
        return kind ? { kind, ...learned } : null;
      }
      case "LAUGH": {
        const kind = actionName(manifest, ["laugh"]);
        return kind ? { kind, ...learned } : null;
      }
      case "MOVE_TOWARDS": {
        const kind = actionName(manifest, ["move", "move towards", "move_towards"]);
        return kind && target ? { kind, target, ...learned } : null;
      }
      case "EAT": {
        if (!target) return null;
        const hand = snapshot.records.find(
          (record) => record.band === "body" && record.instanceId && record.values.item_held === target,
        );
        if (hand) {
          const kind = actionName(manifest, ["eat"]);
          return kind ? { kind, instrument: hand.instanceId, target, ...learned } : null;
        }
        // Compatibility with Village's two-step mechanics. The S9 fixture
        // learns EAT(reachable food); Village first requires Pickup(food).
        const kind = actionName(manifest, ["pickup", "pick up", "pick_up"]);
        return kind ? { kind, target, ...learned } : null;
      }
      case "LOOK": {
        const kind = actionName(manifest, ["look", "rotate"]);
        return kind ? { kind, ...(target ? { target } : {}), ...learned } : null;
      }
      default:
        return null;
    }
  }
}
