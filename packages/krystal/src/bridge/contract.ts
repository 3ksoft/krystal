import { BRAIN_FRAME_BANDS } from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";
import type { SimQuantityField } from "./agent.ts";
import { quantize } from "./quantize.ts";

export const SENSORY_BANDS = BRAIN_FRAME_BANDS.map((band) => band.kind).filter(
  (kind) =>
    kind !== "system" &&
    kind !== "memory" &&
    kind !== "focus" &&
    kind !== "query" &&
    kind !== "catalog",
);

export type SensoryBand = (typeof SENSORY_BANDS)[number];

export function isSensoryBand(value: string): value is SensoryBand {
  return (SENSORY_BANDS as readonly string[]).includes(value);
}

export interface RawQuantityV2 {
  /** Must name a quantity field declared in the grammar. */
  readonly field: string;
  readonly value: number;
}

export interface RawRecordV2 {
  readonly band: SensoryBand;
  readonly modality: string;
  readonly schema: string;
  readonly instanceId?: string;
  readonly objectInstanceId?: string;
  readonly tokens: readonly string[];
  readonly quantities?: readonly RawQuantityV2[];
  readonly count?: number;
  readonly salience?: number;
  readonly observedAt: number;
  readonly emptiness?: "void" | "unavailable";
}

export interface RawMotionV2 {
  readonly instanceId: string;
  readonly radial: number;
  readonly angular?: number;
}
export interface RawEventV2 {
  readonly relation: string;
  readonly subject: ConceptOperandV2;
  readonly object?: ConceptOperandV2;
  readonly intensity?: number;
  readonly salience?: number;
  readonly observedAt: number;
}

export interface RawSelfMotionV2 {
  readonly speed: number;
  readonly turning?: number;
}

export interface RawSnapshotV2 {
  readonly contract: "pira-raw-sensory@2";
  readonly tick: number;
  readonly deltaMillis: number;
  readonly valence: number;
  readonly actorId: string;
  readonly records: readonly RawRecordV2[];
  readonly motion?: readonly RawMotionV2[];
  readonly events?: readonly RawEventV2[];
  readonly selfMotion?: RawSelfMotionV2 | undefined;
}

export type ConceptOperandV2 =
  | { readonly kind: "instance"; readonly instanceId: string }
  | { readonly kind: "symbol"; readonly symbol: string }
  | { readonly kind: "unknown" }
  | { readonly kind: "something" };

export interface AgentIntentV2 {
  readonly relation: string;
  readonly subject: ConceptOperandV2;
  readonly object: ConceptOperandV2;
  readonly intensity: number;
  readonly commitment: number;
  readonly intentRef?: number;
  readonly source: "learned";
}

export class SensoryContractError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SensoryContractError";
  }
}

export interface LoweringDiagnostics {
  unknownSymbols: Map<string, number>;
  unknownBands: Map<string, number>;
  droppedRecords: number;
}

export function emptyDiagnostics(): LoweringDiagnostics {
  return { unknownSymbols: new Map(), unknownBands: new Map(), droppedRecords: 0 };
}

export function validateSnapshot(
  snapshot: RawSnapshotV2,
  tokenBySymbol: ReadonlyMap<string, number>,
  options: {
    onUnknown?: "throw" | "count";
    /** Declared quantity fields; when given, values are checked against them. */
    quantities?: ReadonlyMap<string, SimQuantityField>;
  } = {},
): LoweringDiagnostics {
  const quantities = options.quantities;
  if (snapshot.contract !== "pira-raw-sensory@2") {
    throw new SensoryContractError(`Unsupported sensory contract '${snapshot.contract}'`);
  }
  if (!Number.isFinite(snapshot.deltaMillis) || snapshot.deltaMillis < 0) {
    throw new SensoryContractError(`snapshot.deltaMillis must be a non-negative number`);
  }
  if (!Number.isFinite(snapshot.valence) || snapshot.valence < 0 || snapshot.valence > 1) {
    throw new SensoryContractError(
      `snapshot.valence must be within 0..1, where 0 is dead; got ${snapshot.valence}`,
    );
  }

  const strict = (options.onUnknown ?? "throw") === "throw";
  const diagnostics = emptyDiagnostics();

  const instances = new Set<string>();
  for (const record of snapshot.records) {
    if (record.instanceId) instances.add(record.instanceId);
    for (const quantity of record.quantities ?? []) {
      if (!Number.isFinite(quantity.value)) {
        throw new SensoryContractError(
          `${record.schema}.${quantity.field}: quantity must be a finite number, not a pre-banded label — the engine owns the thresholds`,
        );
      }
      const declared = quantities?.get(quantity.field);
      if (quantities && !declared) {
        throw new SensoryContractError(
          `${record.schema}.${quantity.field}: no such quantity field in the grammar; declare its kind before sending values`,
        );
      }
      if (declared) {
        try {
          quantize(quantity.value, declared.kind, declared.polarity);
        } catch (error) {
          throw new SensoryContractError(`${record.schema}.${quantity.field}: ${(error as Error).message}`);
        }
      }
    }
    if (!isSensoryBand(record.band)) {
      if (strict) throw new SensoryContractError(`record band '${record.band}' is not writable by the simulation`);
      diagnostics.unknownBands.set(record.band, (diagnostics.unknownBands.get(record.band) ?? 0) + 1);
      diagnostics.droppedRecords++;
      continue;
    }
    if (!tokenBySymbol.has(record.schema)) {
      if (strict) {
        throw new SensoryContractError(
          `record schema '${record.schema}' is not in the compiled grammar; declare it or the brain cannot perceive it`,
        );
      }
      diagnostics.unknownSymbols.set(record.schema, (diagnostics.unknownSymbols.get(record.schema) ?? 0) + 1);
      diagnostics.droppedRecords++;
    }
  }

  for (const event of snapshot.events ?? []) {
    if (!tokenBySymbol.has(event.relation)) {
      throw new SensoryContractError(
        `event relation '${event.relation}' is not in the compiled grammar`,
      );
    }
    for (const participant of [event.subject, event.object]) {
      if (participant === undefined) continue;
      if (participant.kind === "instance") {
        if (!instances.has(participant.instanceId) && participant.instanceId !== snapshot.actorId) {
          throw new SensoryContractError(
            `event '${event.relation}' names instance '${participant.instanceId}', which is not perceived in this snapshot; ` +
              `report an unidentified participant as { kind: "unknown" } rather than dropping the event`,
          );
        }
      } else if (participant.kind === "symbol" && !tokenBySymbol.has(participant.symbol)) {
        throw new SensoryContractError(
          `event '${event.relation}' names symbol '${participant.symbol}', which is not in the compiled grammar`,
        );
      }
    }
  }

        for (const motion of snapshot.motion ?? []) {
    if (!instances.has(motion.instanceId)) {
      throw new SensoryContractError(
        `motion references instance '${motion.instanceId}', which is not perceived in this snapshot`,
      );
    }
    if (!Number.isFinite(motion.radial) || Math.abs(motion.radial) > 1) {
      throw new SensoryContractError(`motion '${motion.instanceId}': radial rate must be within -1..1`);
    }
  }
  return diagnostics;
}

export function toAgentIntents(
  intentSet: v1_0_0.IntentSet,
  relationOf: (intentId: number) => string | undefined,
  instanceOf: (refToken: number) => string | undefined,
): AgentIntentV2[] {
  const intents: AgentIntentV2[] = [];
  for (const proposal of intentSet.proposals) {
    if (proposal.lifecycle === "empty") continue;
    const relation = relationOf(proposal.intentId);
    if (relation === undefined) continue;

    const operand = (side: v1_0_0.SelectedConceptRef): ConceptOperandV2 | undefined => {
      const token = side.concept.handle.tokenId;
      const instanceId = token === 0 ? undefined : instanceOf(token);
      return instanceId === undefined ? undefined : { kind: "instance", instanceId };
    };

    const subject = operand(proposal.subject);
    const object = operand(proposal.object);
    if (!subject || !object) continue;

    intents.push({
      relation,
      subject,
      object,
      intensity: proposal.intensity,
      commitment: proposal.commitment,
      intentRef:
        proposal.intentRef.tokenId === 0 ? undefined : proposal.intentRef.tokenId,
      source: "learned",
    });
  }
  return intents;
}
