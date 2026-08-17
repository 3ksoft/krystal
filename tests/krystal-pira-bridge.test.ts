import { expect, test } from "bun:test";
import { PiraPolicyBridge, type PiraRawSnapshotLike } from "../packages/krystal/src/bridge/pira.ts";

const manifest = {
  header: {},
  symbols: [
    { symbol: "resource:Apple", displayName: "Apple", source: "resource", sourceId: "apple" },
    { symbol: "action:EAT", displayName: "Eat", source: "action", sourceId: "eat" },
    { symbol: "action:PICKUP", displayName: "Pickup", source: "action", sourceId: "pickup" },
    { symbol: "action:MOVE", displayName: "Move", source: "action", sourceId: "move" },
  ],
};

function snapshot(held = false): PiraRawSnapshotLike {
  return {
    contract: "pira-raw-sensory@1",
    tick: 4,
    actorId: "child",
    records: [
      { band: "homeostasis", schema: "comfort", instanceId: "need", values: { level: 0 }, observedAt: 4 },
      { band: "body", schema: "child", instanceId: "child", values: { x: 0, y: 0 }, observedAt: 4 },
      { band: "body", schema: "hand", instanceId: "hand", values: { item_held: held ? "apple-1" : "" }, observedAt: 4 },
      { band: "vision", schema: "apple", instanceId: "apple-1", values: { x: 2, y: 0, color: ["Red"] }, observedAt: 4 },
    ],
  };
}

test("Pira bridge lowers exact refs and adapts reachable EAT to Village pickup/eat", () => {
  const bridge = new PiraPolicyBridge();
  const lowered = bridge.lower(snapshot(), manifest);
  expect(lowered.frame.comfort).toBe(-1);
  expect(lowered.frame.resources[0]).toMatchObject({ kind: "apple", band: "vision", properties: ["NEAR", "RED"] });
  const refToken = lowered.frame.resources[0]!.refToken;
  expect(bridge.intent({ action: "EAT", refToken }, snapshot(), manifest)).toMatchObject({ kind: "Pickup", target: "apple-1" });
  expect(bridge.intent({ action: "EAT", refToken }, snapshot(true), manifest)).toMatchObject({ kind: "Eat", instrument: "hand", target: "apple-1" });
  expect(bridge.intent({ action: "MOVE_TOWARDS", refToken }, snapshot(), manifest)).toMatchObject({ kind: "Move", target: "apple-1" });
});
