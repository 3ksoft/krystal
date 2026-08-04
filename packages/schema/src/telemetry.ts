import { scope, wgsl } from "@schema-pop/schema";

export const TELEMETRY = {
  MAX_ENTRIES: 256,
};

export const telemetry = scope({
  ...wgsl.import(),

  DecodeTelemetryEntry: {
    step: "u32",
    tokenId: "u32",
    position: "u32",
    status: "u32",
  },

  DecodeTelemetry: {
    enabled: "u32",
    cursor: "au32",
    entries: `DecodeTelemetryEntry[] == ${TELEMETRY.MAX_ENTRIES}`,
  },
});