import {
  QUANTITY_BANDS,
  KRYSTAL_TOKEN_RANGES,
} from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";

type QuantityKind = v1_0_0.QuantityKind;

const STRUCTURE_BASE = KRYSTAL_TOKEN_RANGES.structure[0];

export const BAND_SYMBOLS = {
  neither: "NEITHER",
  worse: "WORSE",
  better: "BETTER",
  mild: "MAG_MILD",
  moderate: "MAG_MODERATE",
  severe: "MAG_SEVERE",
  few: "CNT_FEW",
  some: "CNT_SOME",
  many: "CNT_MANY",
  none_of: "Q_NONE",
  some_of: "Q_SOME",
  most_of: "Q_MOST",
  all_of: "Q_ALL",
} as const;


export const BAND_TOKEN_IDS: Readonly<Record<string, number>> = Object.freeze(
  Object.fromEntries(
    Object.values(BAND_SYMBOLS).map((symbol, index) => [symbol, STRUCTURE_BASE + 16 + index]),
  ),
);

export class QuantizeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "QuantizeError";
  }
}


export interface Polarity {
  readonly negative: string;
  readonly positive: string;
}

export interface BandedQuantity {
  readonly tokens: readonly string[];
  readonly exact: number;
}

function magnitudeBand(magnitude: number): string {
  const [mild, moderate] = QUANTITY_BANDS.signedMagnitude;
  if (magnitude <= mild!) return BAND_SYMBOLS.mild;
  if (magnitude <= moderate!) return BAND_SYMBOLS.moderate;
  return BAND_SYMBOLS.severe;
}

export function quantize(
  value: number,
  kind: QuantityKind,
  polarity?: Polarity,
): BandedQuantity {
  if (!Number.isFinite(value)) throw new QuantizeError(`value must be finite, got ${value}`);

  switch (kind) {
    case "signed": {
      if (Math.abs(value) > 1) throw new QuantizeError(`signed value ${value} outside -1..1`);
      if (!polarity) throw new QuantizeError("a signed field needs polarity symbols from the grammar");
      if (Math.abs(value) <= QUANTITY_BANDS.signedDeadzone) {
        return { tokens: [BAND_SYMBOLS.neither], exact: value };
      }
      const sign = value < 0 ? polarity.negative : polarity.positive;
      return { tokens: [sign, magnitudeBand(Math.abs(value))], exact: value };
    }

    case "unipolar": {
      if (value < 0 || value > 1) throw new QuantizeError(`unipolar value ${value} outside 0..1`);
      return { tokens: [magnitudeBand(value)], exact: value };
    }

    case "count": {
      if (!Number.isInteger(value) || value < 0) {
        throw new QuantizeError(`count must be a non-negative integer, got ${value}`);
      }
      const [few, some] = QUANTITY_BANDS.count;
      const token =
        value <= few! ? BAND_SYMBOLS.few : value <= some! ? BAND_SYMBOLS.some : BAND_SYMBOLS.many;
      return { tokens: [token], exact: value };
    }

    case "proportion": {
      if (value < 0 || value > 1) throw new QuantizeError(`proportion ${value} outside 0..1`);
      if (value === 0) return { tokens: [BAND_SYMBOLS.none_of], exact: value };
      if (value === 1) return { tokens: [BAND_SYMBOLS.all_of], exact: value };
      if (value > 0.5) return { tokens: [BAND_SYMBOLS.most_of], exact: value };
      return { tokens: [BAND_SYMBOLS.some_of], exact: value };
    }

    default: {
      const exhaustive: never = kind;
      throw new QuantizeError(`unknown quantity kind ${exhaustive}`);
    }
  }
}


export function tokenWidthOf(kind: QuantityKind): number {
  return kind === "signed" ? 2 : 1;
}
