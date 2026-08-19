/**
 * Discretization: exact numbers in, band tokens out.
 *
 * The thresholds live here rather than in the simulation that produces the
 * numbers. A band is a token, a token owns a trained embedding row, so a
 * threshold moved upstream would keep every symbol identical while silently
 * changing what the trained vector denotes — loss would keep falling against a
 * shifted meaning, the same failure class as renumbering a manifest. Keeping
 * the cuts beside the tokens they define also keeps them consistent across
 * senses, without which DIST_NEAR would mean one thing to sight and another to
 * hearing and stop being learnable.
 */
import {
  QUANTITY_BANDS,
  KRYSTAL_TOKEN_RANGES,
} from "../../../schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";

type QuantityKind = v1_0_0.QuantityKind;

/**
 * Band symbols the ENGINE owns, because each is a pure consequence of a
 * threshold defined above. They sit in the reserved `structure` range for that
 * reason: a simulation may not redefine them, and they are the same symbols in
 * every world.
 *
 * Polarity is deliberately NOT here. Which direction of a signed field counts
 * as which concept is domain knowledge — negative comfort is FEEL_BAD, negative
 * radial motion is RECEDING — so a simulation names those per field in its
 * grammar, while the engine only decides how far from zero a value has to be to
 * stop being "neither".
 */
const STRUCTURE_BASE = KRYSTAL_TOKEN_RANGES.structure[0];

export const BAND_SYMBOLS = {
  // Signed zero: a category of its own, not a small magnitude.
  neither: "NEITHER",
  // Polarity of the one signed channel the engine derives for itself: the
  // change in valence. Named for the change rather than the state, because
  // that is what it reports — the level is a separate, unipolar reading.
  worse: "WORSE",
  better: "BETTER",
  // Magnitude, shared by signed and unipolar.
  mild: "MAG_MILD",
  moderate: "MAG_MODERATE",
  severe: "MAG_SEVERE",
  // Count, subitizing.
  few: "CNT_FEW",
  some: "CNT_SOME",
  many: "CNT_MANY",
  // Proportion. Logic, not perception: no threshold is calibratable here.
  none_of: "Q_NONE",
  some_of: "Q_SOME",
  most_of: "Q_MOST",
  all_of: "Q_ALL",
} as const;

/** Reserved ids, offset within the structure range after the case markers. */
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

/** Polarity symbols for a signed field, supplied by the grammar. */
export interface Polarity {
  readonly negative: string;
  readonly positive: string;
}

/**
 * One discretized value. `magnitude` is absent for kinds that carry their whole
 * meaning in one token; `exact` is preserved because the banded token is what
 * the model reads while the runtime still needs the real number (a motion
 * relation's `intensity`, for instance).
 */
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

/**
 * Quantize one value.
 *
 * A `signed` value yields TWO tokens — polarity then magnitude — because zero
 * is a category rather than a boundary: "not moving" is a different percept
 * from "barely approaching" and "barely receding", and collapsing them would
 * make the sign unrecoverable near zero. Such a field therefore consumes two of
 * a record's eight token slots.
 */
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
        // Inside the deadzone the sign carries no information, so reporting one
        // would invent a direction the observation does not support.
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
      // Exact at the endpoints on purpose. `all` licenses inference about any
      // member and `most` licenses none, so letting 0.99 read as ALL would not
      // be a rounding error but the loss of an operator.
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

/** Token slots a field of this kind consumes in a record. */
export function tokenWidthOf(kind: QuantityKind): number {
  return kind === "signed" ? 2 : 1;
}
