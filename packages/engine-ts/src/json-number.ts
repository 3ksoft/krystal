export interface JsonNumberConstraint {
  readonly integer: boolean;
  readonly minText?: string;
  readonly maxText?: string;
  /** CPU oracle compatibility. GPU ABI v1 rejects step/multipleOf. */
  readonly step?: number;
}

type NormalizedDecimal = {
  sign: -1 | 0 | 1;
  digits: string;
  magnitude: number;
};

function normalizeJsonDecimal(text: string): NormalizedDecimal {
  let cursor = 0;
  let sign: -1 | 1 = 1;
  if (text[cursor] === "-") {
    sign = -1;
    cursor++;
  }

  const lowerE = text.indexOf("e", cursor);
  const upperE = text.indexOf("E", cursor);
  const exponentAt = lowerE < 0 ? upperE : upperE < 0 ? lowerE : Math.min(lowerE, upperE);
  const mantissaEnd = exponentAt >= 0 ? exponentAt : text.length;
  const dotAt = text.indexOf(".", cursor);
  const hasDot = dotAt >= 0 && dotAt < mantissaEnd;
  const fractionDigits = hasDot ? mantissaEnd - dotAt - 1 : 0;

  let digits = "";
  for (let i = cursor; i < mantissaEnd; i++) if (text[i] !== ".") digits += text[i];
  digits = digits.replace(/^0+/, "");
  if (digits.length === 0) return { sign: 0, digits: "0", magnitude: 0 };

  let exponent = 0;
  if (exponentAt >= 0) {
    let i = exponentAt + 1;
    let exponentSign = 1;
    if (text[i] === "+" || text[i] === "-") {
      if (text[i] === "-") exponentSign = -1;
      i++;
    }
    // Candidate exponents can use most of the 32-byte lexeme. Saturating far
    // outside binary64's finite domain avoids integer overflow and mirrors the
    // bounded i32 arithmetic used by WGSL.
    let absolute = 0;
    for (; i < text.length; i++) {
      absolute = Math.min(1_000_000, absolute * 10 + (text.charCodeAt(i) - 0x30));
    }
    exponent = exponentSign * absolute;
  }

  return {
    sign,
    digits,
    magnitude: digits.length + exponent - fractionDigits,
  };
}

/** Exact mathematical comparison of two syntactically complete JSON decimals. */
export function compareJsonDecimal(leftText: string, rightText: string): number {
  const left = normalizeJsonDecimal(leftText);
  const right = normalizeJsonDecimal(rightText);
  if (left.sign !== right.sign) return left.sign < right.sign ? -1 : 1;
  if (left.sign === 0) return 0;

  let absolute = 0;
  if (left.magnitude !== right.magnitude) absolute = left.magnitude < right.magnitude ? -1 : 1;
  else {
    const width = Math.max(left.digits.length, right.digits.length);
    for (let i = 0; i < width; i++) {
      const a = i < left.digits.length ? left.digits.charCodeAt(i) : 0x30;
      const b = i < right.digits.length ? right.digits.charCodeAt(i) : 0x30;
      if (a !== b) {
        absolute = a < b ? -1 : 1;
        break;
      }
    }
  }
  return left.sign < 0 ? -absolute : absolute;
}

/** Shared completion/range oracle for the CPU NFA and upload-blob VM. */
export function isJsonNumberComplete(
  text: string,
  constraint: JsonNumberConstraint,
): boolean {
  const pattern = constraint.integer
    ? /^-?(?:0|[1-9]\d*)$/
    : /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/;
  if (!pattern.test(text)) return false;

  // Preserve the existing CPU oracle's finite-binary64 envelope for now. This
  // is independent from explicit schema min/max and is easy to mirror on GPU
  // as a fixed decimal bound if/when root scalar overflow becomes relevant.
  const value = Number(text);
  if (!Number.isFinite(value)) return false;

  if (constraint.minText !== undefined && compareJsonDecimal(text, constraint.minText) < 0) return false;
  if (constraint.maxText !== undefined && compareJsonDecimal(text, constraint.maxText) > 0) return false;

  if (constraint.step !== undefined) {
    const base = constraint.minText === undefined ? 0 : Number(constraint.minText);
    const scaled = (value - base) / constraint.step;
    if (Math.abs(scaled - Math.round(scaled)) > 1e-9 * Math.max(1, Math.abs(scaled))) return false;
  }
  return true;
}
