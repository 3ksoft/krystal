// Standalone constraint-mask kernel.
//
// One invocation owns one output u32 and checks 32 vocabulary entries. This
// avoids atomics entirely: for a 65,536-token vocabulary the dispatch is only
// 2,048 invocations (32 workgroups at workgroup_size(64)).
//
// Bindings are deliberately raw u32 blobs so the shader ABI stays independent
// from schema-pop/Sandblaster resource declarations:
//   0: GpuConstraintProgram.blob
//   1: GpuConstraintTokenizer.blob
//   2: 64-byte ConstraintDecoderState
//   3: exact bit mask, ceil(vocab / 32) u32 words

struct ConstraintState {
  node: u32,
  local0: u32,
  local1: u32,
  local2: u32,
  reserved0: u32,
  reserved1: u32,
  reserved2: u32,
  reserved3: u32,
  number0: u32,
  number1: u32,
  number2: u32,
  number3: u32,
  number4: u32,
  number5: u32,
  number6: u32,
  number7: u32,
};

struct DecimalInfo {
  sign: i32,
  magnitude: i32,
  significantDigits: u32,
  significantStart: u32,
  mantissaEnd: u32,
};

@group(0) @binding(0) var<storage, read> constraintProgram: array<u32>;
@group(0) @binding(1) var<storage, read> constraintTokenizer: array<u32>;
@group(0) @binding(2) var<storage, read> constraintState: ConstraintState;
@group(0) @binding(3) var<storage, read_write> constraintMask: array<u32>;

const INVALID_NODE: u32 = 0xffffffffu;
const NODE_WORDS: u32 = 12u;

const NODE_LITERAL: u32 = 0u;
const NODE_SWITCH: u32 = 1u;
const NODE_STRING: u32 = 2u;
const NODE_NUMBER: u32 = 3u;
const NODE_ACCEPT: u32 = 4u;

const NUMBER_INTEGER: u32 = 1u;
const NUMBER_HAS_MIN: u32 = 2u;
const NUMBER_HAS_MAX: u32 = 4u;
const NUMBER_HAS_STEP: u32 = 8u;

const TOKEN_LENGTH_MASK: u32 = 0xffu;
const TOKEN_SPECIAL: u32 = 0x100u;

const NUMBER_START: u32 = 0u;
const NUMBER_SIGN: u32 = 1u;
const NUMBER_ZERO: u32 = 2u;
const NUMBER_INTEGER_PHASE: u32 = 3u;
const NUMBER_DOT: u32 = 4u;
const NUMBER_FRACTION: u32 = 5u;
const NUMBER_EXPONENT: u32 = 6u;
const NUMBER_EXPONENT_SIGN: u32 = 7u;
const NUMBER_EXPONENT_DIGITS: u32 = 8u;

// Exact binary64 overflow midpoint: 2^1024 - 2^970. The first 32
// significant decimal digits are enough for every <=32-byte candidate lexeme.
const F64_OVERFLOW_MAGNITUDE: i32 = 309i;
const F64_OVERFLOW_DIGITS: array<u32, 8> = array<u32, 8>(
  0x37393731u, 0x31333936u, 0x36383433u, 0x35313332u,
  0x39373038u, 0x38323733u, 0x34313739u, 0x30333530u,
);

fn program_node_offset() -> u32 { return constraintProgram[4]; }
fn program_node_count() -> u32 { return constraintProgram[5]; }
fn program_edge_offset() -> u32 { return constraintProgram[6]; }
fn program_byte_offset() -> u32 { return constraintProgram[8]; }
fn program_byte_length() -> u32 { return constraintProgram[9]; }
fn program_accept_node() -> u32 { return constraintProgram[3]; }

fn node_word(node: u32, word: u32) -> u32 {
  if (node >= program_node_count()) { return INVALID_NODE; }
  return constraintProgram[program_node_offset() + node * NODE_WORDS + word];
}

fn program_byte(byteOffset: u32) -> u32 {
  if (byteOffset >= program_byte_length()) { return 0xffffffffu; }
  let word = constraintProgram[program_byte_offset() + (byteOffset >> 2u)];
  return (word >> ((byteOffset & 3u) * 8u)) & 0xffu;
}

fn tokenizer_vocab_size() -> u32 { return constraintTokenizer[0]; }
fn tokenizer_eos_token() -> u32 { return constraintTokenizer[1]; }
fn tokenizer_entry_offset() -> u32 { return constraintTokenizer[2]; }
fn tokenizer_byte_offset() -> u32 { return constraintTokenizer[3]; }

fn tokenizer_entry(token: u32, word: u32) -> u32 {
  return constraintTokenizer[tokenizer_entry_offset() + token * 2u + word];
}

fn tokenizer_byte(byteOffset: u32) -> u32 {
  let word = constraintTokenizer[tokenizer_byte_offset() + (byteOffset >> 2u)];
  return (word >> ((byteOffset & 3u) * 8u)) & 0xffu;
}

fn state_number_word(state: ptr<function, ConstraintState>, word: u32) -> u32 {
  switch word {
    case 0u: { return (*state).number0; }
    case 1u: { return (*state).number1; }
    case 2u: { return (*state).number2; }
    case 3u: { return (*state).number3; }
    case 4u: { return (*state).number4; }
    case 5u: { return (*state).number5; }
    case 6u: { return (*state).number6; }
    default: { return (*state).number7; }
  }
}

fn set_state_number_word(state: ptr<function, ConstraintState>, word: u32, value: u32) {
  switch word {
    case 0u: { (*state).number0 = value; }
    case 1u: { (*state).number1 = value; }
    case 2u: { (*state).number2 = value; }
    case 3u: { (*state).number3 = value; }
    case 4u: { (*state).number4 = value; }
    case 5u: { (*state).number5 = value; }
    case 6u: { (*state).number6 = value; }
    default: { (*state).number7 = value; }
  }
}

fn number_byte(state: ptr<function, ConstraintState>, offset: u32) -> u32 {
  let word = state_number_word(state, offset >> 2u);
  return (word >> ((offset & 3u) * 8u)) & 0xffu;
}

fn set_number_byte(state: ptr<function, ConstraintState>, offset: u32, byte: u32) {
  let wordIndex = offset >> 2u;
  let shift = (offset & 3u) * 8u;
  let mask = 0xffu << shift;
  let previous = state_number_word(state, wordIndex);
  set_state_number_word(state, wordIndex, (previous & ~mask) | ((byte & 0xffu) << shift));
}

fn goto_node(state: ptr<function, ConstraintState>, node: u32) {
  (*state).node = node;
  (*state).local0 = 0u;
  (*state).local1 = 0u;
  (*state).local2 = 0u;
}

fn is_hex(byte: u32) -> bool {
  return (byte >= 0x30u && byte <= 0x39u)
    || (byte >= 0x41u && byte <= 0x46u)
    || (byte >= 0x61u && byte <= 0x66u);
}

fn find_switch_target(node: u32, byte: u32) -> u32 {
  let edgeOffset = node_word(node, 2u);
  let edgeCount = node_word(node, 3u);
  var lo = 0u;
  var hi = edgeCount;
  loop {
    if (lo >= hi) { break; }
    let mid = (lo + hi) >> 1u;
    let packed = constraintProgram[program_edge_offset() + edgeOffset + mid];
    let edgeByte = packed & 0xffu;
    if (edgeByte < byte) { lo = mid + 1u; }
    else { hi = mid; }
  }
  if (lo >= edgeCount) { return INVALID_NODE; }
  let packed = constraintProgram[program_edge_offset() + edgeOffset + lo];
  if ((packed & 0xffu) != byte) { return INVALID_NODE; }
  return (packed >> 8u) & 0xffffu;
}

fn number_next_phase(phase: u32, byte: u32, integerOnly: bool) -> u32 {
  let digit = byte >= 0x30u && byte <= 0x39u;
  let nonZeroDigit = byte >= 0x31u && byte <= 0x39u;

  if (phase == NUMBER_START) {
    if (byte == 0x2du) { return NUMBER_SIGN; }
    if (byte == 0x30u) { return NUMBER_ZERO; }
    if (nonZeroDigit) { return NUMBER_INTEGER_PHASE; }
    return INVALID_NODE;
  }
  if (phase == NUMBER_SIGN) {
    if (byte == 0x30u) { return NUMBER_ZERO; }
    if (nonZeroDigit) { return NUMBER_INTEGER_PHASE; }
    return INVALID_NODE;
  }
  if (phase == NUMBER_ZERO) {
    if (!integerOnly && byte == 0x2eu) { return NUMBER_DOT; }
    if (!integerOnly && (byte == 0x65u || byte == 0x45u)) { return NUMBER_EXPONENT; }
    return INVALID_NODE;
  }
  if (phase == NUMBER_INTEGER_PHASE) {
    if (digit) { return NUMBER_INTEGER_PHASE; }
    if (!integerOnly && byte == 0x2eu) { return NUMBER_DOT; }
    if (!integerOnly && (byte == 0x65u || byte == 0x45u)) { return NUMBER_EXPONENT; }
    return INVALID_NODE;
  }
  if (phase == NUMBER_DOT) {
    if (digit) { return NUMBER_FRACTION; }
    return INVALID_NODE;
  }
  if (phase == NUMBER_FRACTION) {
    if (digit) { return NUMBER_FRACTION; }
    if (byte == 0x65u || byte == 0x45u) { return NUMBER_EXPONENT; }
    return INVALID_NODE;
  }
  if (phase == NUMBER_EXPONENT) {
    if (byte == 0x2bu || byte == 0x2du) { return NUMBER_EXPONENT_SIGN; }
    if (digit) { return NUMBER_EXPONENT_DIGITS; }
    return INVALID_NODE;
  }
  if (phase == NUMBER_EXPONENT_SIGN) {
    if (digit) { return NUMBER_EXPONENT_DIGITS; }
    return INVALID_NODE;
  }
  if (phase == NUMBER_EXPONENT_DIGITS) {
    if (digit) { return NUMBER_EXPONENT_DIGITS; }
    return INVALID_NODE;
  }
  return INVALID_NODE;
}

fn number_phase_complete(phase: u32) -> bool {
  return phase == NUMBER_ZERO
    || phase == NUMBER_INTEGER_PHASE
    || phase == NUMBER_FRACTION
    || phase == NUMBER_EXPONENT_DIGITS;
}

// source=0 reads the candidate number lexeme from ConstraintState.
// source=1 reads an ASCII bound from the program byte pool.
fn decimal_byte(
  source: u32,
  offset: u32,
  index: u32,
  state: ptr<function, ConstraintState>,
) -> u32 {
  if (source == 0u) { return number_byte(state, index); }
  return program_byte(offset + index);
}

fn decimal_info(
  source: u32,
  offset: u32,
  length: u32,
  state: ptr<function, ConstraintState>,
) -> DecimalInfo {
  var cursor = 0u;
  var sign = 1i;
  if (length > 0u && decimal_byte(source, offset, 0u, state) == 0x2du) {
    sign = -1i;
    cursor = 1u;
  }

  var exponentAt = length;
  var dotAt = length;
  var i = cursor;
  loop {
    if (i >= length) { break; }
    let byte = decimal_byte(source, offset, i, state);
    if (byte == 0x65u || byte == 0x45u) {
      exponentAt = i;
      break;
    }
    if (byte == 0x2eu) { dotAt = i; }
    i += 1u;
  }

  let mantissaEnd = exponentAt;
  var fractionDigits = 0u;
  if (dotAt < mantissaEnd) { fractionDigits = mantissaEnd - dotAt - 1u; }

  var totalDigits = 0u;
  var leadingZeros = 0u;
  var seenNonZero = false;
  var significantStart = mantissaEnd;
  i = cursor;
  loop {
    if (i >= mantissaEnd) { break; }
    let byte = decimal_byte(source, offset, i, state);
    if (byte != 0x2eu) {
      totalDigits += 1u;
      if (!seenNonZero) {
        if (byte == 0x30u) { leadingZeros += 1u; }
        else {
          seenNonZero = true;
          significantStart = i;
        }
      }
    }
    i += 1u;
  }

  let significantDigits = totalDigits - leadingZeros;
  if (significantDigits == 0u) {
    return DecimalInfo(0i, 0i, 0u, mantissaEnd, mantissaEnd);
  }

  var exponent = 0i;
  if (exponentAt < length) {
    i = exponentAt + 1u;
    var exponentSign = 1i;
    if (i < length) {
      let marker = decimal_byte(source, offset, i, state);
      if (marker == 0x2bu || marker == 0x2du) {
        if (marker == 0x2du) { exponentSign = -1i; }
        i += 1u;
      }
    }
    var absolute = 0u;
    loop {
      if (i >= length) { break; }
      let digit = decimal_byte(source, offset, i, state) - 0x30u;
      absolute = min(1000000u, absolute * 10u + digit);
      i += 1u;
    }
    exponent = exponentSign * i32(absolute);
  }

  let magnitude = i32(significantDigits) + exponent - i32(fractionDigits);
  return DecimalInfo(sign, magnitude, significantDigits, significantStart, mantissaEnd);
}

fn decimal_significant_digit(
  source: u32,
  offset: u32,
  info: DecimalInfo,
  ordinal: u32,
  state: ptr<function, ConstraintState>,
) -> u32 {
  if (ordinal >= info.significantDigits) { return 0x30u; }
  var found = 0u;
  var i = info.significantStart;
  loop {
    if (i >= info.mantissaEnd) { break; }
    let byte = decimal_byte(source, offset, i, state);
    if (byte != 0x2eu) {
      if (found == ordinal) { return byte; }
      found += 1u;
    }
    i += 1u;
  }
  return 0x30u;
}

fn f64_overflow_digit(ordinal: u32) -> u32 {
  if (ordinal >= 32u) { return 0x30u; }
  let word = F64_OVERFLOW_DIGITS[ordinal >> 2u];
  return (word >> ((ordinal & 3u) * 8u)) & 0xffu;
}

fn decimal_within_f64(state: ptr<function, ConstraintState>) -> bool {
  let info = decimal_info(0u, 0u, (*state).local0, state);
  if (info.sign == 0i) { return true; }
  if (info.magnitude < F64_OVERFLOW_MAGNITUDE) { return true; }
  if (info.magnitude > F64_OVERFLOW_MAGNITUDE) { return false; }

  let width = max(info.significantDigits, 32u);
  var i = 0u;
  loop {
    if (i >= width) { break; }
    let candidate = decimal_significant_digit(0u, 0u, info, i, state);
    let limit = f64_overflow_digit(i);
    if (candidate < limit) { return true; }
    if (candidate > limit) { return false; }
    i += 1u;
  }
  return true;
}

fn compare_decimal(
  leftSource: u32,
  leftOffset: u32,
  leftLength: u32,
  rightSource: u32,
  rightOffset: u32,
  rightLength: u32,
  state: ptr<function, ConstraintState>,
) -> i32 {
  let left = decimal_info(leftSource, leftOffset, leftLength, state);
  let right = decimal_info(rightSource, rightOffset, rightLength, state);

  if (left.sign != right.sign) {
    if (left.sign < right.sign) { return -1i; }
    return 1i;
  }
  if (left.sign == 0i) { return 0i; }

  var absolute = 0i;
  if (left.magnitude != right.magnitude) {
    if (left.magnitude < right.magnitude) { absolute = -1i; }
    else { absolute = 1i; }
  } else {
    let width = max(left.significantDigits, right.significantDigits);
    var i = 0u;
    loop {
      if (i >= width) { break; }
      let a = decimal_significant_digit(leftSource, leftOffset, left, i, state);
      let b = decimal_significant_digit(rightSource, rightOffset, right, i, state);
      if (a != b) {
        if (a < b) { absolute = -1i; }
        else { absolute = 1i; }
        break;
      }
      i += 1u;
    }
  }

  if (left.sign < 0i) { return -absolute; }
  return absolute;
}

fn number_complete(state: ptr<function, ConstraintState>, node: u32) -> bool {
  if (!number_phase_complete((*state).local1)) { return false; }

  let flags = node_word(node, 4u);
  if ((flags & NUMBER_HAS_STEP) != 0u) { return false; }
  if (!decimal_within_f64(state)) { return false; }

  let length = (*state).local0;
  if ((flags & NUMBER_HAS_MIN) != 0u) {
    let minOffset = node_word(node, 6u);
    let minLength = node_word(node, 7u);
    if (compare_decimal(0u, 0u, length, 1u, minOffset, minLength, state) < 0i) { return false; }
  }
  if ((flags & NUMBER_HAS_MAX) != 0u) {
    let maxOffset = node_word(node, 8u);
    let maxLength = node_word(node, 9u);
    if (compare_decimal(0u, 0u, length, 1u, maxOffset, maxLength, state) > 0i) { return false; }
  }
  return true;
}

fn feed_byte(state: ptr<function, ConstraintState>, byte: u32) -> bool {
  // Number completion can consume zero bytes and retry the same delimiter on
  // its continuation. 64 is intentionally the same hard cap as the CPU VM.
  var retry = 0u;
  loop {
    if (retry >= 64u) { return false; }
    retry += 1u;

    let node = (*state).node;
    let kind = node_word(node, 0u);
    let next = node_word(node, 1u);

    if (kind == NODE_ACCEPT) { return false; }

    if (kind == NODE_LITERAL) {
      let cursor = (*state).local0;
      let byteOffset = node_word(node, 2u);
      let byteCount = node_word(node, 3u);
      if (cursor >= byteCount) { return false; }
      if (program_byte(byteOffset + cursor) != byte) { return false; }
      let advanced = cursor + 1u;
      if (advanced == byteCount) { goto_node(state, next); }
      else { (*state).local0 = advanced; }
      return true;
    }

    if (kind == NODE_SWITCH) {
      let target = find_switch_target(node, byte);
      if (target == INVALID_NODE) { return false; }
      goto_node(state, target);
      return true;
    }

    if (kind == NODE_STRING) {
      let phase = (*state).local0;
      let decodedLength = (*state).local1;
      let minLength = node_word(node, 4u);
      let maxLength = node_word(node, 5u);

      if (phase == 0u) {
        if (byte != 0x22u) { return false; }
        (*state).local0 = 1u;
        return true;
      }
      if (phase == 2u) {
        if (byte == 0x75u) {
          (*state).local0 = 3u;
          (*state).local2 = 4u;
          return true;
        }
        let simpleEscape = byte == 0x22u || byte == 0x5cu || byte == 0x2fu
          || byte == 0x62u || byte == 0x66u || byte == 0x6eu
          || byte == 0x72u || byte == 0x74u;
        if (!simpleEscape || decodedLength >= maxLength) { return false; }
        (*state).local1 = decodedLength + 1u;
        (*state).local0 = 1u;
        return true;
      }
      if (phase == 3u) {
        if (!is_hex(byte) || (*state).local2 == 0u) { return false; }
        let remaining = (*state).local2 - 1u;
        (*state).local2 = remaining;
        if (remaining == 0u) {
          if (decodedLength >= maxLength) { return false; }
          (*state).local1 = decodedLength + 1u;
          (*state).local0 = 1u;
        }
        return true;
      }
      if (phase != 1u) { return false; }

      if (byte == 0x22u) {
        if (decodedLength < minLength) { return false; }
        goto_node(state, next);
        return true;
      }
      if (byte == 0x5cu) {
        if (decodedLength >= maxLength) { return false; }
        (*state).local0 = 2u;
        return true;
      }
      if (byte < 0x20u || decodedLength >= maxLength) { return false; }
      (*state).local1 = decodedLength + 1u;
      return true;
    }

    if (kind == NODE_NUMBER) {
      let length = (*state).local0;
      let phase = (*state).local1;
      let flags = node_word(node, 4u);
      let maxChars = node_word(node, 5u);
      let integerOnly = (flags & NUMBER_INTEGER) != 0u;
      var nextPhase = INVALID_NODE;
      if (length < maxChars) { nextPhase = number_next_phase(phase, byte, integerOnly); }

      if (nextPhase != INVALID_NODE) {
        set_number_byte(state, length, byte);
        (*state).local0 = length + 1u;
        (*state).local1 = nextPhase;
        if (length + 1u == maxChars && !number_complete(state, node)) { return false; }
        return true;
      }

      if (!number_complete(state, node)) { return false; }
      goto_node(state, next);
      continue;
    }

    return false;
  }
}

fn token_survives(token: u32) -> bool {
  let vocab = tokenizer_vocab_size();
  if (token >= vocab) { return false; }

  let eos = tokenizer_eos_token();
  if (constraintState.node == program_accept_node()) { return token == eos; }
  if (token == eos) { return false; }

  let byteOffset = tokenizer_entry(token, 0u);
  let meta = tokenizer_entry(token, 1u);
  let byteLength = meta & TOKEN_LENGTH_MASK;
  if ((meta & TOKEN_SPECIAL) != 0u || byteLength == 0u) { return false; }

  var candidate = constraintState;
  var i = 0u;
  loop {
    if (i >= byteLength) { break; }
    if (!feed_byte(&candidate, tokenizer_byte(byteOffset + i))) { return false; }
    i += 1u;
  }
  return true;
}

@compute @workgroup_size(64)
fn constraint_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
  let maskWord = gid.x;
  let vocab = tokenizer_vocab_size();
  let maskWords = (vocab + 31u) >> 5u;
  if (maskWord >= maskWords) { return; }

  let baseToken = maskWord << 5u;
  var bits = 0u;
  var lane = 0u;
  loop {
    if (lane >= 32u) { break; }
    let token = baseToken + lane;
    if (token < vocab && token_survives(token)) { bits |= 1u << lane; }
    lane += 1u;
  }
  constraintMask[maskWord] = bits;
}
