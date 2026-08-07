// Commit helpers used only by constrained argmax. The mask kernel keeps
// constraintState read-only; this include is linked only into the write path.

const CONSTRAINT_ARGMAX_INVALID_TOKEN: u32 = 0xffffffffu;
const CONSTRAINT_ERROR_DEAD_END: u32 = 0x434e5354u; // "CNST"
const CONSTRAINT_ERROR_COMMIT: u32 = 0x434d4954u;   // "CMIT"

fn store_constraint_state(state: ConstraintState) {
  constraintState[0] = state.node;
  constraintState[1] = state.local0;
  constraintState[2] = state.local1;
  constraintState[3] = state.local2;
  constraintState[4] = state.status;
  constraintState[5] = state.errorCode;
  constraintState[6] = state.reserved0;
  constraintState[7] = state.reserved1;
  constraintState[8] = state.number0;
  constraintState[9] = state.number1;
  constraintState[10] = state.number2;
  constraintState[11] = state.number3;
  constraintState[12] = state.number4;
  constraintState[13] = state.number5;
  constraintState[14] = state.number6;
  constraintState[15] = state.number7;
}

// Commit exactly one token selected from the previously generated exact mask.
// EOS is a terminal control token: at accept it leaves the parser state intact.
fn commit_constraint_token(token: u32) -> bool {
  let vocab = tokenizer_vocab_size();
  if (token >= vocab) { return false; }

  let eos = tokenizer_eos_token();
  if (constraintState[0] == program_accept_node()) {
    return token == eos;
  }
  if (token == eos) { return false; }

  let byteOffset = tokenizer_entry(token, 0u);
  let tokenMeta = tokenizer_entry(token, 1u);
  let byteLength = tokenMeta & TOKEN_LENGTH_MASK;
  if ((tokenMeta & TOKEN_SPECIAL) != 0u || byteLength == 0u) { return false; }

  var nextState = load_constraint_state();
  var i = 0u;
  loop {
    if (i >= byteLength) { break; }
    if (!feed_byte(&nextState, tokenizer_byte(byteOffset + i))) { return false; }
    i += 1u;
  }

  store_constraint_state(nextState);
  return true;
}
