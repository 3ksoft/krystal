// Commit helpers used only by constrained argmax. The mask kernel keeps
// constraintState read-only; this include is linked only into the write path.

const CONSTRAINT_ARGMAX_INVALID_TOKEN: u32 = 0xffffffffu;
const CONSTRAINT_ERROR_DEAD_END: u32 = 0x434e5354u; // "CNST"
const CONSTRAINT_ERROR_COMMIT: u32 = 0x434d4954u;   // "CMIT"

// Commit exactly one token selected from the previously generated exact mask.
// EOS is a terminal control token: at accept it leaves the parser state intact.
fn commit_constraint_token(token: u32) -> bool {
  let vocab = tokenizer_vocab_size();
  if (token >= vocab) { return false; }

  let eos = tokenizer_eos_token();
  var currentState = constraintState;
  if (state_is_terminal(&currentState)) {
    return token == eos;
  }
  if (token == eos) { return false; }

  let byteOffset = tokenizer_entry(token, 0u);
  let tokenMeta = tokenizer_entry(token, 1u);
  let byteLength = tokenMeta & TOKEN_LENGTH_MASK;
  if ((tokenMeta & TOKEN_SPECIAL) != 0u || byteLength == 0u) { return false; }

  var nextState = constraintState;
  var i = 0u;
  loop {
    if (i >= byteLength) { break; }
    if (!feed_byte(&nextState, tokenizer_byte(byteOffset + i))) { return false; }
    i += 1u;
  }

  constraintState = nextState;
  return true;
}
