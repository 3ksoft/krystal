# Structured Generation

Status: Implemented v0  
Date: 2026-08-07

## 1. Public contract

Structured generation is the primary typed generation operation:

```ts
const t = type({
  id: "number",
  name: "string < 64",
});

const result = await engine.generate(t, {
  checkpoint,
  blocks,
});
```

The result type is inferred from the schema.

The schema is a generation constraint, not an automatic post-generation assertion. Callers may explicitly validate with ArkType if desired:

```ts
const validated = t.assert(result);
```

## 2. Root values

The root may be any supported JSON value type.

Examples:

```text
type("string < 64")  -> string
type("number")       -> number
type("boolean")      -> boolean
array schema          -> array
object schema         -> object
```

Chomato does not wrap scalar results in an object envelope.

## 3. Canonical JSON transport value

The model emits one strict JSON value matching the schema.

Examples:

```text
schema                    emitted token bytes      API result
string                    "hello"                  "hello"
number                    42.5                     42.5
boolean                   true                     true
array                     [1,2]                    [1, 2]
object                    {"id":1}                { id: 1 }
```

Once the root value reaches the accept state, normal tokens are no longer allowed and EOS is the only valid continuation. The v0 path does not intentionally generate trailing whitespace after a complete root value.

## 4. Host compilation

The high-level type is converted to a JSON-Schema-like semantic representation and compiled directly to the constraint IR.

Dynamic JSON structures are **not** routed through the binary memory-layout analyzer, because binary layout cannot preserve semantics such as `maxItems` for variable-length arrays.

Conceptually:

```text
Type<T>
  ↓
JSON Schema semantics
  ↓
LayoutConstraintProgram
  ↓
GPU deterministic linker
  ↓
packed constraint program blob
```

The compiler supports bounded dynamic arrays directly rather than expanding every possible array length into a giant variant graph.

## 5. GPU constraint program

The GPU program is a compact deterministic byte VM. Current instruction/state concepts include:

- literals,
- byte switches/tries,
- strings and JSON escapes (the short forms `\" \\ \/ \b \f \n \r \t`; `\uXXXX`
  is deliberately not accepted — see *Output bounds* below),
- numbers,
- epsilon/jump transitions used by bounded repetition,
- accept/terminal state.

Optional fields and enum branches are compiled to deterministic prefix tries. Bounded arrays use explicit repetition/jump structure so `maxItems = 50` does not explode into every possible full array variant.

The VM operates on tokenizer bytes, not token text labels or a canonical tokenization of literals.

## 6. Tokenizer metadata

The model-global tokenizer table maps every token ID to its raw byte sequence and flags special tokens.

For LFM2.5:

```text
vocab entries = 65,536
EOS token     = 7
```

The current real-model table is under 1 MiB and is uploaded once as immutable GPU metadata. The constraint program itself is per schema/generation and typically much smaller.

This byte-level mapping is essential because the same JSON fragment may be represented by different token segmentations. Constraint correctness is defined over the emitted bytes, not over one chosen tokenization.

## 7. Exact token mask

For each decode state the GPU computes the exact set of token IDs that can legally extend the current JSON value.

Mask geometry:

```text
65,536 tokens
÷ 32 tokens per u32
= 2,048 u32 words
= 8,192 bytes
```

Current `constraint_mask` dispatch:

```text
2,048 invocations
× 32 tokens tested per invocation
→ one u32 mask word per invocation
```

Each candidate evaluates against a local copy of decoder state. Mask generation never commits candidate state transitions.

## 8. Sampling and state commit

Structured token selection happens after the model forward has produced logits:

```text
model forward
→ logits
→ constraint_mask
→ constraint_argmax
```

`constraint_argmax`:

1. ignores tokens whose mask bit is clear,
2. preserves global sampler sentinels/reserved-token rules,
3. selects the highest-logit allowed token for the current greedy path,
4. feeds the selected token bytes through the VM,
5. commits the resulting decoder state,
6. updates the normal generation token/runtime state.

This keeps constraint application **before** token selection and does not use top-K-before-validation.

## 9. Dense vs sparse execution

The current implementation is a masked-dense path:

```text
full model logits
→ full exact vocabulary mask
→ masked argmax
```

There is no sparse LM-head row execution yet. Sparse trie edges make individual token validation cheap, but all 65,536 token IDs are still considered by the mask kernel.

Current measurements do not make sparse execution urgent. On an RTX 3060 the mask kernel is 0.07–0.30 ms and the masked argmax 0.14–0.16 ms against ~8.5 ms/token, and `tests/structured-benchmark.test.ts` measures the end-to-end difference between constrained and unconstrained decoding at −1.5% to +2.8% — that is, inside run-to-run noise.

There is a hard ceiling on what sparse LM-head execution could ever be worth, and it is smaller than it looks: `tests/decode-profile.test.ts` puts the LM head at ~8% of a decode step and the block stack at ~90%.

## 10. Output bounds

The typed API does not require a separate user-facing `maxTokens` for ordinary structured generation. A finite generation budget is derived conservatively from the schema.

`compileStructuredGeneration()` walks the constraint graph for the longest JSON
byte string it accepts, and charges **one token per byte** plus EOS. That is not
pessimism: under a legal-token mask the model can always pick a single-byte
token, so one byte is genuinely one token in the worst case.

The per-node costs are therefore what decides whether a schema fits:

| node | worst case | where |
|---|---|---|
| string | `2 + maxLength * 2` | 2 bytes/unit is the short escape; the body charges one unit per byte, so raw UTF-8 costs no more than it measures |
| number | `maxNumberChars` | a flat bound, `JSON_SCHEMA_CONSTRAINT_LIMITS` in `json-schema-constraint.ts` |
| array | per-item cost × `maxItems` | plus separators |

`\uXXXX` is excluded from the accepted language precisely because of the first
row: allowing it would raise the string ceiling from 2 bytes per length unit to
6, and it buys nothing — the body phase accepts every byte `>= 0x20`, so `é` and
`é` were two spellings of a string that stays reachable either way. What is
lost is escaping control characters with no short form.

Schemas used for generation should be bounded where their textual representation is otherwise unbounded, for example:

```ts
type("string < 512")
```

instead of an unconstrained string when a finite response budget is required.

The derived response budget must also fit the runtime context capacity, and the
binding constraint is the context, not `MAX_NEW_TOKENS`: the runtime rejects a
request when `promptTokens + budget - 1 > context`, so the budget actually
available is `context - promptTokens + 1`.

## 11. Current schema coverage

Implemented v0 coverage includes:

- strings with finite length bounds,
- numbers/integers with the implemented range semantics,
- booleans,
- enums/literals,
- objects,
- required/optional fields,
- nested objects,
- bounded arrays with `minItems`/`maxItems`,
- fixed arrays where applicable.

Current non-goals / deferred keywords include:

- unbounded arrays,
- tuple / `prefixItems`,
- general recursive schemas,
- full arbitrary unions,
- `multipleOf` / step semantics,
- full JSON Schema compatibility.

## 12. Correctness testing

The structured path is tested at several levels:

```text
constraint linker
→ CPU reference VM
→ packed CPU mask oracle
→ Dawn AOT mask equivalence
→ constrained argmax + decoder transition
→ public Engine.generate(...) E2E
```

The CPU oracle evaluates the same packed program/tokenizer blobs uploaded to the GPU, avoiding a second unrelated implementation of the input format.

The public E2E suite uses the real model and public API. It does not create test-only compute pipelines after runtime compilation.

## 13. Failure semantics

A constraint state with no valid token continuation is a hard structured-generation failure. The runtime must not silently relax the schema or fall back to unconstrained token selection.

A valid completed root terminates through EOS. A decoder dead-end and an ordinary model EOS are distinct runtime conditions.
