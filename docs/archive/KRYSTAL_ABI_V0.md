# Krystal ABI v0

Status: **PARTIAL FREEZE — v0 machine core frozen; vocabulary remains provisional**
Purpose: minimal semantic IR contract for the first grounded Krystal curriculum.

---

## 1. Scope

Krystal ABI v0 defines the stable execution and representation rules required by the first curriculum-driven brain experiments.

It deliberately does **not** define a complete 4096-token vocabulary. The v0 goal is to freeze:

- the 12-bit token address space,
- token-class ranges,
- structural rules,
- AST linearization,
- operator arity,
- dynamic symbol handling,
- separation between runtime identity and learned semantics,
- versioning rules.

Actual lexical concepts are introduced only when required by curriculum competencies.

### 1.1 Frozen v0 machine contract

The following rules are normative for ABI v0 and require an ABI version change
to break:

1. `0xExx` denotes exact runtime slot identity within one binding epoch.
2. ABI-aware brains may use matching slot identity as a causal attention bias.
3. `RETRIEVE` is a learned pointer over parser-declared compatible payload
   positions.
4. Exact composition such as `EQ_TOKEN`, `COPY` and structural AST transforms
   belongs to the Oracle/runtime.
5. Exact numeric payloads and arithmetic belong to the runtime/ALU; the brain
   receives a typed semantic quantity projection and compiles exact calls.
6. Versioned runtime references carry exact snapshot lifetimes. The brain may
   select a current binding; allocation, invalidation and stale validation
   belong to the runtime.
7. Temporal intent may be learned, but selecting a snapshot, resolving a
   binding in that snapshot and dereferencing its payload are exact runtime
   operations.
8. Historical execution uses runtime-materialized typed local frames. A brain
   need not receive the raw event log or unrelated fields to execute a bounded
   query against one resolved historical view.

These rules freeze the machine boundary, not a particular neural architecture,
attention coefficient or complete vocabulary.

## 2. Design principle

Krystal is not primarily a human language in this layer. It is a compact semantic intermediate representation:

```text
natural language
      ↓
frontend / translator
      ↓
Krystal semantic AST
      ↓
Krystal tape
      ↓
Oracle/runtime + learned executor
```

Human-readable Krystal may exist as a renderer/debugging notation, but the brain does not need to operate on UTF-8 text.

## 3. Token space

The logical vocabulary is fixed at:

```text
4096 token IDs
0x000 .. 0xFFF
12 bits per token
```

Physical storage may initially use `u16` or `u32`. The 12-bit layout is an ABI rule, not a requirement to implement packed `u12` storage immediately.

Token decomposition:

```ts
classOf(token) = token >> 8
valueOf(token) = token & 0xFF
```

The upper nibble identifies an ABI class. The lower byte is a class-local ID.

## 4. Token classes

```text
0x000–0x0FF  SYSTEM / CONTROL
0x100–0x1FF  STRUCTURE / FEATURES
0x200–0x2FF  CORE OPERATIONS
0x300–0x3FF  BASIC OBJECT CONCEPTS
0x400–0x4FF  PROPERTIES / QUALITIES
0x500–0x5FF  QUANTITY / TIME / SPACE
0x600–0x6FF  ACTIONS / STATE CHANGES
0x700–0x7FF  REFERENCES / ROLES
0x800–0x8FF  RELATIONS
0x900–0x9FF  LOGIC / INTENT
0xA00–0xDFF  DOMAIN CONCEPT BANK
0xE00–0xEFF  DYNAMIC CONTEXT SYMBOLS
0xF00–0xFFF  RESERVED / EXPERIMENTAL
```

These ranges are stable even when mostly empty.

## 5. Active-vocabulary rule

Krystal ABI v0 does **not** require all 4096 slots to be populated.

Expected initial active vocabulary:

```text
~200–400 tokens
```

Unused IDs remain reserved. Existing IDs never change meaning after freeze.

## 6. System / control class

`0x0xx` is reserved for protocol mechanics rather than world semantics.

Minimum candidates:

```text
PAD
BOS
EOS
TRUE
FALSE
UNKNOWN
BEGIN
END
REDUCE
QUERY
ANSWER
```

Additional control tokens must be added only when a real protocol needs them.

## 7. AST model

The canonical semantic representation is a typed AST. Nodes are one of:

```text
ATOM
SYMBOL
LITERAL
OPERATOR
```

Operators have fixed arity.

Examples:

```text
NOT/1
LINK/2
AND/2
IF/2
BETWEEN/3
```

Fixed arity is part of the ABI and allows a prefix tape to be parsed without parentheses.

## 8. Canonical linearization

The default Krystal tape format is **prefix notation**.

Example tree:

```text
AND
├── LINK A B
└── NOT
    └── LINK C D
```

Canonical tape:

```text
AND LINK A B NOT LINK C D
```

No brackets are required when every operator has fixed arity.

This choice follows K0 results: prefix was stable between seeds, brackets added substantial cost, and local operator/argument proximity matters strongly for LFM2.

## 9. Locality rule

Immediate operator operands should remain as close as possible on the tape.

Preferred primitive form:

```text
OP ARG1 ARG2
```

rather than:

```text
OP FILL FILL ARG1 FILL FILL ARG2
```

K0 found a causal interaction between LFM2 convolution window and operand distance: moving arguments outside the local convolution window caused a sharp performance cliff, and increasing the convolution kernel shifted that cliff by the predicted amount.

Therefore locality is an ABI design objective.

## 10. Oracle/runtime boundary

Krystal does not require the learned model to perform deterministic structural work.

The Oracle/runtime owns operations that can be exact without learned semantics:

```text
AST traversal
redex selection
arity validation
state storage
splice / replacement
copying unchanged structure
stopping
symbol identity
pointer / slot equality
binding lookup
payload dereference
structural validation
exact token equality
copy from a validated pointer
```

The learned executor should receive only a local operation whose result requires learned behavior.

Design rule:

> Learn only what cannot be computed exactly.

## 11. Local learned execution

The K1 execution contract is the reference v0 model.

The learned executor receives only a ready local redex, for example:

```text
NOT TRUE
AND FALSE TRUE
IF TRUE FALSE
```

or a learned domain predicate over already-resolved payloads.

It returns a typed local result. The Oracle writes the result back to the external AST and schedules the next redex.

Recursion therefore belongs to runtime iteration rather than latent model depth.

## 12. Symbol identity versus semantic payload

A symbol ID is not itself its semantic vector.

K2 established that learned token embeddings are a poor interface for truly open runtime symbols. Therefore Krystal ABI v0 separates:

```text
symbol identity
semantic payload
```

Dynamic context IDs:

```text
0xE00 .. 0xEFF
```

are runtime handles / slots. Their lower byte is an exact slot number, not a
learned semantic feature:

```text
class(E17) = 0xE
slot(E17)  = 0x17

E17 == E17  exact within the current binding epoch
E17 != E31  exact within the current binding epoch
```

The Oracle owns identity, allocation, lifetime and binding. A slot may be reused
after its binding epoch ends; ABI identity therefore does not imply a universal
or permanent entity meaning.

A learned tool should receive resolved payload data when the operation depends on payload semantics.

Example:

```text
E17 -> payload P17
E31 -> payload P31
```

Instead of requiring the model to learn associative lookup, the runtime may dereference the symbols before invoking the learned operator.

## 13. Dynamic symbols

`0xExx` is reserved for temporary session/world entities.

Examples:

```text
specific child
specific candy
specific box
specific person
specific location
specific object instance
```

Their meaning may change between examples or sessions. They are analogous to registers, variables, object handles or local symbols.

A model must not rely on a permanent learned embedding meaning for an `0xExx` slot.

### 13.1 Optional slot forwarding

An ABI-aware brain implementation may expose exact dynamic-slot equality as an
attention-logit bias. If parser-labelled query and key spans carry the same
`0xExx` slot, matching positions may receive an implementation-defined positive
bias.

Slot forwarding:

- communicates identity equality only, never semantic similarity,
- must preserve causal masking,
- may propagate a slot label across parser-declared record payload spans,
- does not replace exact Oracle dereference,
- is optional for conforming ABI implementations.

The bias strength and layer placement are implementation parameters. The
experimental `alpha=4` default is not an ABI constant.

### 13.2 Learned retrieval and exact lowering

`RETRIEVE` is an internal execution primitive, not necessarily a serialized
tape token:

```text
RETRIEVE(address_key, role_key, compatible_payload_positions)
    -> PayloadPointer
```

The parser declares the finite set of type/role-compatible payload positions.
The learned retrieval unit selects one position from that set. The runtime
validates the pointer and performs dereference exactly.

Answer-only copy supervision is sufficient to learn this pointer in the v0
property assay. Exact source-position labels may be used as auxiliary curriculum
or diagnostic metadata, but are not required by the execution contract.

Property operators lower canonically as follows:

```text
GET_PROPERTY(e, p)
    -> ptr = RETRIEVE(e, p, parser.compatible_payloads)
    -> COPY(ptr.payload)

HAS_PROPERTY(e, p, v)
    -> ptr = RETRIEVE(e, p, parser.compatible_payloads)
    -> EQ_TOKEN(ptr.payload, v)
```

`COPY` and `EQ_TOKEN` are exact runtime micro-ops. `EQ_TOKEN` compares canonical
token identity; it must not be used as a substitute for learned semantic
similarity. A future semantic-equivalence predicate requires a distinct learned
operator.

### 13.3 Versioned references and snapshot validity

A persistent runtime binding may not silently change the payload denoted by the
same live reference. Mutating a versioned value uses copy-on-write semantics:

```text
QREF_A valid_from=t0, invalid_at=t1
QREF_B valid_from=t1, invalid_at=null

entity.quantity_ref @ t0 = QREF_A
entity.quantity_ref @ t1 = QREF_B
```

The normative validity rule is a half-open snapshot interval:

```text
VALID(ref, snapshot)
    iff ref.valid_from <= snapshot < ref.invalid_at
```

An absent `invalid_at` denotes a currently live binding. The exact runtime owns:

```text
ALLOC_QREF(exact_payload, snapshot) -> fresh_ref
INVALIDATE_REF(ref, snapshot)
VALIDATE_REF_AT(ref, snapshot) -> LIVE | STALE | NOT_YET_VALID
```

The learned executor may select `CURRENT_REF(entity, role)` from the current
frame. It must not be asked to infer lifetime validity that is already known to
the machine.

Staleness is relative to a snapshot. A ref invalid in the current snapshot may
remain valid for an explicit historical dereference:

```text
VALIDATE_REF_AT(QREF_A, t1) -> STALE
VALIDATE_REF_AT(QREF_A, t0) -> LIVE
```

Slot reuse is a separate binding-epoch issue. An implementation must either
retain a generation/epoch sidecar or avoid reusing a slot while an older epoch
can still be addressed. The serialized `0xExx` slot token alone is insufficient
to distinguish an ABA reuse.

### 13.4 Temporal selectors and historical reads

Temporal query compilation is separated from history storage and exact
dereference. A brain may compile one of:

```text
CURRENT
AT_TIME(snapshot_id)
PREVIOUS
```

The machine lowers that selector through:

```text
snapshot = SELECT_SNAPSHOT(selector, optional_id, current_snapshot)
ref      = RESOLVE_BINDING_AT(snapshot, entity, role, history_store)
status   = VALIDATE_REF_AT(ref, snapshot)
value    = COPY(ref.payload)  // only when status == LIVE
```

`PREVIOUS` means the predecessor of the current snapshot/event boundary;
`AT_TIME` names an explicit accessible snapshot. Snapshot existence, future
access rejection, binding resolution and lifetime checks are exact. The learned
component is evaluated on selection of semantic temporal intent, not on
remembering historical payloads or reproducing their exact values.

### 13.5 Event anchors and bounded historical frames

An event-relative selector may name a typed boundary rather than an absolute
snapshot:

```text
BEFORE_LAST_EVENT(MOVE)
BEFORE_LAST_EVENT(TRANSFER_QUANTITY)
```

The runtime resolves the last matching event and returns its predecessor
snapshot. Missing events, future boundaries and unknown families are exact
errors. The brain is not required to scan a serialized event log when the
runtime already indexes event boundaries.

After resolving the anchor, the runtime may project a bounded typed frame:

```text
MATERIALIZE_BOUNDED_FRAME(snapshot, GET_PROPERTY)
    -> entity identity + property bindings

MATERIALIZE_BOUNDED_FRAME(snapshot, WHERE)
    -> entity identity + live location edges

MATERIALIZE_BOUNDED_FRAME(snapshot, COMPARE_QUANTITY)
    -> entity identity + live quantity bindings
```

The projection must preserve exact identity and snapshot validity while
excluding fields unrelated to the selected datapath. Learned retrieval may run
inside the projected frame; exact dereference, equality, arithmetic and
lifetime validation remain runtime operations.

Bounded projection does not authorize slot reuse. While an older epoch remains
addressable, implementations must retain fresh slot identity or add a separate
generation tag as required by section 13.3.

## 14. Domain concept bank

`0xA00..0xDFF` contains domain-level reusable concepts. Its concrete contents may differ between checkpoints or compiled tools.

Examples:

```text
coder:
BUFFER
POINTER
TYPE
FUNCTION

legal:
CONTRACT
CLAIM
LIABILITY
JURISDICTION
```

The IDs in this bank are checkpoint/domain ABI, not necessarily universal semantic IDs. Every domain bank must therefore have a manifest.

## 15. Core versus domain versus context

Krystal distinguishes three lifetimes of meaning:

```text
CORE
stable across the Krystal ABI

DOMAIN
stable for a specific checkpoint/tool/corpus

CONTEXT
dynamic for a specific world/session
```

This distinction should remain explicit in tooling and serialized metadata.

## 16. Grounding rule

Abstract operators or concepts should not be added merely because they are linguistically convenient.

Whenever practical, concepts should be grounded through many concrete generated worlds before being used as abstract curriculum primitives.

Example:

```text
2 candies
+ receives 1 candy
= 3 candies
```

repeated across different objects and actors should precede relying on an abstract arithmetic relation as if its meaning were self-evident.

The curriculum is the source of semantic grounding.

## 17. Literals

v0 should support a minimal literal layer sufficient for early curriculum.

Required first:

```text
small integers
Boolean values
```

Potential later additions:

```text
decimal values
units
short strings
coordinates
timestamps
```

Literal encoding should remain distinct from domain concept IDs.

## 18. Quantities and exact numeric payloads

A quantity is not a naked number. It is a typed property value with two
machine views:

```text
Quantity
├── semantic projection -> brain
└── exact canonical payload -> runtime / ALU
```

For the first curriculum, exact payloads should support at least `0..20`.
Payload identity and arithmetic remain exact even when the brain receives only
a geometric or lossy magnitude projection.

Exact integers are machine literals/payloads, not required fundamental
brain-facing semantic atoms. `NUM_0..NUM_N` may remain available for runtime,
debugging, translation, and controlled ablations without defining the neural
quantity representation.

The quantity type minimally records:

```text
quantity kind: discrete count | continuous measure
carrier/property/dimension
semantic magnitude projection
exact canonical payload reference
optional scale and unit
```

The same quantity contract must be reused across:

```text
counting
addition
subtraction
comparison
ordering
object quantities
```

This encourages grounding of quantity as a shared abstraction without asking
the neural executor to reproduce integer arithmetic.

Exact questions lower to typed runtime calls, for example:

```text
ADD_EXACT(q1, q2)
    -> CALL ALU.ADD_INT(q1.exact, q2.exact)

COMPARE_EXACT(q1, q2)
    -> CALL ALU.COMPARE_INT(q1.exact, q2.exact)
```

A semantic answer may be produced directly only when it is invariant over all
exact payloads compatible with the visible projection. Otherwise the brain
must request exact execution. Implementations should report required-call
recall, unnecessary-call rate and unsafe direct-answer rate.

The replicated Q0-Q3 assay supports this hybrid contract, including
near-perfect typed call construction from a lossy interval view. A 2/4/8/16
granularity ablation identifies four carrier-relative bands as the current
best non-degenerate setting for the tiny LFM assay. Band count remains
type/domain metadata, not a universal ABI constant. The ABI does not yet freeze
projection geometry, interval boundaries, unit vocabulary, or serialized
quantity syntax.

## 19. Typed operators

Each operator must declare a signature.

Example:

```text
NOT(bool) -> bool
AND(bool, bool) -> bool

HAS(agent, object) -> bool
INSIDE(entity, container) -> bool

GIVE(agent, recipient, object) -> event
COUNT(collection) -> quantity
ADD(quantity, quantity) -> quantity
```

Types are semantic constraints used by the generator, validator and Oracle. They do not need to correspond one-to-one with token classes.

## 20. Exact versus learned operators

Operators should be classified by execution ownership.

Example:

```text
exact/runtime:
ID_EQUAL
integer addition
integer comparison
unit conversion
binding lookup
AST traversal

learned:
soft category membership
naturalistic compatibility
ambiguous causal relation
domain-specific judgment

hybrid lowering:
learned RETRIEVE + exact COPY
learned RETRIEVE + exact EQ_TOKEN
learned ALU routing/call construction + exact numeric execution
```

A function that is easily exact should not be delegated to the learned executor merely to make the model appear more capable.

## 21. ABI versioning

Every dataset, checkpoint and domain bank must declare:

```text
krystal_abi_version
vocab_manifest_version
curriculum_generator_version
```

Existing stable token meanings cannot be reassigned within an ABI version. Removed concepts leave tombstones/reserved IDs. Experimental meanings belong in `0xFxx`.

## 22. Required machine-readable artifacts

The eventual implementation should be generated from or accompanied by:

```text
abi.ts
vocab.ts
operators.ts
types.ts
manifest.json
```

Each operator entry should minimally expose:

```ts
{
  id,
  name,
  arity,
  inputTypes,
  outputType,
  execution: "oracle" | "learned" | "hybrid",
  lowering?: RuntimeStep[]
}
```

Runtime micro-ops may be declared separately without consuming token IDs. Every
step referenced by `lowering` must resolve to a declared runtime primitive.

## 23. Non-goals for v0

Krystal ABI v0 does not freeze:

- a complete human-language grammar,
- the final 4096-word lexicon,
- a universal ontology,
- domain-bank contents,
- a full natural-language tokenizer,
- packed 12-bit storage,
- typed LM heads,
- grammar masking,
- multimodal payloads,
- tool routing,
- long-term memory policy.

Those require later evidence.

## 24. Freeze criterion

ABI v0 is ready to freeze when the first curriculum generator can express every required competency without ad-hoc structural exceptions.

If a competency forces a structural exception, prefer revising the ABI before freezing rather than encoding the exception into training data.

The vocabulary itself should grow from curriculum requirements.

## 25. Core principle

Krystal ABI v0 exists to preserve three experimentally supported properties:

```text
canonical identity
local semantic structure
deterministic runtime orchestration
```

The frozen v0 property datapath is therefore:

```text
parser-declared roles
      +
exact 0xExx identity / optional slot forwarding
      ↓
learned RETRIEVE pointer
      ↓
exact COPY / EQ_TOKEN / structural runtime operation
```

The learned model is not required to reconstruct structure that the system already knows exactly.
