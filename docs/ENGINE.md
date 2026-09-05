# The Krystal engine

**Status:** describes `master` as of the `gpu parity` commit. Verified against
the code, not against intent: every number here was read out of the tree, and
the test suite it describes runs 78 pass / 0 fail via `bun test packages`.

For how to *use* this from a simulation, see [`API.md`](./API.md). For the
superseded v2 contract and the milestone notes, see
[`archive/`](./archive/README.md).

---

## 1. What the engine is

Krystal is a small, record-based decision model. One pass does exactly one
thing:

> **records in → one pointer per question out**

A frame is a list of *records*. Each record is up to eight tokens. Some records
are marked as *questions*; the rest form the *bank*. For every question the
model returns one bank record, plus the distribution it came from.

That is the whole output. There is no generated text, no opcode, no plan
structure, no exact reference. The model learns which record is relevant, and
nothing else.

### What it deliberately does not do

- **It does not know what a record means.** Not what a band is, not which
  record is an apple, not what the chosen record will be used for.
- **It does not own a vocabulary.** Token ids are the host's; the session is
  handed a `tokenRows` mapping and a brain is only meaningful together with the
  mapping it learned under.
- **It does not decide what is legal.** Every candidate restriction arrives as
  a host closure, `allows(query, record)`. Krystal applies the answer and never
  asks why.
- **It does not hold world state.** No caching, no history, no handles. The
  host re-sends what the creature can currently see and remember.

The design pressure behind all four is one rule: *anything that is a fact about
one world must not be compiled into the engine that runs every world.*

---

## 2. Relations are records, and roles are questions

This is the change that the v2 document predates entirely, and it happened by
**deletion rather than by extension**.

v2 had an asymmetry. An action was picked from a static `ActionIntent` catalog
by one selector; its arguments were picked from the bank by other selectors;
the subject was resolved structurally and was almost always `Self`. Predicates
and participants were different kinds of thing, reached by different machinery.

They are now the same kind of thing. A relation is an ordinary record in the
bank, indistinguishable to the engine from the apple it might be about. Its
roles are ordinary questions — several query records in the same frame — and
each is answered by the same single selector, under whatever mask the host
supplied for it.

So a creature deciding "eat *what*?" and a creature deciding "*who* eats?" run
identical code. What separates the two is the content of the query record and
the mask the host attached to it.

Concretely, everything reified away:

| Gone | Where the responsibility went |
|---|---|
| `ActionIntent` catalog records, `intentSchemaId` | A relation is a bank record like any other. |
| Separate intent selector vs. argument selector | One `selectorOracle`, run once per question. |
| Hardcoded `subjectFilter` / `droppedNoSubject` | `Self` is a bank candidate; the host's mask admits it or does not. |
| `RelationRole = "subject" \| "object"` | A role is a query record. The engine never names one. |
| `roleAdmitsRecord`, `accepts: ["category:Edible"]` | The host's `allows` closure. |

`BRAIN_LIMITS.relationArity`, `maxActionIntents` and the `RelationRole`
reference in [krystal-engine-schema.ts](../packages/schema/src/krystal-engine-schema.ts)
are **dead** — comments and unread constants. Nothing in the engine reads them.

### The consequence for arity

A role costs a *question*, not a token slot and not a reference. A frame asking
four things about one relation carries four query records. Cost scales with
questions, and questions are cheap: the encode is shared, and each additional
question is one selector pass over the bank.

---

## 3. The frame

### 3.1 What the host sends

```ts
interface HostRecord {
  schemaId?: number;                        // free for the host to assign
  band?: number;                            // an embedding row, and QUERY_BAND
  query?: boolean;                          // shorthand for band: QUERY_BAND
  tokens: readonly (number | HostToken)[];  // at most 8
}
interface HostToken { token: number; role?: number }
```

`packHostFrame` ([frame.ts](../packages/krystal/src/host/frame.ts)) serializes a
list of these. **The frame is sized to the list it was given.** Three records
are three records — not three records padded into a fixed geometry. The
432-slot frame in the schema is a *device arena ceiling*, not a shape the host
must fill.

### 3.2 What the model reads

Five flat SoA arrays, and nothing else:

```ts
BrainFrameGpu {
  tokenIds:             u32[slots * 8]
  fieldRoles:           u32[slots * 8]
  schemaIds:            u32[slots]
  bandIds:              u32[slots]
  activeRecordIndices:  u32[slots]
}
```

Three buffers were removed when it turned out nothing read them, and each
removal is load-bearing:

- **`attentionMask`** — written every frame, never looked at. Padding is found
  by the `PAD` sentinel.
- **`runtimeRefs`** — a reference to a world entity is the host's to resolve.
  The model never learned anything from one.
- **`recordFlags`** — served role filters that belonged to a world contract the
  host now owns.

v2's `recordDescriptors { tokenOffset, tokenLength, schemaId, bandId, flags,
handle }` is gone with them. **There is no handle.** The model answers with a
record slot index in the host's own numbering; mapping that back to a world
object is the host's business and never crosses the boundary.

### 3.3 Record width is eight, and it is structural

Eight token positions per record, one learned position embedding each. A record
with nine tokens is **refused**, not truncated — a truncated record would be a
different record with no way to tell. Slot `PAD` inside a record is simply
absent from the active list; a hole is a hole, not a token.

This is the one geometric rule left in the engine. Changing it changes the
model's shape.

### 3.4 Bands are structural, never sensory

v2 named the bands `vision`, `audio`, `touch`, `proprioception`,
`interoception`, `communication`. That was the engine asserting which senses
exist — a fact about one world, frozen into the ABI of all of them. A creature
may live in a world with echolocation and no eyes.

So there is **one** `perception` band, and which sense a record came from is a
*token inside the record*, drawn from that world's own vocabulary. Per-channel
quotas are the simulation's business.

The eight bands are `system`, `homeostasis`, `body`, `perception`, `focus`,
`query`, `catalog`, `memory`. To the model a band is one embedding row and
nothing more — with a single exception: `query`, which is what tells a question
from a fact.

> ### ⚠️ A live trap: two band orderings disagree
>
> `BRAIN_FRAME_BANDS` is `[...PERCEPT_FRAME_BANDS, ...MEMORY_FRAME_BANDS]`, so
> `memory` lands last:
>
> ```
> runtime  (BRAIN_FRAME_BANDS):  0 system  1 homeostasis  2 body  3 perception  4 focus  5 query  6 catalog  7 memory
> generated (BrainBandKind):     0 system  1 homeostasis  2 body  3 perception  4 memory  5 focus  6 query  7 catalog
> ```
>
> `QUERY_BAND_INDEX` — what the engine actually reads — is **5**. The generated
> enum says `query: 6`, which at runtime is `catalog`. A host that reached for
> `v1_0_0.BrainBandKind.query` would file its questions in the catalog band and
> silently get no answers.
>
> Nothing in the tree does this today, because `frame.ts` exports
> `QUERY_BAND`. **Always use `QUERY_BAND` or `record.query = true`.** The two
> orderings should be reconciled in code; this document is not a fix.

### 3.5 The active frame

`compileActiveFrame` ([masks.ts](../packages/krystal/src/forward/masks.ts))
turns the packed frame into what the forward walks:

- `activeTokens` — compact list of non-`PAD` frame token indices, record-major;
- `recordCompactOffset` / `recordCompactCount` — each slot's range in that list;
- `streamIds` — `STREAM_QUERY` for query-band slots, `STREAM_RECORD` otherwise;
- `bankRecords` / `queryRecords` — the two halves, in slot order.

Everything downstream is indexed in *bank rows* and *query rows*, not slots. The
host's slot numbering is restored at the boundary, in `HostSelection.record`.

---

## 4. Vocabulary and embedding rows

The token space is 16-bit and split in half:

| Range | Meaning | Embedding rows |
|---|---|---|
| `0x0000`–`0x7fff` | semantic symbols | `semanticEmbeddingRows` = 4096, one per manifest index |
| `0x8000`–`0xfffe` | references to world entities | `refEmbeddingRows` = 256, a shared pool |
| `0xffff` | reserved empty | — |

**Ids are never used as rows directly.** `config.tokenRows[id]` projects an id
to a row. Indexing by id would cost 32768 rows to carry a few hundred live
symbols, and the reference half has no stable identity to learn anyway — every
reference folds into a small shared pool.

`tokenRows` is *required* at session construction and carried in the config,
not held as a module constant. The reason is stated plainly in
[model.ts](../packages/krystal/src/forward/model.ts): a global made the forward
silently correct for exactly one world. `BRAIN_FORWARD_CONFIG.tokenRows` is an
empty array and deliberately unusable.

### Sentinels

`KRYSTAL_SENTINEL_TOKENS` reserves `pad`, `bos`, `eos`, `boolTrue`, `boolFalse`,
`unknown`, `begin`, `end`, `void`, `unavailable`, `something`, the NSM logical
primes `not` / `isA` / `partOf`, the temporal primes `before` / `now` / `then`,
and `want`.

**Only `pad` is read by any code today.** The rest are a reservation awaiting a
host that uses them. The reasoning behind the list is in
[archive/RESERVED_VOCABULARY_AUDIT.md](./archive/RESERVED_VOCABULARY_AUDIT.md);
the reasoning behind `want` specifically — that a creature must never be able to
assert its own wanting, because it would learn a label instead of a connection —
is in the schema itself and is still in force as a design rule.

---

## 5. The model graph

```
tokens                                          per active token
  │
  ├─ 6 additive embedding lookups ─────────────► fieldStates [T, H]
  │    token · field · schema · band · stream · pos
  │
  ├─ 2 encoder blocks ─────────────────────────► fieldStates [T, H]
  │    record-local self-attention + ReLU FFN, residual
  │
  ├─ learned-query pooling ────────────────────► key + value per record
  │
  ├───────────────┬─────────────────────────────
  │  bank         │  queries
  │  keys/values  │  queryValues
  │  [R, H]       │  [Q, H]
  │               │
  │               ├─ 2 mixer blocks ───────────► queryOutput [Q, H]
  │               │    query → bank cross-attention + ReLU FFN
  │               │
  └───────────────┴─ selector ─────────────────► p [Q, R], gather [Q, H], index [Q]
                        score + host mask, softmax, soft gather, argmax or sample
```

The CPU reference is [oracle.ts](../packages/krystal/src/forward/oracle.ts); the
GPU path dispatches the same math and is compared against it.

### 5.1 Input embedding — six terms, not five

```
x_t = E_token[row(id_t)] + E_field[row(role_t)] + E_schema[schema_t]
    + E_band[band_t] + E_stream[stream_t] + E_pos[t mod 8]
```

`E_field` projects through the *same row mapping* as `E_token`, because field
roles are ordinary semantic tokens. `E_pos` is the record-local position; there
is no RoPE and no `E_recordIndex`. Absolute record position must not encode
identity — that invariant survived from v2 intact.

The six-way sum has a consequence that shapes the whole training story: two
questions about one relation share `schema`, `band`, `stream`, `pos` and the
relation token. Initially only the field token tells them apart, so the
distinguishing part of the vector is about a hundredth of it. See §7.3.

### 5.2 Record encoder — local by construction, not by mask

Two blocks: multi-head self-attention, residual, ReLU FFN, residual. No
ShortConv (v2 left it open; the first profile does not use it).

Attention is block-diagonal because **no token ever attends across a record
boundary**. The CPU path implements this as a *range*, not a mask: `recordRanges`
gives each token its own record's span, and the attention visits only that. A
full `T × T` score matrix would be quadratic work to produce zeros — 9.4 MB per
frame at a full one.

The `[T, T]` mask still exists for the device path, and it carries one optional
extra: the same-word additive bias (`WordBias`, `alpha` on the logit when two
tokens share a local word id). `alpha = 0` is bit-identical to the unbiased
mask. **It is reachable only through the GPU runner directly** — `BrainSession`
and `gpuBackend` do not expose it. See
[archive/word_attention_bias.md](./archive/word_attention_bias.md).

### 5.3 Pooling — two learned queries, not an average

Each record is pooled twice, by two learned query vectors held in `pool [2, H]`:
row 0 gives the **key** (what the record is retrieved by), row 1 gives the
**value** (what is read off it once chosen). Softmax over the record's own
tokens, then a weighted sum.

Key and value are not forced into the same pooling, exactly as v2 required.
Token averaging is not part of the contract.

The same pooling runs over query records, producing `queryValues [Q, H]`.

### 5.4 Mixer — cross-attention, one direction

Two blocks of query → bank cross-attention plus ReLU FFN, residual on both.
Queries read the bank; the bank does not read the queries, and records do not
attend to each other.

The mixer mask is **unconstrained** when driven through `BrainSession`: what a
question may *attend to* while it thinks is not the same as what it may
*choose*, and only the second is the host's grammar. This is a deliberate
choice made in both [session.ts](../packages/krystal/src/host/session.ts) and
[gradients.ts](../packages/krystal/src/host/gradients.ts), and the two must
agree or the update differentiates a distribution the creature never sampled.

### 5.5 Selector — score, mask, gather, choose

```
score[q,i] = dot(Wq · queryOutput[q], Wk · bankKey[i]) / sqrt(H) + mask[q,i]
p[q]       = softmax(score[q])
gather[q]  = Σ_i p[q,i] · bankValue[i]
index[q]   = argmax(p[q])        or a draw from p[q]
```

**One shared `Wq`/`Wk` pair serves every question.** Per-slot projections are
noted in the code as "a later ablation" and are the leading suspect for why the
policy is slow to separate two questions asked of one relation.

Three behaviours worth knowing:

- **A masked record has probability exactly zero.** Sampling narrows nothing the
  mask allowed and admits nothing it forbade.
- **A fully blocked row comes back open, not uniform-over-the-impossible.** The
  scores are zeroed and `index` is `0xffffffff`. A row of `-1e30` softmaxed
  would look like an answer and is not one.
- **Sampling is the host's noise, not the engine's.** `choose` takes a `sample`
  closure returning a uniform in `[0,1)`; `sampleRow` does inverse-CDF. A weight
  that is not a number is skipped like a zero, so a row of NaN chooses `-1`
  rather than its last entry — measured, that failure once looked like a
  creature deliberately choosing the one act nothing may do, 120 times over.

Why sample at all: a deterministic policy has no behaviour to reinforce. An
untrained selector's distribution is nearly uniform, so sampling tries
everything, and exploration decays on its own as the policy sharpens. There is
no temperature to schedule — the policy's own certainty is the schedule.

### 5.6 The value head, and the decision head that does not run

Both read the same concatenated `3H` context:

```
ctx[q] = [ queryOutput[q] , gather[q] , third[q] ]
```

**Value head** `[1, 3H]` — predicts the change in valence for the next tick.
Squared error against an observed number, so the target needs no labelling: next
tick's valence is simply read off. That is what makes live play trainable once a
gold curriculum stops. It is a REINFORCE baseline, which is why the third block
of its context is `context: "available"` — the mean bank value over what the
grammar *allows*, a state feature. A critic conditioned on which action was
drawn would make two frames with identical outcomes disagree about how they
went.

**Decision head** `[routeKindCount=4, 3H]` — v2's route classifier. It is
initialized, checkpointed, differentiated by both the CPU oracle and the GPU
shaders, and **never invoked by `BrainSession`**. It is dead weight in the
current profile: 1536 parameters carried for a route decision nothing asks for.

### 5.7 The profile

| Parameter | Value |
|---|---:|
| Hidden size `H` | 128 |
| Heads × head dim | 4 × 32 |
| FFN | 384 |
| Encoder blocks | 2 |
| Mixer blocks | 2 |
| Record width | 8 |
| Block type | bidirectional self-attention + ReLU FFN (no ShortConv, no RoPE) |
| Route kinds | 4 (unused) |
| Embedding rows | token 4352, field 4352, schema 256, band 8, stream 2, pos 8 |

Parameters, as built by `createBrainForwardWeights`:

| Page | Elements |
|---|---:|
| embeddings | 1 149 184 |
| encoder blocks (2) | 294 912 |
| mixer blocks (2) | 294 912 |
| selector `Wq`+`Wk` | 32 768 |
| pool | 256 |
| decision head | 1 536 |
| value head | 384 |
| **total** | **1 773 952** (6.77 MiB as f32) |

Two thirds of the brain is the embedding page, and the tables are also the part
that does most of the learning (§7.3).

Initialization is deterministic from a seed: Xavier/Glorot uniform for matrices,
normal `std = 0.02` for tables, no biases anywhere in the graph. The same seed
gives byte-identical weights on CPU and GPU, which is what the parity tests
rest on.

---

## 6. Masks: the host's grammar

There are two masks in the graph and they answer different questions.

**The record mask** (encoder) is structural and the engine owns it: a token
attends within its record.

**The selection mask** (selector) is the host's grammar and the engine never
authors it. `selectionMask(active, allows)` builds `[Q, R]` with `-1e30` on
every record a question may not choose, by calling the host's closure per
(question, record).

One rule inside it is worth stating: **a question nothing admits is left open**
rather than uniformly forbidden. All-`-1e30` comes back as a uniform over the
impossible, which looks like an answer.

The same function builds the mask for *choosing* and for *learning*. A mask that
differed between the two would compute the gradient of a distribution the
creature never sampled from — and that is the reason `HostExperience.allows` is
documented as "the SAME predicate `think` was given".

Where the mask came from is entirely outside the engine. Affordances, type
compatibility, "must be visible", "must not be Mother" — all of it is a closure.
This is the layer boundary the project cares most about: *a mask restriction
must be a fact about the frame, never knowledge about the world.*

---

## 7. Learning

Two mechanisms, separate on purpose, answering different questions:

| | question | signal | default |
|---|---|---|---|
| `teach` | what **can** be said here | a demonstration | tables unfrozen |
| `learn` | what is **worth** saying | an outcome | tables frozen |

Folding them together would make a shown act compete with a felt one on the same
scale, and the creature would learn that being taught feels good.

### 7.1 `teach` — being shown

Pointer loss and nothing else. No baseline, no advantage, no critic, no reward.
The value head is provably untouched by teaching, and there is a test asserting
exactly that.

- `gold[q]` — the slot this question should have chosen. Cross-entropy toward it.
- `forbidden[q]` — "not this". Unlikelihood loss `-log(1 - p)`, whose gradient is
  `p/(1-p) · (onehot - p)`: a choice the policy barely makes is barely touched,
  one it is sure of is pushed as hard as a demonstration pulls. A "no" names
  nothing to do instead, and the loss does not pretend it did.
- A question with both keeps its gold. What to do says more than what not to.

Encoded in one word: the `AWAY` bit (`0x80000000`) on the target row, so every
reader — CPU oracle, device shader, invalid-row check — sees one array with one
meaning per entry.

Three guards, each of which was added after a measured failure:

- **`maxUpdateAbs` (0.05)** — the largest move any one weight may make; the whole
  update is scaled down, direction kept. Without it: a five-lesson curriculum
  drove the largest weight `0.69 → 5.6 → 441 → 23 million` in three showings,
  then NaN.
- **`decay` (1e-3)** — cross-entropy's fixed point is at infinity, so something
  has to pull back. The pull grows with the weight; the push does not.
- **`maxParameterAbs` (1)** — a last line. A showing that still reaches it is
  rolled back to the last weight, and the report counts it.

`meanAgreement` is the measurement: how much probability the policy already put
on what it was being shown, *before* the pass changed anything. It starts near
`1/candidates` and should climb.

### 7.2 `learn` — living with what happened

REINFORCE with the value head as baseline. The trick that lets it reuse the
supervised machinery unchanged: the gradient of cross-entropy toward the choice
that was **actually made** is exactly the direction that makes that choice more
likely, so scaling it by the advantage turns "push toward the right answer" into
"push toward what was done, in proportion to how much better than expected it
turned out". A negative advantage flips the sign.

- **Advantage must be standardised across the batch** (`(a − mean)/σ`, clipped at
  ±3). This is not tuning. Un-standardised, `policyScale = 1` moved nothing and
  `policyScale = 20` reached `loss = 2.7e6` in an actor–critic feedback loop.
- **Gradients are collected before any is applied.** A per-frame step would make
  each update depend on batch order, and would leave the advantage on whatever
  scale the world's sense of well-being happens to use.
- **A batch where everything went equally well does not push at all.** Zero
  deviation is a guard, not an epsilon — an epsilon amplifies noise into
  confidence.
- **A turn with no `reward` is skipped for the actor but still teaches the
  critic.** The first frame of a life has nothing to difference against, and
  calling that zero is a claim that nothing happened.
- **The whole update is one transaction.** Non-finite or over-limit, and
  representation, actor and critic all roll back together. Keeping two after the
  third diverged creates a checkpoint corresponding to no policy that produced
  the batch.

Reported: `meanEntropy` (0..1, normalized against uniform over what the grammar
allowed) and `meanConfidence`. Entropy falling is the only visible sign that
anything was learned — reward can improve because the world got easier, and an
argmax looks the same however sure of itself it is.

### 7.3 What actually moves

The trainable surface is small and staged, and the staging is the finding:

| Part | `teach` | `learn` | Why |
|---|---|---|---|
| selector `Wq`/`Wk` | ✅ | ✅ | the actor |
| value head | ✗ | ✅ | the critic |
| token + field tables | ✅ default | opt-in `unfreeze.tokens` | **the only lever that can make two records look different** |
| pool | ✗ | opt-in `unfreeze.pool` | shared by every record; cannot create a missing difference |
| schema / band / stream / pos tables | ✗ | ✗ | shared by every record of a kind; moving them biases everything at once |
| encoder + mixer blocks | ✗ | ✗ | frozen; their gradients exist on the device and nobody applies them |

**Teaching with the tables frozen is a no-op that looks like slow teaching.**
Measured on two records and one question: frozen, 200 showings move the policy
`0.498 → 0.510`. Unfrozen at the same rate it reaches `0.996` in fifty, and at
`lr 0.2` in about twenty. One shared selector projection cannot pull apart two
records whose representations are fixed random vectors. This is why
`unfreezeTokens` defaults **on** in `teach` and **off** in `learn`.

A token gradient is a pure scatter — a token state is the *sum* of six rows, so
the gradient at the state is the gradient of each — and only the two tables that
say *what this is* are written.

One known caveat, stated in the code: unfreezing tables in `learn` was once
reported unstable, but that run was **off-policy replay** — the same choices
replayed hundreds of times after the policy had moved away from them. Fresh
on-policy samples stay finite. Do not hide stale replay with an optimiser;
replay needs an importance ratio or an explicitly off-policy objective.

### 7.4 Telling two roles of one relation apart

This is where reification meets the training story, and the behaviour is
measured rather than argued. The setup: two people in the bank, and two
questions — *who gives* and *who receives* — with Ada as the right giver and Bo
as the right receiver.

| Setup | Result |
|---|---|
| Two query rows with **identical tokens** | The two answers are not merely similar, they are **bit-identical**. Nothing in the frame distinguishes them, so nothing can. A giver could never differ from a receiver. |
| Query rows that **name their role** (one token differs), one question credited | The credited question learns decisively (`\|Δ\| > 0.15`) — the policy gradient is not weak. |
| Same, reading the *uncredited* question | It moved almost exactly as far (`\|Δ₀ − Δ₁\| < 0.01`). Naming the role reaches the model; frozen, it does not *separate* it. |
| Opposing targets on both rows, tables frozen | They very nearly cancel: the residue is three orders of magnitude below what one question alone learns. Both questions drift the *same* way whatever the reward said. |
| Same, tables unfrozen (`tokens: true`, 300 rounds) | Finally opposite: measured `+0.910 / −0.880`. At 150 rounds it is still `+0.019 / +0.000` — the actor now moves the representation alone and takes about twice as long. |
| Fresh on-policy sampling, 700 batches of 8, small geometry | Converged: `p(Ada \| giver) = 0.99997`, `p(Bo \| receiver) = 0.98991`, every weight finite. |

Two practical notes from those runs:

- **A role must be named in its query row.** Two questions that say the same
  thing are one question asked twice, and no amount of training fixes it.
- **The learning rate belongs to the geometry, not to the host.** `tokenRate 0.5`
  is right for the reduced test geometry; on the full profile it is too hot —
  107 of 500 fresh batches were rolled back and both questions collapsed onto
  one record. A real game runs `0.1`, where the rows plateau around `|0.20|` over
  2000 batches and never trip the ceiling.

The frozen-model tests in `learn.test.ts` document a **limit, not a
requirement**. When the pool, encoder and mixer come unfrozen they should start
failing, and the right response is to invert them.

---

## 8. Backends

### 8.1 The seam

Reading a frame is the expensive half of thinking; asking questions of it is the
cheap half — measured in a simulation at **20 s of `consider` against 0.17 s of
`choose`** over sixty ticks. So the seam is placed there, and it is three
matrices wide:

```ts
EncodedFrame { queryOutput [Q,H], bankKeys [R,H], bankValues [R,H] }
```

Everything downstream — every question, its distribution, its choice — is a
function of these three and the host's mask.

Learning has the same shape. Differentiating a frame was **68% of a learning
tick** in scalar JS; what the update does with the gradients is a few thousand
multiplies. So a backend may also differentiate, handing back exactly the arrays
the update consumes — but *the update itself always runs on the host*, on the
host's own weights, whoever computed the numbers.

`BackwardResult` carries `dSelectorWq`, `dSelectorWk`, `dPool`, `dFieldStates`,
`dValueWv`, plus `policy` and `valuePrediction`. Nothing about the encoder or
mixer blocks: the device computes those gradients, and a gradient nobody applies
is a readback nobody needed.

The backend is **injected**, never reached for. A brain that could construct its
own device backend would import the WebGPU package, and every browser bundle
that only meant to think on the CPU would carry a shader artifact it never runs.

### 8.2 Weight synchronisation

Every device backend holds a *copy* of the weights, so after `learn` or `teach`
the two disagree until told. Telling it everything cost 7 MB a batch for a change
to two projections and a few embedding rows — so an update reports exactly what
it touched via `WeightChanges`, including the element offsets of the individual
embedding rows that moved. Absent means all of it, which is what a restored
checkpoint needs.

Nothing is said after a rollback: the arrays are as the backend last saw them.

In `teach`, the source is told *between* showings, not after the loop — the
second showing must be differentiated against the brain the first one left.

### 8.3 The WebGPU path

23 WGSL shaders under
[packages/webgpu/src/shaders/training/](../packages/webgpu/src/shaders/training/),
compiled ahead of time into a serialized artifact
(`krystal.artifact.generated.ts`) so no arktype scope or link step enters the
runtime. `bun run build:webgpu` rebuilds it.

Device arena capacities — ceilings, not ABI:

| Constant | Value |
|---|---:|
| `KRYSTAL_MAX_TOKENS` | 1536 |
| `KRYSTAL_MAX_RECORDS` | 432 (`frameRecordSlots`) |
| `KRYSTAL_MAX_QUERIES` | 128 |
| `KRYSTAL_MAX_H` / `MAX_FFN` / `MAX_HEADS` / `MAX_BLOCKS` | 128 / 384 / 4 / 2 |
| `KRYSTAL_MAX_ROUTE_KINDS` | 8 |

Note that 432 records × 8 tokens is 3456, well past the 1536-token ceiling. **The
binding constraint on the device is active tokens, not slots** — an unoccupied
slot is nearly free, an occupied one is not, because attention is quadratic in
tokens and only linear in records.

Every mask comes from the host, including the mixer's. The runner used to compile
that one itself; when the host stopped shipping a reference table, every cell
came back blocked, the mixer output collapsed to zero, and the GPU quietly
disagreed with the CPU. One layer deciding another's business.

Parity is tested at two levels: per-operator, and — in
[test/session.test.ts](../packages/webgpu/test/session.test.ts) — at the surface
a simulation actually calls, with the same records, the same grammar closure and
the same seed on both backends, through `think`, `learn` and `teach`.

**Any change to the arena regions or the shader list requires
`bun run build:webgpu`.** The arena capacity lives in the serialized artifact, so
without a rebuild the failure is `copy range does not fit in krystal.arena` at
runtime rather than a compile error.

---

## 9. Checkpoints

`KRY1`, version 1. Header of 11 words: magic, version, seven geometry numbers
(`hiddenSize`, `ffnSize`, `encoderBlocks`, `mixerBlocks`, `headCount`, `headDim`,
`routeKindCount`), then `tokenRows.length` and the total weight element count.
Then the row mapping, then every weight array in the one order both directions
walk.

Three things ride together because weights alone can be loaded into the wrong
brain *without failing* — they would simply denote something else, quietly, for
the rest of that creature's life. So both the geometry and the row mapping are
checked, and a mismatch is refused:

`"not a krystal checkpoint"` · `"a later checkpoint format"` ·
`"a different geometry"` · `"a different vocabulary"` · `"truncated"`

`restore` writes **in place**, because a session hands out its weights;
replacing the arrays would leave anything holding the old ones training a brain
nobody is thinking with.

---

## 10. Invariants that hold today

The v2 list, corrected against the code. These are the ones the implementation
actually enforces:

1. **Whole-record framing.** A record wider than eight tokens is refused, never
   truncated into an illegal shape.
2. **No identity shortcut.** No `E_recordIndex`, no RoPE. Absolute record
   position cannot encode identity.
3. **Permutation contract.** Records are encoded independently and the mixer is
   order-free, so permuting the bank permutes pointer indices only.
4. **No cross-record attention.** Enforced by construction on the CPU (ranges)
   and by mask on the device.
5. **Exact masks.** Forbidden candidates get `-1e30`, not a learned negative
   bias, and carry probability exactly zero.
6. **One mask for choosing and for learning.** Built by the same function, or
   the gradient is of a distribution nobody sampled.
7. **The engine never authors grammar.** Every selection restriction is a host
   closure.
8. **The model never reconstructs a reference.** It answers with a slot index in
   the host's numbering.
9. **Checkpoint guard.** Geometry and vocabulary mismatch is refused, not
   coerced.
10. **Updates are transactional.** Non-finite or over-limit rolls back
    representation, actor and critic together.
11. **CPU/GPU parity at the session surface**, not merely per operator.
12. **A brain is not a value.** `learn`, `teach` and `restore` mutate in place;
    copying to hand back a new brain would make "which of these two is the
    creature" a question with no answer.

Dropped from v2 because nothing implements them: stable-vocab tombstones,
cache equivalence (there is no cache), and backward compatibility of vacant
vocab slots.

---

## 11. In the tree but not wired

Honest inventory, so nobody mistakes presence for use:

| Thing | State |
|---|---|
| Decision head / `routeKindCount` | Initialized, checkpointed, differentiated on both paths. Never invoked by `BrainSession`. |
| `WordBias` | Honoured by the GPU runner and by `compileRecordMask`. Not reachable through `BrainSession`. |
| `BRAIN_LIMITS.relationArity`, `maxActionIntents`, `maxIntentProposals`, `maxQueries`, `maxMemorySlots`, `maxRecordFields`, `maxReferencesPerRecord`, `maxTutorial*` | Unread constants. |
| `PERCEPT_FRAME_BANDS` / `MEMORY_FRAME_BANDS` capacities, `placement`, `overflow` | Only the concatenated list (for band indices) and the totals (as GPU ceilings) are read. Placement and overflow policies are documentation. |
| Sentinels other than `pad` | Reserved, unread. |
| `BrainValueKind` | Generated, unreferenced. |
| `masks.ts` `allBlocked`, the S2–S10 mask contract comment | Vestigial; describes a curriculum that no longer exists. |
| `packages/shared` | Empty. |

---

## 12. What changed since v2

For readers who knew [archive/KRYSTAL_BRAIN_ARCHITECTURE_V2.md](./archive/KRYSTAL_BRAIN_ARCHITECTURE_V2.md).

| v2 said | Now |
|---|---|
| §4.1 Records need no fixed length; `recordDescriptors` carry offset, length, flags, handle | Exactly 8 token positions; five flat SoA arrays; `handle` and `flags` deleted |
| §4.3 Five additive embedding terms | Six — record-local `pos` is part of the contract |
| §5 Sensory bands: vision, audio, touch, proprioception, interoception, communication | Eight *structural* bands, one `perception`; the sense is a token in the record |
| §5 288 record slots, 2304 tokens, 1536 active | 432 slots / 3456 tokens as a device ceiling; the host frame has no fixed geometry at all |
| §7 Candidate masks derived from the ABI by slot type | Host closure `allows(query, record)`; the engine never asks why |
| §8 `ActionIntent` catalog, `RouteHead`, hierarchical typed argument selectors | Deleted. A relation is a bank record; a role is a question; one selector serves both |
| §9 `TypedPlan { routeKind, controllerHandle, argumentHandles[], … }` | `HostSelection { query, record, probability, distribution }` |
| §10 Runtime may cache `RecordState` | No cache exists. `consider` caches one encode for the life of a deliberation, and that is all |
| §11 Physical vocab 256, ~183–200 concepts | 16-bit split token space, 4352 embedding rows, host-supplied `tokenRows` |
| §6.1 ShortConv blocks possible | Plain bidirectional attention + ReLU FFN |
| Out of scope: backward pass, optimizer, training | `teach`, `learn`, the value head and the CPU/GPU backward are the larger half of the engine |
| Not mentioned | The value head / critic, the backend seam, the freezing schedule, checkpoints, sampling |

The through-line: **v2 tried to give the engine a model of the world — types,
catalogs, plans, handles. v3 took all of it out.** What is left is a pointer
machine that knows about records, questions and masks, and a host that knows
everything else.

---

## 13. Open questions

Not profiling details — these are unresolved and they matter.

1. **One selector projection for every question.** `Wq`/`Wk` are shared across
   the relation question and the role questions. "Choose EAT" and "choose the
   apple" train the same weights and may cancel. Per-slot projections are the
   named next step, and the code itself calls the shared pair "a later
   ablation".
2. **Whether the tables can move under `learn` on-policy at scale.** The one
   instability observed was off-policy replay. Unproven either way.
3. **The decision head.** Either wire a route decision or delete 1536 parameters
   and the shaders behind them.
4. **The two band orderings** (§3.4). A latent silent failure.
5. **Whether questions need to condition on each other.** Roles of one relation
   are currently answered independently; nothing lets the patient question see
   what the agent question chose.
