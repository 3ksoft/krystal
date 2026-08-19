# What Krystal reserves — NSM semantic primes as a sanity check

## The test

A token belongs to the engine only if it is meaningful **before any world
exists**: no declared entities, no declared senses, no assumption that space or
matter is part of the world at all. Everything else the simulation declares.

NSM's semantic primes (Wierzbicka & Goddard, 65 items in 12 categories) are a
useful cross-check because they were arrived at independently and by a different
method — what is *indefinable* in human languages. Where the two criteria agree,
the case is strong. Where they disagree, NSM is usually keeping something that
presupposes a world, because it describes human language and humans have one.

The primes are a check, not a specification. NSM is a hypothesis about language,
several of its items are contested, and some (`say`, `words`) presuppose
communication a pre-linguistic creature does not need first.

**The engine list is strictly smaller than NSM.** That is the expected result,
not a shortfall.

## Why "indefinable" is the right property to select for

A prime cannot be composed from other primes. Applied here: a reserved token is
one the creature could never *learn* as a combination of tokens it already has.
Everything composable can be acquired, so it can be left to the world to name.

This also gives the curriculum a non-arbitrary success criterion. If a creature
knows the primes and still cannot form a concept it has not seen, then either a
prime is missing or the architecture does not compose — and both answers are
useful. A curriculum that teaches `EAT(apple)` cannot tell you either.

## Verdict by category

`RESERVE` — survives an empty world; the engine owns it.
`WORLD` — presupposes entities, senses, space or matter; the simulation owns it.
`OPEN` — genuinely undecided, listed with the argument on both sides.

### 1. I~me, you, someone, something, people, body, kind, part

| prime | verdict | why |
|---|---|---|
| I~me | RESERVE | There is always exactly one perceiver, and its record slot is fixed by the engine (`BRAIN_FIXED_RECORDS.self`). |
| something | RESERVE | Pure existential. Already present as `SOMETHING`. |
| someone | WORLD | The animate/inanimate split is a fact about a world's contents. |
| you, people, body | WORLD | Require other agents and embodiment to exist. |
| kind, part | WORLD | Taxonomy and composition are world structure. The *algebra* over them (`RELATION_FLAGS.transitive`) is engine-owned; the relation tokens are not. |

### 2. this, the same, other

| prime | verdict | why |
|---|---|---|
| the same | RESERVE | The engine already computes reference identity from handle epochs — it derives this and currently has no token to say it. |
| other | RESERVE | The complement of identity; same argument. |
| this | RESERVE | Deixis into the current focus. The focus band exists regardless of what a world contains. |

### 3. one, two, much~many, little~few, some, all

All **RESERVE**. Counting and quantification need nothing to exist in order to be
defined. Present as `CNT_*` and `Q_*`.

Two observations. NSM lists `one` and `two` as separate primes and then jumps to
`much~many` — independent support for subitizing, but with the cut at **1, 2,
many** where `QUANTITY_BANDS.count` currently uses `[1, 3]`. And NSM has no
`most`: `Q_MOST` is not primitive but is logically definable, so it stays as a
derived reserved token rather than a prime.

### 4. good, bad, big, small

| prime | verdict | why |
|---|---|---|
| big, small | RESERVE | Magnitude bands; present as `MAG_*`. |
| good, bad | RESERVE | Not because valence is world-free, but because the *comfort channel* is engine structure: it is the reward channel and exists whatever the world is made of. What makes a thing good is world knowledge; that there is a signed axis at all is not. |

### 5. think, know, want, feel, see, hear

| prime | verdict | why |
|---|---|---|
| want, don't want | RESERVE | An intent is engine machinery. The creature wanting is what the engine produces. |
| know | RESERVE | The epistemic axis; `UNKNOWN` is its negative pole. |
| see, hear | WORLD | **Senses are not imposed.** A modality is declared, never assumed. |
| feel | WORLD (as a sense) | The comfort channel stays as structure; interoception as a *modality* is declared like any other. |
| think | OPEN | An internal action. Actions are cataloged by the world, which argues WORLD; but "the creature did something with no outward effect" may be structural. |

### 6. say, words, true

| prime | verdict | why |
|---|---|---|
| true | RESERVE | Logical value; present as `TRUE`/`FALSE`. |
| say, words | WORLD | Presuppose communication and other agents. |

### 7. do, happen, move

| prime | verdict | why |
|---|---|---|
| do | STRUCTURAL | Already carried by the subject slot; see below. |
| happen | STRUCTURAL | Same. |
| move | WORLD | Motion presupposes space, and a world need not have any. |

Agency needs no token, because the relation form already encodes it. Under the
reflexive default a unary relation puts the thing itself in the subject slot,
and a transitive one puts the actor there:

    BURNING(candle, candle)    reflexive — it changed by itself   = happen
    BURN(self, candle)         self is the subject                = do

NSM needs two words because a natural language has no structural slot for the
distinction. Asking "did I do it" is asking whether `self` occupies the subject
slot, which is a read of structure rather than of a symbol.

### 8. be somewhere, there is, be someone, is mine

| prime | verdict | why |
|---|---|---|
| there is | RESERVE | Existence. `VOID` is already its negation. |
| be somewhere | WORLD | Space. |
| is mine | WORLD | Possession. |

### 9. live, die

Both **WORLD**.

### 10. time, now, before, after, a long time, a short time, moment

All **RESERVE**. Time is engine structure — the tick advances whatever the world
contains, and `deltaMillis` and `previousObservedAt` are already carried.

The gap here is **duration**: the temporal design carries rate but has no way to
say *how long*. A creature cannot currently think "that lasted a long time".

### 11. place, here, above, below, far, near, side, inside, touch

All **WORLD** — every one presupposes space.

This is the sharpest consequence of the test, and it is worth stating plainly:
`DIST_HERE`/`NEAR`/`FAR` must **not** be engine-reserved. A simulation declares a
distance field, the engine bands it as an ordinary `unipolar` quantity into
`MAG_*`, and "near" as a *concept* is a world token. The engine supplies the
banding, never the spatial reading of it.

### 12. not, maybe, can, because, if, very, more, like

| prime | verdict | why |
|---|---|---|
| not | RESERVE | Logic, and required by the entailment machinery. |
| maybe | RESERVE | Uncertainty, distinct from `UNKNOWN`: "not sure which" and "probably" are different states, and the engine already carries confidence. |
| because | STRUCTURAL / WORLD | The one causal fact the engine can know — this outcome followed that intent of mine — is already a record: `IntentFeedback` carries `intentRef` beside `comfortMagnitude`. Causation in general (fire makes smoke) is world knowledge. |
| very | RESERVE | Intensifier; already `MAG_SEVERE`. |
| more | RESERVE | Comparison of two quantities needs no world. |
| like | RESERVE | Similarity, the graded neighbour of `the same`. |
| can | OPEN | Capability. `capabilityClassToken` exists in the schema with no token, and the candidate mask *is* the engine deciding what is possible — but what a creature can do is mostly world. |
| if | OPEN | Only meaningful once something reasons over it. |

## Engine tokens NSM has no reason to contain

NSM describes language; these are frame mechanics.

`PAD` (structural absence, hard-masked), `BOS`, `EOS`, `BEGIN`, `END`,
`UNAVAILABLE` (a sense not reporting — an engine artifact, and distinct from
`VOID`, which is a percept).

So the reserved block has two halves with different justifications: structural
mechanics, and primes that are indefinable. Worth keeping visibly separate, as
the current layout does not.

## Against what exists today

Present and correct: `PAD`, `BOS`, `EOS`, `TRUE`, `FALSE`, `UNKNOWN`, `BEGIN`,
`END`, `VOID`, `UNAVAILABLE`, `SOMETHING`, `MAG_*`, `CNT_*`, `Q_*`, `NEITHER`.

### Already carried, and therefore not missing

An earlier version of this audit listed ten missing tokens. Most were an error of
method: it asked "is this concept present as a symbol" instead of asking, per
item, whether a **slot** or a **quantity** already carries it. With that question
asked first, the list mostly dissolves.

- `DO` / `HAPPEN` / `BECAUSE` — the subject slot and `intentRef`, as above.
- `SELF` — the subject resolves structurally to the fixed body-band record, so
  "me" is grounded by position rather than by a symbol. That is the stronger
  grounding: a learned SELF vector can drift, a fixed slot cannot.
- `CHANGING`, `LONG_TIME`, `SHORT_TIME`, `MAYBE` — all quantities, and there is
  already a banding machine for quantities. Duration, change magnitude and
  confidence each band to `MAG_*`. "It lasted a long time" is a declared
  duration field, not a new prime.

The general rule, and what makes NSM a check rather than a specification:
**primes are what a language must lexicalize because it has no other slot for
them, and an engine with structural slots needs fewer words.**

### Resolved

**`VANISHED` — reserved, and emitted into the temporal band.**

It is the only member of the change family that needs a token of its own. An
appearance is already legible in a record whose `previousObservedAt` is invalid;
a change of content is a magnitude that bands like any other quantity. Only an
absence has nothing left to carry it, because the record is gone and an absence
cannot be observed in the frame it is absent from.

The engine synthesizes it: a stable instance id present in the previous frame and
missing from this one. The simulation could not send it even in principle — it
reports what is perceived, and a vanished thing is precisely what is not. It
carries maximum salience, since it is the last frame in which noticing is still
possible, and its reference outlives perception, which is what gives working
memory something to keep pointing at.

**`SAME` / `OTHER` — not reserved, and not merely deferred.**

The question turned out not to be perceptual. With stable references, identity of
*referents* is given rather than perceived: two mentions of the same reference
are the same thing by construction, and the engine's attention bias already uses
that. What is left — whether two things are alike in their properties — is not a
percept but a computation, `COMPARE(a, b)`, and computation belongs to the ALU
that this schema explicitly places out of scope.

So NSM's `the same` splits in two for us, and both halves land outside the
reserved vocabulary: one is structural, the other is an operation.

**`NOT` — deferred.**

Neither a quantity nor structural, so it would be a genuine addition. But
`Q_NONE` covers quantified negation and `FALSE` the boolean case, which is enough
until there is inference to run over it.

Wrongly assigned today:

- `SELF` sits in a simulation-owned class.
- `NOMINATIVE` / `ACCUSATIVE` are reserved. Grammatical case presupposes
  language, so by this test they are WORLD, not engine. They earn their place
  only as a mechanism the attention bias reads — which is an implementation
  argument, not a semantic one, and should be recorded as such.

## A correction

An earlier reading held that the engine should own `APPROACHING`/`RECEDING`
because `RawMotionV2` fixes the sign convention ("positive approaches"). Under
this test that is wrong: motion presupposes space, so the whole motion channel is
world-level and optional. A world without space simply does not declare it.

The general rule that survives: **the engine owns polarity only for channels that
are engine structure.** Comfort qualifies, because comfort is the reward channel.
Radial motion does not.

## Ordering is an ABI commitment

Reserved symbols occupy embedding rows `0..N-1` and simulation symbols start at
`N`. Inserting a reserved symbol in the middle therefore shifts every simulation
row and invalidates the embeddings trained against them.

The grammar hash does not protect against this, because the reserved block
changes when *Krystal* is updated rather than when the simulation is. So the
block must be append-only, and a version of its own is worth carrying so a
checkpoint can refuse an engine whose reserved block has grown.
