# What pira sends

Two contracts, versioned separately because they change for different reasons.

## Once, at agent creation — `pira-grammar@2`

### symbols

Every concept the world contains. Token ids in classes `object` and above; the
engine owns `system` and `structure` and rejects anything that lands there.

Assignment must be **append-only**. An embedding row is a learned vector indexed
by manifest position, so inserting a symbol in the middle redefines every later
row. This does not have to be perfect — a checkpoint carries the grammar hash and
refuses a grammar it was not trained against — but the choice is between "keep
training" and "start over", and only append-only keeps the first option.

### quantities

Every numeric field the world reports, declared once:

    { field: "size",    kind: "unipolar" }
    { field: "hunger",  kind: "unipolar" }
    { field: "count",   kind: "count" }
    { field: "ripe",    kind: "proportion" }

Kind is a property of the field, not of an observation. Repeating it per record
would let the two disagree between frames.

`signed` additionally names what each direction means, because the engine knows
where the threshold is and not what it signifies:

    { field: "mood", kind: "signed", polarity: { negative: "SAD", positive: "GLAD" } }

### actions

What the creature can do. One binary relation each:

    { relation: "EAT",  object: { accepts: ["apple", "berry"], candidateBands: ["vision"] } }
    { relation: "LOOK", object: { candidateBands: ["vision"] } }
    { relation: "CRY" }

`relation` names the same symbol an observed event uses, and that sharing is what
makes the catalog mean anything: an entry is otherwise an option the creature has
no way to interpret, while a token it has also seen someone else act out carries
a meaning learned by watching.

Omitting `object` declares a unary action whose object mirrors its subject.
Omitting `subject` means the actor. Narrowing nothing admits everything — silence
is not a prohibition.

`accepts` is matched against a record's **tokens**, not against its `schema`,
and a record's schema symbol is simply its first token. One rule therefore
covers both readings: `accepts: ["resource:Apple"]` names an individual, and
`accepts: ["category:Edible"]` names a class — every record carrying that
category is admitted, including one whose kind this world has never shown the
creature. That is what categories are for: prefer them, and a role should
rarely need more than one or two entries. A role may carry at most 16 accepted
tokens, and hitting that ceiling is a sign the world is missing a word rather
than a sign the limit is too low.

Truncation applies here as everywhere: a record holds eight tokens, and its
categories compete for those slots with its reference, its quantity bands and
its own free tokens. A record whose category was cut is a record that role can
no longer admit, so `truncatedRecords` in the response is worth watching.

Declare actions that will often fail. Capability and precondition are
descriptive, never exclusive: a creature that cannot attempt something cannot
learn why it does not work.

### motion

Only for a world with space:

    motion: { radial: { negative: "RECEDING", positive: "APPROACHING" } }

## Every tick — `pira-raw-sensory@2`

    tick, deltaMillis, actorId
    valence        0..1, where 0 is dead
    records[]      what is perceived
    events[]       what happened
    motion[]       what is moving
    selfMotion?    the actor's own movement

### valence

One number for how the actor is doing. How it is computed — from satiation,
warmth, health, whatever the world has — is entirely the simulation's business.

Unipolar rather than signed because death is a floor and an absorbing state:
nothing is as good as death is bad, so a scale symmetric about a neutral point
asserts a symmetry the world does not have. It also spares the simulation from
choosing where "neutral" sits, a calibration that quietly decides whether any
signal exists at all.

Send the level, never its change. The change is what the engine trains on and it
derives that itself.

There is no second comfort channel. Individual comfort dimensions are ordinary
perceptions — unipolar quantities on homeostasis records — and the creature
should perceive "I am hungry" as something distinct from "I am doing badly".

### records

    { band, modality, schema, instanceId?, tokens[], quantities[], count?,
      salience?, observedAt, emptiness? }

`schema` names a grammar symbol. `instanceId` identifies a persistent entity and
is what earns a runtime reference; the **same entity must carry the same
instanceId across senses and across ticks**, or the brain perceives several
things where there is one, and working memory has nothing to hold.

`quantities` carry exact numbers. Never bands: `Size.Medium` and `DIST_NEAR` are
decisions, and they belong to the engine, because a band is a token, a token owns
a trained vector, and a threshold that moved upstream would redefine that vector
without changing a symbol.

`count` lets one record stand for a group, which is how a distant flock stays
inside a bounded band instead of consuming a slot per sheep.

`emptiness` distinguishes the two percepts that are not "nothing was sampled":
`void` is looked-and-found-empty, `unavailable` is the sense not reporting.
Silence should settle a creature; a blocked ear should not.

### events

    { relation, subject, object?, intensity?, salience?, observedAt }

Actions that happened, by anyone. Not outcomes: an outcome would have to say
where its effects end, and dropping a ball on a trampoline has no defensible
answer — the fall, the bounce and the second bounce are equally "the result".
Send that an action occurred; whatever follows arrives as further events.

The actor's own actions are **not** sent back. The engine emitted them and
already knows.

### motion

    { instanceId, radial, angular? }

Signed rate toward the actor, normalized to -1..1. Required because the engine
cannot recover it: distance bands are coarse enough that an animal can cross the
whole of `near` without changing one, and exact positions are not sent because
coordinates are not perceivable while closing distance is.

Send only what actually moved. The temporal band carries change; a world holding
still contributes nothing to it.

## What pira must not send

**Simulation internals.** Patrol phase, spawn ids, child lists. The cost is not
wasted slots: the model learns to use them, is rewarded for predicting movement
it could never have seen, and reaches a world without patrols never having
learned to look.

**Pre-banded values.** See above.

**Outcomes.** See above.

**Undeclared symbols.** Refused, not dropped. A boundary that quietly forgets is
what makes "why can it not see the apple" cost a day.

**Disappearances.** A vanished thing is precisely what is not perceived. The
engine derives it from a stable instanceId missing where it was.

## What krystal returns

    { relation, subject, object, intensity, commitment, intentRef? }

Always a binary relation, both sides populated. `commitment` is how firmly the
network chose; `intensity` is how much of the thing — they are different
quantities and are deliberately not merged.
