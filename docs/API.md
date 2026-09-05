# Using Krystal

How a simulation drives a brain. For what the engine does internally, see
[`ENGINE.md`](./ENGINE.md).

---

## 1. The whole surface

```ts
import { BrainSession, QUERY_BAND } from "@krystal/krystal/host";

const brain = new BrainSession({ tokenRows });

const { selections } = await brain.think([
  { schemaId: 1, band: 3, tokens: [APPLE, RED] },
  { schemaId: 1, band: 3, tokens: [STONE, GREY] },
  { schemaId: 9, query: true, tokens: [EAT, PATIENT] },
]);

selections[0].record;       // 0 or 1 — the slot it chose, as you numbered them
selections[0].probability;  // how sure it was
```

Records of tokens in, one chosen record per question out. Everything else —
`teach`, `learn`, checkpoints, running on a GPU — is built on that one call.

> **Import from `@krystal/krystal/host`, not from `@krystal/krystal`.** The
> package index deliberately exports only the model graph. Importing a brain
> must not drag in a world.

---

## 2. Constructing a session

```ts
new BrainSession({
  tokenRows,          // required: Uint32Array, token id → embedding row
  seed?,              // default 1337; deterministic initial weights
  weights?,           // an existing BrainForwardWeights, instead of a fresh one
  config?,            // a different geometry (see §9)
  backend?,           // where to encode; absent means here, on the CPU
})
```

### `tokenRows` is not optional and not incidental

It maps every token id your world uses to the embedding row that holds its
meaning. **A brain and its mapping are only meaningful together**: the same
weights under a different mapping denote something else entirely, and nothing
signals it. A checkpoint therefore carries the mapping and refuses to load
under a different one.

The array must be long enough to index every token id you will ever send, and
every row must be below `config.tokenSpace` (4352 in the default profile).

```ts
// The simplest possible mapping: id N sits in row N.
const tokenRows = Uint32Array.from({ length: 4096 }, (_, id) => id);
```

Assign rows **append-only**. Inserting a symbol in the middle redefines every
later row, and the weights that were learned for them.

An untrained brain is a real brain with opinions worth nothing, not a special
case to code around. The loop can be closed end to end before there is anything
to say about it.

---

## 3. Records

```ts
interface HostRecord {
  schemaId?: number;  // which kind of record this is. Free for you to assign;
                      //   the model only learns to tell them apart. Default 0.
  band?: number;      // an embedding row, 0..7. Default 0.
  query?: boolean;    // this record is a question. Shorthand for band: QUERY_BAND.
  tokens: readonly (number | HostToken)[];   // at most 8
}

interface HostToken {
  token: number;
  role?: number;      // which field this token fills; an ordinary token id
}
```

Rules that are enforced:

- **At most 8 tokens.** A ninth throws `HostFrameError`. Split it into a base
  record and a continuation rather than truncating — a truncated record is a
  different record with no way to tell.
- **A frame with no records throws.** There is nothing to think about.
- **A `PAD` token (id 0) inside a record is simply absent**, not a position.
- **A frame with no query record answers nothing** — `selections` comes back
  empty. So does a frame with no bank records.

Everything else is yours. Krystal says nothing about what a band means, how many
records a world may have, or which of them is perception and which is memory.
The frame is sized to the list you sent.

> **Mark questions with `query: true` or `band: QUERY_BAND`.** Do not reach for
> `v1_0_0.BrainBandKind.query` from the generated types — its ordering disagrees
> with the runtime's, and `6` is the `catalog` band. See ENGINE.md §3.4.

### Relations and roles

A relation is an ordinary record. Its roles are ordinary questions. There is no
special construct for either:

```ts
const frame = () => [
  { schemaId: 1, band: 3, tokens: [ADA] },              // 0
  { schemaId: 1, band: 3, tokens: [BO]  },              // 1
  { schemaId: 1, band: 3, tokens: [APPLE] },            // 2
  { schemaId: 2, band: 3, tokens: [GIVE] },             // 3  the relation itself
  { schemaId: 9, query: true, tokens: [GIVE, AGENT]    }, // question 0: who gives?
  { schemaId: 9, query: true, tokens: [GIVE, RECIPIENT]}, // question 1: to whom?
  { schemaId: 9, query: true, tokens: [GIVE, PATIENT]  }, // question 2: what?
];
```

Three questions, three answers, one shared encode. Cost scales with questions,
and questions are cheap.

**Each role must be named in its own query row.** Two query records carrying the
same tokens are one question asked twice — their answers come back bit-identical
and no amount of training separates them.

---

## 4. Asking

### `think(records, options?)`

One pass: records in, one choice per question out.

```ts
const { selections, frame } = await brain.think(records, options);

interface HostSelection {
  query: number;             // the question's index, in the order you sent them
  record: number;            // the slot it chose, in YOUR numbering
  probability: number;       // how much mass was on it
  distribution: Float32Array;// over the whole bank, in slot order
}
```

`distribution` is kept because a selection is only as trustworthy as the
distribution it came from — an argmax at 0.02 and one at 0.9 are different
answers to the same question.

### `consider(records)` → `choose(options?)`

Split the pass when you want to ask the same frame more than one thing.

```ts
const deliberation = await brain.consider(records);   // expensive, once
const a = deliberation.choose({ allows: mayEat });    // cheap, synchronous
const b = deliberation.choose({ allows: mayCarry });  // cheap, synchronous
```

The encoder and the mixer depend only on the *records*; the mask depends only on
the *question*. Measured in a simulation over sixty ticks: `consider` took 20 s,
every `choose` together took 0.17 s. A creature deciding "eat what?" after
deciding "eat" is looking at the same world through a different grammar, and
re-encoding it would spend most of the cost of thinking to arrive where it
already was.

`choose` is synchronous even when the frame was encoded on a device — the
encoded frame is already back.

### `allows` — your grammar

```ts
allows?: (query: number, record: number) => boolean
```

Called per (question, record). `false` removes that record from that question's
distribution entirely: probability exactly zero, never chosen, never sampled.

This is where your grammar lands, and it stays yours. Krystal applies the answer
and never asks why. Affordances, type compatibility, "must be visible", "must
not be the mother" — all of it belongs here, not in the engine.

Two behaviours to design around:

- **A question nothing admits comes back open**, as a normal distribution over
  the bank, not as a uniform over the impossible. Check for that case yourself
  if "no legal answer" is meaningful in your world.
- `record` is your slot number, so you can index straight back into the array
  you sent.

### `sample` — where behaviour comes from

```ts
sample?: (query: number) => number   // one uniform in [0, 1) per question
```

Without it, `choose` takes the argmax — which is the right answer to "what does
this policy believe" and the wrong answer to "what does this creature do". **A
deterministic policy has no behaviour to reinforce**: it emits one action per
frame, gets one outcome, and has nothing to compare it against. An untrained
creature will repeat the same act for as long as anyone watches.

Sampling is motor noise. An untrained selector is nearly uniform so it tries
everything, and as learning sharpens the distribution the exploration decays on
its own. There is no temperature to schedule and no epsilon to decay.

The generator is **yours** — reproducibility belongs to whoever owns the run, not
to the brain being replayed.

```ts
const rng = seeded(12345);
await brain.think(records, { allows, sample: () => rng() });
```

---

## 5. Teaching — what *can* be said here

```ts
await brain.teach([{ records, gold: [0], forbidden: [1], allows }], options?)
```

A demonstration is not "this went well", it is "this is what one does". No
reward, no baseline, no critic — the value head is provably untouched.

```ts
interface HostDemonstration {
  records: readonly HostRecord[];
  gold: readonly (number | undefined)[];        // slot each question should have chosen
  forbidden?: readonly (number | undefined)[];  // slot each question should NOT choose
  allows?: (query: number, record: number) => boolean;
}
```

- Index by **question**, value is a **slot**. `undefined` means this
  demonstration says nothing about that question.
- `forbidden` pushes down without naming a replacement. Where the mass goes is
  up to the rest of the distribution.
- A question with both **keeps its gold**. What to do says more than what not to.
- Show it under the same `allows` a real choice would have been made under.

### Options

| Option | Default | Note |
|---|---:|---|
| `learningRate` | `0.2` | |
| `unfreezeTokens` | `true` | **Leave this on.** Frozen, 200 showings move the policy `0.498 → 0.510`. Unfrozen, it reaches `0.996` in fifty. Teaching with frozen tables is a no-op that looks like slow teaching. |
| `tokenRate` | `0.5` | how fast the tables follow the selector |
| `maxUpdateAbs` | `0.05` | largest move any weight may make; the update is scaled whole, direction kept |
| `decay` | `1e-3` | cross-entropy's fixed point is at infinity; this is what makes it settle |
| `maxParameterAbs` | `1` | a showing that still reaches this is rolled back and counted |

### Report

```ts
{ framesSeen, shown, meanAgreement, maxParameterAbs, rejected, clipped }
```

**`meanAgreement` is the measurement.** How much probability the policy already
put on what it was being shown, *before* this pass changed anything. It starts
near `1/candidates` and should climb. `rejected` above zero means a gradient came
back non-finite; `clipped` consistently at the batch size means you are teaching
at the cap — slower than asked, and stable.

```ts
let report = await brain.teach(lessons);
while (report.meanAgreement < 0.95) report = await brain.teach(lessons);
```

---

## 6. Learning — what is *worth* saying

```ts
await brain.learn([{ records, chosen: [0], allows, reward: 0.4 }], options?)
```

REINFORCE with the value head as baseline.

```ts
interface HostExperience {
  records: readonly HostRecord[];
  chosen?: readonly (number | undefined)[];  // the slot each question actually chose
  allows?: (query: number, record: number) => boolean;
  reward?: number;                           // the change in how the creature was doing
}
```

Four rules that decide whether this works at all:

1. **`records`, `chosen` and `allows` must describe the same event `think` did.**
   Same records, same order, same numbering, same predicate. A different mask
   here computes the gradient of a distribution the creature never sampled from.
2. **`reward` is a *change*, not a level.** It is the difference in the
   creature's well-being that followed this turn.
3. **Omit `reward` when there is nothing to difference against** — the first
   frame of a life. The frame is then skipped rather than counted as a zero,
   which would be a claim that nothing happened. (It still teaches the critic if
   you pass one.)
4. **Batches want variety.** A batch where everything went equally well does not
   push at all: nothing in it says *which* choice was responsible. This is a
   guard, not a bug.

### Options

| Option | Default | Note |
|---|---:|---|
| `learningRate` | `0.05` | |
| `policyScale` | `1` | actor step relative to the critic's |
| `advantageClip` | `3` | in batch deviations |
| `maxParameterAbs` | `Infinity` | set one if you unfreeze anything; treat a rejection as a diagnostic |
| `unfreeze.tokens` | `false` | the token + field tables. The only lever that can make two questions look different |
| `unfreeze.tokenRate` | `0.1` | |
| `unfreeze.pool` | `false` | an explicit experiment, not the next stage |

### Report

```ts
{ framesSeen, meanAdvantage, meanValueLoss, reinforced, discouraged,
  meanEntropy, meanConfidence, updateApplied, rejected?, health }
```

**Watch `meanEntropy`.** 0..1, normalized against guessing uniformly among the
same records. `1` is a creature choosing at random, and falling entropy is the
only visible sign that anything has been learned — reward can improve because
the world got easier, and an argmax looks the same however sure of itself it is.

`updateApplied: false` with `rejected: "non-finite" | "parameter-limit"` means
the whole update was rolled back as one transaction: representation, actor and
critic together. The brain is exactly as it was.

### Whether to unfreeze

Start frozen. Unfreeze the tables when you need two questions about one relation
to pull in opposite directions — frozen, they cannot, and the shape of that
failure is documented in ENGINE.md §7.4.

When you do:

- Set a `maxParameterAbs` and treat rejections as a signal, not noise.
- **Pick `tokenRate` for your geometry.** `0.5` suits a reduced test geometry;
  on the full profile it is too hot (107 of 500 fresh batches rolled back, both
  questions collapsed onto one record). `0.1` is the working value.
- **Sample fresh batches from the current policy.** Replaying one batch after
  the policy has moved is off-policy and eventually keeps crediting an action
  the current policy would never draw. That, not unfreezing, was the source of
  the one runaway anyone measured.

```ts
for (let round = 0; round < rounds; round++) {
  const batch: HostExperience[] = [];
  for (let sample = 0; sample < 8; sample++) {
    const records = frame();
    const chosen = (await brain.think(records, { allows, sample: () => rng() }))
      .selections.map((s) => s.record);
    batch.push({ records, chosen, allows, reward: outcomeOf(chosen) });
  }
  await brain.learn(batch, { learningRate: 0.05, unfreeze: { tokens: true, tokenRate: 0.1 } });
}
```

---

## 7. Checkpoints

```ts
const bytes = brain.snapshot();              // Uint8Array

const refusal = brain.restore(bytes);        // null when taken up
if (refusal) console.error(refusal);
```

The blob carries the weights, the geometry they were shaped for, and the
`tokenRows` they learned their meanings under. A mismatch is **refused**, not
coerced:

| Refusal | Meaning |
|---|---|
| `"not a krystal checkpoint"` | wrong magic |
| `"a later checkpoint format"` | written by a newer version |
| `"a different geometry"` | hidden size, FFN, block or head counts, route kinds |
| `"a different vocabulary"` | `tokenRows` differs in length or content |
| `"truncated"` | short file |

`restore` writes **in place**. Anything already holding `brain.weights` is now
training the brain the creature is thinking with, which is the point.

---

## 8. Running on a GPU

```ts
import { BrainSession } from "@krystal/krystal/host";
import { gpuBackend, createWebGpuDevice } from "@krystal/webgpu";

const { device } = await createWebGpuDevice();
const brain = new BrainSession({ tokenRows, seed: 7, backend: gpuBackend(device) });

// ... identical from here on ...

brain.destroy();   // release what the backend holds
```

**Nothing else about the session changes.** `think`, `consider`, `choose`,
`learn`, `teach`, checkpoints — all still run on the host, on the host's own
arrays, with the host's own closures. What moves is the arithmetic: the encode
as one submit with one readback, and the backward as one submit with one
readback of the gradients the update consumes.

The parity tests assert this at the surface, not just per operator: the same
records, the same grammar closure and the same seed give the same choice, and
the same batch leaves the same brain.

Notes:

- Pass a device you already have if you have one, so the creature and the
  picture of it share a device instead of competing for two.
- **Call `destroy()`.** The CPU path has nothing to release; the device path
  does.
- Device ceilings: 1536 active tokens, 432 record slots, 128 questions. Note the
  first — 432 × 8 is 3456, so **active tokens bind before slots do**.
- If you change arena regions or the shader list, run `bun run build:webgpu`.
  The capacity lives in the serialized artifact, and without a rebuild you get
  `copy range does not fit in krystal.arena` at runtime rather than a compile
  error.

---

## 9. A different geometry

```ts
import { BRAIN_FORWARD_CONFIG } from "@krystal/krystal";

const small = {
  ...BRAIN_FORWARD_CONFIG,
  hiddenSize: 32, headCount: 1, headDim: 32,
  ffnSize: 64, encoderBlocks: 1, mixerBlocks: 1,
};
const brain = new BrainSession({ tokenRows, config: small });
```

Useful for tests and assays — the reduced geometry exercises the same six tables
and the same update path at a fraction of the cost. It is not a checkpoint-
compatible brain: geometry is part of what a checkpoint checks.

`recordWidth` (8) is **not** a config parameter. It is the model's shape.

---

## 10. The default profile at a glance

| | |
|---|---:|
| Hidden `H` / heads / FFN | 128 / 4 × 32 / 384 |
| Encoder + mixer blocks | 2 + 2 |
| Record width | 8 tokens |
| Parameters | 1 773 952 (6.77 MiB f32) |
| Embedding rows | token 4352 · field 4352 · schema 256 · band 8 · stream 2 · pos 8 |
| Trainable by default | selector `Wq`/`Wk` (32 768), value head (384), token+field rows via `teach` |
| Frozen | encoder, mixer, pool, schema/band/stream/pos tables |

---

## 11. Common mistakes

| Symptom | Cause |
|---|---|
| `selections` is empty | No record marked `query`, or no non-query records to choose from |
| Two questions always answer identically | Their query rows carry the same tokens. Name the role |
| Teaching does nothing | `unfreezeTokens: false`. It is on by default; something turned it off |
| Learning never moves the actor | Every `reward` in the batch is the same, or no `chosen` was supplied, or `reward` was omitted |
| `HostFrameError: record N carries 9 tokens` | Split it into a base record and a continuation |
| The GPU disagrees with the CPU | A different `allows` between `think` and the update, or a stale artifact — rerun `bun run build:webgpu` |
| A choice lands on the one act nothing may do, repeatedly | NaN in the weights. `sampleRow` returns `-1` for an all-NaN row; check `health.finite` from the last `learn` |
| Questions filed in the wrong band | `v1_0_0.BrainBandKind.query` is `6`; the runtime's is `5`. Use `QUERY_BAND` |

---

## 12. Development

```bash
bun install
bun run build          # schema codegen, then the WebGPU artifact
bun test packages      # 78 tests; the GPU ones need Dawn and run headless
```

`bun run build:schema` alone after touching
`packages/schema/src/krystal-engine-schema.ts`; `bun run build:webgpu` alone
after touching arena regions or the shader list.
