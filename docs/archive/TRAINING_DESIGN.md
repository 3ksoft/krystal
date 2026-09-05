# Krystal training design

Status: high-level design draft  
Companion implementation document: `WEBGPU_BACKWARD_PLAN.md`  
Scope: training contract for the brain engine; simulation authoring remains separate

## 1. Purpose

This document describes how the Krystal brain should be trained once the engine ABI, frame geometry, record schemas and `ActionIntent` catalog are compiled.

It covers:

- exposure of the static `ActionIntent` catalog;
- supervised targets produced from exact world state;
- Creator's Touch tutorials and curriculum progression;
- negative and counterexample generation;
- pointer, query, memory and unordered intent-set objectives;
- the WebGPU backward boundary;
- optimizer and checkpoint policy;
- evaluation and versioning.

It does not define world physics, materials, bodies, scenes or tutorial rendering. Those belong to the simulation and scene compiler. Training consumes their compiled frames, exact bindings, transitions and outcomes.

## 2. Governing principle

```text
latent world + compiled engine contract
                  ↓
        deterministic episode
                  ↓
 frames + exact bindings + transitions + tutorial annotations
                  ↓
       learned interpretation and intent selection
```

The runtime computes everything that can be exact:

- frame layout and record schemas;
- reference identity and lifetime;
- typed candidate sets;
- exact dereference;
- action signature validation;
- world transition and physical outcome;
- tutorial ground truth;
- structural equality and exact arithmetic.

The model learns what is genuinely semantic or selective:

- which record matters;
- which compatible reference is intended;
- which concept/property explains an observation;
- which action or concurrent intent set serves the current state;
- which topic should be looked at, remembered, thought of or spoken of;
- which experiences are salient;
- which consequences are likely.

Design rule:

> The compiler narrows the legal language. The world teaches whether a legal idea was useful, successful, awkward, surprising or disastrous.

## 3. Training sample contract

One training sample is derived from an exact episode transition rather than authored as a loose prompt/answer pair.

Conceptually it contains:

```ts
interface BrainTrainingSample {
  frame: BrainFrame;
  catalogVersion: number;

  queries: QueryTarget[];
  pointers: PointerTarget[];
  intentSet?: IntentSetTarget;
  memory?: MemoryTarget;
  predictions?: OutcomeTarget[];

  transition?: WorldTransitionMetadata;
  tutorial?: TutorialTargetMetadata;
  difficulty: DifficultyMetadata;
  provenance: SampleProvenance;
}
```

This is a conceptual contract. The final schema-pop representation should use packed buffers and compiled offsets rather than host objects.

Every sample must be reproducible from at least:

```text
engine ABI version
frame-layout version
record-schema manifest hash
token-vocabulary version
ActionIntent catalog hash
simulation/scenario version
curriculum generator version
seed
```

## 4. ActionIntent catalog exposure

### 4.1 The catalog is a compiled capability bank

`ActionIntent` is a static catalog compiled from engine schemas. Each entry describes one semantic action shape:

```text
intentId
action token
semantic intent token
domain
actor schema
typed argument descriptors
effect class
capability/precondition hints
preferred controller role
durative flag
```

The catalog is not a list of currently possible actions and is not a hard-coded policy. It is the vocabulary of intentions that the brain may propose.

### 4.2 Do not repeat the full catalog in every frame

The catalog should be exposed as a model-side bank, analogous to static learned keys/values:

```text
ActionIntent descriptor
        ↓ catalog encoder / embeddings
intent key + intent value + argument-head descriptors
        ↓
action decoder queries the bank from the current BrainState
```

Because the catalog changes only when the engine contract changes, its encoded representation may be cached. The ordinary 1024-token brain frame should contain current state, perception, memory, queries and feedback—not a repeated serialization of all action definitions.

The catalog manifest and its hash remain explicit model inputs/configuration so a checkpoint cannot silently run against a different action ABI.

### 4.3 Candidate masks

Hard masks are generated only from exact structural incompatibility:

- wrong argument kind;
- incompatible record/schema type;
- reference unavailable in the permitted current or historical scope;
- malformed lifecycle transition;
- missing required argument;
- catalog entry unavailable in this engine build.

Do not hard-mask:

- occupied hands;
- awkward posture;
- excessive load;
- competing concurrent goals;
- probable failure;
- low expected reward;
- physically creative use of a body part or object.

Those remain attemptable. The motor/simulation layer produces `accepted`, `partial`, `failed`, `cancelled`, dropped-object and comfort/world consequences.

The mask therefore means:

```text
can be expressed and resolved safely
```

not:

```text
is a sensible thing to do
```

### 4.4 Typed argument selection

Each reference argument is selected through the same soft-gather mechanism used elsewhere:

```text
argument query
  -> ABI-derived compatible record/field candidates
  -> masked selector logits
  -> softmax
  -> weighted gathered representation during training
  -> exact selected record/field/handle at runtime
```

The runtime never asks the network to synthesize an arbitrary exact handle. The network selects a compatible location; the runtime resolves the sidecar binding exactly.

## 5. Unordered concurrent IntentSet

The output is an unordered set of desired effects, not one exclusive action lane per manipulator.

Multiple proposals may:

- use different controllers;
- overlap in controller preference;
- compete for the same hand or leg;
- maintain an existing durative intent;
- introduce a new intent;
- stop or resume another intent;
- fail jointly even though each is individually expressible.

### 5.1 Training target

An `IntentSetTarget` contains zero or more gold proposals:

```text
lifecycle
intentId
typed arguments
purpose/topic references
activation
priority
persistence
optional expected effect/outcome
```

Gold proposal order has no meaning.

### 5.2 Set matching

During training, match predicted proposal slots to gold proposals using minimum-cost bipartite matching. Matching cost can combine:

```text
intent classification cost
argument pointer cost
lifecycle cost
topic/purpose cost
```

After matching:

- matched slots receive intent and argument losses;
- unmatched predicted slots receive `no proposal`/objectness loss;
- unmatched gold proposals count as misses;
- proposal transport slots are shuffled during training.

This avoids inventing semantic lanes such as `left-hand action slot` and preserves creativity in controller assignment.

The first curriculum stages may use at most one intent. Multi-intent matching should be introduced only after single-intent selection and argument binding are stable.

## 6. Creator's Touch curriculum

Creator's Touch is implemented as deterministic, game-like tutorials generated over the same frame and action ABI used during ordinary episodes.

The Creator is a trusted instructional source, distinct from normal simulated speakers.

A minimal lesson may execute:

```text
reset comforts to neutral
present sensory noise/background
ask: WHAT IS THIS?
present red Apple#12
say: THIS IS RED APPLE
hold for several frames
demonstrate or request LOOK Apple#12
expose focused details
say: APPLE IS EDIBLE
demonstrate EAT Apple#12
show world reaction and satiety change
reset sensory input and working episode state
assess with a new apple instance
```

The tutorial compiler owns scene cues and exact targets. The engine sees ordinary frames, queries, intents and feedback; there is no separate magical neural interface.

### 6.1 Suggested progression

#### T0 — frame and response discipline

- distinguish input bands and query records;
- produce `no proposal` when no action is requested;
- learn stable record/field roles under within-band shuffling;
- ignore deterministic sensory noise that carries no task information.

#### T1 — object identity and properties

- recognize object/category tokens across instances;
- retrieve color, size, shape and sensory properties;
- distinguish `Apple#12` from concept `APPLE`;
- answer with token, boolean or pointer targets.

#### T2 — active perception

- `LOOK(ref)` reveals visual detail;
- `SNIFF(ref)`, `TOUCH(ref)` and other modalities expose their own fields;
- learn when additional information is useful;
- do not reward repeated inspection after information gain reaches zero.

#### T3 — simple actions and consequences

- select one `ActionIntent`;
- bind typed actor/object arguments;
- observe `accepted/succeeded/partial/failed` feedback;
- associate actions with comfort and world-state changes.

#### T4 — memory, topic and familiarity

- `THINK_OF(ref)` gathers from working memory;
- `SPEAK_OF(ref)` projects a current thought/topic into communication;
- relevant interactions increase activation/familiarity;
- stale memories remain observations, not current world truth;
- repeatedly used object instances may become autobiographically familiar.

#### T5 — durative and concurrent intents

- `start`, `maintain`, `stop`, `resume`;
- carrying while looking or speaking;
- competing goals and partial execution;
- surprise may weaken `HOLD` and produce a drop;
- a purpose may decay or be forgotten while an action chain continues.

#### T6 — abstraction and counterexamples

- varied instances precede universal claims;
- `small zebra striped`, `large zebra striped` support `ALL ZEBRA STRIPED`;
- red and green apples refute `ALL APPLE RED`;
- distinguish `TRUE`, `FALSE` and `UNKNOWN`;
- when false, optionally select a counterexample pointer.

#### T7 — open episodes

- mix previously learned competencies;
- reduce direct Creator narration;
- preserve exact world-derived probes;
- introduce novel object/action combinations and longer trajectories.

The stages are scheduling groups, not new ABI modes. A tutorial and an ordinary episode must exercise the same model interfaces.

## 7. Sample generation

One latent episode should produce many compatible views:

```text
current-state query
property query
pointer query
active-perception decision
action demonstration
inverse action question
consequence prediction
before/after comparison
counterexample probe
memory probe
intent lifecycle target
```

Ground truth comes from the scenario compiler and runtime transition, never from guessing the rendered narration.

Surface language, if used, is generated only after the semantic target exists. Multiple utterance forms may map to the same Krystal representation.

## 8. Negative sampling and counterexamples

Negative sampling must make the selection problem difficult without teaching false facts.

### 8.1 Pointer negatives

Use hard negatives drawn from parser/compiler-compatible candidates:

- same schema, different instance;
- same entity, wrong property field;
- same role, wrong temporal snapshot;
- familiar object versus visually similar new object;
- current perception versus stale memory of the same entity.

Candidates rejected by the type system are useful for mask tests, but they are weak learning negatives because the model never needs to score them.

### 8.2 Intent negatives

Useful intent distractors include:

- compatible action with the wrong semantic effect;
- correct action with the wrong object reference;
- correct intent but wrong lifecycle (`start` versus `maintain`);
- plausible alternative satisfying another active comfort;
- action that was useful in a similar earlier state but not in this state.

An action that merely fails physically is not automatically a semantic negative. It may be the correct attempted intent with a `failed` outcome.

### 8.3 Property and universal negatives

Generate negatives from known world facts:

- substitute one observed property;
- swap references between compatible records;
- construct an explicit counterexample instance;
- ask universal questions over complete, declared domains.

Absence of evidence must map to `UNKNOWN`, not automatically `FALSE`.

### 8.4 Consequence negatives

For transition prediction, pair an executed intent with:

- its exact observed outcome;
- plausible but non-occurring outcomes;
- outcomes from a similar action on another material/object;
- success versus partial/failure variants.

Do not leak the answer through fixed record order, reference number, proposal slot, scene seed or tutorial timing.

### 8.5 Negative ratios

Start with all candidates when the sets are small. Sampling is an optimization for large sets, not a semantic requirement.

When sampling is needed:

```text
easy random negatives
hard type-compatible negatives
counterfactual negatives
temporal/stale negatives
```

must be tracked separately in metrics. Hard-negative proportion should rise only after basic grounding is stable.

## 9. Training objectives

The total loss is a weighted collection of typed objectives:

```text
L =
  w_query      * L_query
+ w_pointer    * L_pointer
+ w_intent     * L_intent_set
+ w_argument   * L_arguments
+ w_lifecycle  * L_lifecycle
+ w_memory     * L_memory
+ w_outcome    * L_outcome
+ w_aux        * L_auxiliary
```

Not every sample enables every term. Loss masks are derived from available gold annotations.

### 9.1 Query/value loss

Use the typed head appropriate for the probe:

- categorical cross-entropy for tokens/intents/status;
- boolean or `TRUE/FALSE/UNKNOWN` classification;
- regression only for genuinely continuous semantic projections;
- exact numeric payloads remain runtime/ALU targets, not lossy neural regression.

### 9.2 Pointer loss

For compatible candidates `C`:

```text
p = softmax(masked selector logits over C)
L_pointer = -log p[gold]
gathered = sum_i p[i] * value[i]
```

Soft gather remains in the differentiable training path. Runtime execution uses selected record/field plus exact sidecar handle resolution.

### 9.3 Intent-set loss

After bipartite matching:

```text
objectness/cardinality loss
intentId classification loss
argument pointer/value losses
lifecycle classification loss
optional activation/priority/persistence losses
```

The first version should avoid over-weighting continuous intent metadata. Correct semantic intent and argument bindings matter more than perfectly matching an oracle's arbitrary priority number.

### 9.4 Memory loss

Memory management is partly runtime-defined and partly learned/conditioned.

Possible supervised targets:

- retain/refresh/allow-eviction ranking;
- topic relevance;
- source and observation-time retrieval;
- distinction between current fact and remembered observation.

Do not supervise ownership/familiarity with a special `MY_STICK` class. Familiarity should emerge from repeated interaction history and memory features.

### 9.5 Outcome prediction

An auxiliary head may predict:

- accepted/partial/failed/succeeded;
- effect class;
- comfort delta band;
- progress/outcome magnitude band;
- likely information gain.

This teaches affordances without turning the predictor into a hard runtime gate.

## 10. Curiosity and behavioral learning

Curiosity/engagement is a homeostatic signal, not a reward for arbitrary motion.

Suggested transition rule:

- lack of meaningful intent gradually lowers curiosity comfort;
- merely emitting an action may pause decay but does not restore comfort;
- novelty, information gain or progress restores it;
- repeating a known action-object-outcome tuple yields diminishing or zero gain;
- deliberate `WAIT`, `REST`, `SLEEP` and `OBSERVE` remain valid intents distinct from empty output.

Early supervised curriculum should teach these distinctions directly. Reinforcement learning is not required for grounding, reference binding or basic action selection.

RL may be added later for preferences among multiple valid strategies after the supervised world model and intent interface are stable.

## 11. Scheduling and replay

Each sample carries difficulty metadata such as:

```text
competencies
entity count
candidate count
distractor count
temporal depth
number of concurrent intents
memory pressure
sensory-noise level
novelty
required reasoning steps
```

Initial scheduling is staged but not strictly one-way:

```text
mostly current stage
+ replay of mastered stages
+ small preview of the next stage
```

Promote a competency based on held-out performance, not merely training steps. Preserve replay of old competencies to detect catastrophic forgetting.

Later, adaptive replay may prioritize:

- recent failures;
- high-entropy pointer decisions;
- forgotten older competencies;
- rare intent/argument combinations;
- counterexamples for overgeneralized rules.

## 12. WebGPU backward boundary

The training runtime uses a static, explicitly reversed WebGPU program. It does not record a dynamic autograd graph.

Each differentiable operation declares conceptually:

```text
forward shader
backward shader(s)
saved tensors or recomputation inputs
gradient outputs
trainable parameters
```

The first implementation milestone is specified in `WEBGPU_BACKWARD_PLAN.md`:

```text
embedding
-> f32 linear
-> cross entropy
-> linear backward
-> embedding backward
-> SGD
```

After correctness is established, extend in operator order:

```text
residual
SiLU/gating
RMSNorm
ShortConv
QK normalization + RoPE
attention
record encoder/mixer
soft gather and typed heads
```

Training data, activations, gradients, weights and optimizer state remain GPU-resident during `trainStep`. CPU readback is limited to telemetry and explicit diagnostic tests.

## 13. Parameter policy

Use explicit parameter groups:

```text
frozen pretrained backbone
trainable record encoder/mixer additions
trainable catalog/action heads
trainable pointer/query heads
optional adapters
```

Packed WQ4 tensors are not updated directly.

Recommended initial path:

1. verify backward with tiny f32 weights and SGD;
2. freeze the WQ4 backbone;
3. train new f32/f16 heads and architecture additions;
4. add backward-input through frozen blocks when required;
5. only later consider unquantized master weights or QAT.

Every checkpoint records its trainable/frozen parameter manifest.

## 14. Optimizer plan

### O0 — correctness

```text
precision: f32
optimizer: SGD
momentum: none
weight decay: none
fusion: none
```

Purpose: prove gradients and updates, not achieve production convergence.

### O1 — practical supervised training

```text
trainable weights: f16 or f32 according to kernel support
master/update state: f32
gradient accumulation: f32
optimizer: AdamW
optional global gradient norm clipping
warmup + simple decay schedule
```

Use parameter groups for backbone/adapters/heads and exclude parameters from weight decay when semantically appropriate, such as normalization scales and biases.

Do not fuse optimizer logic until unfused updates match a CPU oracle.

### O2 — memory/performance

Only after correctness and convergence:

- activation checkpointing and recomputation;
- lifetime-based arena reuse;
- fused safe elementwise operations;
- mixed precision with explicit loss scaling if required;
- gradient micro-batching;
- profiling-driven partial reductions.

## 15. Evaluation

Training loss is insufficient. Track metrics per competency and difficulty bucket.

Core metrics:

```text
typed query accuracy
pointer top-1 accuracy
pointer entropy and calibration
intentId accuracy
argument pointer accuracy
IntentSet precision/recall/F1
intent cardinality accuracy
lifecycle accuracy
type-valid output rate
TRUE/FALSE/UNKNOWN calibration
counterexample selection accuracy
outcome/status prediction accuracy
memory current-vs-stale accuracy
```

Behavioral evaluation additionally tracks:

- success, partial and failure distributions;
- information gain versus repetitive action spam;
- retention under memory pressure;
- recovery after forgotten goals;
- novel controller/action combinations;
- whether hard masks reject only structural invalidity.

Physical success rate must not be the sole action metric. Otherwise the model is trained toward immobility and conservative repetition.

## 16. Holdouts and leakage control

Required holdouts include:

- unseen object instances and reference assignments;
- within-band record permutations;
- new property combinations;
- new actor/object/action combinations;
- novel but type-compatible tool use;
- longer temporal chains;
- higher memory pressure;
- unfamiliar concurrent intent combinations;
- counterexamples to frequent correlations;
- new tutorial surface wording;
- new world seeds and scene layouts.

Dynamic `0xExx` references must be rebound/randomized so a slot never acquires permanent semantics.

Proposal slots, irrelevant record order and harmless tutorial timing must also be randomized. A semantic signature should detect train/test duplication even when surface frames differ.

## 17. Checkpoints and compatibility

A training checkpoint must store or identify:

```text
architecture version
engine ABI version
frame layout version
record schema hash
vocabulary version
ActionIntent catalog hash
curriculum/generator version
parameter manifest
optimizer type and state
global step
random seed/state where applicable
evaluation summary
```

A catalog or layout mismatch is a compatibility error unless an explicit migration exists.

Optimizer state is optional for inference exports but required for resumable training checkpoints.

## 18. Initial milestones

### M0 — GPU gradient proof

- complete `WEBGPU_BACKWARD_PLAN.md` milestone 1;
- overfit a tiny deterministic token mapping;
- match CPU and finite-difference gradients.

### M1 — typed perception/query training

- train record encoder/mixer and typed query heads;
- reproduce pointer/soft-gather results on shuffled records;
- validate exact runtime lowering.

### M2 — single ActionIntent

- expose cached catalog bank;
- train intent classification and typed argument binding;
- include success, failure and consequence feedback;
- no multi-intent output yet.

### M3 — Creator's Touch foundation

- object/property tutorials;
- active perception;
- simple action demonstrations;
- deterministic assessment and counterexamples.

### M4 — memory and durative behavior

- working-memory relevance targets;
- `THINK_OF` and `SPEAK_OF`;
- intent lifecycle and feedback;
- familiarity and stale-observation tests.

### M5 — unordered concurrent IntentSet

- set matching;
- shuffled proposal slots;
- overlapping controllers;
- partial/failure outcomes and surprise-driven drops.

### M6 — open-world curriculum

- mixed competencies;
- adaptive replay;
- broader hard negatives;
- novel type-compatible strategies;
- optional later policy optimization.

## 19. Decisions deliberately left open

The following should be determined experimentally rather than frozen now:

- exact catalog encoder architecture;
- intent-set matching cost weights;
- loss weights and curriculum thresholds;
- number of proposal slots used during early training;
- memory supervision strength versus runtime heuristics;
- attention/ShortConv checkpoint boundaries;
- mixed-precision format;
- AdamW hyperparameters;
- when or whether RL becomes useful.

The stable contracts are the typed interfaces, exact/runtime boundary, unordered intent semantics, reproducible curriculum provenance and GPU-resident training step.

## 20. Core training hypothesis

Krystal already exposes typed structure that an ordinary language model must rediscover from token order. Training should exploit only the structure known exactly to the compiler while leaving semantic choice, attention, memory relevance and creative intent composition learnable.

The intended result is neither a scripted agent nor an unconstrained token generator:

```text
exact machine envelope
+ learned grounded selection
+ fallible physical execution
+ persistent but limited memory
+ curriculum-derived concepts
```

That is the boundary the training system should preserve.
