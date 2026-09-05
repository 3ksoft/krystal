# The creature game — teaching loop concept

Date: 2026-08-17. Design concept agreed before the first implementation slice.
This document fixes the *shape* of the game and what it demands of Krystal. It
deliberately does not specify the lesson GUI.

## Premise

One model is one creature. The game is about **teaching** that creature, not
about operating it. A new creature starts incompetent, and every competence it
acquires is visibly the player's work.

This makes the learning mechanism the game mechanic. Nothing here simulates
learning for narrative effect: the player's lessons are training episodes, the
gradient is real, and the failure modes the player sees are the failure modes
the training harness already reports.

## The loop

```text
day     the creature acts in the world; the player observes
correct the player pauses, rewinds, and says what it should have done
night   the corrections train the creature (supervised)
next    the creature acts again, changed
```

Night is a training pass, not a cutscene. Its cost is bounded and small: at the
current batch-1 step latency (~14.5 ms), a twenty-frame lesson over three
epochs is under a second. Immediate re-observation of a taught behaviour is
therefore part of the interaction, not a load screen.

## Teaching modes

**Retroactive correction (v1).** The player pauses free roam, steps back
through the tape, and labels a decision the creature already made: *this is
what you should have done here*. This is the primary mode. It requires no
authoring skill, it is grounded in a situation the player just watched, and the
substrate already exists — the tape holds every decision window.

**Scene authoring (advanced, later).** The player constructs a situation
deliberately — an apple next to a stone — and states the correct response.
More power, and the only way to teach a *concept* rather than a case, but it
asks the player to understand what they are building.

Both produce the same artefact. A lesson is a `PolicyEpisode`:

```text
{ stage, seed, frames: [{ tick, comfort, resources, gold }] }
```

Retroactive correction cuts one out of the tape; scene authoring builds one by
hand. The curriculum runner already mixes such episodes with replay and
adversarial negatives.

## Mechanics that fall out of the mathematics

These are not designed mechanics. They are properties of the training stack
that happen to be legible, and the game should expose rather than hide them.

**Forgetting.** Teaching only new material degrades old competence. This is
measured, not imagined: the S9 slice moved retention from 89/96 to 93/96 purely
by fixing the replay mixture, three regressions remain visible, and one global
change to S1 sampling diverged the established M-B trajectory to NaN. The
60/30/10 current/replay/adversarial mixture becomes the player's revision
budget.

**Overfitting to lessons.** The creature is scored on situations the player
never authored — held-out layouts, resource ids and record order, exactly as
`evalSeeds` works today. A creature that aces its lessons and fails the yard is
the honest consequence of teaching cases instead of concepts, and it is what
eventually motivates the advanced mode.

**Legible mistakes.** The creature does not merely act wrongly; it points at
what it thought. `EAT#e00` names the record it selected, so the UI can say
*it tried to eat the stone*. Pointer accuracy conditional on intent, and the
invalid-pointer rate, are already computed by the harness. A teaching game
needs the learner's error to be inspectable; here it is structural.

## Starting state

A new creature carries reflexes and nothing else: bad comfort cries, good
comfort laughs. That is the existing S1 contract.

Reflexes matter for readability. A randomly initialised selection head points
at arbitrary records, and arbitrary reads as *broken*, not as *young*. What
reads as naive is systematic error — over-generalisation, fixation on the first
record, literalism that learned the apple and does not transfer to the berry.
Those are the errors the curriculum already produces.

## Out of scope for v1

Deferred deliberately, in this order of likely return:

- **Reinforcement learning.** The night consolidates the player's labels; it
  does not explore. A creature that drifts under its own learning can undo a
  lesson, which is hard to make readable while the player is still learning to
  teach. "The creature begins to learn on its own" is a later progression beat.
- **Curiosity and UNKNOWN.** Genuine not-knowing cannot be supervised into
  existence — labelling `LOOK(x)` whenever the fixture knows `x` is unlabelled
  only relocates the label. It needs a payoff for investigation, so it follows
  RL.
- **Token-selection head, `SET_GOAL`, `SET_SUBGOAL`, `FILTER_SENSORY_BAND`.**
  Arguments that select from the vocabulary rather than from bank records need
  a second head with a different shape. Goals only become meaningful when
  something pays for reaching them.

## What v1 demands of Krystal

1. **Catalog integrity.** `buildFixtureActionCatalog` must hash and count its
   argument descriptors. Until it does, widening the catalog cannot be
   detected by the artifact hash.
2. **A wider grounded action catalog.** `PICK_UP`, `HOLD`, `DROP` and the
   arity-0 body actions are all either zero-arity or a single `ResourceRef`, so
   they need no architectural change — and `PICK_UP`/`HOLD` remove the
   `EAT` → `Pickup` compatibility shim at the Pira boundary.
3. **The tape stores the lowered frame.** Retroactive correction requires the
   gradient to reach exactly the input the creature saw. It cannot be
   reconstructed from the raw snapshot: the host bridge is stateful (a growing
   reference-token map and a memory band with a six-tick lifetime), so
   replaying an earlier tick under a later bridge state yields a different
   frame. Storing the packed `BrainFrame` alongside the selected action and
   pointer makes a lesson exactly reproducible by construction. Budget a few
   kilobytes per decision and bound the retained window.
4. **Per-creature weight persistence.** A creature is a file — currently about
   1.71 M parameters, 6.5 MB in fp32. Most of that is two 4096-row embedding
   tables the fixture vocabulary barely uses; sizing them to the game's
   compiled manifest would shrink a creature considerably.
5. **The night hook.** `recordExperience` and `trainAgent` are already declared
   on the Pira `AgentRuntime` interface and unimplemented on the Krystal side.
   Night is supervised consolidation of the player's labels plus replay.
6. **Held-out evaluation as the score.** The existing train/eval seed split
   becomes the game's measurement of what the creature actually knows.

Items 1, 2, 4 and 5 are Krystal-side. Item 3 spans the boundary; item 6 is
mostly presentation.

## Open questions

- How the lesson GUI presents a correction, and how far back the tape can be
  rewound.
- Whether a creature is shareable, and what that implies for vocabulary
  compatibility between two players' games.
- Whether the day/night boundary is fixed or player-triggered.
