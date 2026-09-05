Below is the complete currently discussed catalog. Arity excludes the implicit actor, lifecycle, `intentRef`, and metadata.

## WorldActionIntent — supplied by Pira/modules

| Intent         | Arity | Arguments             | Notes                                   |
| -------------- | ----: | --------------------- | --------------------------------------- |
| `CRY`          |     0 | —                     | Communicative world action              |
| `LAUGH`        |     0 | —                     | Communicative world action              |
| `WAIT`         |     0 | —                     | Deliberate inactivity                   |
| `REST`         |     0 | —                     | Body action                             |
| `SLEEP`        |     0 | —                     | Durative body action                    |
| `DEFECATE`     |     0 | —                     | Biological action                       |
| `EAT`          |     1 | `target: ResourceRef` | Target must satisfy the action contract |
| `PICK_UP`      |     1 | `target: ResourceRef` | Attempts acquisition                    |
| `HOLD`         |     1 | `target: ResourceRef` | Durative control                        |
| `DROP`         |     1 | `target: ResourceRef` | Releases a held target                  |
| `THROW`        |     1 | `target: ResourceRef` | Releases with impulse                   |
| `MOVE_TOWARDS` |     1 | `target: ResourceRef` | Durative locomotion                     |
| `SPEAK`        |     1 | `utterance: TopicRef` | Physical vocal execution                |

## PerceptualIntent — supplied by Krystal

| Intent                | Arity | Arguments                                  | Notes                      |
| --------------------- | ----: | ------------------------------------------ | -------------------------- |
| `OBSERVE`             |     0 | —                                          | General active observation |
| `LOOK`                |     1 | `target: ResourceRef`                      | Focused visual detail      |
| `SNIFF`               |     1 | `target: ResourceRef`                      | Focused olfactory detail   |
| `TOUCH`               |     1 | `target: ResourceRef`                      | Focused tactile detail     |
| `FILTER_SENSORY_BAND` |     2 | `band: SenseBand`, `predicate: ConceptRef` | Example: `VISION, EDIBLE`  |

`FILTER_SENSORY_BAND` may also accept a compiler-lowered predicate:

```text
FILTER_SENSORY_BAND(VISION, TARGET_OF(EAT))
```

## CognitiveIntent — supplied by Krystal

| Intent        | Arity | Arguments                                      | Notes                                   |
| ------------- | ----: | ---------------------------------------------- | --------------------------------------- |
| `SET_GOAL`    |     2 | `dimension: ConceptRef`, `desiredState: Value` | Example: `SATIATION, UP`                |
| `SET_SUBGOAL` |     2 | `relation: ConceptRef`, `target: ResourceRef`  | Example: `REACHABLE, Apple#12`          |
| `CLEAR_GOAL`  |     1 | `goal: GoalRef`                                | Removes/completes a goal                |
| `BIND_TARGET` |     1 | `target: ResourceRef`                          | Stores the selected target; provisional |
| `THINK_OF`    |     1 | `topic: TopicRef`                              | Activates a memory/topic                |
| `SPEAK_OF`    |     1 | `topic: TopicRef`                              | Selects a topic for communication       |

## Common lifecycle

Lifecycle is not part of arity:

```text
START
MAINTAIN
STOP
RESUME
```

Examples:

```text
START MOVE_TOWARDS(Apple#12)
MAINTAIN MOVE_TOWARDS(Apple#12)
STOP HOLD(Glass#7)
```

`NO_ACTION` is an empty `IntentSet`, not an intent. Deliberate inactivity uses `WAIT()`.
