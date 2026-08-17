# Krystal in the Pirapitinga simulation

The current development bridge runs Krystal as a local WebGPU/Dawn service on
port `8801`. Pirapitinga's browser simulation sends a complete
`pira-raw-sensory@1` snapshot and its compiled vocabulary manifest; the service
returns the existing Pira `AgentActionIntent` shape.

Start the policy service:

```bash
cd /home/kr/Projects/krystal
bun run pira:serve
```

The boot policy trains the proven S9 + S1-S8 replay slice in memory. On the
development machine this currently takes about 20 seconds. Wait for:

```text
[krystal] Pira service listening on http://127.0.0.1:8801/
```

Then start Pirapitinga and open the Village simulation:

```bash
cd /home/kr/Projects/pirapitinga
bun run dev
```

Each decision is visible in the service terminal, for example:

```text
[krystal] tick=4 agent=child prediction=EAT#e00 intent=Pickup
```

`GET http://127.0.0.1:8801/health` reports readiness and the boot-policy id.

## Current compatibility slice

The trained fixture and Village do not yet have identical action grammars.
The bridge therefore makes two explicit adaptations:

- `MOVE_TOWARDS(target)` becomes Village `Move(target)`.
- Fixture `EAT(target)` becomes `Pickup(target)` until a body/hand record says
  that exact target is held; the next `EAT` becomes Village
  `Eat(instrument=hand, target)`.

`CRY` and `LAUGH` map directly. `WAIT` produces no intent. This shim is a
temporary integration boundary, not hidden curriculum behavior; a later
general policy should train against the action entries and arities in Pira's
manifest directly.

The service currently trains on every launch and does not persist learned
weights. Override the boot size only for smoke tests:

```bash
KRYSTAL_BOOT_EPISODES=4 KRYSTAL_BOOT_EPOCHS=1 bun run pira:serve
```
