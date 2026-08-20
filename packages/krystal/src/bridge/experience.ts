import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";

/**
 * What the creature chose in one frame, kept so the choice can be reinforced.
 *
 * Bank indices rather than record slots: a policy-gradient update re-runs the
 * selectors over the same frame, and the bank is what they score. Slots are
 * shuffled per frame and mean nothing across one.
 */
export interface RecordedChoice {
  /** Bank index of the chosen catalog record. */
  readonly intentBank: number;
  /** Catalog position of the chosen relation, for conditioning the role mask. */
  readonly intentId: number;
  /** Bank index chosen for the patient role, if one was bound. */
  readonly patientBank?: number | undefined;
}

export interface ExperienceEntry {
  readonly frame: v1_0_0.BrainFrame;
  readonly tick: number;
  /**
   * Change in valence that followed this frame.
   *
   * Written when the NEXT frame arrives — which is what "settled" means here: a
   * frame whose consequences are not yet known cannot be learned from, in
   * either direction.
   */
  readonly target: number | undefined;
  /** What was done in this frame. Absent when nothing was proposed. */
  readonly choice: RecordedChoice | undefined;
}

interface MutableEntry {
  frame: v1_0_0.BrainFrame;
  tick: number;
  target: number | undefined;
  choice: RecordedChoice | undefined;
}

export class ExperienceBuffer {
  private readonly entries: MutableEntry[] = [];

  constructor(private readonly capacity = 256) {}

  get size(): number {
    return this.entries.length;
  }

  record(
    frame: v1_0_0.BrainFrame,
    tick: number,
    valenceDelta: number | undefined,
    choice?: RecordedChoice,
  ): void {
    if (valenceDelta !== undefined && this.entries.length > 0) {
      this.entries[this.entries.length - 1]!.target = valenceDelta;
    }
    this.entries.push({ frame, tick, target: undefined, choice: undefined });
    while (this.entries.length > this.capacity) this.entries.shift();
    if (choice) this.entries[this.entries.length - 1]!.choice = choice;
  }

  /**
   * Attach the choice made in the frame just recorded.
   *
   * Separate from `record` because the decision is taken after the frame is
   * built: the creature has to see the world before it can act in it.
   */
  attachChoice(choice: RecordedChoice): void {
    const last = this.entries[this.entries.length - 1];
    if (last) last.choice = choice;
  }

  settled(): ExperienceEntry[] {
    return this.entries
      .slice(0, Math.max(0, this.entries.length - 1))
      .filter((entry) => entry.target !== undefined)
      .map((entry) => ({ ...entry }));
  }

  drain(): ExperienceEntry[] {
    const taken = this.settled();
    if (taken.length > 0) this.entries.splice(0, taken.length);
    return taken;
  }

  clear(): void {
    this.entries.length = 0;
  }
}
