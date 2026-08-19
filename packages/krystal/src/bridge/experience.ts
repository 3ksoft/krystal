/**
 * Frames waiting to be learned from.
 *
 * The buffer exists mainly to get one thing right: **which frame a valence
 * change belongs to.**
 *
 * The value head predicts, from the state at tick t, how things are about to
 * change. The change is observed at t+1. So the delta that arrives alongside a
 * snapshot settles the *previous* frame's prediction, not that snapshot's own —
 * attaching it to the frame it arrived with would train every prediction
 * against an outcome that had already happened before the prediction was made.
 * Nothing about that error is visible: the loss still falls, on a target that
 * cannot be predicted from the input.
 *
 * Consequently the newest frame is always held back. Its outcome is not known
 * yet, and a frame with no target contributes nothing rather than being trained
 * toward zero.
 */
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";

export interface ExperienceEntry {
  readonly frame: v1_0_0.BrainFrame;
  readonly tick: number;
  /**
   * The valence change that followed this frame — its value-head target.
   * Undefined while the outcome is still in the future.
   */
  readonly target: number | undefined;
}

export class ExperienceBuffer {
  private readonly entries: { frame: v1_0_0.BrainFrame; tick: number; target: number | undefined }[] = [];

  constructor(private readonly capacity = 256) {}

  get size(): number {
    return this.entries.length;
  }

  /**
   * Record a lowered frame, together with the valence change observed at the
   * same tick. That change settles the PREVIOUS entry, and this frame's own
   * target stays open until the next tick arrives.
   */
  record(frame: v1_0_0.BrainFrame, tick: number, valenceDelta: number | undefined): void {
    if (valenceDelta !== undefined && this.entries.length > 0) {
      this.entries[this.entries.length - 1]!.target = valenceDelta;
    }
    this.entries.push({ frame, tick, target: undefined });
    // Oldest first: a bounded buffer that dropped the newest would keep only
    // the past it had already learned from.
    while (this.entries.length > this.capacity) this.entries.shift();
  }

  /**
   * Entries whose outcome is known. The trailing frame is excluded even if
   * something set its target, because nothing has settled it yet.
   */
  settled(): ExperienceEntry[] {
    return this.entries
      .slice(0, Math.max(0, this.entries.length - 1))
      .filter((entry) => entry.target !== undefined)
      .map((entry) => ({ ...entry }));
  }

  /**
   * Take the settled entries out, leaving the unsettled tail in place so the
   * next tick can still complete it.
   */
  drain(): ExperienceEntry[] {
    const taken = this.settled();
    if (taken.length > 0) this.entries.splice(0, taken.length);
    return taken;
  }

  clear(): void {
    this.entries.length = 0;
  }
}
