
import type { v1_0_0 } from "../../../schema/generated/krystal.types.ts";

export interface ExperienceEntry {
  readonly frame: v1_0_0.BrainFrame;
  readonly tick: number;
  readonly target: number | undefined;
}

export class ExperienceBuffer {
  private readonly entries: { frame: v1_0_0.BrainFrame; tick: number; target: number | undefined }[] = [];

  constructor(private readonly capacity = 256) {}

  get size(): number {
    return this.entries.length;
  }

  record(frame: v1_0_0.BrainFrame, tick: number, valenceDelta: number | undefined): void {
    if (valenceDelta !== undefined && this.entries.length > 0) {
      this.entries[this.entries.length - 1]!.target = valenceDelta;
    }
    this.entries.push({ frame, tick, target: undefined });
            while (this.entries.length > this.capacity) this.entries.shift();
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
