/**
 * Window placement for the desktop.
 *
 * Windows tile in the workspace grid until one is dragged, at which point it is
 * *torn off* into free placement and the rest reflow around the gap. This keeps
 * the property that matters for a tool you actually work in — an arbitrary
 * combination of open windows always lands somewhere sane — while still letting
 * a window be put where you want it.
 *
 * A torn-off window can be dropped somewhere useless, so recovery is explicit
 * and cheap: Desk -> Clean Up returns everything to the grid. Placement is keyed
 * by window title and survives reload; nothing else here is persisted.
 */
import { reactive, watch } from "vue";

export interface Placement {
  x: number;
  y: number;
  z: number;
}

const STORAGE_KEY = "chomato.windows.v1";

interface Store {
  placements: Record<string, Placement>;
  topZ: number;
}

function load(): Store {
  const empty: Store = { placements: {}, topZ: 1 };
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return empty;
    const parsed = JSON.parse(raw) as Partial<Store>;
    if (!parsed || typeof parsed !== "object" || !parsed.placements) return empty;
    // Coordinates come back from storage unvalidated; a NaN here would place a
    // window at an unreachable position with no way to grab it.
    const placements: Record<string, Placement> = {};
    for (const [key, value] of Object.entries(parsed.placements)) {
      const { x, y, z } = value as Placement;
      if ([x, y, z].every((n) => typeof n === "number" && Number.isFinite(n))) {
        placements[key] = { x, y, z };
      }
    }
    const topZ = typeof parsed.topZ === "number" && Number.isFinite(parsed.topZ) ? parsed.topZ : 1;
    return { placements, topZ };
  } catch {
    return empty;
  }
}

const store = reactive<Store>(load());

watch(
  () => JSON.stringify(store),
  (serialized) => {
    try {
      localStorage.setItem(STORAGE_KEY, serialized);
    } catch {
      // Private mode or a full quota: placement is a convenience, not state we
      // may fail a session over.
    }
  },
);

export function placementOf(key: string): Placement | undefined {
  return store.placements[key];
}

export function isFloating(key: string): boolean {
  return store.placements[key] !== undefined;
}

/** Bring a floating window to the front. No-op while it is still tiled. */
export function raise(key: string): void {
  const placement = store.placements[key];
  if (!placement) return;
  if (placement.z === store.topZ) return;
  store.topZ += 1;
  placement.z = store.topZ;
}

export function place(key: string, x: number, y: number): void {
  const existing = store.placements[key];
  if (existing) {
    existing.x = x;
    existing.y = y;
    return;
  }
  store.topZ += 1;
  store.placements[key] = { x, y, z: store.topZ };
}

/** Return one window to the grid. */
export function reflow(key: string): void {
  delete store.placements[key];
}

/** GEM's name for it: put every window back on the grid. */
export function cleanUp(): void {
  for (const key of Object.keys(store.placements)) delete store.placements[key];
  store.topZ = 1;
}

export function floatingCount(): number {
  return Object.keys(store.placements).length;
}
