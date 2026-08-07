/**
 * Structural rules for a composed context.
 *
 * Pure on purpose: the engine has no opinion about BOS, so a malformed
 * composition executes happily and simply returns an answer that ignores its
 * own context. That failure is invisible unless something checks the shape,
 * and checking it must not require a GPU.
 */

export interface ContextPart {
  /** Human-readable id, e.g. "b#3" or "ckpt#1". */
  what: string;
  /** BOS at index 0. */
  bosLeading: boolean;
  /** BOS occurrences after index 0. */
  bosInterior: number;
}

export interface ContextShape {
  parts: readonly ContextPart[];
  tokens: number;
  /** Runtime context capacity, when a model is loaded. */
  capacity?: number;
}

/**
 * BOS is a property of the whole sequence, not of a block. It must open the
 * context exactly once: a second BOS further along resets the model, so every
 * block before it stops being visible — which reads as "the model ignores my
 * blocks" rather than as a malformed prompt.
 */
export function contextIssues(shape: ContextShape): string[] {
  const issues: string[] = [];
  const parts = shape.parts;
  if (!parts.length) return issues;

  const first = parts[0]!;
  if (!first.bosLeading) {
    issues.push(`${first.what} does not start with BOS — the first element of a context should`);
  }
  for (const part of parts.slice(1)) {
    if (part.bosLeading) {
      issues.push(`${part.what} starts with BOS but is not first — it resets the context`);
    }
  }
  for (const part of parts) {
    if (part.bosInterior > 0) {
      issues.push(`${part.what} contains BOS inside its own tokens (${part.bosInterior}x)`);
    }
  }
  if (shape.capacity !== undefined && shape.tokens > shape.capacity) {
    issues.push(`context is ${shape.tokens} tokens, capacity is ${shape.capacity}`);
  }
  return issues;
}
