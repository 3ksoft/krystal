import { krystalArtifact } from "./krystal.artifact.generated";
import { defineKrystalFromArtifact } from "./krystal-artifact";

export * from "./krystal-layout";
export * from "./krystal-artifact";
export { krystalArtifact };

/**
 * Runtime definition.
 *
 * Shader sources, layout plans and linking live exclusively in
 * scripts/build-krystal-artifact.ts. At runtime `Sandblaster.fromArtifact()`
 * recreates the typed resource/program handles from the serialized artifact —
 * no arktype scope, no re-declaration of the resource graph, no link().
 */
export const krystal = defineKrystalFromArtifact();

// Compact compatibility exports for the runtime while the legacy scheduler is
// being removed. New code should prefer `krystal.resources/programs/passes`.
export const engine = krystal.engine;
export const resources = krystal.resources;
export const programs = krystal.programs;
export const passes = krystal.passes;
