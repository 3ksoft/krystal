import { lfm2Artifact } from "./lfm2.artifact.generated";
import { defineLfm2FromArtifact } from "./lfm2-artifact";

export * from "./lfm2-layout";
export * from "./lfm2-artifact";
export { lfm2Artifact };

/**
 * Runtime definition.
 *
 * Shader sources, layout plans and linking live exclusively in
 * scripts/build-lfm2-artifact.ts. At runtime `Sandblaster.fromArtifact()`
 * recreates the typed resource/program handles from the serialized artifact —
 * no arktype scope, no re-declaration of the resource graph, no link().
 */
export const lfm2 = defineLfm2FromArtifact();

// Compact compatibility exports for the runtime while the legacy scheduler is
// being removed. New code should prefer `lfm2.resources/programs/passes`.
export const engine = lfm2.engine;
export const resources = lfm2.resources;
export const programs = lfm2.programs;
export const passes = lfm2.passes;
