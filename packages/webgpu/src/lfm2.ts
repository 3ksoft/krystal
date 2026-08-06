import { lfm2Artifact } from "./lfm2.artifact.generated";
import { defineLfm2 } from "./lfm2-definition";

export * from "./lfm2-definition";
export { lfm2Artifact };

/**
 * Browser/runtime definition.
 *
 * Shader sources and linking live exclusively in scripts/build-lfm2-artifact.ts.
 * At runtime we only recreate the typed resource/program handles expected by
 * Sandblaster and bind the serialized link artifact to them. compile() can then
 * go straight to GPU resource/pipeline creation without invoking link().
 */
export const lfm2 = defineLfm2();
lfm2.engine.deserialize(lfm2Artifact);

// Compact compatibility exports for the runtime while the legacy scheduler is
// being removed. New code should prefer `lfm2.resources/programs/passes`.
export const engine = lfm2.engine;
export const resources = lfm2.resources;
export const programs = lfm2.programs;
export const passes = lfm2.passes;
