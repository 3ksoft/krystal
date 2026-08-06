import {
  lfm2,
  type Lfm2OpParams,
  type Lfm2PassSpec,
  type Lfm2ShaderName,
  type Lfm2Workgroups,
  type Lfm2WeightBinding,
} from "./lfm2";

/**
 * Everything below this import boundary is host-side inference orchestration.
 * Shader resources, includes, entry points and dispatch geometry stay in
 * lfm2.ts; this module deals in semantic pass requests.
 */
export interface Lfm2PassRequest {
  readonly name: Lfm2ShaderName;
  readonly program: Lfm2PassSpec["program"];
  readonly op: Readonly<Lfm2OpParams>;
  readonly workgroups: Lfm2Workgroups;
  readonly weight: Lfm2WeightBinding;
}

/** Resolve one semantic runtime operation into a concrete GPU pass request. */
export function lfm2Pass(
  name: Lfm2ShaderName,
  op: Readonly<Lfm2OpParams>,
): Lfm2PassRequest {
  const spec = lfm2.passes[name];
  return {
    name,
    program: spec.program,
    op,
    workgroups: spec.workgroups(op),
    weight: spec.weight,
  };
}

/**
 * Runtime-facing GPU definition. Kept as one import so the actual inference
 * scheduler does not grow a long list of program/resource imports.
 *
 * The next migration step belongs here: write OpParams, select/rebind the
 * concrete weight page when `request.weight !== "none"`, then encode
 * `request.program` using `request.workgroups`.
 */
export const gpu = lfm2;
