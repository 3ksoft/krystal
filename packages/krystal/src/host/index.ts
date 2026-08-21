/**
 * The low-level host surface: records of tokens in, choices out.
 *
 * Deliberately NOT re-exported from the package index. Importing this must not
 * drag in the old simulation bridge, its world contract or its fixture
 * vocabulary — that coupling is the thing this entrypoint exists to end.
 */
export {
  packHostFrame,
  HostFrameError,
  PAD_TOKEN,
  QUERY_BAND,
  RECORD_WIDTH,
  type HostFrame,
  type HostRecord,
  type HostToken,
} from "./frame.ts";

export {
  learnFromExperience,
  selectionMask,
  type HostExperience,
  type LearnOptions,
  type LearnReport,
} from "./learn.ts";

export {
  BrainSession,
  createBrainForwardWeights,
  type BrainForwardConfig,
  type BrainForwardWeights,
  type Deliberation,
  type HostSelection,
  type HostSessionOptions,
  type ThinkOptions,
  type ThinkResult,
} from "./session.ts";
