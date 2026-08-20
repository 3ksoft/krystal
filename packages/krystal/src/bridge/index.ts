/**
 * The whole surface a simulation may touch.
 *
 * One file on purpose: a boundary spread across a package is a boundary nobody
 * can read in one sitting, and this one has two sides that must agree.
 */

export {
  createAgent,
  compileVocabulary,
  compileRelationCatalog,
  schemaIdOf,
  RESERVED_SYMBOLS,
  RESERVED_TOKEN_END,
  SIM_TOKEN_CLASSES,
  CATALOG_SCHEMA_ID,
  VocabularyError,
  AgentCheckpointMismatchError,
  type Agent,
  type AgentCheckpoint,
  type CreateAgentInput,
  type CompiledVocabulary,
  type CompiledCatalog,
} from "./agent.ts";

export {
  validatePercept,
  toAgentIntents,
  emptyDiagnostics,
  PerceptContractError,
  type PerceptDiagnostics,
  type ValidateOptions,
} from "./contract.ts";

export {
  lowerPercept,
  ReferenceTable,
  LoweringError,
  type PerformedRelation,
  type LoweredFrame,
  type BandOverflow,
} from "./lower.ts";

export { ExperienceBuffer, type ExperienceEntry } from "./experience.ts";

export {
  quantize,
  tokenWidthOf,
  BAND_SYMBOLS,
  BAND_TOKEN_IDS,
  QuantizeError,
  type BandedQuantity,
  type Polarity,
} from "./quantize.ts";

// The contract itself: validators for the wire, and the types generated from
// the same scope. A simulation needs both and should not have to know that one
// is written and the other derived.
export {
  world,
  WorldVocabulary,
  WorldChannel,
  WorldQuantity,
  WorldSymbol,
  WorldRelation,
  WorldRelationRole,
  Percept,
  Lesson,
  PerceptRecord,
  PerceptRelation,
  PerceptRoleBinding,
  PerceptOperand,
  PerceptQuantity,
  AgentIntent,
} from "../../../schema/src/world.ts";

export type { v1_0_0 as contract } from "../../../schema/generated/world.types.ts";

export {
  KRYSTAL_ABI,
  KRYSTAL_SENTINEL_TOKENS,
  KRYSTAL_TOKEN_RANGES,
  QUANTITY_BANDS,
  QUANTIFIER_FLAGS,
  RELATION_FLAGS,
  RELATION_ROLES,
  RELATION_ROLE_INDEX,
  type RelationRoleName,
} from "../../../schema/src/krystal-engine-schema.ts";
