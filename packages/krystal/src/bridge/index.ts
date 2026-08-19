export {
  createAgent,
  compileGrammar,
  RESERVED_SYMBOLS,
  RESERVED_TOKEN_END,
  SIM_TOKEN_CLASSES,
  AgentGrammarError,
  AgentCheckpointMismatchError,
  type Agent,
  type AgentCheckpoint,
  type CreateAgentInput,
  type CompiledGrammar,
  type SimGrammar,
  type SimGrammarSymbol,
  type SimQuantityField,
  type SimActionV2,
  type RelationRoleV2,
  CATALOG_SCHEMA_ID,
  compileActionCatalog,
  schemaIdOf,
  type CompiledCatalog,
} from "./agent.ts";

export {
  validateSnapshot,
  isSensoryBand,
  emptyDiagnostics,
  toAgentIntents,
  SENSORY_BANDS,
  SensoryContractError,
  type SensoryBand,
  type RawSnapshotV2,
  type RawRecordV2,
  type RawQuantityV2,
  type RawMotionV2,
  type RawEventV2,
  type RawSelfMotionV2,
  type LoweringDiagnostics,
} from "./contract.ts";

export type { AgentIntentV2, ConceptOperandV2 } from "./contract.ts";

export {
  lowerSnapshot,
  ReferenceTable,
  LoweringError,
  type PerformedAction,
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

export {
  KRYSTAL_ABI,
  KRYSTAL_SENTINEL_TOKENS,
  KRYSTAL_TOKEN_RANGES,
  QUANTITY_BANDS,
  QUANTIFIER_FLAGS,
  RELATION_FLAGS,
} from "../../../schema/src/krystal-engine-schema.ts";
