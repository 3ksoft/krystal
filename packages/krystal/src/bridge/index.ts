/**
 * The Krystal agent API, version 2 — the complete surface a simulation talks to.
 *
 * Everything a simulation needs is re-exported here and nothing else is: if a
 * symbol is not on this list, it is engine internals and a simulation should
 * not reach for it. The point of a single entry is that the boundary is
 * enumerable — you can read what crosses it in one screen.
 *
 * Two contracts, versioned independently because they change for different
 * reasons:
 *
 *   pira-grammar@2        what exists in the world. Sent once, at agent
 *                         construction, and bound to the weights.
 *   pira-raw-sensory@2    what the actor perceives. Sent every tick.
 *
 * The shape of the exchange:
 *
 *   1. createAgent({ grammar })          declare the world's vocabulary
 *   2. validateSnapshot(snapshot, ...)   check a tick's perception
 *   3. lowerSnapshot(snapshot, ...)      perception becomes a BrainFrame
 *
 * Three rules that are not obvious from the types, and that the validators
 * enforce rather than merely document:
 *
 *   The simulation sends exact numbers, never bands. `Size.Medium` is already a
 *   decision, and it is the engine's: a band is a token, a token owns a trained
 *   embedding row, and a threshold that moved upstream would redefine that row
 *   without changing a single symbol.
 *
 *   The simulation sends only what the actor can perceive. A patrol's phase, a
 *   spawner id and a child list are the simulation's bookkeeping. Leaking them
 *   does not merely waste record slots — the model learns to use them, and then
 *   fails in a world without patrols having never learned to look.
 *
 *   Vocabulary is declared before it is used. An unknown symbol is refused, not
 *   dropped, because a boundary that silently forgets is a boundary that costs a
 *   day of wondering why the brain cannot see an apple.
 */

// -- Agent construction and grammar ----------------------------------------
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

// -- The per-tick sensory contract -----------------------------------------
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

// -- What comes back out ----------------------------------------------------
export type { AgentIntentV2, ConceptOperandV2 } from "./contract.ts";

// -- Lowering: perception becomes the model's input ------------------------
export {
  lowerSnapshot,
  ReferenceTable,
  LoweringError,
  type PerformedAction,
  type LoweredFrame,
  type BandOverflow,
} from "./lower.ts";

// -- Experience: frames waiting to be learned from -------------------------
export { ExperienceBuffer, type ExperienceEntry } from "./experience.ts";

// -- Discretization ---------------------------------------------------------
export {
  quantize,
  tokenWidthOf,
  BAND_SYMBOLS,
  BAND_TOKEN_IDS,
  QuantizeError,
  type BandedQuantity,
  type Polarity,
} from "./quantize.ts";

// -- ABI constants a simulation needs to respect ---------------------------
export {
  KRYSTAL_ABI,
  KRYSTAL_SENTINEL_TOKENS,
  KRYSTAL_TOKEN_RANGES,
  QUANTITY_BANDS,
  QUANTIFIER_FLAGS,
  RELATION_FLAGS,
} from "../../../schema/src/krystal-engine-schema.ts";
