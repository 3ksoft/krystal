// THIS FILE IS AUTO-GENERATED - DO NOT CHANGE

// THIS FILE IS FOR REFERENCE ONLY!! DO NOT INCLUDE IT DIRECTLY!!
const SMB_system: u32 = 0u;
const SMB_structure: u32 = 1u;
const SMB_operation: u32 = 2u;
const SMB_object: u32 = 3u;
const SMB_property: u32 = 4u;
const SMB_quantity: u32 = 5u;
const SMB_action: u32 = 6u;
const SMB_reference: u32 = 7u;
const SMB_relation: u32 = 8u;
const SMB_logic: u32 = 9u;
const SMB_domain: u32 = 10u;
const SMB_context: u32 = 11u;
const SMB_experimental: u32 = 12u;
const SMB_homeostasis: u32 = 13u;
const SMB_body: u32 = 14u;
const SMB_vision: u32 = 15u;
const SMB_audio: u32 = 16u;
const SMB_olfaction: u32 = 17u;
const SMB_taste: u32 = 18u;
const SMB_touch: u32 = 19u;
const SMB_memory: u32 = 20u;
const SMB_focus: u32 = 21u;
const SMB_query: u32 = 22u;
const SMB_fixed: u32 = 23u;
const SMB_shuffled_records: u32 = 24u;
const SMB_stable_resident: u32 = 25u;
const SMB_error: u32 = 26u;
const SMB_truncate_low_salience: u32 = 27u;
const SMB_evict_low_priority: u32 = 28u;
const SMB_drop_oldest: u32 = 29u;
const SMB_runtime: u32 = 30u;
const SMB_sensor: u32 = 31u;
const SMB_creator: u32 = 32u;
const SMB_intent_feedback: u32 = 33u;
const SMB_none: u32 = 34u;
const SMB_entity: u32 = 35u;
const SMB_value: u32 = 36u;
const SMB_event: u32 = 37u;
const SMB_goal: u32 = 38u;
const SMB_intent: u32 = 39u;
const SMB_snapshot: u32 = 40u;
const SMB_controller: u32 = 41u;
const SMB_topic: u32 = 42u;
const SMB_invalid: u32 = 43u;
const SMB_live: u32 = 44u;
const SMB_stale: u32 = 45u;
const SMB_historical: u32 = 46u;
const SMB_destroyed: u32 = 47u;
const SMB_token: u32 = 48u;
const SMB_context_ref: u32 = 49u;
const SMB_record_ref: u32 = 50u;
const SMB_boolean_class: u32 = 51u;
const SMB_scalar_band: u32 = 52u;
const SMB_quantity_projection: u32 = 53u;
const SMB_opaque_payload: u32 = 54u;
const SMB_tutorial: u32 = 55u;
const SMB_external: u32 = 56u;
const SMB_internal: u32 = 57u;
const SMB_continuation: u32 = 58u;
const SMB_runtime_feedback: u32 = 59u;
const SMB_observation: u32 = 60u;
const SMB_empty: u32 = 61u;
const SMB_active: u32 = 62u;
const SMB_evictable: u32 = 63u;
const SMB_evicted: u32 = 64u;
const SMB_look: u32 = 65u;
const SMB_interaction: u32 = 66u;
const SMB_comfort_delta: u32 = 67u;
const SMB_retrieval: u32 = 68u;
const SMB_rehearsal: u32 = 69u;
const SMB_decay: u32 = 70u;
const SMB_perceptual: u32 = 71u;
const SMB_communicative: u32 = 72u;
const SMB_postural: u32 = 73u;
const SMB_selected: u32 = 74u;
const SMB_masked: u32 = 75u;
const SMB_ambiguous: u32 = 76u;
const SMB_start: u32 = 77u;
const SMB_maintain: u32 = 78u;
const SMB_stop: u32 = 79u;
const SMB_resume: u32 = 80u;
const SMB_proposed: u32 = 81u;
const SMB_accepted: u32 = 82u;
const SMB_succeeded: u32 = 83u;
const SMB_partial: u32 = 84u;
const SMB_failed: u32 = 85u;
const SMB_cancelled: u32 = 86u;
const SMB_forgotten: u32 = 87u;
const SMB_narrate: u32 = 88u;
const SMB_ask: u32 = 89u;
const SMB_present: u32 = 90u;
const SMB_demonstrate: u32 = 91u;
const SMB_show_reaction: u32 = 92u;
const SMB_assess: u32 = 93u;
const SMB_wait: u32 = 94u;
const SMB_reset: u32 = 95u;
const SMB_boolean: u32 = 96u;
const SMB_pointer: u32 = 97u;
const SMB_consequence: u32 = 98u;
const SMB_counterexample: u32 = 99u;
const SMB_unknown: u32 = 100u;
const SMB_idle: u32 = 101u;
const SMB_demonstrating: u32 = 102u;
const SMB_waiting: u32 = 103u;
const SMB_assessing: u32 = 104u;
const SMB_passed: u32 = 105u;
const SMB_done: u32 = 106u;
const SMB_assembling_frame: u32 = 107u;
const SMB_running: u32 = 108u;
const SMB_executing: u32 = 109u;

alias BandMask = u32;

alias KrystalTokenId = BandMask;

alias SchemaId = BandMask;

alias FieldId = BandMask;

alias IntentId = BandMask;

alias RecordIndex = BandMask;

alias LocalTokenIndex = BandMask;

alias KrystalTokenClass = u32;
const KrystalTokenClass_system: KrystalTokenClass = 0u;
const KrystalTokenClass_structure: KrystalTokenClass = 1u;
const KrystalTokenClass_operation: KrystalTokenClass = 2u;
const KrystalTokenClass_object: KrystalTokenClass = 3u;
const KrystalTokenClass_property: KrystalTokenClass = 4u;
const KrystalTokenClass_quantity: KrystalTokenClass = 5u;
const KrystalTokenClass_action: KrystalTokenClass = 6u;
const KrystalTokenClass_reference: KrystalTokenClass = 7u;
const KrystalTokenClass_relation: KrystalTokenClass = 8u;
const KrystalTokenClass_logic: KrystalTokenClass = 9u;
const KrystalTokenClass_domain: KrystalTokenClass = 10u;
const KrystalTokenClass_context: KrystalTokenClass = 11u;
const KrystalTokenClass_experimental: KrystalTokenClass = 12u;

alias BrainBandKind = u32;
const BrainBandKind_system: BrainBandKind = 0u;
const BrainBandKind_homeostasis: BrainBandKind = 1u;
const BrainBandKind_body: BrainBandKind = 2u;
const BrainBandKind_vision: BrainBandKind = 3u;
const BrainBandKind_audio: BrainBandKind = 4u;
const BrainBandKind_olfaction: BrainBandKind = 5u;
const BrainBandKind_taste: BrainBandKind = 6u;
const BrainBandKind_touch: BrainBandKind = 7u;
const BrainBandKind_memory: BrainBandKind = 8u;
const BrainBandKind_focus: BrainBandKind = 9u;
const BrainBandKind_query: BrainBandKind = 10u;

alias BandPlacementPolicy = u32;
const BandPlacementPolicy_fixed: BandPlacementPolicy = 0u;
const BandPlacementPolicy_shuffled_records: BandPlacementPolicy = 1u;
const BandPlacementPolicy_stable_resident: BandPlacementPolicy = 2u;

alias BandOverflowPolicy = u32;
const BandOverflowPolicy_error: BandOverflowPolicy = 0u;
const BandOverflowPolicy_truncate_low_salience: BandOverflowPolicy = 1u;
const BandOverflowPolicy_evict_low_priority: BandOverflowPolicy = 2u;
const BandOverflowPolicy_drop_oldest: BandOverflowPolicy = 3u;

alias RecordSource = u32;
const RecordSource_runtime: RecordSource = 0u;
const RecordSource_sensor: RecordSource = 1u;
const RecordSource_body: RecordSource = 2u;
const RecordSource_homeostasis: RecordSource = 3u;
const RecordSource_memory: RecordSource = 4u;
const RecordSource_focus: RecordSource = 5u;
const RecordSource_query: RecordSource = 6u;
const RecordSource_creator: RecordSource = 7u;
const RecordSource_intent_feedback: RecordSource = 8u;

alias RuntimeRefKind = u32;
const RuntimeRefKind_none: RuntimeRefKind = 0u;
const RuntimeRefKind_entity: RuntimeRefKind = 1u;
const RuntimeRefKind_value: RuntimeRefKind = 2u;
const RuntimeRefKind_memory: RuntimeRefKind = 3u;
const RuntimeRefKind_event: RuntimeRefKind = 4u;
const RuntimeRefKind_goal: RuntimeRefKind = 5u;
const RuntimeRefKind_intent: RuntimeRefKind = 6u;
const RuntimeRefKind_snapshot: RuntimeRefKind = 7u;
const RuntimeRefKind_controller: RuntimeRefKind = 8u;
const RuntimeRefKind_topic: RuntimeRefKind = 9u;

alias RuntimeRefStatus = u32;
const RuntimeRefStatus_invalid: RuntimeRefStatus = 0u;
const RuntimeRefStatus_live: RuntimeRefStatus = 1u;
const RuntimeRefStatus_stale: RuntimeRefStatus = 2u;
const RuntimeRefStatus_historical: RuntimeRefStatus = 3u;
const RuntimeRefStatus_destroyed: RuntimeRefStatus = 4u;

struct RuntimeRefHandle {
	tokenId: BandMask,
	generation: BandMask,
	kind: RuntimeRefKind,
	status: RuntimeRefStatus,
};


struct VocabManifestHeader {
	tokenAbiVersion: BandMask,
	manifestVersion: BandMask,
	vocabSize: BandMask,
	activeTokenCount: BandMask,
	manifestHashLo: BandMask,
	manifestHashHi: BandMask,
	reserved0: BandMask,
	reserved1: BandMask,
};


struct VocabManifestEntry {
	tokenId: BandMask,
	tokenClass: KrystalTokenClass,
	flags: BandMask,
	arity: BandMask,
	semanticTypeToken: BandMask,
	inverseToken: BandMask,
	reserved0: BandMask,
	reserved1: BandMask,
};


alias BrainValueKind = u32;
const BrainValueKind_none: BrainValueKind = 0u;
const BrainValueKind_token: BrainValueKind = 1u;
const BrainValueKind_context_ref: BrainValueKind = 2u;
const BrainValueKind_record_ref: BrainValueKind = 3u;
const BrainValueKind_boolean_class: BrainValueKind = 4u;
const BrainValueKind_scalar_band: BrainValueKind = 5u;
const BrainValueKind_quantity_projection: BrainValueKind = 6u;
const BrainValueKind_opaque_payload: BrainValueKind = 7u;

struct RecordSchemaManifestHeader {
	version: BandMask,
	schemaCount: BandMask,
	fieldCount: BandMask,
	maxRecordTokens: BandMask,
	schemaHashLo: BandMask,
	schemaHashHi: BandMask,
	reserved0: BandMask,
	reserved1: BandMask,
};


struct RecordSchemaEntry {
	schemaId: BandMask,
	familyToken: BandMask,
	defaultBand: BrainBandKind,
	tokenCount: BandMask,
	fieldOffset: BandMask,
	fieldCount: BandMask,
	flags: BandMask,
	reserved0: BandMask,
};


struct RecordFieldEntry {
	schemaId: BandMask,
	fieldId: BandMask,
	localTokenIndex: BandMask,
	tokenWidth: BandMask,
	roleToken: BandMask,
	valueKind: BrainValueKind,
	acceptedSchemaId: BandMask,
	allowedBandMask: BandMask,
	flags: BandMask,
	reserved0: BandMask,
};


struct TokenAuthoringSpec {
	id: u32,
	symbol: u32,
	tokenClass: KrystalTokenClass,
	semanticType: u32,
	arity: u32,
	doc: u32,
};


struct RecordFieldAuthoringSpec {
	name: u32,
	localTokenIndex: u32,
	roleToken: u32,
	valueKind: BrainValueKind,
	acceptedSchema: u32,
	allowedBands: u32,
	required: u32,
	exactRuntime: u32,
	doc: u32,
};


struct RecordSchemaAuthoringSpec {
	name: u32,
	familyToken: u32,
	defaultBand: BrainBandKind,
	fields: array<RecordFieldAuthoringSpec, 0>,
	doc: u32,
};


struct BrainBandLayout {
	kind: BrainBandKind,
	recordOffset: BandMask,
	recordCapacity: BandMask,
	tokenOffset: BandMask,
	tokenCapacity: BandMask,
	placement: BandPlacementPolicy,
	overflow: BandOverflowPolicy,
	flags: BandMask,
	reserved0: BandMask,
};


struct FixedRecordBinding {
	roleToken: BandMask,
	recordIndex: BandMask,
	expectedSchemaId: BandMask,
	flags: BandMask,
};


struct BrainFrameLayoutHeader {
	tokenAbiVersion: BandMask,
	architectureVersion: BandMask,
	layoutVersion: BandMask,
	recordWidth: BandMask,
	recordSlots: BandMask,
	tokenCapacity: BandMask,
	bandCount: BandMask,
	fixedRecordCount: BandMask,
	flags: BandMask,
	layoutHashLo: BandMask,
	layoutHashHi: BandMask,
};


struct BrainFrameLayout {
	header: BrainFrameLayoutHeader,
	bands: array<BrainBandLayout, 11>,
	fixedRecords: array<FixedRecordBinding, 32>,
};

const BRAIN_FRAME_LAYOUT_BANDS_LEN: u32 = 11u;
const BRAIN_FRAME_LAYOUT_FIXEDRECORDS_LEN: u32 = 32u;

struct BrainTokenMeta {
	fieldId: BandMask,
	roleToken: BandMask,
	flags: BandMask,
	referenceBinding: BandMask,
};


struct BrainReferenceBinding {
	localTokenIndex: BandMask,
	fieldId: BandMask,
	flags: BandMask,
	reserved0: BandMask,
	handle: RuntimeRefHandle,
};


struct BrainRecordHeader {
	schemaId: BandMask,
	band: BrainBandKind,
	source: RecordSource,
	flags: BandMask,
	tokenCount: BandMask,
	referenceCount: BandMask,
	observedAt: BandMask,
	revision: BandMask,
	primaryReference: BandMask,
	continuationRecord: BandMask,
	salience: f32,
	freshness: f32,
};


struct BrainRecordSlot {
	header: BrainRecordHeader,
	tokens: array<BandMask, 8>,
	tokenMeta: array<BrainTokenMeta, 8>,
	references: array<BrainReferenceBinding, 8>,
};

const BRAIN_RECORD_SLOT_TOKENS_LEN: u32 = 8u;
const BRAIN_RECORD_SLOT_TOKENMETA_LEN: u32 = 8u;
const BRAIN_RECORD_SLOT_REFERENCES_LEN: u32 = 8u;

struct BrainBandState {
	kind: BrainBandKind,
	activeRecords: BandMask,
	activeTokens: BandMask,
	overflowRecords: BandMask,
	truncatedRecords: BandMask,
	revision: BandMask,
	flags: BandMask,
	reserved0: BandMask,
};


struct BrainFrameHeader {
	tokenAbiVersion: BandMask,
	architectureVersion: BandMask,
	layoutVersion: BandMask,
	tick: BandMask,
	snapshot: BandMask,
	activeRecordCount: BandMask,
	activeTokenCount: BandMask,
	activeQueryRecord: BandMask,
	actorRecord: BandMask,
	frameRevision: BandMask,
	memoryRevision: BandMask,
	intentRevision: BandMask,
	flags: BandMask,
};


struct BrainFrame {
	header: BrainFrameHeader,
	bands: array<BrainBandState, 11>,
	records: array<BrainRecordSlot, 128>,
};

const BRAIN_FRAME_BANDS_LEN: u32 = 11u;
const BRAIN_FRAME_RECORDS_LEN: u32 = 128u;

struct HomeostasisSignal {
	channelToken: BandMask,
	currentStateToken: BandMask,
	desiredStateToken: BandMask,
	flags: BandMask,
	currentValue: f32,
	targetValue: f32,
	urgency: f32,
	delta: f32,
	source: RuntimeRefHandle,
};


alias BrainQueryKind = u32;
const BrainQueryKind_none: BrainQueryKind = 0u;
const BrainQueryKind_homeostasis: BrainQueryKind = 1u;
const BrainQueryKind_tutorial: BrainQueryKind = 2u;
const BrainQueryKind_external: BrainQueryKind = 3u;
const BrainQueryKind_internal: BrainQueryKind = 4u;
const BrainQueryKind_continuation: BrainQueryKind = 5u;
const BrainQueryKind_runtime_feedback: BrainQueryKind = 6u;

struct BrainQueryState {
	queryRef: RuntimeRefHandle,
	kind: BrainQueryKind,
	routeToken: BandMask,
	predicateToken: BandMask,
	subject: RuntimeRefHandle,
	objectToken: BandMask,
	objectRef: RuntimeRefHandle,
	urgency: f32,
	createdAt: BandMask,
	expiresAt: BandMask,
	flags: BandMask,
};


struct BrainQuerySet {
	count: BandMask,
	primary: BandMask,
	revision: BandMask,
	reserved0: BandMask,
	queries: array<BrainQueryState, 8>,
};

const BRAIN_QUERY_SET_QUERIES_LEN: u32 = 8u;

alias MemoryTraceKind = u32;
const MemoryTraceKind_none: MemoryTraceKind = 0u;
const MemoryTraceKind_entity: MemoryTraceKind = 1u;
const MemoryTraceKind_event: MemoryTraceKind = 2u;
const MemoryTraceKind_goal: MemoryTraceKind = 3u;
const MemoryTraceKind_intent: MemoryTraceKind = 4u;
const MemoryTraceKind_topic: MemoryTraceKind = 5u;
const MemoryTraceKind_observation: MemoryTraceKind = 6u;

alias MemorySlotState = u32;
const MemorySlotState_empty: MemorySlotState = 0u;
const MemorySlotState_active: MemorySlotState = 1u;
const MemorySlotState_evictable: MemorySlotState = 2u;
const MemorySlotState_evicted: MemorySlotState = 3u;

alias MemoryUpdateReason = u32;
const MemoryUpdateReason_observation: MemoryUpdateReason = 0u;
const MemoryUpdateReason_look: MemoryUpdateReason = 1u;
const MemoryUpdateReason_interaction: MemoryUpdateReason = 2u;
const MemoryUpdateReason_comfort_delta: MemoryUpdateReason = 3u;
const MemoryUpdateReason_retrieval: MemoryUpdateReason = 4u;
const MemoryUpdateReason_rehearsal: MemoryUpdateReason = 5u;
const MemoryUpdateReason_goal: MemoryUpdateReason = 6u;
const MemoryUpdateReason_intent: MemoryUpdateReason = 7u;
const MemoryUpdateReason_decay: MemoryUpdateReason = 8u;

struct MemoryConfig {
	slotCount: BandMask,
	activationDecay: f32,
	familiarityDecay: f32,
	familiarityGain: f32,
	evictionHysteresis: f32,
	minimumResidenceTicks: BandMask,
	flags: BandMask,
	reserved0: BandMask,
};


struct MemoryTrace {
	memoryRef: RuntimeRefHandle,
	subject: RuntimeRefHandle,
	kind: MemoryTraceKind,
	state: MemorySlotState,
	flags: BandMask,
	createdAt: BandMask,
	lastObservedAt: BandMask,
	lastAccessedAt: BandMask,
	interactionCount: BandMask,
	activation: f32,
	familiarity: f32,
	affectMagnitude: f32,
	reserved0: f32,
	rememberedRecord: BrainRecordSlot,
};


struct MemoryUpdate {
	subject: RuntimeRefHandle,
	reason: MemoryUpdateReason,
	interactionToken: BandMask,
	flags: BandMask,
	activationDelta: f32,
	familiarityDelta: f32,
	affectMagnitude: f32,
	tick: BandMask,
};


struct WorkingMemoryState {
	revision: BandMask,
	activeCount: BandMask,
	evictedCount: BandMask,
	flags: BandMask,
	slots: array<MemoryTrace, 32>,
};

const WORKING_MEMORY_STATE_SLOTS_LEN: u32 = 32u;

alias ActionIntentDomain = u32;
const ActionIntentDomain_external: ActionIntentDomain = 0u;
const ActionIntentDomain_perceptual: ActionIntentDomain = 1u;
const ActionIntentDomain_internal: ActionIntentDomain = 2u;
const ActionIntentDomain_communicative: ActionIntentDomain = 3u;
const ActionIntentDomain_postural: ActionIntentDomain = 4u;

struct ActionIntentCatalogHeader {
	version: BandMask,
	intentCount: BandMask,
	argumentCount: BandMask,
	flags: BandMask,
	catalogHashLo: BandMask,
	catalogHashHi: BandMask,
	reserved0: BandMask,
	reserved1: BandMask,
};


struct ActionIntentDescriptor {
	intentId: BandMask,
	actionToken: BandMask,
	semanticIntentToken: BandMask,
	domain: ActionIntentDomain,
	actorSchemaId: BandMask,
	argumentOffset: BandMask,
	argumentCount: BandMask,
	flags: BandMask,
	effectClassToken: BandMask,
	capabilityClassToken: BandMask,
	preconditionClassToken: BandMask,
	preferredControllerRole: BandMask,
};


struct ActionArgumentDescriptor {
	intentId: BandMask,
	argumentIndex: BandMask,
	roleToken: BandMask,
	valueKind: BrainValueKind,
	acceptedSchemaId: BandMask,
	candidateBandMask: BandMask,
	flags: BandMask,
	reserved0: BandMask,
};


struct ActionArgumentAuthoringSpec {
	name: u32,
	roleToken: u32,
	valueKind: BrainValueKind,
	acceptedSchema: u32,
	candidateBands: u32,
	required: u32,
	doc: u32,
};


struct ActionIntentAuthoringSpec {
	name: u32,
	actionToken: u32,
	semanticIntentToken: u32,
	domain: ActionIntentDomain,
	arguments: array<ActionArgumentAuthoringSpec, 0>,
	effectClassToken: u32,
	capabilityClassToken: u32,
	preconditionClassToken: u32,
	preferredControllerRole: u32,
	durative: u32,
	doc: u32,
};


alias SoftGatherStatus = u32;
const SoftGatherStatus_empty: SoftGatherStatus = 0u;
const SoftGatherStatus_selected: SoftGatherStatus = 1u;
const SoftGatherStatus_masked: SoftGatherStatus = 2u;
const SoftGatherStatus_ambiguous: SoftGatherStatus = 3u;
const SoftGatherStatus_error: SoftGatherStatus = 4u;

struct SoftGatherResult {
	status: SoftGatherStatus,
	selectedRecord: BandMask,
	selectedField: BandMask,
	selectedReference: BandMask,
	candidateCount: BandMask,
	probability: f32,
	entropy: f32,
	reserved0: BandMask,
};


struct TypedArgumentValue {
	kind: BrainValueKind,
	token: BandMask,
	flags: BandMask,
	reserved0: BandMask,
	handle: RuntimeRefHandle,
	selector: SoftGatherResult,
};


alias IntentLifecycle = u32;
const IntentLifecycle_empty: IntentLifecycle = 0u;
const IntentLifecycle_start: IntentLifecycle = 1u;
const IntentLifecycle_maintain: IntentLifecycle = 2u;
const IntentLifecycle_stop: IntentLifecycle = 3u;
const IntentLifecycle_resume: IntentLifecycle = 4u;

alias IntentExecutionStatus = u32;
const IntentExecutionStatus_empty: IntentExecutionStatus = 0u;
const IntentExecutionStatus_proposed: IntentExecutionStatus = 1u;
const IntentExecutionStatus_accepted: IntentExecutionStatus = 2u;
const IntentExecutionStatus_active: IntentExecutionStatus = 3u;
const IntentExecutionStatus_succeeded: IntentExecutionStatus = 4u;
const IntentExecutionStatus_partial: IntentExecutionStatus = 5u;
const IntentExecutionStatus_failed: IntentExecutionStatus = 6u;
const IntentExecutionStatus_cancelled: IntentExecutionStatus = 7u;
const IntentExecutionStatus_forgotten: IntentExecutionStatus = 8u;

struct IntentProposal {
	proposalSlot: BandMask,
	lifecycle: IntentLifecycle,
	intentId: BandMask,
	flags: BandMask,
	intentRef: RuntimeRefHandle,
	purposeGoal: RuntimeRefHandle,
	controllerHint: RuntimeRefHandle,
	topic: RuntimeRefHandle,
	activation: f32,
	priority: f32,
	persistence: f32,
	confidence: f32,
	arguments: array<TypedArgumentValue, 4>,
};

const INTENT_PROPOSAL_ARGUMENTS_LEN: u32 = 4u;

struct IntentSet {
	tick: BandMask,
	count: BandMask,
	revision: BandMask,
	flags: BandMask,
	proposals: array<IntentProposal, 8>,
};

const INTENT_SET_PROPOSALS_LEN: u32 = 8u;

struct ActiveIntentState {
	intentRef: RuntimeRefHandle,
	purposeGoal: RuntimeRefHandle,
	intentId: BandMask,
	status: IntentExecutionStatus,
	flags: BandMask,
	startedAt: BandMask,
	lastMaintainedAt: BandMask,
	completedAt: BandMask,
	activation: f32,
	progress: f32,
	outcomeMagnitude: f32,
	reserved0: f32,
};


struct ActiveIntentTable {
	revision: BandMask,
	activeCount: BandMask,
	completedCount: BandMask,
	flags: BandMask,
	intents: array<ActiveIntentState, 16>,
};

const ACTIVE_INTENT_TABLE_INTENTS_LEN: u32 = 16u;

struct IntentFeedback {
	intentRef: RuntimeRefHandle,
	status: IntentExecutionStatus,
	effectClassToken: BandMask,
	resultToken: BandMask,
	progress: f32,
	outcomeMagnitude: f32,
	comfortMagnitude: f32,
	tick: BandMask,
	feedbackRecord: BandMask,
	flags: BandMask,
};


alias TutorialBeatKind = u32;
const TutorialBeatKind_narrate: TutorialBeatKind = 0u;
const TutorialBeatKind_ask: TutorialBeatKind = 1u;
const TutorialBeatKind_present: TutorialBeatKind = 2u;
const TutorialBeatKind_focus: TutorialBeatKind = 3u;
const TutorialBeatKind_demonstrate: TutorialBeatKind = 4u;
const TutorialBeatKind_show_reaction: TutorialBeatKind = 5u;
const TutorialBeatKind_assess: TutorialBeatKind = 6u;
const TutorialBeatKind_wait: TutorialBeatKind = 7u;
const TutorialBeatKind_reset: TutorialBeatKind = 8u;

alias TutorialProbeKind = u32;
const TutorialProbeKind_token: TutorialProbeKind = 0u;
const TutorialProbeKind_boolean: TutorialProbeKind = 1u;
const TutorialProbeKind_pointer: TutorialProbeKind = 2u;
const TutorialProbeKind_property: TutorialProbeKind = 3u;
const TutorialProbeKind_intent: TutorialProbeKind = 4u;
const TutorialProbeKind_consequence: TutorialProbeKind = 5u;
const TutorialProbeKind_counterexample: TutorialProbeKind = 6u;
const TutorialProbeKind_unknown: TutorialProbeKind = 7u;

struct TutorialProgramHeader {
	version: BandMask,
	lessonToken: BandMask,
	beatOffset: BandMask,
	beatCount: BandMask,
	probeOffset: BandMask,
	probeCount: BandMask,
	creatorTokenOffset: BandMask,
	creatorTokenCount: BandMask,
	flags: BandMask,
	reserved0: BandMask,
};


struct TutorialBeat {
	kind: TutorialBeatKind,
	sceneCue: BandMask,
	utteranceOffset: BandMask,
	utteranceCount: BandMask,
	holdFrames: BandMask,
	probeIndex: BandMask,
	expectedIntentId: BandMask,
	flags: BandMask,
};


struct TutorialProbe {
	kind: TutorialProbeKind,
	querySchemaId: BandMask,
	expectedToken: BandMask,
	expectedIntentId: BandMask,
	expectedRecord: BandMask,
	expectedField: BandMask,
	oracleBinding: BandMask,
	flags: BandMask,
	reserved0: BandMask,
};


alias TutorialRuntimeStatus = u32;
const TutorialRuntimeStatus_idle: TutorialRuntimeStatus = 0u;
const TutorialRuntimeStatus_demonstrating: TutorialRuntimeStatus = 1u;
const TutorialRuntimeStatus_waiting: TutorialRuntimeStatus = 2u;
const TutorialRuntimeStatus_assessing: TutorialRuntimeStatus = 3u;
const TutorialRuntimeStatus_passed: TutorialRuntimeStatus = 4u;
const TutorialRuntimeStatus_failed: TutorialRuntimeStatus = 5u;
const TutorialRuntimeStatus_done: TutorialRuntimeStatus = 6u;

struct TutorialRuntimeState {
	program: BandMask,
	beat: BandMask,
	probe: BandMask,
	status: TutorialRuntimeStatus,
	frameInBeat: BandMask,
	attempts: BandMask,
	correct: BandMask,
	incorrect: BandMask,
};


struct TutorialProbeAuthoringSpec {
	kind: TutorialProbeKind,
	querySchema: u32,
	expectedToken: u32,
	expectedIntent: u32,
	oracleBinding: u32,
	doc: u32,
};


struct TutorialBeatAuthoringSpec {
	kind: TutorialBeatKind,
	sceneCue: u32,
	utterance: u32,
	holdFrames: u32,
	expectedIntent: u32,
	probe: u32,
	doc: u32,
};


struct TutorialAuthoringSpec {
	name: u32,
	lessonTokens: array<u32, 0>,
	prerequisites: u32,
	beats: array<TutorialBeatAuthoringSpec, 0>,
	doc: u32,
};


alias BrainRuntimeStatus = u32;
const BrainRuntimeStatus_idle: BrainRuntimeStatus = 0u;
const BrainRuntimeStatus_assembling_frame: BrainRuntimeStatus = 1u;
const BrainRuntimeStatus_running: BrainRuntimeStatus = 2u;
const BrainRuntimeStatus_executing: BrainRuntimeStatus = 3u;
const BrainRuntimeStatus_done: BrainRuntimeStatus = 4u;
const BrainRuntimeStatus_error: BrainRuntimeStatus = 5u;

struct BrainModelConfig {
	vocabSize: BandMask,
	contextTokens: BandMask,
	recordWidth: BandMask,
	recordSlots: BandMask,
	hiddenSize: BandMask,
	recordSize: BandMask,
	layerCount: BandMask,
	attentionHeads: BandMask,
	maxIntentProposals: BandMask,
	flags: BandMask,
};


struct BrainRuntimeConfig {
	tokenAbiVersion: BandMask,
	architectureVersion: BandMask,
	frameLayoutVersion: BandMask,
	vocabManifestVersion: BandMask,
	recordManifestVersion: BandMask,
	actionCatalogVersion: BandMask,
	tutorialVersion: BandMask,
	flags: BandMask,
	reserved0: BandMask,
	model: BrainModelConfig,
	memory: MemoryConfig,
};


struct BrainRuntimeState {
	status: BrainRuntimeStatus,
	tick: BandMask,
	snapshot: BandMask,
	frameRevision: BandMask,
	memoryRevision: BandMask,
	queryRevision: BandMask,
	intentRevision: BandMask,
	errorCode: BandMask,
};


struct BrainStepTelemetry {
	tick: BandMask,
	activeRecords: BandMask,
	activeTokens: BandMask,
	truncatedRecords: BandMask,
	intentCount: BandMask,
	activeIntentCount: BandMask,
	memoryCount: BandMask,
	queryCount: BandMask,
	frameBuildMs: f32,
	localEncodeMs: f32,
	recordMixMs: f32,
	gatherMs: f32,
	decideMs: f32,
	runtimeMs: f32,
	meanGatherEntropy: f32,
	minGatherProbability: f32,
	flags: BandMask,
	errorCode: BandMask,
};


struct BrainStepResult {
	state: BrainRuntimeState,
	intents: IntentSet,
	telemetry: BrainStepTelemetry,
};



// ==========================================
// MEMORY HELPERS (Storage Buffers interop)
// ==========================================

fn unpack_words_to_band_mask(raw: u32) -> BandMask {
	var out: BandMask;
	out = bitcast<u32>(raw);
	return out;
}

fn unpack_band_mask(raw: u32) -> BandMask {
	return unpack_words_to_band_mask(raw);
}

fn pack_band_mask_to_words(unpacked: BandMask) -> u32 {
	var out: u32 = 0u;
	out = bitcast<u32>(unpacked);
	return out;
}

fn pack_band_mask(unpacked: BandMask) -> u32 {
	return pack_band_mask_to_words(unpacked);
}

fn unpack_words_to_krystal_token_id(raw: u32) -> KrystalTokenId {
	var out: KrystalTokenId;
	out = unpack_words_to_band_mask(raw);
	return out;
}

fn unpack_krystal_token_id(raw: u32) -> KrystalTokenId {
	return unpack_words_to_krystal_token_id(raw);
}

fn pack_krystal_token_id_to_words(unpacked: KrystalTokenId) -> u32 {
	var out: u32 = 0u;
	out = pack_band_mask_to_words(unpacked);
	return out;
}

fn pack_krystal_token_id(unpacked: KrystalTokenId) -> u32 {
	return pack_krystal_token_id_to_words(unpacked);
}

fn unpack_words_to_schema_id(raw: u32) -> SchemaId {
	var out: SchemaId;
	out = unpack_words_to_band_mask(raw);
	return out;
}

fn unpack_schema_id(raw: u32) -> SchemaId {
	return unpack_words_to_schema_id(raw);
}

fn pack_schema_id_to_words(unpacked: SchemaId) -> u32 {
	var out: u32 = 0u;
	out = pack_band_mask_to_words(unpacked);
	return out;
}

fn pack_schema_id(unpacked: SchemaId) -> u32 {
	return pack_schema_id_to_words(unpacked);
}

fn unpack_words_to_field_id(raw: u32) -> FieldId {
	var out: FieldId;
	out = unpack_words_to_band_mask(raw);
	return out;
}

fn unpack_field_id(raw: u32) -> FieldId {
	return unpack_words_to_field_id(raw);
}

fn pack_field_id_to_words(unpacked: FieldId) -> u32 {
	var out: u32 = 0u;
	out = pack_band_mask_to_words(unpacked);
	return out;
}

fn pack_field_id(unpacked: FieldId) -> u32 {
	return pack_field_id_to_words(unpacked);
}

fn unpack_words_to_intent_id(raw: u32) -> IntentId {
	var out: IntentId;
	out = unpack_words_to_band_mask(raw);
	return out;
}

fn unpack_intent_id(raw: u32) -> IntentId {
	return unpack_words_to_intent_id(raw);
}

fn pack_intent_id_to_words(unpacked: IntentId) -> u32 {
	var out: u32 = 0u;
	out = pack_band_mask_to_words(unpacked);
	return out;
}

fn pack_intent_id(unpacked: IntentId) -> u32 {
	return pack_intent_id_to_words(unpacked);
}

fn unpack_words_to_record_index(raw: u32) -> RecordIndex {
	var out: RecordIndex;
	out = unpack_words_to_band_mask(raw);
	return out;
}

fn unpack_record_index(raw: u32) -> RecordIndex {
	return unpack_words_to_record_index(raw);
}

fn pack_record_index_to_words(unpacked: RecordIndex) -> u32 {
	var out: u32 = 0u;
	out = pack_band_mask_to_words(unpacked);
	return out;
}

fn pack_record_index(unpacked: RecordIndex) -> u32 {
	return pack_record_index_to_words(unpacked);
}

fn unpack_words_to_local_token_index(raw: u32) -> LocalTokenIndex {
	var out: LocalTokenIndex;
	out = unpack_words_to_band_mask(raw);
	return out;
}

fn unpack_local_token_index(raw: u32) -> LocalTokenIndex {
	return unpack_words_to_local_token_index(raw);
}

fn pack_local_token_index_to_words(unpacked: LocalTokenIndex) -> u32 {
	var out: u32 = 0u;
	out = pack_band_mask_to_words(unpacked);
	return out;
}

fn pack_local_token_index(unpacked: LocalTokenIndex) -> u32 {
	return pack_local_token_index_to_words(unpacked);
}

fn unpack_words_to_runtime_ref_handle(raw: array<u32, 3>) -> RuntimeRefHandle {
	var out: RuntimeRefHandle;
	out.tokenId = unpack_words_to_band_mask(raw[0u]);
	out.generation = unpack_words_to_band_mask(raw[1u]);
	out.kind = RuntimeRefKind(extractBits(raw[2u], 0u, 8u));
	out.status = RuntimeRefStatus(extractBits(raw[2u], 8u, 8u));
	return out;
}

fn unpack_runtime_ref_handle(raw: array<u32, 3>) -> RuntimeRefHandle {
	return unpack_words_to_runtime_ref_handle(raw);
}

fn pack_runtime_ref_handle_to_words(unpacked: RuntimeRefHandle) -> array<u32, 3> {
	var out: array<u32, 3>;
	for (var w = 0u; w < 3u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.tokenId);
	out[1u] = pack_band_mask_to_words(unpacked.generation);
	out[2u] = insertBits(out[2u], u32(unpacked.kind), 0u, 8u);
	out[2u] = insertBits(out[2u], u32(unpacked.status), 8u, 8u);
	return out;
}

fn pack_runtime_ref_handle(unpacked: RuntimeRefHandle) -> array<u32, 3> {
	return pack_runtime_ref_handle_to_words(unpacked);
}

fn unpack_words_to_vocab_manifest_header(raw: array<u32, 8>) -> VocabManifestHeader {
	var out: VocabManifestHeader;
	out.tokenAbiVersion = unpack_words_to_band_mask(raw[0u]);
	out.manifestVersion = unpack_words_to_band_mask(raw[1u]);
	out.vocabSize = unpack_words_to_band_mask(raw[2u]);
	out.activeTokenCount = unpack_words_to_band_mask(raw[3u]);
	out.manifestHashLo = unpack_words_to_band_mask(raw[4u]);
	out.manifestHashHi = unpack_words_to_band_mask(raw[5u]);
	out.reserved0 = unpack_words_to_band_mask(raw[6u]);
	out.reserved1 = unpack_words_to_band_mask(raw[7u]);
	return out;
}

fn unpack_vocab_manifest_header(raw: array<u32, 8>) -> VocabManifestHeader {
	return unpack_words_to_vocab_manifest_header(raw);
}

fn pack_vocab_manifest_header_to_words(unpacked: VocabManifestHeader) -> array<u32, 8> {
	var out: array<u32, 8>;
	for (var w = 0u; w < 8u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.tokenAbiVersion);
	out[1u] = pack_band_mask_to_words(unpacked.manifestVersion);
	out[2u] = pack_band_mask_to_words(unpacked.vocabSize);
	out[3u] = pack_band_mask_to_words(unpacked.activeTokenCount);
	out[4u] = pack_band_mask_to_words(unpacked.manifestHashLo);
	out[5u] = pack_band_mask_to_words(unpacked.manifestHashHi);
	out[6u] = pack_band_mask_to_words(unpacked.reserved0);
	out[7u] = pack_band_mask_to_words(unpacked.reserved1);
	return out;
}

fn pack_vocab_manifest_header(unpacked: VocabManifestHeader) -> array<u32, 8> {
	return pack_vocab_manifest_header_to_words(unpacked);
}

fn unpack_words_to_vocab_manifest_entry(raw: array<u32, 8>) -> VocabManifestEntry {
	var out: VocabManifestEntry;
	out.tokenId = unpack_words_to_band_mask(raw[0u]);
	out.tokenClass = KrystalTokenClass(extractBits(raw[1u], 0u, 8u));
	out.flags = unpack_words_to_band_mask(raw[2u]);
	out.arity = unpack_words_to_band_mask(raw[3u]);
	out.semanticTypeToken = unpack_words_to_band_mask(raw[4u]);
	out.inverseToken = unpack_words_to_band_mask(raw[5u]);
	out.reserved0 = unpack_words_to_band_mask(raw[6u]);
	out.reserved1 = unpack_words_to_band_mask(raw[7u]);
	return out;
}

fn unpack_vocab_manifest_entry(raw: array<u32, 8>) -> VocabManifestEntry {
	return unpack_words_to_vocab_manifest_entry(raw);
}

fn pack_vocab_manifest_entry_to_words(unpacked: VocabManifestEntry) -> array<u32, 8> {
	var out: array<u32, 8>;
	for (var w = 0u; w < 8u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.tokenId);
	out[1u] = insertBits(out[1u], u32(unpacked.tokenClass), 0u, 8u);
	out[2u] = pack_band_mask_to_words(unpacked.flags);
	out[3u] = pack_band_mask_to_words(unpacked.arity);
	out[4u] = pack_band_mask_to_words(unpacked.semanticTypeToken);
	out[5u] = pack_band_mask_to_words(unpacked.inverseToken);
	out[6u] = pack_band_mask_to_words(unpacked.reserved0);
	out[7u] = pack_band_mask_to_words(unpacked.reserved1);
	return out;
}

fn pack_vocab_manifest_entry(unpacked: VocabManifestEntry) -> array<u32, 8> {
	return pack_vocab_manifest_entry_to_words(unpacked);
}

fn unpack_words_to_record_schema_manifest_header(raw: array<u32, 8>) -> RecordSchemaManifestHeader {
	var out: RecordSchemaManifestHeader;
	out.version = unpack_words_to_band_mask(raw[0u]);
	out.schemaCount = unpack_words_to_band_mask(raw[1u]);
	out.fieldCount = unpack_words_to_band_mask(raw[2u]);
	out.maxRecordTokens = unpack_words_to_band_mask(raw[3u]);
	out.schemaHashLo = unpack_words_to_band_mask(raw[4u]);
	out.schemaHashHi = unpack_words_to_band_mask(raw[5u]);
	out.reserved0 = unpack_words_to_band_mask(raw[6u]);
	out.reserved1 = unpack_words_to_band_mask(raw[7u]);
	return out;
}

fn unpack_record_schema_manifest_header(raw: array<u32, 8>) -> RecordSchemaManifestHeader {
	return unpack_words_to_record_schema_manifest_header(raw);
}

fn pack_record_schema_manifest_header_to_words(unpacked: RecordSchemaManifestHeader) -> array<u32, 8> {
	var out: array<u32, 8>;
	for (var w = 0u; w < 8u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.version);
	out[1u] = pack_band_mask_to_words(unpacked.schemaCount);
	out[2u] = pack_band_mask_to_words(unpacked.fieldCount);
	out[3u] = pack_band_mask_to_words(unpacked.maxRecordTokens);
	out[4u] = pack_band_mask_to_words(unpacked.schemaHashLo);
	out[5u] = pack_band_mask_to_words(unpacked.schemaHashHi);
	out[6u] = pack_band_mask_to_words(unpacked.reserved0);
	out[7u] = pack_band_mask_to_words(unpacked.reserved1);
	return out;
}

fn pack_record_schema_manifest_header(unpacked: RecordSchemaManifestHeader) -> array<u32, 8> {
	return pack_record_schema_manifest_header_to_words(unpacked);
}

fn unpack_words_to_record_schema_entry(raw: array<u32, 8>) -> RecordSchemaEntry {
	var out: RecordSchemaEntry;
	out.schemaId = unpack_words_to_band_mask(raw[0u]);
	out.familyToken = unpack_words_to_band_mask(raw[1u]);
	out.defaultBand = BrainBandKind(extractBits(raw[2u], 0u, 8u));
	out.tokenCount = unpack_words_to_band_mask(raw[3u]);
	out.fieldOffset = unpack_words_to_band_mask(raw[4u]);
	out.fieldCount = unpack_words_to_band_mask(raw[5u]);
	out.flags = unpack_words_to_band_mask(raw[6u]);
	out.reserved0 = unpack_words_to_band_mask(raw[7u]);
	return out;
}

fn unpack_record_schema_entry(raw: array<u32, 8>) -> RecordSchemaEntry {
	return unpack_words_to_record_schema_entry(raw);
}

fn pack_record_schema_entry_to_words(unpacked: RecordSchemaEntry) -> array<u32, 8> {
	var out: array<u32, 8>;
	for (var w = 0u; w < 8u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.schemaId);
	out[1u] = pack_band_mask_to_words(unpacked.familyToken);
	out[2u] = insertBits(out[2u], u32(unpacked.defaultBand), 0u, 8u);
	out[3u] = pack_band_mask_to_words(unpacked.tokenCount);
	out[4u] = pack_band_mask_to_words(unpacked.fieldOffset);
	out[5u] = pack_band_mask_to_words(unpacked.fieldCount);
	out[6u] = pack_band_mask_to_words(unpacked.flags);
	out[7u] = pack_band_mask_to_words(unpacked.reserved0);
	return out;
}

fn pack_record_schema_entry(unpacked: RecordSchemaEntry) -> array<u32, 8> {
	return pack_record_schema_entry_to_words(unpacked);
}

fn unpack_words_to_record_field_entry(raw: array<u32, 10>) -> RecordFieldEntry {
	var out: RecordFieldEntry;
	out.schemaId = unpack_words_to_band_mask(raw[0u]);
	out.fieldId = unpack_words_to_band_mask(raw[1u]);
	out.localTokenIndex = unpack_words_to_band_mask(raw[2u]);
	out.tokenWidth = unpack_words_to_band_mask(raw[3u]);
	out.roleToken = unpack_words_to_band_mask(raw[4u]);
	out.valueKind = BrainValueKind(extractBits(raw[5u], 0u, 8u));
	out.acceptedSchemaId = unpack_words_to_band_mask(raw[6u]);
	out.allowedBandMask = unpack_words_to_band_mask(raw[7u]);
	out.flags = unpack_words_to_band_mask(raw[8u]);
	out.reserved0 = unpack_words_to_band_mask(raw[9u]);
	return out;
}

fn unpack_record_field_entry(raw: array<u32, 10>) -> RecordFieldEntry {
	return unpack_words_to_record_field_entry(raw);
}

fn pack_record_field_entry_to_words(unpacked: RecordFieldEntry) -> array<u32, 10> {
	var out: array<u32, 10>;
	for (var w = 0u; w < 10u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.schemaId);
	out[1u] = pack_band_mask_to_words(unpacked.fieldId);
	out[2u] = pack_band_mask_to_words(unpacked.localTokenIndex);
	out[3u] = pack_band_mask_to_words(unpacked.tokenWidth);
	out[4u] = pack_band_mask_to_words(unpacked.roleToken);
	out[5u] = insertBits(out[5u], u32(unpacked.valueKind), 0u, 8u);
	out[6u] = pack_band_mask_to_words(unpacked.acceptedSchemaId);
	out[7u] = pack_band_mask_to_words(unpacked.allowedBandMask);
	out[8u] = pack_band_mask_to_words(unpacked.flags);
	out[9u] = pack_band_mask_to_words(unpacked.reserved0);
	return out;
}

fn pack_record_field_entry(unpacked: RecordFieldEntry) -> array<u32, 10> {
	return pack_record_field_entry_to_words(unpacked);
}

fn unpack_words_to_token_authoring_spec(raw: array<u32, 11>) -> TokenAuthoringSpec {
	var out: TokenAuthoringSpec;
	out.id = bitcast<u32>(raw[0u]);
		out.tokenClass = KrystalTokenClass(extractBits(raw[3u], 0u, 8u));
				return out;
}

fn unpack_token_authoring_spec(raw: array<u32, 11>) -> TokenAuthoringSpec {
	return unpack_words_to_token_authoring_spec(raw);
}

fn pack_token_authoring_spec_to_words(unpacked: TokenAuthoringSpec) -> array<u32, 11> {
	var out: array<u32, 11>;
	for (var w = 0u; w < 11u; w++) { out[w] = 0u; }
	out[0u] = bitcast<u32>(unpacked.id);
		out[3u] = insertBits(out[3u], u32(unpacked.tokenClass), 0u, 8u);
				return out;
}

fn pack_token_authoring_spec(unpacked: TokenAuthoringSpec) -> array<u32, 11> {
	return pack_token_authoring_spec_to_words(unpacked);
}

fn unpack_words_to_record_field_authoring_spec(raw: array<u32, 16>) -> RecordFieldAuthoringSpec {
	var out: RecordFieldAuthoringSpec;
		out.localTokenIndex = bitcast<u32>(raw[1u]);
	out.roleToken = bitcast<u32>(raw[3u]);
	out.valueKind = BrainValueKind(extractBits(raw[5u], 0u, 8u));
						return out;
}

fn unpack_record_field_authoring_spec(raw: array<u32, 16>) -> RecordFieldAuthoringSpec {
	return unpack_words_to_record_field_authoring_spec(raw);
}

fn pack_record_field_authoring_spec_to_words(unpacked: RecordFieldAuthoringSpec) -> array<u32, 16> {
	var out: array<u32, 16>;
	for (var w = 0u; w < 16u; w++) { out[w] = 0u; }
		out[1u] = bitcast<u32>(unpacked.localTokenIndex);
	out[3u] = bitcast<u32>(unpacked.roleToken);
	out[5u] = insertBits(out[5u], u32(unpacked.valueKind), 0u, 8u);
						return out;
}

fn pack_record_field_authoring_spec(unpacked: RecordFieldAuthoringSpec) -> array<u32, 16> {
	return pack_record_field_authoring_spec_to_words(unpacked);
}

fn unpack_words_to_record_schema_authoring_spec(raw: array<u32, 7>) -> RecordSchemaAuthoringSpec {
	var out: RecordSchemaAuthoringSpec;
		out.familyToken = bitcast<u32>(raw[1u]);
	out.defaultBand = BrainBandKind(extractBits(raw[3u], 0u, 8u));
	for (var i_0 = 0u; i_0 < 0u; i_0++) {
		{
		var tmp: array<u32, 16>;
		for (var j_1 = 0u; j_1 < 16u; j_1++) { tmp[j_1] = raw[4u + (i_0 * 16u) + j_1]; }
		out.fields[i_0] = unpack_words_to_record_field_authoring_spec(tmp);
	}
	}
		return out;
}

fn unpack_record_schema_authoring_spec(raw: array<u32, 7>) -> RecordSchemaAuthoringSpec {
	return unpack_words_to_record_schema_authoring_spec(raw);
}

fn pack_record_schema_authoring_spec_to_words(unpacked: RecordSchemaAuthoringSpec) -> array<u32, 7> {
	var out: array<u32, 7>;
	for (var w = 0u; w < 7u; w++) { out[w] = 0u; }
		out[1u] = bitcast<u32>(unpacked.familyToken);
	out[3u] = insertBits(out[3u], u32(unpacked.defaultBand), 0u, 8u);
	for (var i_0 = 0u; i_0 < 0u; i_0++) {
			{
		let tmp = pack_record_field_authoring_spec_to_words(unpacked.fields[i_0]);
		for (var j_1 = 0u; j_1 < 16u; j_1++) { out[4u + (i_0 * 16u) + j_1] = tmp[j_1]; }
	}
		}
		return out;
}

fn pack_record_schema_authoring_spec(unpacked: RecordSchemaAuthoringSpec) -> array<u32, 7> {
	return pack_record_schema_authoring_spec_to_words(unpacked);
}

fn unpack_words_to_brain_band_layout(raw: array<u32, 8>) -> BrainBandLayout {
	var out: BrainBandLayout;
	out.kind = BrainBandKind(extractBits(raw[0u], 0u, 8u));
	out.recordOffset = unpack_words_to_band_mask(raw[1u]);
	out.recordCapacity = unpack_words_to_band_mask(raw[2u]);
	out.tokenOffset = unpack_words_to_band_mask(raw[3u]);
	out.tokenCapacity = unpack_words_to_band_mask(raw[4u]);
	out.placement = BandPlacementPolicy(extractBits(raw[5u], 0u, 8u));
	out.overflow = BandOverflowPolicy(extractBits(raw[5u], 8u, 8u));
	out.flags = unpack_words_to_band_mask(raw[6u]);
	out.reserved0 = unpack_words_to_band_mask(raw[7u]);
	return out;
}

fn unpack_brain_band_layout(raw: array<u32, 8>) -> BrainBandLayout {
	return unpack_words_to_brain_band_layout(raw);
}

fn pack_brain_band_layout_to_words(unpacked: BrainBandLayout) -> array<u32, 8> {
	var out: array<u32, 8>;
	for (var w = 0u; w < 8u; w++) { out[w] = 0u; }
	out[0u] = insertBits(out[0u], u32(unpacked.kind), 0u, 8u);
	out[1u] = pack_band_mask_to_words(unpacked.recordOffset);
	out[2u] = pack_band_mask_to_words(unpacked.recordCapacity);
	out[3u] = pack_band_mask_to_words(unpacked.tokenOffset);
	out[4u] = pack_band_mask_to_words(unpacked.tokenCapacity);
	out[5u] = insertBits(out[5u], u32(unpacked.placement), 0u, 8u);
	out[5u] = insertBits(out[5u], u32(unpacked.overflow), 8u, 8u);
	out[6u] = pack_band_mask_to_words(unpacked.flags);
	out[7u] = pack_band_mask_to_words(unpacked.reserved0);
	return out;
}

fn pack_brain_band_layout(unpacked: BrainBandLayout) -> array<u32, 8> {
	return pack_brain_band_layout_to_words(unpacked);
}

fn unpack_words_to_fixed_record_binding(raw: array<u32, 4>) -> FixedRecordBinding {
	var out: FixedRecordBinding;
	out.roleToken = unpack_words_to_band_mask(raw[0u]);
	out.recordIndex = unpack_words_to_band_mask(raw[1u]);
	out.expectedSchemaId = unpack_words_to_band_mask(raw[2u]);
	out.flags = unpack_words_to_band_mask(raw[3u]);
	return out;
}

fn unpack_fixed_record_binding(raw: array<u32, 4>) -> FixedRecordBinding {
	return unpack_words_to_fixed_record_binding(raw);
}

fn pack_fixed_record_binding_to_words(unpacked: FixedRecordBinding) -> array<u32, 4> {
	var out: array<u32, 4>;
	for (var w = 0u; w < 4u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.roleToken);
	out[1u] = pack_band_mask_to_words(unpacked.recordIndex);
	out[2u] = pack_band_mask_to_words(unpacked.expectedSchemaId);
	out[3u] = pack_band_mask_to_words(unpacked.flags);
	return out;
}

fn pack_fixed_record_binding(unpacked: FixedRecordBinding) -> array<u32, 4> {
	return pack_fixed_record_binding_to_words(unpacked);
}

fn unpack_words_to_brain_frame_layout_header(raw: array<u32, 11>) -> BrainFrameLayoutHeader {
	var out: BrainFrameLayoutHeader;
	out.tokenAbiVersion = unpack_words_to_band_mask(raw[0u]);
	out.architectureVersion = unpack_words_to_band_mask(raw[1u]);
	out.layoutVersion = unpack_words_to_band_mask(raw[2u]);
	out.recordWidth = unpack_words_to_band_mask(raw[3u]);
	out.recordSlots = unpack_words_to_band_mask(raw[4u]);
	out.tokenCapacity = unpack_words_to_band_mask(raw[5u]);
	out.bandCount = unpack_words_to_band_mask(raw[6u]);
	out.fixedRecordCount = unpack_words_to_band_mask(raw[7u]);
	out.flags = unpack_words_to_band_mask(raw[8u]);
	out.layoutHashLo = unpack_words_to_band_mask(raw[9u]);
	out.layoutHashHi = unpack_words_to_band_mask(raw[10u]);
	return out;
}

fn unpack_brain_frame_layout_header(raw: array<u32, 11>) -> BrainFrameLayoutHeader {
	return unpack_words_to_brain_frame_layout_header(raw);
}

fn pack_brain_frame_layout_header_to_words(unpacked: BrainFrameLayoutHeader) -> array<u32, 11> {
	var out: array<u32, 11>;
	for (var w = 0u; w < 11u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.tokenAbiVersion);
	out[1u] = pack_band_mask_to_words(unpacked.architectureVersion);
	out[2u] = pack_band_mask_to_words(unpacked.layoutVersion);
	out[3u] = pack_band_mask_to_words(unpacked.recordWidth);
	out[4u] = pack_band_mask_to_words(unpacked.recordSlots);
	out[5u] = pack_band_mask_to_words(unpacked.tokenCapacity);
	out[6u] = pack_band_mask_to_words(unpacked.bandCount);
	out[7u] = pack_band_mask_to_words(unpacked.fixedRecordCount);
	out[8u] = pack_band_mask_to_words(unpacked.flags);
	out[9u] = pack_band_mask_to_words(unpacked.layoutHashLo);
	out[10u] = pack_band_mask_to_words(unpacked.layoutHashHi);
	return out;
}

fn pack_brain_frame_layout_header(unpacked: BrainFrameLayoutHeader) -> array<u32, 11> {
	return pack_brain_frame_layout_header_to_words(unpacked);
}

fn unpack_words_to_brain_frame_layout(raw: array<u32, 227>) -> BrainFrameLayout {
	var out: BrainFrameLayout;
	{
		var tmp: array<u32, 11>;
		for (var j_0 = 0u; j_0 < 11u; j_0++) { tmp[j_0] = raw[0u + j_0]; }
		out.header = unpack_words_to_brain_frame_layout_header(tmp);
	}
	for (var i_0 = 0u; i_0 < 11u; i_0++) {
		{
		var tmp: array<u32, 8>;
		for (var j_1 = 0u; j_1 < 8u; j_1++) { tmp[j_1] = raw[11u + (i_0 * 8u) + j_1]; }
		out.bands[i_0] = unpack_words_to_brain_band_layout(tmp);
	}
	}
	for (var i_0 = 0u; i_0 < 32u; i_0++) {
		{
		var tmp: array<u32, 4>;
		for (var j_1 = 0u; j_1 < 4u; j_1++) { tmp[j_1] = raw[99u + (i_0 * 4u) + j_1]; }
		out.fixedRecords[i_0] = unpack_words_to_fixed_record_binding(tmp);
	}
	}
	return out;
}

fn unpack_brain_frame_layout(raw: array<u32, 227>) -> BrainFrameLayout {
	return unpack_words_to_brain_frame_layout(raw);
}

fn pack_brain_frame_layout_to_words(unpacked: BrainFrameLayout) -> array<u32, 227> {
	var out: array<u32, 227>;
	for (var w = 0u; w < 227u; w++) { out[w] = 0u; }
	{
		let tmp = pack_brain_frame_layout_header_to_words(unpacked.header);
		for (var j_0 = 0u; j_0 < 11u; j_0++) { out[0u + j_0] = tmp[j_0]; }
	}
	for (var i_0 = 0u; i_0 < 11u; i_0++) {
			{
		let tmp = pack_brain_band_layout_to_words(unpacked.bands[i_0]);
		for (var j_1 = 0u; j_1 < 8u; j_1++) { out[11u + (i_0 * 8u) + j_1] = tmp[j_1]; }
	}
		}
	for (var i_0 = 0u; i_0 < 32u; i_0++) {
			{
		let tmp = pack_fixed_record_binding_to_words(unpacked.fixedRecords[i_0]);
		for (var j_1 = 0u; j_1 < 4u; j_1++) { out[99u + (i_0 * 4u) + j_1] = tmp[j_1]; }
	}
		}
	return out;
}

fn pack_brain_frame_layout(unpacked: BrainFrameLayout) -> array<u32, 227> {
	return pack_brain_frame_layout_to_words(unpacked);
}

fn unpack_words_to_brain_token_meta(raw: array<u32, 4>) -> BrainTokenMeta {
	var out: BrainTokenMeta;
	out.fieldId = unpack_words_to_band_mask(raw[0u]);
	out.roleToken = unpack_words_to_band_mask(raw[1u]);
	out.flags = unpack_words_to_band_mask(raw[2u]);
	out.referenceBinding = unpack_words_to_band_mask(raw[3u]);
	return out;
}

fn unpack_brain_token_meta(raw: array<u32, 4>) -> BrainTokenMeta {
	return unpack_words_to_brain_token_meta(raw);
}

fn pack_brain_token_meta_to_words(unpacked: BrainTokenMeta) -> array<u32, 4> {
	var out: array<u32, 4>;
	for (var w = 0u; w < 4u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.fieldId);
	out[1u] = pack_band_mask_to_words(unpacked.roleToken);
	out[2u] = pack_band_mask_to_words(unpacked.flags);
	out[3u] = pack_band_mask_to_words(unpacked.referenceBinding);
	return out;
}

fn pack_brain_token_meta(unpacked: BrainTokenMeta) -> array<u32, 4> {
	return pack_brain_token_meta_to_words(unpacked);
}

fn unpack_words_to_brain_reference_binding(raw: array<u32, 7>) -> BrainReferenceBinding {
	var out: BrainReferenceBinding;
	out.localTokenIndex = unpack_words_to_band_mask(raw[0u]);
	out.fieldId = unpack_words_to_band_mask(raw[1u]);
	out.flags = unpack_words_to_band_mask(raw[2u]);
	out.reserved0 = unpack_words_to_band_mask(raw[3u]);
	{
		var tmp: array<u32, 3>;
		for (var j_0 = 0u; j_0 < 3u; j_0++) { tmp[j_0] = raw[4u + j_0]; }
		out.handle = unpack_words_to_runtime_ref_handle(tmp);
	}
	return out;
}

fn unpack_brain_reference_binding(raw: array<u32, 7>) -> BrainReferenceBinding {
	return unpack_words_to_brain_reference_binding(raw);
}

fn pack_brain_reference_binding_to_words(unpacked: BrainReferenceBinding) -> array<u32, 7> {
	var out: array<u32, 7>;
	for (var w = 0u; w < 7u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.localTokenIndex);
	out[1u] = pack_band_mask_to_words(unpacked.fieldId);
	out[2u] = pack_band_mask_to_words(unpacked.flags);
	out[3u] = pack_band_mask_to_words(unpacked.reserved0);
	{
		let tmp = pack_runtime_ref_handle_to_words(unpacked.handle);
		for (var j_0 = 0u; j_0 < 3u; j_0++) { out[4u + j_0] = tmp[j_0]; }
	}
	return out;
}

fn pack_brain_reference_binding(unpacked: BrainReferenceBinding) -> array<u32, 7> {
	return pack_brain_reference_binding_to_words(unpacked);
}

fn unpack_words_to_brain_record_header(raw: array<u32, 11>) -> BrainRecordHeader {
	var out: BrainRecordHeader;
	out.schemaId = unpack_words_to_band_mask(raw[0u]);
	out.band = BrainBandKind(extractBits(raw[1u], 0u, 8u));
	out.source = RecordSource(extractBits(raw[1u], 8u, 8u));
	out.flags = unpack_words_to_band_mask(raw[2u]);
	out.tokenCount = unpack_words_to_band_mask(raw[3u]);
	out.referenceCount = unpack_words_to_band_mask(raw[4u]);
	out.observedAt = unpack_words_to_band_mask(raw[5u]);
	out.revision = unpack_words_to_band_mask(raw[6u]);
	out.primaryReference = unpack_words_to_band_mask(raw[7u]);
	out.continuationRecord = unpack_words_to_band_mask(raw[8u]);
	out.salience = bitcast<f32>(raw[9u]);
	out.freshness = bitcast<f32>(raw[10u]);
	return out;
}

fn unpack_brain_record_header(raw: array<u32, 11>) -> BrainRecordHeader {
	return unpack_words_to_brain_record_header(raw);
}

fn pack_brain_record_header_to_words(unpacked: BrainRecordHeader) -> array<u32, 11> {
	var out: array<u32, 11>;
	for (var w = 0u; w < 11u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.schemaId);
	out[1u] = insertBits(out[1u], u32(unpacked.band), 0u, 8u);
	out[1u] = insertBits(out[1u], u32(unpacked.source), 8u, 8u);
	out[2u] = pack_band_mask_to_words(unpacked.flags);
	out[3u] = pack_band_mask_to_words(unpacked.tokenCount);
	out[4u] = pack_band_mask_to_words(unpacked.referenceCount);
	out[5u] = pack_band_mask_to_words(unpacked.observedAt);
	out[6u] = pack_band_mask_to_words(unpacked.revision);
	out[7u] = pack_band_mask_to_words(unpacked.primaryReference);
	out[8u] = pack_band_mask_to_words(unpacked.continuationRecord);
	out[9u] = bitcast<u32>(unpacked.salience);
	out[10u] = bitcast<u32>(unpacked.freshness);
	return out;
}

fn pack_brain_record_header(unpacked: BrainRecordHeader) -> array<u32, 11> {
	return pack_brain_record_header_to_words(unpacked);
}

fn unpack_words_to_brain_record_slot(raw: array<u32, 107>) -> BrainRecordSlot {
	var out: BrainRecordSlot;
	{
		var tmp: array<u32, 11>;
		for (var j_0 = 0u; j_0 < 11u; j_0++) { tmp[j_0] = raw[0u + j_0]; }
		out.header = unpack_words_to_brain_record_header(tmp);
	}
	for (var i_0 = 0u; i_0 < 8u; i_0++) {
		out.tokens[i_0] = unpack_words_to_band_mask(raw[11u + (i_0 * 1u)]);
	}
	for (var i_0 = 0u; i_0 < 8u; i_0++) {
		{
		var tmp: array<u32, 4>;
		for (var j_1 = 0u; j_1 < 4u; j_1++) { tmp[j_1] = raw[19u + (i_0 * 4u) + j_1]; }
		out.tokenMeta[i_0] = unpack_words_to_brain_token_meta(tmp);
	}
	}
	for (var i_0 = 0u; i_0 < 8u; i_0++) {
		{
		var tmp: array<u32, 7>;
		for (var j_1 = 0u; j_1 < 7u; j_1++) { tmp[j_1] = raw[51u + (i_0 * 7u) + j_1]; }
		out.references[i_0] = unpack_words_to_brain_reference_binding(tmp);
	}
	}
	return out;
}

fn unpack_brain_record_slot(raw: array<u32, 107>) -> BrainRecordSlot {
	return unpack_words_to_brain_record_slot(raw);
}

fn pack_brain_record_slot_to_words(unpacked: BrainRecordSlot) -> array<u32, 107> {
	var out: array<u32, 107>;
	for (var w = 0u; w < 107u; w++) { out[w] = 0u; }
	{
		let tmp = pack_brain_record_header_to_words(unpacked.header);
		for (var j_0 = 0u; j_0 < 11u; j_0++) { out[0u + j_0] = tmp[j_0]; }
	}
	for (var i_0 = 0u; i_0 < 8u; i_0++) {
			out[11u + (i_0 * 1u)] = pack_band_mask_to_words(unpacked.tokens[i_0]);
		}
	for (var i_0 = 0u; i_0 < 8u; i_0++) {
			{
		let tmp = pack_brain_token_meta_to_words(unpacked.tokenMeta[i_0]);
		for (var j_1 = 0u; j_1 < 4u; j_1++) { out[19u + (i_0 * 4u) + j_1] = tmp[j_1]; }
	}
		}
	for (var i_0 = 0u; i_0 < 8u; i_0++) {
			{
		let tmp = pack_brain_reference_binding_to_words(unpacked.references[i_0]);
		for (var j_1 = 0u; j_1 < 7u; j_1++) { out[51u + (i_0 * 7u) + j_1] = tmp[j_1]; }
	}
		}
	return out;
}

fn pack_brain_record_slot(unpacked: BrainRecordSlot) -> array<u32, 107> {
	return pack_brain_record_slot_to_words(unpacked);
}

fn unpack_words_to_brain_band_state(raw: array<u32, 8>) -> BrainBandState {
	var out: BrainBandState;
	out.kind = BrainBandKind(extractBits(raw[0u], 0u, 8u));
	out.activeRecords = unpack_words_to_band_mask(raw[1u]);
	out.activeTokens = unpack_words_to_band_mask(raw[2u]);
	out.overflowRecords = unpack_words_to_band_mask(raw[3u]);
	out.truncatedRecords = unpack_words_to_band_mask(raw[4u]);
	out.revision = unpack_words_to_band_mask(raw[5u]);
	out.flags = unpack_words_to_band_mask(raw[6u]);
	out.reserved0 = unpack_words_to_band_mask(raw[7u]);
	return out;
}

fn unpack_brain_band_state(raw: array<u32, 8>) -> BrainBandState {
	return unpack_words_to_brain_band_state(raw);
}

fn pack_brain_band_state_to_words(unpacked: BrainBandState) -> array<u32, 8> {
	var out: array<u32, 8>;
	for (var w = 0u; w < 8u; w++) { out[w] = 0u; }
	out[0u] = insertBits(out[0u], u32(unpacked.kind), 0u, 8u);
	out[1u] = pack_band_mask_to_words(unpacked.activeRecords);
	out[2u] = pack_band_mask_to_words(unpacked.activeTokens);
	out[3u] = pack_band_mask_to_words(unpacked.overflowRecords);
	out[4u] = pack_band_mask_to_words(unpacked.truncatedRecords);
	out[5u] = pack_band_mask_to_words(unpacked.revision);
	out[6u] = pack_band_mask_to_words(unpacked.flags);
	out[7u] = pack_band_mask_to_words(unpacked.reserved0);
	return out;
}

fn pack_brain_band_state(unpacked: BrainBandState) -> array<u32, 8> {
	return pack_brain_band_state_to_words(unpacked);
}

fn unpack_words_to_brain_frame_header(raw: array<u32, 13>) -> BrainFrameHeader {
	var out: BrainFrameHeader;
	out.tokenAbiVersion = unpack_words_to_band_mask(raw[0u]);
	out.architectureVersion = unpack_words_to_band_mask(raw[1u]);
	out.layoutVersion = unpack_words_to_band_mask(raw[2u]);
	out.tick = unpack_words_to_band_mask(raw[3u]);
	out.snapshot = unpack_words_to_band_mask(raw[4u]);
	out.activeRecordCount = unpack_words_to_band_mask(raw[5u]);
	out.activeTokenCount = unpack_words_to_band_mask(raw[6u]);
	out.activeQueryRecord = unpack_words_to_band_mask(raw[7u]);
	out.actorRecord = unpack_words_to_band_mask(raw[8u]);
	out.frameRevision = unpack_words_to_band_mask(raw[9u]);
	out.memoryRevision = unpack_words_to_band_mask(raw[10u]);
	out.intentRevision = unpack_words_to_band_mask(raw[11u]);
	out.flags = unpack_words_to_band_mask(raw[12u]);
	return out;
}

fn unpack_brain_frame_header(raw: array<u32, 13>) -> BrainFrameHeader {
	return unpack_words_to_brain_frame_header(raw);
}

fn pack_brain_frame_header_to_words(unpacked: BrainFrameHeader) -> array<u32, 13> {
	var out: array<u32, 13>;
	for (var w = 0u; w < 13u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.tokenAbiVersion);
	out[1u] = pack_band_mask_to_words(unpacked.architectureVersion);
	out[2u] = pack_band_mask_to_words(unpacked.layoutVersion);
	out[3u] = pack_band_mask_to_words(unpacked.tick);
	out[4u] = pack_band_mask_to_words(unpacked.snapshot);
	out[5u] = pack_band_mask_to_words(unpacked.activeRecordCount);
	out[6u] = pack_band_mask_to_words(unpacked.activeTokenCount);
	out[7u] = pack_band_mask_to_words(unpacked.activeQueryRecord);
	out[8u] = pack_band_mask_to_words(unpacked.actorRecord);
	out[9u] = pack_band_mask_to_words(unpacked.frameRevision);
	out[10u] = pack_band_mask_to_words(unpacked.memoryRevision);
	out[11u] = pack_band_mask_to_words(unpacked.intentRevision);
	out[12u] = pack_band_mask_to_words(unpacked.flags);
	return out;
}

fn pack_brain_frame_header(unpacked: BrainFrameHeader) -> array<u32, 13> {
	return pack_brain_frame_header_to_words(unpacked);
}

fn unpack_words_to_brain_frame(raw: array<u32, 13797>) -> BrainFrame {
	var out: BrainFrame;
	{
		var tmp: array<u32, 13>;
		for (var j_0 = 0u; j_0 < 13u; j_0++) { tmp[j_0] = raw[0u + j_0]; }
		out.header = unpack_words_to_brain_frame_header(tmp);
	}
	for (var i_0 = 0u; i_0 < 11u; i_0++) {
		{
		var tmp: array<u32, 8>;
		for (var j_1 = 0u; j_1 < 8u; j_1++) { tmp[j_1] = raw[13u + (i_0 * 8u) + j_1]; }
		out.bands[i_0] = unpack_words_to_brain_band_state(tmp);
	}
	}
	for (var i_0 = 0u; i_0 < 128u; i_0++) {
		{
		var tmp: array<u32, 107>;
		for (var j_1 = 0u; j_1 < 107u; j_1++) { tmp[j_1] = raw[101u + (i_0 * 107u) + j_1]; }
		out.records[i_0] = unpack_words_to_brain_record_slot(tmp);
	}
	}
	return out;
}

fn unpack_brain_frame(raw: array<u32, 13797>) -> BrainFrame {
	return unpack_words_to_brain_frame(raw);
}

fn pack_brain_frame_to_words(unpacked: BrainFrame) -> array<u32, 13797> {
	var out: array<u32, 13797>;
	for (var w = 0u; w < 13797u; w++) { out[w] = 0u; }
	{
		let tmp = pack_brain_frame_header_to_words(unpacked.header);
		for (var j_0 = 0u; j_0 < 13u; j_0++) { out[0u + j_0] = tmp[j_0]; }
	}
	for (var i_0 = 0u; i_0 < 11u; i_0++) {
			{
		let tmp = pack_brain_band_state_to_words(unpacked.bands[i_0]);
		for (var j_1 = 0u; j_1 < 8u; j_1++) { out[13u + (i_0 * 8u) + j_1] = tmp[j_1]; }
	}
		}
	for (var i_0 = 0u; i_0 < 128u; i_0++) {
			{
		let tmp = pack_brain_record_slot_to_words(unpacked.records[i_0]);
		for (var j_1 = 0u; j_1 < 107u; j_1++) { out[101u + (i_0 * 107u) + j_1] = tmp[j_1]; }
	}
		}
	return out;
}

fn pack_brain_frame(unpacked: BrainFrame) -> array<u32, 13797> {
	return pack_brain_frame_to_words(unpacked);
}

fn unpack_words_to_homeostasis_signal(raw: array<u32, 11>) -> HomeostasisSignal {
	var out: HomeostasisSignal;
	out.channelToken = unpack_words_to_band_mask(raw[0u]);
	out.currentStateToken = unpack_words_to_band_mask(raw[1u]);
	out.desiredStateToken = unpack_words_to_band_mask(raw[2u]);
	out.flags = unpack_words_to_band_mask(raw[3u]);
	out.currentValue = bitcast<f32>(raw[4u]);
	out.targetValue = bitcast<f32>(raw[5u]);
	out.urgency = bitcast<f32>(raw[6u]);
	out.delta = bitcast<f32>(raw[7u]);
	{
		var tmp: array<u32, 3>;
		for (var j_0 = 0u; j_0 < 3u; j_0++) { tmp[j_0] = raw[8u + j_0]; }
		out.source = unpack_words_to_runtime_ref_handle(tmp);
	}
	return out;
}

fn unpack_homeostasis_signal(raw: array<u32, 11>) -> HomeostasisSignal {
	return unpack_words_to_homeostasis_signal(raw);
}

fn pack_homeostasis_signal_to_words(unpacked: HomeostasisSignal) -> array<u32, 11> {
	var out: array<u32, 11>;
	for (var w = 0u; w < 11u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.channelToken);
	out[1u] = pack_band_mask_to_words(unpacked.currentStateToken);
	out[2u] = pack_band_mask_to_words(unpacked.desiredStateToken);
	out[3u] = pack_band_mask_to_words(unpacked.flags);
	out[4u] = bitcast<u32>(unpacked.currentValue);
	out[5u] = bitcast<u32>(unpacked.targetValue);
	out[6u] = bitcast<u32>(unpacked.urgency);
	out[7u] = bitcast<u32>(unpacked.delta);
	{
		let tmp = pack_runtime_ref_handle_to_words(unpacked.source);
		for (var j_0 = 0u; j_0 < 3u; j_0++) { out[8u + j_0] = tmp[j_0]; }
	}
	return out;
}

fn pack_homeostasis_signal(unpacked: HomeostasisSignal) -> array<u32, 11> {
	return pack_homeostasis_signal_to_words(unpacked);
}

fn unpack_words_to_brain_query_state(raw: array<u32, 17>) -> BrainQueryState {
	var out: BrainQueryState;
	{
		var tmp: array<u32, 3>;
		for (var j_0 = 0u; j_0 < 3u; j_0++) { tmp[j_0] = raw[0u + j_0]; }
		out.queryRef = unpack_words_to_runtime_ref_handle(tmp);
	}
	out.kind = BrainQueryKind(extractBits(raw[3u], 0u, 8u));
	out.routeToken = unpack_words_to_band_mask(raw[4u]);
	out.predicateToken = unpack_words_to_band_mask(raw[5u]);
	{
		var tmp: array<u32, 3>;
		for (var j_0 = 0u; j_0 < 3u; j_0++) { tmp[j_0] = raw[6u + j_0]; }
		out.subject = unpack_words_to_runtime_ref_handle(tmp);
	}
	out.objectToken = unpack_words_to_band_mask(raw[9u]);
	{
		var tmp: array<u32, 3>;
		for (var j_0 = 0u; j_0 < 3u; j_0++) { tmp[j_0] = raw[10u + j_0]; }
		out.objectRef = unpack_words_to_runtime_ref_handle(tmp);
	}
	out.urgency = bitcast<f32>(raw[13u]);
	out.createdAt = unpack_words_to_band_mask(raw[14u]);
	out.expiresAt = unpack_words_to_band_mask(raw[15u]);
	out.flags = unpack_words_to_band_mask(raw[16u]);
	return out;
}

fn unpack_brain_query_state(raw: array<u32, 17>) -> BrainQueryState {
	return unpack_words_to_brain_query_state(raw);
}

fn pack_brain_query_state_to_words(unpacked: BrainQueryState) -> array<u32, 17> {
	var out: array<u32, 17>;
	for (var w = 0u; w < 17u; w++) { out[w] = 0u; }
	{
		let tmp = pack_runtime_ref_handle_to_words(unpacked.queryRef);
		for (var j_0 = 0u; j_0 < 3u; j_0++) { out[0u + j_0] = tmp[j_0]; }
	}
	out[3u] = insertBits(out[3u], u32(unpacked.kind), 0u, 8u);
	out[4u] = pack_band_mask_to_words(unpacked.routeToken);
	out[5u] = pack_band_mask_to_words(unpacked.predicateToken);
	{
		let tmp = pack_runtime_ref_handle_to_words(unpacked.subject);
		for (var j_0 = 0u; j_0 < 3u; j_0++) { out[6u + j_0] = tmp[j_0]; }
	}
	out[9u] = pack_band_mask_to_words(unpacked.objectToken);
	{
		let tmp = pack_runtime_ref_handle_to_words(unpacked.objectRef);
		for (var j_0 = 0u; j_0 < 3u; j_0++) { out[10u + j_0] = tmp[j_0]; }
	}
	out[13u] = bitcast<u32>(unpacked.urgency);
	out[14u] = pack_band_mask_to_words(unpacked.createdAt);
	out[15u] = pack_band_mask_to_words(unpacked.expiresAt);
	out[16u] = pack_band_mask_to_words(unpacked.flags);
	return out;
}

fn pack_brain_query_state(unpacked: BrainQueryState) -> array<u32, 17> {
	return pack_brain_query_state_to_words(unpacked);
}

fn unpack_words_to_brain_query_set(raw: array<u32, 140>) -> BrainQuerySet {
	var out: BrainQuerySet;
	out.count = unpack_words_to_band_mask(raw[0u]);
	out.primary = unpack_words_to_band_mask(raw[1u]);
	out.revision = unpack_words_to_band_mask(raw[2u]);
	out.reserved0 = unpack_words_to_band_mask(raw[3u]);
	for (var i_0 = 0u; i_0 < 8u; i_0++) {
		{
		var tmp: array<u32, 17>;
		for (var j_1 = 0u; j_1 < 17u; j_1++) { tmp[j_1] = raw[4u + (i_0 * 17u) + j_1]; }
		out.queries[i_0] = unpack_words_to_brain_query_state(tmp);
	}
	}
	return out;
}

fn unpack_brain_query_set(raw: array<u32, 140>) -> BrainQuerySet {
	return unpack_words_to_brain_query_set(raw);
}

fn pack_brain_query_set_to_words(unpacked: BrainQuerySet) -> array<u32, 140> {
	var out: array<u32, 140>;
	for (var w = 0u; w < 140u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.count);
	out[1u] = pack_band_mask_to_words(unpacked.primary);
	out[2u] = pack_band_mask_to_words(unpacked.revision);
	out[3u] = pack_band_mask_to_words(unpacked.reserved0);
	for (var i_0 = 0u; i_0 < 8u; i_0++) {
			{
		let tmp = pack_brain_query_state_to_words(unpacked.queries[i_0]);
		for (var j_1 = 0u; j_1 < 17u; j_1++) { out[4u + (i_0 * 17u) + j_1] = tmp[j_1]; }
	}
		}
	return out;
}

fn pack_brain_query_set(unpacked: BrainQuerySet) -> array<u32, 140> {
	return pack_brain_query_set_to_words(unpacked);
}

fn unpack_words_to_memory_config(raw: array<u32, 8>) -> MemoryConfig {
	var out: MemoryConfig;
	out.slotCount = unpack_words_to_band_mask(raw[0u]);
	out.activationDecay = bitcast<f32>(raw[1u]);
	out.familiarityDecay = bitcast<f32>(raw[2u]);
	out.familiarityGain = bitcast<f32>(raw[3u]);
	out.evictionHysteresis = bitcast<f32>(raw[4u]);
	out.minimumResidenceTicks = unpack_words_to_band_mask(raw[5u]);
	out.flags = unpack_words_to_band_mask(raw[6u]);
	out.reserved0 = unpack_words_to_band_mask(raw[7u]);
	return out;
}

fn unpack_memory_config(raw: array<u32, 8>) -> MemoryConfig {
	return unpack_words_to_memory_config(raw);
}

fn pack_memory_config_to_words(unpacked: MemoryConfig) -> array<u32, 8> {
	var out: array<u32, 8>;
	for (var w = 0u; w < 8u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.slotCount);
	out[1u] = bitcast<u32>(unpacked.activationDecay);
	out[2u] = bitcast<u32>(unpacked.familiarityDecay);
	out[3u] = bitcast<u32>(unpacked.familiarityGain);
	out[4u] = bitcast<u32>(unpacked.evictionHysteresis);
	out[5u] = pack_band_mask_to_words(unpacked.minimumResidenceTicks);
	out[6u] = pack_band_mask_to_words(unpacked.flags);
	out[7u] = pack_band_mask_to_words(unpacked.reserved0);
	return out;
}

fn pack_memory_config(unpacked: MemoryConfig) -> array<u32, 8> {
	return pack_memory_config_to_words(unpacked);
}

fn unpack_words_to_memory_trace(raw: array<u32, 123>) -> MemoryTrace {
	var out: MemoryTrace;
	{
		var tmp: array<u32, 3>;
		for (var j_0 = 0u; j_0 < 3u; j_0++) { tmp[j_0] = raw[0u + j_0]; }
		out.memoryRef = unpack_words_to_runtime_ref_handle(tmp);
	}
	{
		var tmp: array<u32, 3>;
		for (var j_0 = 0u; j_0 < 3u; j_0++) { tmp[j_0] = raw[3u + j_0]; }
		out.subject = unpack_words_to_runtime_ref_handle(tmp);
	}
	out.kind = MemoryTraceKind(extractBits(raw[6u], 0u, 8u));
	out.state = MemorySlotState(extractBits(raw[6u], 8u, 8u));
	out.flags = unpack_words_to_band_mask(raw[7u]);
	out.createdAt = unpack_words_to_band_mask(raw[8u]);
	out.lastObservedAt = unpack_words_to_band_mask(raw[9u]);
	out.lastAccessedAt = unpack_words_to_band_mask(raw[10u]);
	out.interactionCount = unpack_words_to_band_mask(raw[11u]);
	out.activation = bitcast<f32>(raw[12u]);
	out.familiarity = bitcast<f32>(raw[13u]);
	out.affectMagnitude = bitcast<f32>(raw[14u]);
	out.reserved0 = bitcast<f32>(raw[15u]);
	{
		var tmp: array<u32, 107>;
		for (var j_0 = 0u; j_0 < 107u; j_0++) { tmp[j_0] = raw[16u + j_0]; }
		out.rememberedRecord = unpack_words_to_brain_record_slot(tmp);
	}
	return out;
}

fn unpack_memory_trace(raw: array<u32, 123>) -> MemoryTrace {
	return unpack_words_to_memory_trace(raw);
}

fn pack_memory_trace_to_words(unpacked: MemoryTrace) -> array<u32, 123> {
	var out: array<u32, 123>;
	for (var w = 0u; w < 123u; w++) { out[w] = 0u; }
	{
		let tmp = pack_runtime_ref_handle_to_words(unpacked.memoryRef);
		for (var j_0 = 0u; j_0 < 3u; j_0++) { out[0u + j_0] = tmp[j_0]; }
	}
	{
		let tmp = pack_runtime_ref_handle_to_words(unpacked.subject);
		for (var j_0 = 0u; j_0 < 3u; j_0++) { out[3u + j_0] = tmp[j_0]; }
	}
	out[6u] = insertBits(out[6u], u32(unpacked.kind), 0u, 8u);
	out[6u] = insertBits(out[6u], u32(unpacked.state), 8u, 8u);
	out[7u] = pack_band_mask_to_words(unpacked.flags);
	out[8u] = pack_band_mask_to_words(unpacked.createdAt);
	out[9u] = pack_band_mask_to_words(unpacked.lastObservedAt);
	out[10u] = pack_band_mask_to_words(unpacked.lastAccessedAt);
	out[11u] = pack_band_mask_to_words(unpacked.interactionCount);
	out[12u] = bitcast<u32>(unpacked.activation);
	out[13u] = bitcast<u32>(unpacked.familiarity);
	out[14u] = bitcast<u32>(unpacked.affectMagnitude);
	out[15u] = bitcast<u32>(unpacked.reserved0);
	{
		let tmp = pack_brain_record_slot_to_words(unpacked.rememberedRecord);
		for (var j_0 = 0u; j_0 < 107u; j_0++) { out[16u + j_0] = tmp[j_0]; }
	}
	return out;
}

fn pack_memory_trace(unpacked: MemoryTrace) -> array<u32, 123> {
	return pack_memory_trace_to_words(unpacked);
}

fn unpack_words_to_memory_update(raw: array<u32, 10>) -> MemoryUpdate {
	var out: MemoryUpdate;
	{
		var tmp: array<u32, 3>;
		for (var j_0 = 0u; j_0 < 3u; j_0++) { tmp[j_0] = raw[0u + j_0]; }
		out.subject = unpack_words_to_runtime_ref_handle(tmp);
	}
	out.reason = MemoryUpdateReason(extractBits(raw[3u], 0u, 8u));
	out.interactionToken = unpack_words_to_band_mask(raw[4u]);
	out.flags = unpack_words_to_band_mask(raw[5u]);
	out.activationDelta = bitcast<f32>(raw[6u]);
	out.familiarityDelta = bitcast<f32>(raw[7u]);
	out.affectMagnitude = bitcast<f32>(raw[8u]);
	out.tick = unpack_words_to_band_mask(raw[9u]);
	return out;
}

fn unpack_memory_update(raw: array<u32, 10>) -> MemoryUpdate {
	return unpack_words_to_memory_update(raw);
}

fn pack_memory_update_to_words(unpacked: MemoryUpdate) -> array<u32, 10> {
	var out: array<u32, 10>;
	for (var w = 0u; w < 10u; w++) { out[w] = 0u; }
	{
		let tmp = pack_runtime_ref_handle_to_words(unpacked.subject);
		for (var j_0 = 0u; j_0 < 3u; j_0++) { out[0u + j_0] = tmp[j_0]; }
	}
	out[3u] = insertBits(out[3u], u32(unpacked.reason), 0u, 8u);
	out[4u] = pack_band_mask_to_words(unpacked.interactionToken);
	out[5u] = pack_band_mask_to_words(unpacked.flags);
	out[6u] = bitcast<u32>(unpacked.activationDelta);
	out[7u] = bitcast<u32>(unpacked.familiarityDelta);
	out[8u] = bitcast<u32>(unpacked.affectMagnitude);
	out[9u] = pack_band_mask_to_words(unpacked.tick);
	return out;
}

fn pack_memory_update(unpacked: MemoryUpdate) -> array<u32, 10> {
	return pack_memory_update_to_words(unpacked);
}

fn unpack_words_to_working_memory_state(raw: array<u32, 3940>) -> WorkingMemoryState {
	var out: WorkingMemoryState;
	out.revision = unpack_words_to_band_mask(raw[0u]);
	out.activeCount = unpack_words_to_band_mask(raw[1u]);
	out.evictedCount = unpack_words_to_band_mask(raw[2u]);
	out.flags = unpack_words_to_band_mask(raw[3u]);
	for (var i_0 = 0u; i_0 < 32u; i_0++) {
		{
		var tmp: array<u32, 123>;
		for (var j_1 = 0u; j_1 < 123u; j_1++) { tmp[j_1] = raw[4u + (i_0 * 123u) + j_1]; }
		out.slots[i_0] = unpack_words_to_memory_trace(tmp);
	}
	}
	return out;
}

fn unpack_working_memory_state(raw: array<u32, 3940>) -> WorkingMemoryState {
	return unpack_words_to_working_memory_state(raw);
}

fn pack_working_memory_state_to_words(unpacked: WorkingMemoryState) -> array<u32, 3940> {
	var out: array<u32, 3940>;
	for (var w = 0u; w < 3940u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.revision);
	out[1u] = pack_band_mask_to_words(unpacked.activeCount);
	out[2u] = pack_band_mask_to_words(unpacked.evictedCount);
	out[3u] = pack_band_mask_to_words(unpacked.flags);
	for (var i_0 = 0u; i_0 < 32u; i_0++) {
			{
		let tmp = pack_memory_trace_to_words(unpacked.slots[i_0]);
		for (var j_1 = 0u; j_1 < 123u; j_1++) { out[4u + (i_0 * 123u) + j_1] = tmp[j_1]; }
	}
		}
	return out;
}

fn pack_working_memory_state(unpacked: WorkingMemoryState) -> array<u32, 3940> {
	return pack_working_memory_state_to_words(unpacked);
}

fn unpack_words_to_action_intent_catalog_header(raw: array<u32, 8>) -> ActionIntentCatalogHeader {
	var out: ActionIntentCatalogHeader;
	out.version = unpack_words_to_band_mask(raw[0u]);
	out.intentCount = unpack_words_to_band_mask(raw[1u]);
	out.argumentCount = unpack_words_to_band_mask(raw[2u]);
	out.flags = unpack_words_to_band_mask(raw[3u]);
	out.catalogHashLo = unpack_words_to_band_mask(raw[4u]);
	out.catalogHashHi = unpack_words_to_band_mask(raw[5u]);
	out.reserved0 = unpack_words_to_band_mask(raw[6u]);
	out.reserved1 = unpack_words_to_band_mask(raw[7u]);
	return out;
}

fn unpack_action_intent_catalog_header(raw: array<u32, 8>) -> ActionIntentCatalogHeader {
	return unpack_words_to_action_intent_catalog_header(raw);
}

fn pack_action_intent_catalog_header_to_words(unpacked: ActionIntentCatalogHeader) -> array<u32, 8> {
	var out: array<u32, 8>;
	for (var w = 0u; w < 8u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.version);
	out[1u] = pack_band_mask_to_words(unpacked.intentCount);
	out[2u] = pack_band_mask_to_words(unpacked.argumentCount);
	out[3u] = pack_band_mask_to_words(unpacked.flags);
	out[4u] = pack_band_mask_to_words(unpacked.catalogHashLo);
	out[5u] = pack_band_mask_to_words(unpacked.catalogHashHi);
	out[6u] = pack_band_mask_to_words(unpacked.reserved0);
	out[7u] = pack_band_mask_to_words(unpacked.reserved1);
	return out;
}

fn pack_action_intent_catalog_header(unpacked: ActionIntentCatalogHeader) -> array<u32, 8> {
	return pack_action_intent_catalog_header_to_words(unpacked);
}

fn unpack_words_to_action_intent_descriptor(raw: array<u32, 12>) -> ActionIntentDescriptor {
	var out: ActionIntentDescriptor;
	out.intentId = unpack_words_to_band_mask(raw[0u]);
	out.actionToken = unpack_words_to_band_mask(raw[1u]);
	out.semanticIntentToken = unpack_words_to_band_mask(raw[2u]);
	out.domain = ActionIntentDomain(extractBits(raw[3u], 0u, 8u));
	out.actorSchemaId = unpack_words_to_band_mask(raw[4u]);
	out.argumentOffset = unpack_words_to_band_mask(raw[5u]);
	out.argumentCount = unpack_words_to_band_mask(raw[6u]);
	out.flags = unpack_words_to_band_mask(raw[7u]);
	out.effectClassToken = unpack_words_to_band_mask(raw[8u]);
	out.capabilityClassToken = unpack_words_to_band_mask(raw[9u]);
	out.preconditionClassToken = unpack_words_to_band_mask(raw[10u]);
	out.preferredControllerRole = unpack_words_to_band_mask(raw[11u]);
	return out;
}

fn unpack_action_intent_descriptor(raw: array<u32, 12>) -> ActionIntentDescriptor {
	return unpack_words_to_action_intent_descriptor(raw);
}

fn pack_action_intent_descriptor_to_words(unpacked: ActionIntentDescriptor) -> array<u32, 12> {
	var out: array<u32, 12>;
	for (var w = 0u; w < 12u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.intentId);
	out[1u] = pack_band_mask_to_words(unpacked.actionToken);
	out[2u] = pack_band_mask_to_words(unpacked.semanticIntentToken);
	out[3u] = insertBits(out[3u], u32(unpacked.domain), 0u, 8u);
	out[4u] = pack_band_mask_to_words(unpacked.actorSchemaId);
	out[5u] = pack_band_mask_to_words(unpacked.argumentOffset);
	out[6u] = pack_band_mask_to_words(unpacked.argumentCount);
	out[7u] = pack_band_mask_to_words(unpacked.flags);
	out[8u] = pack_band_mask_to_words(unpacked.effectClassToken);
	out[9u] = pack_band_mask_to_words(unpacked.capabilityClassToken);
	out[10u] = pack_band_mask_to_words(unpacked.preconditionClassToken);
	out[11u] = pack_band_mask_to_words(unpacked.preferredControllerRole);
	return out;
}

fn pack_action_intent_descriptor(unpacked: ActionIntentDescriptor) -> array<u32, 12> {
	return pack_action_intent_descriptor_to_words(unpacked);
}

fn unpack_words_to_action_argument_descriptor(raw: array<u32, 8>) -> ActionArgumentDescriptor {
	var out: ActionArgumentDescriptor;
	out.intentId = unpack_words_to_band_mask(raw[0u]);
	out.argumentIndex = unpack_words_to_band_mask(raw[1u]);
	out.roleToken = unpack_words_to_band_mask(raw[2u]);
	out.valueKind = BrainValueKind(extractBits(raw[3u], 0u, 8u));
	out.acceptedSchemaId = unpack_words_to_band_mask(raw[4u]);
	out.candidateBandMask = unpack_words_to_band_mask(raw[5u]);
	out.flags = unpack_words_to_band_mask(raw[6u]);
	out.reserved0 = unpack_words_to_band_mask(raw[7u]);
	return out;
}

fn unpack_action_argument_descriptor(raw: array<u32, 8>) -> ActionArgumentDescriptor {
	return unpack_words_to_action_argument_descriptor(raw);
}

fn pack_action_argument_descriptor_to_words(unpacked: ActionArgumentDescriptor) -> array<u32, 8> {
	var out: array<u32, 8>;
	for (var w = 0u; w < 8u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.intentId);
	out[1u] = pack_band_mask_to_words(unpacked.argumentIndex);
	out[2u] = pack_band_mask_to_words(unpacked.roleToken);
	out[3u] = insertBits(out[3u], u32(unpacked.valueKind), 0u, 8u);
	out[4u] = pack_band_mask_to_words(unpacked.acceptedSchemaId);
	out[5u] = pack_band_mask_to_words(unpacked.candidateBandMask);
	out[6u] = pack_band_mask_to_words(unpacked.flags);
	out[7u] = pack_band_mask_to_words(unpacked.reserved0);
	return out;
}

fn pack_action_argument_descriptor(unpacked: ActionArgumentDescriptor) -> array<u32, 8> {
	return pack_action_argument_descriptor_to_words(unpacked);
}

fn unpack_words_to_action_argument_authoring_spec(raw: array<u32, 12>) -> ActionArgumentAuthoringSpec {
	var out: ActionArgumentAuthoringSpec;
		out.roleToken = bitcast<u32>(raw[1u]);
	out.valueKind = BrainValueKind(extractBits(raw[3u], 0u, 8u));
					return out;
}

fn unpack_action_argument_authoring_spec(raw: array<u32, 12>) -> ActionArgumentAuthoringSpec {
	return unpack_words_to_action_argument_authoring_spec(raw);
}

fn pack_action_argument_authoring_spec_to_words(unpacked: ActionArgumentAuthoringSpec) -> array<u32, 12> {
	var out: array<u32, 12>;
	for (var w = 0u; w < 12u; w++) { out[w] = 0u; }
		out[1u] = bitcast<u32>(unpacked.roleToken);
	out[3u] = insertBits(out[3u], u32(unpacked.valueKind), 0u, 8u);
					return out;
}

fn pack_action_argument_authoring_spec(unpacked: ActionArgumentAuthoringSpec) -> array<u32, 12> {
	return pack_action_argument_authoring_spec_to_words(unpacked);
}

fn unpack_words_to_action_intent_authoring_spec(raw: array<u32, 23>) -> ActionIntentAuthoringSpec {
	var out: ActionIntentAuthoringSpec;
		out.actionToken = bitcast<u32>(raw[1u]);
	out.semanticIntentToken = bitcast<u32>(raw[3u]);
	out.domain = ActionIntentDomain(extractBits(raw[5u], 0u, 8u));
	for (var i_0 = 0u; i_0 < 0u; i_0++) {
		{
		var tmp: array<u32, 12>;
		for (var j_1 = 0u; j_1 < 12u; j_1++) { tmp[j_1] = raw[6u + (i_0 * 12u) + j_1]; }
		out.arguments[i_0] = unpack_words_to_action_argument_authoring_spec(tmp);
	}
	}
							return out;
}

fn unpack_action_intent_authoring_spec(raw: array<u32, 23>) -> ActionIntentAuthoringSpec {
	return unpack_words_to_action_intent_authoring_spec(raw);
}

fn pack_action_intent_authoring_spec_to_words(unpacked: ActionIntentAuthoringSpec) -> array<u32, 23> {
	var out: array<u32, 23>;
	for (var w = 0u; w < 23u; w++) { out[w] = 0u; }
		out[1u] = bitcast<u32>(unpacked.actionToken);
	out[3u] = bitcast<u32>(unpacked.semanticIntentToken);
	out[5u] = insertBits(out[5u], u32(unpacked.domain), 0u, 8u);
	for (var i_0 = 0u; i_0 < 0u; i_0++) {
			{
		let tmp = pack_action_argument_authoring_spec_to_words(unpacked.arguments[i_0]);
		for (var j_1 = 0u; j_1 < 12u; j_1++) { out[6u + (i_0 * 12u) + j_1] = tmp[j_1]; }
	}
		}
							return out;
}

fn pack_action_intent_authoring_spec(unpacked: ActionIntentAuthoringSpec) -> array<u32, 23> {
	return pack_action_intent_authoring_spec_to_words(unpacked);
}

fn unpack_words_to_soft_gather_result(raw: array<u32, 8>) -> SoftGatherResult {
	var out: SoftGatherResult;
	out.status = SoftGatherStatus(extractBits(raw[0u], 0u, 8u));
	out.selectedRecord = unpack_words_to_band_mask(raw[1u]);
	out.selectedField = unpack_words_to_band_mask(raw[2u]);
	out.selectedReference = unpack_words_to_band_mask(raw[3u]);
	out.candidateCount = unpack_words_to_band_mask(raw[4u]);
	out.probability = bitcast<f32>(raw[5u]);
	out.entropy = bitcast<f32>(raw[6u]);
	out.reserved0 = unpack_words_to_band_mask(raw[7u]);
	return out;
}

fn unpack_soft_gather_result(raw: array<u32, 8>) -> SoftGatherResult {
	return unpack_words_to_soft_gather_result(raw);
}

fn pack_soft_gather_result_to_words(unpacked: SoftGatherResult) -> array<u32, 8> {
	var out: array<u32, 8>;
	for (var w = 0u; w < 8u; w++) { out[w] = 0u; }
	out[0u] = insertBits(out[0u], u32(unpacked.status), 0u, 8u);
	out[1u] = pack_band_mask_to_words(unpacked.selectedRecord);
	out[2u] = pack_band_mask_to_words(unpacked.selectedField);
	out[3u] = pack_band_mask_to_words(unpacked.selectedReference);
	out[4u] = pack_band_mask_to_words(unpacked.candidateCount);
	out[5u] = bitcast<u32>(unpacked.probability);
	out[6u] = bitcast<u32>(unpacked.entropy);
	out[7u] = pack_band_mask_to_words(unpacked.reserved0);
	return out;
}

fn pack_soft_gather_result(unpacked: SoftGatherResult) -> array<u32, 8> {
	return pack_soft_gather_result_to_words(unpacked);
}

fn unpack_words_to_typed_argument_value(raw: array<u32, 15>) -> TypedArgumentValue {
	var out: TypedArgumentValue;
	out.kind = BrainValueKind(extractBits(raw[0u], 0u, 8u));
	out.token = unpack_words_to_band_mask(raw[1u]);
	out.flags = unpack_words_to_band_mask(raw[2u]);
	out.reserved0 = unpack_words_to_band_mask(raw[3u]);
	{
		var tmp: array<u32, 3>;
		for (var j_0 = 0u; j_0 < 3u; j_0++) { tmp[j_0] = raw[4u + j_0]; }
		out.handle = unpack_words_to_runtime_ref_handle(tmp);
	}
	{
		var tmp: array<u32, 8>;
		for (var j_0 = 0u; j_0 < 8u; j_0++) { tmp[j_0] = raw[7u + j_0]; }
		out.selector = unpack_words_to_soft_gather_result(tmp);
	}
	return out;
}

fn unpack_typed_argument_value(raw: array<u32, 15>) -> TypedArgumentValue {
	return unpack_words_to_typed_argument_value(raw);
}

fn pack_typed_argument_value_to_words(unpacked: TypedArgumentValue) -> array<u32, 15> {
	var out: array<u32, 15>;
	for (var w = 0u; w < 15u; w++) { out[w] = 0u; }
	out[0u] = insertBits(out[0u], u32(unpacked.kind), 0u, 8u);
	out[1u] = pack_band_mask_to_words(unpacked.token);
	out[2u] = pack_band_mask_to_words(unpacked.flags);
	out[3u] = pack_band_mask_to_words(unpacked.reserved0);
	{
		let tmp = pack_runtime_ref_handle_to_words(unpacked.handle);
		for (var j_0 = 0u; j_0 < 3u; j_0++) { out[4u + j_0] = tmp[j_0]; }
	}
	{
		let tmp = pack_soft_gather_result_to_words(unpacked.selector);
		for (var j_0 = 0u; j_0 < 8u; j_0++) { out[7u + j_0] = tmp[j_0]; }
	}
	return out;
}

fn pack_typed_argument_value(unpacked: TypedArgumentValue) -> array<u32, 15> {
	return pack_typed_argument_value_to_words(unpacked);
}

fn unpack_words_to_intent_proposal(raw: array<u32, 80>) -> IntentProposal {
	var out: IntentProposal;
	out.proposalSlot = unpack_words_to_band_mask(raw[0u]);
	out.lifecycle = IntentLifecycle(extractBits(raw[1u], 0u, 8u));
	out.intentId = unpack_words_to_band_mask(raw[2u]);
	out.flags = unpack_words_to_band_mask(raw[3u]);
	{
		var tmp: array<u32, 3>;
		for (var j_0 = 0u; j_0 < 3u; j_0++) { tmp[j_0] = raw[4u + j_0]; }
		out.intentRef = unpack_words_to_runtime_ref_handle(tmp);
	}
	{
		var tmp: array<u32, 3>;
		for (var j_0 = 0u; j_0 < 3u; j_0++) { tmp[j_0] = raw[7u + j_0]; }
		out.purposeGoal = unpack_words_to_runtime_ref_handle(tmp);
	}
	{
		var tmp: array<u32, 3>;
		for (var j_0 = 0u; j_0 < 3u; j_0++) { tmp[j_0] = raw[10u + j_0]; }
		out.controllerHint = unpack_words_to_runtime_ref_handle(tmp);
	}
	{
		var tmp: array<u32, 3>;
		for (var j_0 = 0u; j_0 < 3u; j_0++) { tmp[j_0] = raw[13u + j_0]; }
		out.topic = unpack_words_to_runtime_ref_handle(tmp);
	}
	out.activation = bitcast<f32>(raw[16u]);
	out.priority = bitcast<f32>(raw[17u]);
	out.persistence = bitcast<f32>(raw[18u]);
	out.confidence = bitcast<f32>(raw[19u]);
	for (var i_0 = 0u; i_0 < 4u; i_0++) {
		{
		var tmp: array<u32, 15>;
		for (var j_1 = 0u; j_1 < 15u; j_1++) { tmp[j_1] = raw[20u + (i_0 * 15u) + j_1]; }
		out.arguments[i_0] = unpack_words_to_typed_argument_value(tmp);
	}
	}
	return out;
}

fn unpack_intent_proposal(raw: array<u32, 80>) -> IntentProposal {
	return unpack_words_to_intent_proposal(raw);
}

fn pack_intent_proposal_to_words(unpacked: IntentProposal) -> array<u32, 80> {
	var out: array<u32, 80>;
	for (var w = 0u; w < 80u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.proposalSlot);
	out[1u] = insertBits(out[1u], u32(unpacked.lifecycle), 0u, 8u);
	out[2u] = pack_band_mask_to_words(unpacked.intentId);
	out[3u] = pack_band_mask_to_words(unpacked.flags);
	{
		let tmp = pack_runtime_ref_handle_to_words(unpacked.intentRef);
		for (var j_0 = 0u; j_0 < 3u; j_0++) { out[4u + j_0] = tmp[j_0]; }
	}
	{
		let tmp = pack_runtime_ref_handle_to_words(unpacked.purposeGoal);
		for (var j_0 = 0u; j_0 < 3u; j_0++) { out[7u + j_0] = tmp[j_0]; }
	}
	{
		let tmp = pack_runtime_ref_handle_to_words(unpacked.controllerHint);
		for (var j_0 = 0u; j_0 < 3u; j_0++) { out[10u + j_0] = tmp[j_0]; }
	}
	{
		let tmp = pack_runtime_ref_handle_to_words(unpacked.topic);
		for (var j_0 = 0u; j_0 < 3u; j_0++) { out[13u + j_0] = tmp[j_0]; }
	}
	out[16u] = bitcast<u32>(unpacked.activation);
	out[17u] = bitcast<u32>(unpacked.priority);
	out[18u] = bitcast<u32>(unpacked.persistence);
	out[19u] = bitcast<u32>(unpacked.confidence);
	for (var i_0 = 0u; i_0 < 4u; i_0++) {
			{
		let tmp = pack_typed_argument_value_to_words(unpacked.arguments[i_0]);
		for (var j_1 = 0u; j_1 < 15u; j_1++) { out[20u + (i_0 * 15u) + j_1] = tmp[j_1]; }
	}
		}
	return out;
}

fn pack_intent_proposal(unpacked: IntentProposal) -> array<u32, 80> {
	return pack_intent_proposal_to_words(unpacked);
}

fn unpack_words_to_intent_set(raw: array<u32, 644>) -> IntentSet {
	var out: IntentSet;
	out.tick = unpack_words_to_band_mask(raw[0u]);
	out.count = unpack_words_to_band_mask(raw[1u]);
	out.revision = unpack_words_to_band_mask(raw[2u]);
	out.flags = unpack_words_to_band_mask(raw[3u]);
	for (var i_0 = 0u; i_0 < 8u; i_0++) {
		{
		var tmp: array<u32, 80>;
		for (var j_1 = 0u; j_1 < 80u; j_1++) { tmp[j_1] = raw[4u + (i_0 * 80u) + j_1]; }
		out.proposals[i_0] = unpack_words_to_intent_proposal(tmp);
	}
	}
	return out;
}

fn unpack_intent_set(raw: array<u32, 644>) -> IntentSet {
	return unpack_words_to_intent_set(raw);
}

fn pack_intent_set_to_words(unpacked: IntentSet) -> array<u32, 644> {
	var out: array<u32, 644>;
	for (var w = 0u; w < 644u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.tick);
	out[1u] = pack_band_mask_to_words(unpacked.count);
	out[2u] = pack_band_mask_to_words(unpacked.revision);
	out[3u] = pack_band_mask_to_words(unpacked.flags);
	for (var i_0 = 0u; i_0 < 8u; i_0++) {
			{
		let tmp = pack_intent_proposal_to_words(unpacked.proposals[i_0]);
		for (var j_1 = 0u; j_1 < 80u; j_1++) { out[4u + (i_0 * 80u) + j_1] = tmp[j_1]; }
	}
		}
	return out;
}

fn pack_intent_set(unpacked: IntentSet) -> array<u32, 644> {
	return pack_intent_set_to_words(unpacked);
}

fn unpack_words_to_active_intent_state(raw: array<u32, 16>) -> ActiveIntentState {
	var out: ActiveIntentState;
	{
		var tmp: array<u32, 3>;
		for (var j_0 = 0u; j_0 < 3u; j_0++) { tmp[j_0] = raw[0u + j_0]; }
		out.intentRef = unpack_words_to_runtime_ref_handle(tmp);
	}
	{
		var tmp: array<u32, 3>;
		for (var j_0 = 0u; j_0 < 3u; j_0++) { tmp[j_0] = raw[3u + j_0]; }
		out.purposeGoal = unpack_words_to_runtime_ref_handle(tmp);
	}
	out.intentId = unpack_words_to_band_mask(raw[6u]);
	out.status = IntentExecutionStatus(extractBits(raw[7u], 0u, 8u));
	out.flags = unpack_words_to_band_mask(raw[8u]);
	out.startedAt = unpack_words_to_band_mask(raw[9u]);
	out.lastMaintainedAt = unpack_words_to_band_mask(raw[10u]);
	out.completedAt = unpack_words_to_band_mask(raw[11u]);
	out.activation = bitcast<f32>(raw[12u]);
	out.progress = bitcast<f32>(raw[13u]);
	out.outcomeMagnitude = bitcast<f32>(raw[14u]);
	out.reserved0 = bitcast<f32>(raw[15u]);
	return out;
}

fn unpack_active_intent_state(raw: array<u32, 16>) -> ActiveIntentState {
	return unpack_words_to_active_intent_state(raw);
}

fn pack_active_intent_state_to_words(unpacked: ActiveIntentState) -> array<u32, 16> {
	var out: array<u32, 16>;
	for (var w = 0u; w < 16u; w++) { out[w] = 0u; }
	{
		let tmp = pack_runtime_ref_handle_to_words(unpacked.intentRef);
		for (var j_0 = 0u; j_0 < 3u; j_0++) { out[0u + j_0] = tmp[j_0]; }
	}
	{
		let tmp = pack_runtime_ref_handle_to_words(unpacked.purposeGoal);
		for (var j_0 = 0u; j_0 < 3u; j_0++) { out[3u + j_0] = tmp[j_0]; }
	}
	out[6u] = pack_band_mask_to_words(unpacked.intentId);
	out[7u] = insertBits(out[7u], u32(unpacked.status), 0u, 8u);
	out[8u] = pack_band_mask_to_words(unpacked.flags);
	out[9u] = pack_band_mask_to_words(unpacked.startedAt);
	out[10u] = pack_band_mask_to_words(unpacked.lastMaintainedAt);
	out[11u] = pack_band_mask_to_words(unpacked.completedAt);
	out[12u] = bitcast<u32>(unpacked.activation);
	out[13u] = bitcast<u32>(unpacked.progress);
	out[14u] = bitcast<u32>(unpacked.outcomeMagnitude);
	out[15u] = bitcast<u32>(unpacked.reserved0);
	return out;
}

fn pack_active_intent_state(unpacked: ActiveIntentState) -> array<u32, 16> {
	return pack_active_intent_state_to_words(unpacked);
}

fn unpack_words_to_active_intent_table(raw: array<u32, 260>) -> ActiveIntentTable {
	var out: ActiveIntentTable;
	out.revision = unpack_words_to_band_mask(raw[0u]);
	out.activeCount = unpack_words_to_band_mask(raw[1u]);
	out.completedCount = unpack_words_to_band_mask(raw[2u]);
	out.flags = unpack_words_to_band_mask(raw[3u]);
	for (var i_0 = 0u; i_0 < 16u; i_0++) {
		{
		var tmp: array<u32, 16>;
		for (var j_1 = 0u; j_1 < 16u; j_1++) { tmp[j_1] = raw[4u + (i_0 * 16u) + j_1]; }
		out.intents[i_0] = unpack_words_to_active_intent_state(tmp);
	}
	}
	return out;
}

fn unpack_active_intent_table(raw: array<u32, 260>) -> ActiveIntentTable {
	return unpack_words_to_active_intent_table(raw);
}

fn pack_active_intent_table_to_words(unpacked: ActiveIntentTable) -> array<u32, 260> {
	var out: array<u32, 260>;
	for (var w = 0u; w < 260u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.revision);
	out[1u] = pack_band_mask_to_words(unpacked.activeCount);
	out[2u] = pack_band_mask_to_words(unpacked.completedCount);
	out[3u] = pack_band_mask_to_words(unpacked.flags);
	for (var i_0 = 0u; i_0 < 16u; i_0++) {
			{
		let tmp = pack_active_intent_state_to_words(unpacked.intents[i_0]);
		for (var j_1 = 0u; j_1 < 16u; j_1++) { out[4u + (i_0 * 16u) + j_1] = tmp[j_1]; }
	}
		}
	return out;
}

fn pack_active_intent_table(unpacked: ActiveIntentTable) -> array<u32, 260> {
	return pack_active_intent_table_to_words(unpacked);
}

fn unpack_words_to_intent_feedback(raw: array<u32, 12>) -> IntentFeedback {
	var out: IntentFeedback;
	{
		var tmp: array<u32, 3>;
		for (var j_0 = 0u; j_0 < 3u; j_0++) { tmp[j_0] = raw[0u + j_0]; }
		out.intentRef = unpack_words_to_runtime_ref_handle(tmp);
	}
	out.status = IntentExecutionStatus(extractBits(raw[3u], 0u, 8u));
	out.effectClassToken = unpack_words_to_band_mask(raw[4u]);
	out.resultToken = unpack_words_to_band_mask(raw[5u]);
	out.progress = bitcast<f32>(raw[6u]);
	out.outcomeMagnitude = bitcast<f32>(raw[7u]);
	out.comfortMagnitude = bitcast<f32>(raw[8u]);
	out.tick = unpack_words_to_band_mask(raw[9u]);
	out.feedbackRecord = unpack_words_to_band_mask(raw[10u]);
	out.flags = unpack_words_to_band_mask(raw[11u]);
	return out;
}

fn unpack_intent_feedback(raw: array<u32, 12>) -> IntentFeedback {
	return unpack_words_to_intent_feedback(raw);
}

fn pack_intent_feedback_to_words(unpacked: IntentFeedback) -> array<u32, 12> {
	var out: array<u32, 12>;
	for (var w = 0u; w < 12u; w++) { out[w] = 0u; }
	{
		let tmp = pack_runtime_ref_handle_to_words(unpacked.intentRef);
		for (var j_0 = 0u; j_0 < 3u; j_0++) { out[0u + j_0] = tmp[j_0]; }
	}
	out[3u] = insertBits(out[3u], u32(unpacked.status), 0u, 8u);
	out[4u] = pack_band_mask_to_words(unpacked.effectClassToken);
	out[5u] = pack_band_mask_to_words(unpacked.resultToken);
	out[6u] = bitcast<u32>(unpacked.progress);
	out[7u] = bitcast<u32>(unpacked.outcomeMagnitude);
	out[8u] = bitcast<u32>(unpacked.comfortMagnitude);
	out[9u] = pack_band_mask_to_words(unpacked.tick);
	out[10u] = pack_band_mask_to_words(unpacked.feedbackRecord);
	out[11u] = pack_band_mask_to_words(unpacked.flags);
	return out;
}

fn pack_intent_feedback(unpacked: IntentFeedback) -> array<u32, 12> {
	return pack_intent_feedback_to_words(unpacked);
}

fn unpack_words_to_tutorial_program_header(raw: array<u32, 10>) -> TutorialProgramHeader {
	var out: TutorialProgramHeader;
	out.version = unpack_words_to_band_mask(raw[0u]);
	out.lessonToken = unpack_words_to_band_mask(raw[1u]);
	out.beatOffset = unpack_words_to_band_mask(raw[2u]);
	out.beatCount = unpack_words_to_band_mask(raw[3u]);
	out.probeOffset = unpack_words_to_band_mask(raw[4u]);
	out.probeCount = unpack_words_to_band_mask(raw[5u]);
	out.creatorTokenOffset = unpack_words_to_band_mask(raw[6u]);
	out.creatorTokenCount = unpack_words_to_band_mask(raw[7u]);
	out.flags = unpack_words_to_band_mask(raw[8u]);
	out.reserved0 = unpack_words_to_band_mask(raw[9u]);
	return out;
}

fn unpack_tutorial_program_header(raw: array<u32, 10>) -> TutorialProgramHeader {
	return unpack_words_to_tutorial_program_header(raw);
}

fn pack_tutorial_program_header_to_words(unpacked: TutorialProgramHeader) -> array<u32, 10> {
	var out: array<u32, 10>;
	for (var w = 0u; w < 10u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.version);
	out[1u] = pack_band_mask_to_words(unpacked.lessonToken);
	out[2u] = pack_band_mask_to_words(unpacked.beatOffset);
	out[3u] = pack_band_mask_to_words(unpacked.beatCount);
	out[4u] = pack_band_mask_to_words(unpacked.probeOffset);
	out[5u] = pack_band_mask_to_words(unpacked.probeCount);
	out[6u] = pack_band_mask_to_words(unpacked.creatorTokenOffset);
	out[7u] = pack_band_mask_to_words(unpacked.creatorTokenCount);
	out[8u] = pack_band_mask_to_words(unpacked.flags);
	out[9u] = pack_band_mask_to_words(unpacked.reserved0);
	return out;
}

fn pack_tutorial_program_header(unpacked: TutorialProgramHeader) -> array<u32, 10> {
	return pack_tutorial_program_header_to_words(unpacked);
}

fn unpack_words_to_tutorial_beat(raw: array<u32, 8>) -> TutorialBeat {
	var out: TutorialBeat;
	out.kind = TutorialBeatKind(extractBits(raw[0u], 0u, 8u));
	out.sceneCue = unpack_words_to_band_mask(raw[1u]);
	out.utteranceOffset = unpack_words_to_band_mask(raw[2u]);
	out.utteranceCount = unpack_words_to_band_mask(raw[3u]);
	out.holdFrames = unpack_words_to_band_mask(raw[4u]);
	out.probeIndex = unpack_words_to_band_mask(raw[5u]);
	out.expectedIntentId = unpack_words_to_band_mask(raw[6u]);
	out.flags = unpack_words_to_band_mask(raw[7u]);
	return out;
}

fn unpack_tutorial_beat(raw: array<u32, 8>) -> TutorialBeat {
	return unpack_words_to_tutorial_beat(raw);
}

fn pack_tutorial_beat_to_words(unpacked: TutorialBeat) -> array<u32, 8> {
	var out: array<u32, 8>;
	for (var w = 0u; w < 8u; w++) { out[w] = 0u; }
	out[0u] = insertBits(out[0u], u32(unpacked.kind), 0u, 8u);
	out[1u] = pack_band_mask_to_words(unpacked.sceneCue);
	out[2u] = pack_band_mask_to_words(unpacked.utteranceOffset);
	out[3u] = pack_band_mask_to_words(unpacked.utteranceCount);
	out[4u] = pack_band_mask_to_words(unpacked.holdFrames);
	out[5u] = pack_band_mask_to_words(unpacked.probeIndex);
	out[6u] = pack_band_mask_to_words(unpacked.expectedIntentId);
	out[7u] = pack_band_mask_to_words(unpacked.flags);
	return out;
}

fn pack_tutorial_beat(unpacked: TutorialBeat) -> array<u32, 8> {
	return pack_tutorial_beat_to_words(unpacked);
}

fn unpack_words_to_tutorial_probe(raw: array<u32, 9>) -> TutorialProbe {
	var out: TutorialProbe;
	out.kind = TutorialProbeKind(extractBits(raw[0u], 0u, 8u));
	out.querySchemaId = unpack_words_to_band_mask(raw[1u]);
	out.expectedToken = unpack_words_to_band_mask(raw[2u]);
	out.expectedIntentId = unpack_words_to_band_mask(raw[3u]);
	out.expectedRecord = unpack_words_to_band_mask(raw[4u]);
	out.expectedField = unpack_words_to_band_mask(raw[5u]);
	out.oracleBinding = unpack_words_to_band_mask(raw[6u]);
	out.flags = unpack_words_to_band_mask(raw[7u]);
	out.reserved0 = unpack_words_to_band_mask(raw[8u]);
	return out;
}

fn unpack_tutorial_probe(raw: array<u32, 9>) -> TutorialProbe {
	return unpack_words_to_tutorial_probe(raw);
}

fn pack_tutorial_probe_to_words(unpacked: TutorialProbe) -> array<u32, 9> {
	var out: array<u32, 9>;
	for (var w = 0u; w < 9u; w++) { out[w] = 0u; }
	out[0u] = insertBits(out[0u], u32(unpacked.kind), 0u, 8u);
	out[1u] = pack_band_mask_to_words(unpacked.querySchemaId);
	out[2u] = pack_band_mask_to_words(unpacked.expectedToken);
	out[3u] = pack_band_mask_to_words(unpacked.expectedIntentId);
	out[4u] = pack_band_mask_to_words(unpacked.expectedRecord);
	out[5u] = pack_band_mask_to_words(unpacked.expectedField);
	out[6u] = pack_band_mask_to_words(unpacked.oracleBinding);
	out[7u] = pack_band_mask_to_words(unpacked.flags);
	out[8u] = pack_band_mask_to_words(unpacked.reserved0);
	return out;
}

fn pack_tutorial_probe(unpacked: TutorialProbe) -> array<u32, 9> {
	return pack_tutorial_probe_to_words(unpacked);
}

fn unpack_words_to_tutorial_runtime_state(raw: array<u32, 8>) -> TutorialRuntimeState {
	var out: TutorialRuntimeState;
	out.program = unpack_words_to_band_mask(raw[0u]);
	out.beat = unpack_words_to_band_mask(raw[1u]);
	out.probe = unpack_words_to_band_mask(raw[2u]);
	out.status = TutorialRuntimeStatus(extractBits(raw[3u], 0u, 8u));
	out.frameInBeat = unpack_words_to_band_mask(raw[4u]);
	out.attempts = unpack_words_to_band_mask(raw[5u]);
	out.correct = unpack_words_to_band_mask(raw[6u]);
	out.incorrect = unpack_words_to_band_mask(raw[7u]);
	return out;
}

fn unpack_tutorial_runtime_state(raw: array<u32, 8>) -> TutorialRuntimeState {
	return unpack_words_to_tutorial_runtime_state(raw);
}

fn pack_tutorial_runtime_state_to_words(unpacked: TutorialRuntimeState) -> array<u32, 8> {
	var out: array<u32, 8>;
	for (var w = 0u; w < 8u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.program);
	out[1u] = pack_band_mask_to_words(unpacked.beat);
	out[2u] = pack_band_mask_to_words(unpacked.probe);
	out[3u] = insertBits(out[3u], u32(unpacked.status), 0u, 8u);
	out[4u] = pack_band_mask_to_words(unpacked.frameInBeat);
	out[5u] = pack_band_mask_to_words(unpacked.attempts);
	out[6u] = pack_band_mask_to_words(unpacked.correct);
	out[7u] = pack_band_mask_to_words(unpacked.incorrect);
	return out;
}

fn pack_tutorial_runtime_state(unpacked: TutorialRuntimeState) -> array<u32, 8> {
	return pack_tutorial_runtime_state_to_words(unpacked);
}

fn unpack_words_to_tutorial_probe_authoring_spec(raw: array<u32, 12>) -> TutorialProbeAuthoringSpec {
	var out: TutorialProbeAuthoringSpec;
	out.kind = TutorialProbeKind(extractBits(raw[0u], 0u, 8u));
						return out;
}

fn unpack_tutorial_probe_authoring_spec(raw: array<u32, 12>) -> TutorialProbeAuthoringSpec {
	return unpack_words_to_tutorial_probe_authoring_spec(raw);
}

fn pack_tutorial_probe_authoring_spec_to_words(unpacked: TutorialProbeAuthoringSpec) -> array<u32, 12> {
	var out: array<u32, 12>;
	for (var w = 0u; w < 12u; w++) { out[w] = 0u; }
	out[0u] = insertBits(out[0u], u32(unpacked.kind), 0u, 8u);
						return out;
}

fn pack_tutorial_probe_authoring_spec(unpacked: TutorialProbeAuthoringSpec) -> array<u32, 12> {
	return pack_tutorial_probe_authoring_spec_to_words(unpacked);
}

fn unpack_words_to_tutorial_beat_authoring_spec(raw: array<u32, 25>) -> TutorialBeatAuthoringSpec {
	var out: TutorialBeatAuthoringSpec;
	out.kind = TutorialBeatKind(extractBits(raw[0u], 0u, 8u));
							return out;
}

fn unpack_tutorial_beat_authoring_spec(raw: array<u32, 25>) -> TutorialBeatAuthoringSpec {
	return unpack_words_to_tutorial_beat_authoring_spec(raw);
}

fn pack_tutorial_beat_authoring_spec_to_words(unpacked: TutorialBeatAuthoringSpec) -> array<u32, 25> {
	var out: array<u32, 25>;
	for (var w = 0u; w < 25u; w++) { out[w] = 0u; }
	out[0u] = insertBits(out[0u], u32(unpacked.kind), 0u, 8u);
							return out;
}

fn pack_tutorial_beat_authoring_spec(unpacked: TutorialBeatAuthoringSpec) -> array<u32, 25> {
	return pack_tutorial_beat_authoring_spec_to_words(unpacked);
}

fn unpack_words_to_tutorial_authoring_spec(raw: array<u32, 7>) -> TutorialAuthoringSpec {
	var out: TutorialAuthoringSpec;
		for (var i_0 = 0u; i_0 < 0u; i_0++) {
		out.lessonTokens[i_0] = bitcast<u32>(raw[1u + (i_0 * 1u)]);
	}
		for (var i_0 = 0u; i_0 < 0u; i_0++) {
		{
		var tmp: array<u32, 25>;
		for (var j_1 = 0u; j_1 < 25u; j_1++) { tmp[j_1] = raw[4u + (i_0 * 25u) + j_1]; }
		out.beats[i_0] = unpack_words_to_tutorial_beat_authoring_spec(tmp);
	}
	}
		return out;
}

fn unpack_tutorial_authoring_spec(raw: array<u32, 7>) -> TutorialAuthoringSpec {
	return unpack_words_to_tutorial_authoring_spec(raw);
}

fn pack_tutorial_authoring_spec_to_words(unpacked: TutorialAuthoringSpec) -> array<u32, 7> {
	var out: array<u32, 7>;
	for (var w = 0u; w < 7u; w++) { out[w] = 0u; }
		for (var i_0 = 0u; i_0 < 0u; i_0++) {
			out[1u + (i_0 * 1u)] = bitcast<u32>(unpacked.lessonTokens[i_0]);
		}
		for (var i_0 = 0u; i_0 < 0u; i_0++) {
			{
		let tmp = pack_tutorial_beat_authoring_spec_to_words(unpacked.beats[i_0]);
		for (var j_1 = 0u; j_1 < 25u; j_1++) { out[4u + (i_0 * 25u) + j_1] = tmp[j_1]; }
	}
		}
		return out;
}

fn pack_tutorial_authoring_spec(unpacked: TutorialAuthoringSpec) -> array<u32, 7> {
	return pack_tutorial_authoring_spec_to_words(unpacked);
}

fn unpack_words_to_brain_model_config(raw: array<u32, 10>) -> BrainModelConfig {
	var out: BrainModelConfig;
	out.vocabSize = unpack_words_to_band_mask(raw[0u]);
	out.contextTokens = unpack_words_to_band_mask(raw[1u]);
	out.recordWidth = unpack_words_to_band_mask(raw[2u]);
	out.recordSlots = unpack_words_to_band_mask(raw[3u]);
	out.hiddenSize = unpack_words_to_band_mask(raw[4u]);
	out.recordSize = unpack_words_to_band_mask(raw[5u]);
	out.layerCount = unpack_words_to_band_mask(raw[6u]);
	out.attentionHeads = unpack_words_to_band_mask(raw[7u]);
	out.maxIntentProposals = unpack_words_to_band_mask(raw[8u]);
	out.flags = unpack_words_to_band_mask(raw[9u]);
	return out;
}

fn unpack_brain_model_config(raw: array<u32, 10>) -> BrainModelConfig {
	return unpack_words_to_brain_model_config(raw);
}

fn pack_brain_model_config_to_words(unpacked: BrainModelConfig) -> array<u32, 10> {
	var out: array<u32, 10>;
	for (var w = 0u; w < 10u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.vocabSize);
	out[1u] = pack_band_mask_to_words(unpacked.contextTokens);
	out[2u] = pack_band_mask_to_words(unpacked.recordWidth);
	out[3u] = pack_band_mask_to_words(unpacked.recordSlots);
	out[4u] = pack_band_mask_to_words(unpacked.hiddenSize);
	out[5u] = pack_band_mask_to_words(unpacked.recordSize);
	out[6u] = pack_band_mask_to_words(unpacked.layerCount);
	out[7u] = pack_band_mask_to_words(unpacked.attentionHeads);
	out[8u] = pack_band_mask_to_words(unpacked.maxIntentProposals);
	out[9u] = pack_band_mask_to_words(unpacked.flags);
	return out;
}

fn pack_brain_model_config(unpacked: BrainModelConfig) -> array<u32, 10> {
	return pack_brain_model_config_to_words(unpacked);
}

fn unpack_words_to_brain_runtime_config(raw: array<u32, 27>) -> BrainRuntimeConfig {
	var out: BrainRuntimeConfig;
	out.tokenAbiVersion = unpack_words_to_band_mask(raw[0u]);
	out.architectureVersion = unpack_words_to_band_mask(raw[1u]);
	out.frameLayoutVersion = unpack_words_to_band_mask(raw[2u]);
	out.vocabManifestVersion = unpack_words_to_band_mask(raw[3u]);
	out.recordManifestVersion = unpack_words_to_band_mask(raw[4u]);
	out.actionCatalogVersion = unpack_words_to_band_mask(raw[5u]);
	out.tutorialVersion = unpack_words_to_band_mask(raw[6u]);
	out.flags = unpack_words_to_band_mask(raw[7u]);
	out.reserved0 = unpack_words_to_band_mask(raw[8u]);
	{
		var tmp: array<u32, 10>;
		for (var j_0 = 0u; j_0 < 10u; j_0++) { tmp[j_0] = raw[9u + j_0]; }
		out.model = unpack_words_to_brain_model_config(tmp);
	}
	{
		var tmp: array<u32, 8>;
		for (var j_0 = 0u; j_0 < 8u; j_0++) { tmp[j_0] = raw[19u + j_0]; }
		out.memory = unpack_words_to_memory_config(tmp);
	}
	return out;
}

fn unpack_brain_runtime_config(raw: array<u32, 27>) -> BrainRuntimeConfig {
	return unpack_words_to_brain_runtime_config(raw);
}

fn pack_brain_runtime_config_to_words(unpacked: BrainRuntimeConfig) -> array<u32, 27> {
	var out: array<u32, 27>;
	for (var w = 0u; w < 27u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.tokenAbiVersion);
	out[1u] = pack_band_mask_to_words(unpacked.architectureVersion);
	out[2u] = pack_band_mask_to_words(unpacked.frameLayoutVersion);
	out[3u] = pack_band_mask_to_words(unpacked.vocabManifestVersion);
	out[4u] = pack_band_mask_to_words(unpacked.recordManifestVersion);
	out[5u] = pack_band_mask_to_words(unpacked.actionCatalogVersion);
	out[6u] = pack_band_mask_to_words(unpacked.tutorialVersion);
	out[7u] = pack_band_mask_to_words(unpacked.flags);
	out[8u] = pack_band_mask_to_words(unpacked.reserved0);
	{
		let tmp = pack_brain_model_config_to_words(unpacked.model);
		for (var j_0 = 0u; j_0 < 10u; j_0++) { out[9u + j_0] = tmp[j_0]; }
	}
	{
		let tmp = pack_memory_config_to_words(unpacked.memory);
		for (var j_0 = 0u; j_0 < 8u; j_0++) { out[19u + j_0] = tmp[j_0]; }
	}
	return out;
}

fn pack_brain_runtime_config(unpacked: BrainRuntimeConfig) -> array<u32, 27> {
	return pack_brain_runtime_config_to_words(unpacked);
}

fn unpack_words_to_brain_runtime_state(raw: array<u32, 8>) -> BrainRuntimeState {
	var out: BrainRuntimeState;
	out.status = BrainRuntimeStatus(extractBits(raw[0u], 0u, 8u));
	out.tick = unpack_words_to_band_mask(raw[1u]);
	out.snapshot = unpack_words_to_band_mask(raw[2u]);
	out.frameRevision = unpack_words_to_band_mask(raw[3u]);
	out.memoryRevision = unpack_words_to_band_mask(raw[4u]);
	out.queryRevision = unpack_words_to_band_mask(raw[5u]);
	out.intentRevision = unpack_words_to_band_mask(raw[6u]);
	out.errorCode = unpack_words_to_band_mask(raw[7u]);
	return out;
}

fn unpack_brain_runtime_state(raw: array<u32, 8>) -> BrainRuntimeState {
	return unpack_words_to_brain_runtime_state(raw);
}

fn pack_brain_runtime_state_to_words(unpacked: BrainRuntimeState) -> array<u32, 8> {
	var out: array<u32, 8>;
	for (var w = 0u; w < 8u; w++) { out[w] = 0u; }
	out[0u] = insertBits(out[0u], u32(unpacked.status), 0u, 8u);
	out[1u] = pack_band_mask_to_words(unpacked.tick);
	out[2u] = pack_band_mask_to_words(unpacked.snapshot);
	out[3u] = pack_band_mask_to_words(unpacked.frameRevision);
	out[4u] = pack_band_mask_to_words(unpacked.memoryRevision);
	out[5u] = pack_band_mask_to_words(unpacked.queryRevision);
	out[6u] = pack_band_mask_to_words(unpacked.intentRevision);
	out[7u] = pack_band_mask_to_words(unpacked.errorCode);
	return out;
}

fn pack_brain_runtime_state(unpacked: BrainRuntimeState) -> array<u32, 8> {
	return pack_brain_runtime_state_to_words(unpacked);
}

fn unpack_words_to_brain_step_telemetry(raw: array<u32, 18>) -> BrainStepTelemetry {
	var out: BrainStepTelemetry;
	out.tick = unpack_words_to_band_mask(raw[0u]);
	out.activeRecords = unpack_words_to_band_mask(raw[1u]);
	out.activeTokens = unpack_words_to_band_mask(raw[2u]);
	out.truncatedRecords = unpack_words_to_band_mask(raw[3u]);
	out.intentCount = unpack_words_to_band_mask(raw[4u]);
	out.activeIntentCount = unpack_words_to_band_mask(raw[5u]);
	out.memoryCount = unpack_words_to_band_mask(raw[6u]);
	out.queryCount = unpack_words_to_band_mask(raw[7u]);
	out.frameBuildMs = bitcast<f32>(raw[8u]);
	out.localEncodeMs = bitcast<f32>(raw[9u]);
	out.recordMixMs = bitcast<f32>(raw[10u]);
	out.gatherMs = bitcast<f32>(raw[11u]);
	out.decideMs = bitcast<f32>(raw[12u]);
	out.runtimeMs = bitcast<f32>(raw[13u]);
	out.meanGatherEntropy = bitcast<f32>(raw[14u]);
	out.minGatherProbability = bitcast<f32>(raw[15u]);
	out.flags = unpack_words_to_band_mask(raw[16u]);
	out.errorCode = unpack_words_to_band_mask(raw[17u]);
	return out;
}

fn unpack_brain_step_telemetry(raw: array<u32, 18>) -> BrainStepTelemetry {
	return unpack_words_to_brain_step_telemetry(raw);
}

fn pack_brain_step_telemetry_to_words(unpacked: BrainStepTelemetry) -> array<u32, 18> {
	var out: array<u32, 18>;
	for (var w = 0u; w < 18u; w++) { out[w] = 0u; }
	out[0u] = pack_band_mask_to_words(unpacked.tick);
	out[1u] = pack_band_mask_to_words(unpacked.activeRecords);
	out[2u] = pack_band_mask_to_words(unpacked.activeTokens);
	out[3u] = pack_band_mask_to_words(unpacked.truncatedRecords);
	out[4u] = pack_band_mask_to_words(unpacked.intentCount);
	out[5u] = pack_band_mask_to_words(unpacked.activeIntentCount);
	out[6u] = pack_band_mask_to_words(unpacked.memoryCount);
	out[7u] = pack_band_mask_to_words(unpacked.queryCount);
	out[8u] = bitcast<u32>(unpacked.frameBuildMs);
	out[9u] = bitcast<u32>(unpacked.localEncodeMs);
	out[10u] = bitcast<u32>(unpacked.recordMixMs);
	out[11u] = bitcast<u32>(unpacked.gatherMs);
	out[12u] = bitcast<u32>(unpacked.decideMs);
	out[13u] = bitcast<u32>(unpacked.runtimeMs);
	out[14u] = bitcast<u32>(unpacked.meanGatherEntropy);
	out[15u] = bitcast<u32>(unpacked.minGatherProbability);
	out[16u] = pack_band_mask_to_words(unpacked.flags);
	out[17u] = pack_band_mask_to_words(unpacked.errorCode);
	return out;
}

fn pack_brain_step_telemetry(unpacked: BrainStepTelemetry) -> array<u32, 18> {
	return pack_brain_step_telemetry_to_words(unpacked);
}

fn unpack_words_to_brain_step_result(raw: array<u32, 670>) -> BrainStepResult {
	var out: BrainStepResult;
	{
		var tmp: array<u32, 8>;
		for (var j_0 = 0u; j_0 < 8u; j_0++) { tmp[j_0] = raw[0u + j_0]; }
		out.state = unpack_words_to_brain_runtime_state(tmp);
	}
	{
		var tmp: array<u32, 644>;
		for (var j_0 = 0u; j_0 < 644u; j_0++) { tmp[j_0] = raw[8u + j_0]; }
		out.intents = unpack_words_to_intent_set(tmp);
	}
	{
		var tmp: array<u32, 18>;
		for (var j_0 = 0u; j_0 < 18u; j_0++) { tmp[j_0] = raw[652u + j_0]; }
		out.telemetry = unpack_words_to_brain_step_telemetry(tmp);
	}
	return out;
}

fn unpack_brain_step_result(raw: array<u32, 670>) -> BrainStepResult {
	return unpack_words_to_brain_step_result(raw);
}

fn pack_brain_step_result_to_words(unpacked: BrainStepResult) -> array<u32, 670> {
	var out: array<u32, 670>;
	for (var w = 0u; w < 670u; w++) { out[w] = 0u; }
	{
		let tmp = pack_brain_runtime_state_to_words(unpacked.state);
		for (var j_0 = 0u; j_0 < 8u; j_0++) { out[0u + j_0] = tmp[j_0]; }
	}
	{
		let tmp = pack_intent_set_to_words(unpacked.intents);
		for (var j_0 = 0u; j_0 < 644u; j_0++) { out[8u + j_0] = tmp[j_0]; }
	}
	{
		let tmp = pack_brain_step_telemetry_to_words(unpacked.telemetry);
		for (var j_0 = 0u; j_0 < 18u; j_0++) { out[652u + j_0] = tmp[j_0]; }
	}
	return out;
}

fn pack_brain_step_result(unpacked: BrainStepResult) -> array<u32, 670> {
	return pack_brain_step_result_to_words(unpacked);
}

