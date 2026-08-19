// THIS FILE IS AUTO-GENERATED - DO NOT CHANGE

export namespace v1_0_0 {
	/** unsigned 32-bit integer */
	export type BandMask = number;
	
	/** unsigned 32-bit integer */
	export type KrystalTokenId = BandMask;
	
	/** unsigned 32-bit integer */
	export type SchemaId = BandMask;
	
	/** unsigned 32-bit integer */
	export type FieldId = BandMask;
	
	/** unsigned 32-bit integer */
	export type IntentId = BandMask;
	
	/** unsigned 32-bit integer */
	export type RecordIndex = BandMask;
	
	/** unsigned 32-bit integer */
	export type LocalTokenIndex = BandMask;
	
	export const KrystalTokenClass = {
		system: 0,
		structure: 1,
		operation: 2,
		object: 3,
		property: 4,
		quantity: 5,
		action: 6,
		reference: 7,
		relation: 8,
		logic: 9,
		temporal: 10,
		domain: 11,
		experimental: 12,
		context: 13,
	} as const;
	export type KrystalTokenClass = "system" | "structure" | "operation" | "object" | "property" | "quantity" | "action" | "reference" | "relation" | "logic" | "temporal" | "domain" | "experimental" | "context";
	
	export const BrainBandKind = {
		system: 0,
		homeostasis: 1,
		body: 2,
		vision: 3,
		audio: 4,
		olfaction: 5,
		taste: 6,
		touch: 7,
		memory: 8,
		focus: 9,
		query: 10,
		catalog: 11,
	} as const;
	export type BrainBandKind = "system" | "homeostasis" | "body" | "vision" | "audio" | "olfaction" | "taste" | "touch" | "memory" | "focus" | "query" | "catalog";
	
	export const PropositionModality = {
		declarative: 0,
		imperative: 1,
		interrogative: 2,
		implicative: 3,
	} as const;
	export type PropositionModality = "declarative" | "imperative" | "interrogative" | "implicative";
	
	export const BandPlacementPolicy = {
		fixed: 0,
		shuffled_records: 1,
		stable_resident: 2,
	} as const;
	export type BandPlacementPolicy = "fixed" | "shuffled_records" | "stable_resident";
	
	export const BandOverflowPolicy = {
		error: 0,
		truncate_low_salience: 1,
		evict_low_priority: 2,
		drop_oldest: 3,
	} as const;
	export type BandOverflowPolicy = "error" | "truncate_low_salience" | "evict_low_priority" | "drop_oldest";
	
	export const RecordSource = {
		runtime: 0,
		sensor: 1,
		body: 2,
		homeostasis: 3,
		memory: 4,
		focus: 5,
		query: 6,
		creator: 7,
		intent_feedback: 8,
	} as const;
	export type RecordSource = "runtime" | "sensor" | "body" | "homeostasis" | "memory" | "focus" | "query" | "creator" | "intent_feedback";
	
	export const RuntimeRefKind = {
		none: 0,
		entity: 1,
		value: 2,
		memory: 3,
		event: 4,
		goal: 5,
		intent: 6,
		snapshot: 7,
		controller: 8,
		topic: 9,
	} as const;
	export type RuntimeRefKind = "none" | "entity" | "value" | "memory" | "event" | "goal" | "intent" | "snapshot" | "controller" | "topic";
	
	export const RuntimeRefStatus = {
		invalid: 0,
		live: 1,
		stale: 2,
		historical: 3,
		destroyed: 4,
	} as const;
	export type RuntimeRefStatus = "invalid" | "live" | "stale" | "historical" | "destroyed";
	
	export interface RuntimeRefHandle {
		tokenId: BandMask;
		generation: BandMask;
		kind: RuntimeRefKind;
		status: RuntimeRefStatus;
	}
	
	export interface VocabManifestHeader {
		tokenAbiVersion: BandMask;
		manifestVersion: BandMask;
		vocabSize: BandMask;
		activeTokenCount: BandMask;
		embeddingRows: BandMask;
		manifestHashLo: BandMask;
		manifestHashHi: BandMask;
		reserved0: BandMask;
		reserved1: BandMask;
	}
	
	export interface VocabManifestEntry {
		tokenId: BandMask;
		tokenClass: KrystalTokenClass;
		flags: BandMask;
		arity: BandMask;
		semanticTypeToken: BandMask;
		inverseToken: BandMask;
		reserved0: BandMask;
		reserved1: BandMask;
	}
	
	export const BrainValueKind = {
		none: 0,
		token: 1,
		context_ref: 2,
		record_ref: 3,
		boolean_class: 4,
		scalar_band: 5,
		quantity_projection: 6,
		opaque_payload: 7,
	} as const;
	export type BrainValueKind = "none" | "token" | "context_ref" | "record_ref" | "boolean_class" | "scalar_band" | "quantity_projection" | "opaque_payload";
	
	export const QuantityKind = {
		signed: 0,
		unipolar: 1,
		count: 2,
		proportion: 3,
	} as const;
	export type QuantityKind = "signed" | "unipolar" | "count" | "proportion";
	
	export interface RecordSchemaManifestHeader {
		version: BandMask;
		schemaCount: BandMask;
		fieldCount: BandMask;
		maxRecordTokens: BandMask;
		schemaHashLo: BandMask;
		schemaHashHi: BandMask;
		reserved0: BandMask;
		reserved1: BandMask;
	}
	
	export interface RecordSchemaEntry {
		schemaId: BandMask;
		familyToken: BandMask;
		defaultBand: BrainBandKind;
		tokenCount: BandMask;
		fieldOffset: BandMask;
		fieldCount: BandMask;
		flags: BandMask;
		reserved0: BandMask;
	}
	
	export interface RecordFieldEntry {
		schemaId: BandMask;
		fieldId: BandMask;
		localTokenIndex: BandMask;
		tokenWidth: BandMask;
		roleToken: BandMask;
		valueKind: BrainValueKind;
		quantityKind: QuantityKind;
		acceptedSchemaId: BandMask;
		allowedBandMask: BandMask;
		flags: BandMask;
		reserved0: BandMask;
	}
	
	export interface TokenAuthoringSpec {
		id: number;
		symbol: string;
		tokenClass: KrystalTokenClass;
		semanticType?: string;
		arity?: number;
		doc?: string;
	}
	
	export interface RecordFieldAuthoringSpec {
		name: string;
		localTokenIndex: number;
		roleToken: number;
		valueKind: BrainValueKind;
		acceptedSchema?: string;
		allowedBands?: BrainBandKind[];
		required?: boolean;
		exactRuntime?: boolean;
		doc?: string;
	}
	
	export interface RecordSchemaAuthoringSpec {
		name: string;
		familyToken: number;
		defaultBand: BrainBandKind;
		fields: RecordFieldAuthoringSpec[];
		doc?: string;
	}
	
	export interface BrainBandLayout {
		kind: BrainBandKind;
		recordOffset: BandMask;
		recordCapacity: BandMask;
		tokenOffset: BandMask;
		tokenCapacity: BandMask;
		placement: BandPlacementPolicy;
		overflow: BandOverflowPolicy;
		flags: BandMask;
		reserved0: BandMask;
	}
	
	export interface FixedRecordBinding {
		roleToken: BandMask;
		recordIndex: BandMask;
		expectedSchemaId: BandMask;
		flags: BandMask;
	}
	
	export interface BrainFrameLayoutHeader {
		tokenAbiVersion: BandMask;
		architectureVersion: BandMask;
		layoutVersion: BandMask;
		recordWidth: BandMask;
		recordSlots: BandMask;
		tokenCapacity: BandMask;
		bandCount: BandMask;
		fixedRecordCount: BandMask;
		flags: BandMask;
		layoutHashLo: BandMask;
		layoutHashHi: BandMask;
	}
	
	export interface BrainFrameLayout {
		header: BrainFrameLayoutHeader;
		bands: BrainBandLayout[];
		fixedRecords: FixedRecordBinding[];
	}
	
	export interface BrainTokenMeta {
		fieldId: BandMask;
		roleToken: BandMask;
		flags: BandMask;
		referenceBinding: BandMask;
	}
	
	export interface BrainReferenceBinding {
		localTokenIndex: BandMask;
		fieldId: BandMask;
		flags: BandMask;
		reserved0: BandMask;
		handle: RuntimeRefHandle;
	}
	
	export interface BrainRecordHeader {
		schemaId: BandMask;
		band: BrainBandKind;
		source: RecordSource;
		modality: PropositionModality;
		flags: BandMask;
		tokenCount: BandMask;
		referenceCount: BandMask;
		observedAt: BandMask;
		revision: BandMask;
		primaryReference: BandMask;
		continuationRecord: BandMask;
		salience: number;
		freshness: number;
		previousObservedAt: BandMask;
		changeMagnitude: number;
		reserved0: BandMask;
		reserved1: BandMask;
	}
	
	export interface BrainRecordSlot {
		header: BrainRecordHeader;
		tokens: BandMask[];
		tokenMeta: BrainTokenMeta[];
		references: BrainReferenceBinding[];
	}
	
	export interface BrainBandState {
		kind: BrainBandKind;
		activeRecords: BandMask;
		activeTokens: BandMask;
		overflowRecords: BandMask;
		truncatedRecords: BandMask;
		revision: BandMask;
		flags: BandMask;
		reserved0: BandMask;
	}
	
	export interface BrainFrameHeader {
		tokenAbiVersion: BandMask;
		architectureVersion: BandMask;
		layoutVersion: BandMask;
		tick: BandMask;
		snapshot: BandMask;
		deltaMillis: number;
		activeRecordCount: BandMask;
		activeTokenCount: BandMask;
		activeQueryRecord: BandMask;
		actorRecord: BandMask;
		frameRevision: BandMask;
		memoryRevision: BandMask;
		intentRevision: BandMask;
		flags: BandMask;
	}
	
	export interface BrainFrame {
		header: BrainFrameHeader;
		bands: BrainBandState[];
		records: BrainRecordSlot[];
	}
	
	export interface BinaryLayoutPlanHeader {
		planVersion: BandMask;
		layoutVersion: BandMask;
		bufferCount: BandMask;
		recordSlots: BandMask;
		recordWidth: BandMask;
		tokenCapacity: BandMask;
		maxReferencesPerRecord: BandMask;
		planHashLo: BandMask;
		planHashHi: BandMask;
		flags: BandMask;
		reserved0: BandMask;
	}
	
	export interface BinaryLayoutBufferDesc {
		bufferId: BandMask;
		elementCount: BandMask;
		byteSize: BandMask;
		flags: BandMask;
	}
	
	export interface BinaryLayoutPlan {
		header: BinaryLayoutPlanHeader;
		buffers: BinaryLayoutBufferDesc[];
	}
	
	export interface BrainFrameGpu {
		header: BinaryLayoutPlanHeader;
		tokenIds: BandMask[];
		fieldRoles: BandMask[];
		attentionMask: BandMask[];
		schemaIds: BandMask[];
		bandIds: BandMask[];
		runtimeRefs: BandMask[];
		recordFlags: BandMask[];
		activeRecordIndices: BandMask[];
	}
	
	export interface HomeostasisSignal {
		channelToken: BandMask;
		currentStateToken: BandMask;
		desiredStateToken: BandMask;
		flags: BandMask;
		currentValue: number;
		targetValue: number;
		urgency: number;
		delta: number;
		source: RuntimeRefHandle;
	}
	
	export const BrainQueryKind = {
		none: 0,
		homeostasis: 1,
		tutorial: 2,
		external: 3,
		internal: 4,
		continuation: 5,
		runtime_feedback: 6,
	} as const;
	export type BrainQueryKind = "none" | "homeostasis" | "tutorial" | "external" | "internal" | "continuation" | "runtime_feedback";
	
	export interface ConceptRef {
		kind: BrainValueKind;
		token: BandMask;
		flags: BandMask;
		reserved0: BandMask;
		handle: RuntimeRefHandle;
	}
	
	export interface BrainQueryState {
		queryRef: RuntimeRefHandle;
		kind: BrainQueryKind;
		modality: PropositionModality;
		routeToken: BandMask;
		predicateToken: BandMask;
		subject: ConceptRef;
		object: ConceptRef;
		urgency: number;
		createdAt: BandMask;
		expiresAt: BandMask;
		flags: BandMask;
	}
	
	export interface BrainQuerySet {
		count: BandMask;
		primary: BandMask;
		revision: BandMask;
		reserved0: BandMask;
		queries: BrainQueryState[];
	}
	
	export const MemoryTraceKind = {
		none: 0,
		entity: 1,
		event: 2,
		goal: 3,
		intent: 4,
		rule: 5,
		topic: 6,
		observation: 7,
	} as const;
	export type MemoryTraceKind = "none" | "entity" | "event" | "goal" | "intent" | "rule" | "topic" | "observation";
	
	export const MemorySlotState = {
		empty: 0,
		active: 1,
		evictable: 2,
		evicted: 3,
	} as const;
	export type MemorySlotState = "empty" | "active" | "evictable" | "evicted";
	
	export const MemoryUpdateReason = {
		observation: 0,
		look: 1,
		interaction: 2,
		comfort_delta: 3,
		retrieval: 4,
		rehearsal: 5,
		goal: 6,
		intent: 7,
		decay: 8,
	} as const;
	export type MemoryUpdateReason = "observation" | "look" | "interaction" | "comfort_delta" | "retrieval" | "rehearsal" | "goal" | "intent" | "decay";
	
	export interface MemoryConfig {
		slotCount: BandMask;
		activationDecay: number;
		familiarityDecay: number;
		familiarityGain: number;
		evictionHysteresis: number;
		minimumResidenceTicks: BandMask;
		flags: BandMask;
		reserved0: BandMask;
	}
	
	export interface MemoryTrace {
		memoryRef: RuntimeRefHandle;
		subject: RuntimeRefHandle;
		kind: MemoryTraceKind;
		modality: PropositionModality;
		state: MemorySlotState;
		flags: BandMask;
		createdAt: BandMask;
		lastObservedAt: BandMask;
		lastAccessedAt: BandMask;
		interactionCount: BandMask;
		activation: number;
		familiarity: number;
		affectMagnitude: number;
		reserved0: number;
		rememberedRecord: BrainRecordSlot;
	}
	
	export interface MemoryUpdate {
		subject: RuntimeRefHandle;
		reason: MemoryUpdateReason;
		interactionToken: BandMask;
		flags: BandMask;
		activationDelta: number;
		familiarityDelta: number;
		affectMagnitude: number;
		tick: BandMask;
	}
	
	export interface WorkingMemoryState {
		revision: BandMask;
		activeCount: BandMask;
		evictedCount: BandMask;
		flags: BandMask;
		slots: MemoryTrace[];
	}
	
	export const ActionIntentDomain = {
		external: 0,
		perceptual: 1,
		internal: 2,
		communicative: 3,
		postural: 4,
	} as const;
	export type ActionIntentDomain = "external" | "perceptual" | "internal" | "communicative" | "postural";
	
	export interface ActionIntentCatalogHeader {
		version: BandMask;
		intentCount: BandMask;
		relationArity: BandMask;
		flags: BandMask;
		catalogHashLo: BandMask;
		catalogHashHi: BandMask;
		reserved0: BandMask;
		reserved1: BandMask;
	}
	
	export interface RelationRoleDescriptor {
		roleToken: BandMask;
		valueKind: BrainValueKind;
		acceptedTokens: BandMask[];
		candidateBandMask: BandMask;
		flags: BandMask;
		reserved0: BandMask;
	}
	
	export interface ActionIntentDescriptor {
		intentId: BandMask;
		actionToken: BandMask;
		semanticIntentToken: BandMask;
		domain: ActionIntentDomain;
		subjectSchemaId: BandMask;
		flags: BandMask;
		effectClassToken: BandMask;
		capabilityClassToken: BandMask;
		preconditionClassToken: BandMask;
		preferredControllerRole: BandMask;
		reserved0: BandMask;
		reserved1: BandMask;
		subjectRole: RelationRoleDescriptor;
		objectRole: RelationRoleDescriptor;
	}
	
	export interface RelationRoleAuthoringSpec {
		name: string;
		roleToken: number;
		valueKind: BrainValueKind;
		acceptedSchema?: string;
		candidateBands?: BrainBandKind[];
		doc?: string;
	}
	
	export interface ActionIntentAuthoringSpec {
		name: string;
		actionToken: number;
		semanticIntentToken: number;
		domain: ActionIntentDomain;
		subject: RelationRoleAuthoringSpec;
		object?: RelationRoleAuthoringSpec;
		effectClassToken?: number;
		capabilityClassToken?: number;
		preconditionClassToken?: number;
		preferredControllerRole?: number;
		durative?: boolean;
		doc?: string;
	}
	
	export const SoftGatherStatus = {
		empty: 0,
		selected: 1,
		masked: 2,
		ambiguous: 3,
		error: 4,
	} as const;
	export type SoftGatherStatus = "empty" | "selected" | "masked" | "ambiguous" | "error";
	
	export interface SoftGatherResult {
		status: SoftGatherStatus;
		selectedRecord: BandMask;
		selectedField: BandMask;
		selectedReference: BandMask;
		candidateCount: BandMask;
		probability: number;
		entropy: number;
		reserved0: BandMask;
	}
	
	export interface SelectedConceptRef {
		concept: ConceptRef;
		selector: SoftGatherResult;
	}
	
	export const IntentLifecycle = {
		empty: 0,
		start: 1,
		maintain: 2,
		stop: 3,
		resume: 4,
	} as const;
	export type IntentLifecycle = "empty" | "start" | "maintain" | "stop" | "resume";
	
	export const IntentExecutionStatus = {
		empty: 0,
		proposed: 1,
		accepted: 2,
		active: 3,
		succeeded: 4,
		partial: 5,
		failed: 6,
		cancelled: 7,
		forgotten: 8,
	} as const;
	export type IntentExecutionStatus = "empty" | "proposed" | "accepted" | "active" | "succeeded" | "partial" | "failed" | "cancelled" | "forgotten";
	
	export interface IntentProposal {
		proposalSlot: BandMask;
		lifecycle: IntentLifecycle;
		modality: PropositionModality;
		intentId: BandMask;
		flags: BandMask;
		intentRef: RuntimeRefHandle;
		purposeGoal: RuntimeRefHandle;
		controllerHint: RuntimeRefHandle;
		topic: RuntimeRefHandle;
		activation: number;
		priority: number;
		commitment: number;
		intensity: number;
		persistence: number;
		confidence: number;
		subject: SelectedConceptRef;
		object: SelectedConceptRef;
	}
	
	export interface IntentSet {
		tick: BandMask;
		count: BandMask;
		revision: BandMask;
		flags: BandMask;
		proposals: IntentProposal[];
	}
	
	export interface ActiveIntentState {
		intentRef: RuntimeRefHandle;
		purposeGoal: RuntimeRefHandle;
		intentId: BandMask;
		status: IntentExecutionStatus;
		flags: BandMask;
		startedAt: BandMask;
		lastMaintainedAt: BandMask;
		completedAt: BandMask;
		activation: number;
		progress: number;
		outcomeMagnitude: number;
		reserved0: number;
	}
	
	export interface ActiveIntentTable {
		revision: BandMask;
		activeCount: BandMask;
		completedCount: BandMask;
		flags: BandMask;
		intents: ActiveIntentState[];
	}
	
	export interface IntentFeedback {
		intentRef: RuntimeRefHandle;
		status: IntentExecutionStatus;
		effectClassToken: BandMask;
		resultToken: BandMask;
		progress: number;
		outcomeMagnitude: number;
		comfortMagnitude: number;
		tick: BandMask;
		feedbackRecord: BandMask;
		flags: BandMask;
	}
	
	export const TutorialBeatKind = {
		narrate: 0,
		ask: 1,
		present: 2,
		focus: 3,
		demonstrate: 4,
		show_reaction: 5,
		assess: 6,
		wait: 7,
		reset: 8,
	} as const;
	export type TutorialBeatKind = "narrate" | "ask" | "present" | "focus" | "demonstrate" | "show_reaction" | "assess" | "wait" | "reset";
	
	export const TutorialProbeKind = {
		token: 0,
		boolean: 1,
		pointer: 2,
		property: 3,
		intent: 4,
		consequence: 5,
		counterexample: 6,
		unknown: 7,
	} as const;
	export type TutorialProbeKind = "token" | "boolean" | "pointer" | "property" | "intent" | "consequence" | "counterexample" | "unknown";
	
	export interface TutorialProgramHeader {
		version: BandMask;
		lessonToken: BandMask;
		beatOffset: BandMask;
		beatCount: BandMask;
		probeOffset: BandMask;
		probeCount: BandMask;
		creatorTokenOffset: BandMask;
		creatorTokenCount: BandMask;
		flags: BandMask;
		reserved0: BandMask;
	}
	
	export interface TutorialBeat {
		kind: TutorialBeatKind;
		sceneCue: BandMask;
		utteranceOffset: BandMask;
		utteranceCount: BandMask;
		holdFrames: BandMask;
		probeIndex: BandMask;
		expectedIntentId: BandMask;
		flags: BandMask;
	}
	
	export interface TutorialProbe {
		kind: TutorialProbeKind;
		querySchemaId: BandMask;
		expectedToken: BandMask;
		expectedIntentId: BandMask;
		expectedRecord: BandMask;
		expectedField: BandMask;
		oracleBinding: BandMask;
		flags: BandMask;
		reserved0: BandMask;
	}
	
	export const TutorialRuntimeStatus = {
		idle: 0,
		demonstrating: 1,
		waiting: 2,
		assessing: 3,
		passed: 4,
		failed: 5,
		done: 6,
	} as const;
	export type TutorialRuntimeStatus = "idle" | "demonstrating" | "waiting" | "assessing" | "passed" | "failed" | "done";
	
	export interface TutorialRuntimeState {
		program: BandMask;
		beat: BandMask;
		probe: BandMask;
		status: TutorialRuntimeStatus;
		frameInBeat: BandMask;
		attempts: BandMask;
		correct: BandMask;
		incorrect: BandMask;
	}
	
	export interface TutorialProbeAuthoringSpec {
		kind: TutorialProbeKind;
		querySchema?: string;
		expectedToken?: number;
		expectedIntent?: string;
		oracleBinding?: string;
		doc?: string;
	}
	
	export interface TutorialBeatAuthoringSpec {
		kind: TutorialBeatKind;
		sceneCue?: string;
		utterance?: number[];
		holdFrames?: number;
		expectedIntent?: string;
		probe?: TutorialProbeAuthoringSpec;
		doc?: string;
	}
	
	export interface TutorialAuthoringSpec {
		name: string;
		lessonTokens: number[];
		prerequisites?: string[];
		beats: TutorialBeatAuthoringSpec[];
		doc?: string;
	}
	
	export const BrainRuntimeStatus = {
		idle: 0,
		assembling_frame: 1,
		running: 2,
		executing: 3,
		done: 4,
		error: 5,
	} as const;
	export type BrainRuntimeStatus = "idle" | "assembling_frame" | "running" | "executing" | "done" | "error";
	
	export interface BrainModelConfig {
		vocabSize: BandMask;
		refEmbeddingRows: BandMask;
		contextTokens: BandMask;
		recordWidth: BandMask;
		recordSlots: BandMask;
		hiddenSize: BandMask;
		recordSize: BandMask;
		layerCount: BandMask;
		attentionHeads: BandMask;
		maxIntentProposals: BandMask;
		flags: BandMask;
	}
	
	export interface BrainRuntimeConfig {
		tokenAbiVersion: BandMask;
		architectureVersion: BandMask;
		frameLayoutVersion: BandMask;
		vocabManifestVersion: BandMask;
		recordManifestVersion: BandMask;
		actionCatalogVersion: BandMask;
		tutorialVersion: BandMask;
		flags: BandMask;
		reserved0: BandMask;
		model: BrainModelConfig;
		memory: MemoryConfig;
	}
	
	export interface BrainRuntimeState {
		status: BrainRuntimeStatus;
		tick: BandMask;
		snapshot: BandMask;
		frameRevision: BandMask;
		memoryRevision: BandMask;
		queryRevision: BandMask;
		intentRevision: BandMask;
		errorCode: BandMask;
	}
	
	export interface BrainStepTelemetry {
		tick: BandMask;
		activeRecords: BandMask;
		activeTokens: BandMask;
		truncatedRecords: BandMask;
		intentCount: BandMask;
		activeIntentCount: BandMask;
		memoryCount: BandMask;
		queryCount: BandMask;
		frameBuildMs: number;
		localEncodeMs: number;
		recordMixMs: number;
		gatherMs: number;
		decideMs: number;
		runtimeMs: number;
		meanGatherEntropy: number;
		minGatherProbability: number;
		flags: BandMask;
		errorCode: BandMask;
	}
	
	export interface BrainStepResult {
		state: BrainRuntimeState;
		intents: IntentSet;
		telemetry: BrainStepTelemetry;
	}
	
	}
