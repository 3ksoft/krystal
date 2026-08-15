// THIS FILE IS AUTO-GENERATED - DO NOT CHANGE

import type { v1_0_0 } from "./krystal.types.ts";

type BandMask = v1_0_0.BandMask;
type KrystalTokenId = v1_0_0.KrystalTokenId;
type SchemaId = v1_0_0.SchemaId;
type FieldId = v1_0_0.FieldId;
type IntentId = v1_0_0.IntentId;
type RecordIndex = v1_0_0.RecordIndex;
type LocalTokenIndex = v1_0_0.LocalTokenIndex;
type KrystalTokenClass = v1_0_0.KrystalTokenClass;
type BrainBandKind = v1_0_0.BrainBandKind;
type BandPlacementPolicy = v1_0_0.BandPlacementPolicy;
type BandOverflowPolicy = v1_0_0.BandOverflowPolicy;
type RecordSource = v1_0_0.RecordSource;
type RuntimeRefKind = v1_0_0.RuntimeRefKind;
type RuntimeRefStatus = v1_0_0.RuntimeRefStatus;
type RuntimeRefHandle = v1_0_0.RuntimeRefHandle;
type VocabManifestHeader = v1_0_0.VocabManifestHeader;
type VocabManifestEntry = v1_0_0.VocabManifestEntry;
type BrainValueKind = v1_0_0.BrainValueKind;
type RecordSchemaManifestHeader = v1_0_0.RecordSchemaManifestHeader;
type RecordSchemaEntry = v1_0_0.RecordSchemaEntry;
type RecordFieldEntry = v1_0_0.RecordFieldEntry;
type TokenAuthoringSpec = v1_0_0.TokenAuthoringSpec;
type RecordFieldAuthoringSpec = v1_0_0.RecordFieldAuthoringSpec;
type RecordSchemaAuthoringSpec = v1_0_0.RecordSchemaAuthoringSpec;
type BrainBandLayout = v1_0_0.BrainBandLayout;
type FixedRecordBinding = v1_0_0.FixedRecordBinding;
type BrainFrameLayoutHeader = v1_0_0.BrainFrameLayoutHeader;
type BrainFrameLayout = v1_0_0.BrainFrameLayout;
type BrainTokenMeta = v1_0_0.BrainTokenMeta;
type BrainReferenceBinding = v1_0_0.BrainReferenceBinding;
type BrainRecordHeader = v1_0_0.BrainRecordHeader;
type BrainRecordSlot = v1_0_0.BrainRecordSlot;
type BrainBandState = v1_0_0.BrainBandState;
type BrainFrameHeader = v1_0_0.BrainFrameHeader;
type BrainFrame = v1_0_0.BrainFrame;
type HomeostasisSignal = v1_0_0.HomeostasisSignal;
type BrainQueryKind = v1_0_0.BrainQueryKind;
type BrainQueryState = v1_0_0.BrainQueryState;
type BrainQuerySet = v1_0_0.BrainQuerySet;
type MemoryTraceKind = v1_0_0.MemoryTraceKind;
type MemorySlotState = v1_0_0.MemorySlotState;
type MemoryUpdateReason = v1_0_0.MemoryUpdateReason;
type MemoryConfig = v1_0_0.MemoryConfig;
type MemoryTrace = v1_0_0.MemoryTrace;
type MemoryUpdate = v1_0_0.MemoryUpdate;
type WorkingMemoryState = v1_0_0.WorkingMemoryState;
type ActionIntentDomain = v1_0_0.ActionIntentDomain;
type ActionIntentCatalogHeader = v1_0_0.ActionIntentCatalogHeader;
type ActionIntentDescriptor = v1_0_0.ActionIntentDescriptor;
type ActionArgumentDescriptor = v1_0_0.ActionArgumentDescriptor;
type ActionArgumentAuthoringSpec = v1_0_0.ActionArgumentAuthoringSpec;
type ActionIntentAuthoringSpec = v1_0_0.ActionIntentAuthoringSpec;
type SoftGatherStatus = v1_0_0.SoftGatherStatus;
type SoftGatherResult = v1_0_0.SoftGatherResult;
type TypedArgumentValue = v1_0_0.TypedArgumentValue;
type IntentLifecycle = v1_0_0.IntentLifecycle;
type IntentExecutionStatus = v1_0_0.IntentExecutionStatus;
type IntentProposal = v1_0_0.IntentProposal;
type IntentSet = v1_0_0.IntentSet;
type ActiveIntentState = v1_0_0.ActiveIntentState;
type ActiveIntentTable = v1_0_0.ActiveIntentTable;
type IntentFeedback = v1_0_0.IntentFeedback;
type TutorialBeatKind = v1_0_0.TutorialBeatKind;
type TutorialProbeKind = v1_0_0.TutorialProbeKind;
type TutorialProgramHeader = v1_0_0.TutorialProgramHeader;
type TutorialBeat = v1_0_0.TutorialBeat;
type TutorialProbe = v1_0_0.TutorialProbe;
type TutorialRuntimeStatus = v1_0_0.TutorialRuntimeStatus;
type TutorialRuntimeState = v1_0_0.TutorialRuntimeState;
type TutorialProbeAuthoringSpec = v1_0_0.TutorialProbeAuthoringSpec;
type TutorialBeatAuthoringSpec = v1_0_0.TutorialBeatAuthoringSpec;
type TutorialAuthoringSpec = v1_0_0.TutorialAuthoringSpec;
type BrainRuntimeStatus = v1_0_0.BrainRuntimeStatus;
type BrainModelConfig = v1_0_0.BrainModelConfig;
type BrainRuntimeConfig = v1_0_0.BrainRuntimeConfig;
type BrainRuntimeState = v1_0_0.BrainRuntimeState;
type BrainStepTelemetry = v1_0_0.BrainStepTelemetry;
type BrainStepResult = v1_0_0.BrainStepResult;

const __textDecoder = typeof TextDecoder !== "undefined" ? new TextDecoder() : null;
const __textEncoder = typeof TextEncoder !== "undefined" ? new TextEncoder() : null;

export const SIZEOF_BandMask = 4;
export const SIZEOF_KrystalTokenId = 4;
export const SIZEOF_SchemaId = 4;
export const SIZEOF_FieldId = 4;
export const SIZEOF_IntentId = 4;
export const SIZEOF_RecordIndex = 4;
export const SIZEOF_LocalTokenIndex = 4;
export const SIZEOF_RuntimeRefHandle = 12;
export const SIZEOF_VocabManifestHeader = 32;
export const SIZEOF_VocabManifestEntry = 32;
export const SIZEOF_RecordSchemaManifestHeader = 32;
export const SIZEOF_RecordSchemaEntry = 32;
export const SIZEOF_RecordFieldEntry = 40;
export const SIZEOF_TokenAuthoringSpec = 44;
export const SIZEOF_RecordFieldAuthoringSpec = 64;
export const SIZEOF_RecordSchemaAuthoringSpec = 28;
export const SIZEOF_BrainBandLayout = 32;
export const SIZEOF_FixedRecordBinding = 16;
export const SIZEOF_BrainFrameLayoutHeader = 44;
export const SIZEOF_BrainFrameLayout = 908;
export const BRAIN_FRAME_LAYOUT_BANDS_LEN = 11;
export const BRAIN_FRAME_LAYOUT_FIXEDRECORDS_LEN = 32;
export const SIZEOF_BrainTokenMeta = 16;
export const SIZEOF_BrainReferenceBinding = 28;
export const SIZEOF_BrainRecordHeader = 44;
export const SIZEOF_BrainRecordSlot = 428;
export const BRAIN_RECORD_SLOT_TOKENS_LEN = 8;
export const BRAIN_RECORD_SLOT_TOKENMETA_LEN = 8;
export const BRAIN_RECORD_SLOT_REFERENCES_LEN = 8;
export const SIZEOF_BrainBandState = 32;
export const SIZEOF_BrainFrameHeader = 52;
export const SIZEOF_BrainFrame = 55188;
export const BRAIN_FRAME_BANDS_LEN = 11;
export const BRAIN_FRAME_RECORDS_LEN = 128;
export const SIZEOF_HomeostasisSignal = 44;
export const SIZEOF_BrainQueryState = 68;
export const SIZEOF_BrainQuerySet = 560;
export const BRAIN_QUERY_SET_QUERIES_LEN = 8;
export const SIZEOF_MemoryConfig = 32;
export const SIZEOF_MemoryTrace = 492;
export const SIZEOF_MemoryUpdate = 40;
export const SIZEOF_WorkingMemoryState = 15760;
export const WORKING_MEMORY_STATE_SLOTS_LEN = 32;
export const SIZEOF_ActionIntentCatalogHeader = 32;
export const SIZEOF_ActionIntentDescriptor = 48;
export const SIZEOF_ActionArgumentDescriptor = 32;
export const SIZEOF_ActionArgumentAuthoringSpec = 48;
export const SIZEOF_ActionIntentAuthoringSpec = 92;
export const SIZEOF_SoftGatherResult = 32;
export const SIZEOF_TypedArgumentValue = 60;
export const SIZEOF_IntentProposal = 320;
export const INTENT_PROPOSAL_ARGUMENTS_LEN = 4;
export const SIZEOF_IntentSet = 2576;
export const INTENT_SET_PROPOSALS_LEN = 8;
export const SIZEOF_ActiveIntentState = 64;
export const SIZEOF_ActiveIntentTable = 1040;
export const ACTIVE_INTENT_TABLE_INTENTS_LEN = 16;
export const SIZEOF_IntentFeedback = 48;
export const SIZEOF_TutorialProgramHeader = 40;
export const SIZEOF_TutorialBeat = 32;
export const SIZEOF_TutorialProbe = 36;
export const SIZEOF_TutorialRuntimeState = 32;
export const SIZEOF_TutorialProbeAuthoringSpec = 48;
export const SIZEOF_TutorialBeatAuthoringSpec = 100;
export const SIZEOF_TutorialAuthoringSpec = 28;
export const SIZEOF_BrainModelConfig = 40;
export const SIZEOF_BrainRuntimeConfig = 108;
export const SIZEOF_BrainRuntimeState = 32;
export const SIZEOF_BrainStepTelemetry = 72;
export const SIZEOF_BrainStepResult = 2680;

export function deserializeBandMask(view: DataView, offset: number): BandMask {
	return view.getUint32(offset, true) as any;
}

export function serializeBandMask(val: BandMask, view: DataView, offset: number): void {
	view.setUint32(offset, val, true);
}

export function deserializeKrystalTokenId(view: DataView, offset: number): KrystalTokenId {
	return view.getUint32(offset, true) as any;
}

export function serializeKrystalTokenId(val: KrystalTokenId, view: DataView, offset: number): void {
	view.setUint32(offset, val, true);
}

export function deserializeSchemaId(view: DataView, offset: number): SchemaId {
	return view.getUint32(offset, true) as any;
}

export function serializeSchemaId(val: SchemaId, view: DataView, offset: number): void {
	view.setUint32(offset, val, true);
}

export function deserializeFieldId(view: DataView, offset: number): FieldId {
	return view.getUint32(offset, true) as any;
}

export function serializeFieldId(val: FieldId, view: DataView, offset: number): void {
	view.setUint32(offset, val, true);
}

export function deserializeIntentId(view: DataView, offset: number): IntentId {
	return view.getUint32(offset, true) as any;
}

export function serializeIntentId(val: IntentId, view: DataView, offset: number): void {
	view.setUint32(offset, val, true);
}

export function deserializeRecordIndex(view: DataView, offset: number): RecordIndex {
	return view.getUint32(offset, true) as any;
}

export function serializeRecordIndex(val: RecordIndex, view: DataView, offset: number): void {
	view.setUint32(offset, val, true);
}

export function deserializeLocalTokenIndex(view: DataView, offset: number): LocalTokenIndex {
	return view.getUint32(offset, true) as any;
}

export function serializeLocalTokenIndex(val: LocalTokenIndex, view: DataView, offset: number): void {
	view.setUint32(offset, val, true);
}

export function deserializeKrystalTokenClass(view: DataView, offset: number): KrystalTokenClass {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "system";
		case 1: return "structure";
		case 2: return "operation";
		case 3: return "object";
		case 4: return "property";
		case 5: return "quantity";
		case 6: return "action";
		case 7: return "reference";
		case 8: return "relation";
		case 9: return "logic";
		case 10: return "domain";
		case 11: return "context";
		case 12: return "experimental";
		default: throw new Error("Unknown Enum value for KrystalTokenClass: " + v);
	}
}

export function serializeKrystalTokenClass(val: KrystalTokenClass, view: DataView, offset: number): void {
	if(val === "system") { view.setUint8(offset, 0); return; }
	if(val === "structure") { view.setUint8(offset, 1); return; }
	if(val === "operation") { view.setUint8(offset, 2); return; }
	if(val === "object") { view.setUint8(offset, 3); return; }
	if(val === "property") { view.setUint8(offset, 4); return; }
	if(val === "quantity") { view.setUint8(offset, 5); return; }
	if(val === "action") { view.setUint8(offset, 6); return; }
	if(val === "reference") { view.setUint8(offset, 7); return; }
	if(val === "relation") { view.setUint8(offset, 8); return; }
	if(val === "logic") { view.setUint8(offset, 9); return; }
	if(val === "domain") { view.setUint8(offset, 10); return; }
	if(val === "context") { view.setUint8(offset, 11); return; }
	if(val === "experimental") { view.setUint8(offset, 12); return; }
}

export function deserializeBrainBandKind(view: DataView, offset: number): BrainBandKind {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "system";
		case 1: return "homeostasis";
		case 2: return "body";
		case 3: return "vision";
		case 4: return "audio";
		case 5: return "olfaction";
		case 6: return "taste";
		case 7: return "touch";
		case 8: return "memory";
		case 9: return "focus";
		case 10: return "query";
		default: throw new Error("Unknown Enum value for BrainBandKind: " + v);
	}
}

export function serializeBrainBandKind(val: BrainBandKind, view: DataView, offset: number): void {
	if(val === "system") { view.setUint8(offset, 0); return; }
	if(val === "homeostasis") { view.setUint8(offset, 1); return; }
	if(val === "body") { view.setUint8(offset, 2); return; }
	if(val === "vision") { view.setUint8(offset, 3); return; }
	if(val === "audio") { view.setUint8(offset, 4); return; }
	if(val === "olfaction") { view.setUint8(offset, 5); return; }
	if(val === "taste") { view.setUint8(offset, 6); return; }
	if(val === "touch") { view.setUint8(offset, 7); return; }
	if(val === "memory") { view.setUint8(offset, 8); return; }
	if(val === "focus") { view.setUint8(offset, 9); return; }
	if(val === "query") { view.setUint8(offset, 10); return; }
}

export function deserializeBandPlacementPolicy(view: DataView, offset: number): BandPlacementPolicy {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "fixed";
		case 1: return "shuffled_records";
		case 2: return "stable_resident";
		default: throw new Error("Unknown Enum value for BandPlacementPolicy: " + v);
	}
}

export function serializeBandPlacementPolicy(val: BandPlacementPolicy, view: DataView, offset: number): void {
	if(val === "fixed") { view.setUint8(offset, 0); return; }
	if(val === "shuffled_records") { view.setUint8(offset, 1); return; }
	if(val === "stable_resident") { view.setUint8(offset, 2); return; }
}

export function deserializeBandOverflowPolicy(view: DataView, offset: number): BandOverflowPolicy {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "error";
		case 1: return "truncate_low_salience";
		case 2: return "evict_low_priority";
		case 3: return "drop_oldest";
		default: throw new Error("Unknown Enum value for BandOverflowPolicy: " + v);
	}
}

export function serializeBandOverflowPolicy(val: BandOverflowPolicy, view: DataView, offset: number): void {
	if(val === "error") { view.setUint8(offset, 0); return; }
	if(val === "truncate_low_salience") { view.setUint8(offset, 1); return; }
	if(val === "evict_low_priority") { view.setUint8(offset, 2); return; }
	if(val === "drop_oldest") { view.setUint8(offset, 3); return; }
}

export function deserializeRecordSource(view: DataView, offset: number): RecordSource {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "runtime";
		case 1: return "sensor";
		case 2: return "body";
		case 3: return "homeostasis";
		case 4: return "memory";
		case 5: return "focus";
		case 6: return "query";
		case 7: return "creator";
		case 8: return "intent_feedback";
		default: throw new Error("Unknown Enum value for RecordSource: " + v);
	}
}

export function serializeRecordSource(val: RecordSource, view: DataView, offset: number): void {
	if(val === "runtime") { view.setUint8(offset, 0); return; }
	if(val === "sensor") { view.setUint8(offset, 1); return; }
	if(val === "body") { view.setUint8(offset, 2); return; }
	if(val === "homeostasis") { view.setUint8(offset, 3); return; }
	if(val === "memory") { view.setUint8(offset, 4); return; }
	if(val === "focus") { view.setUint8(offset, 5); return; }
	if(val === "query") { view.setUint8(offset, 6); return; }
	if(val === "creator") { view.setUint8(offset, 7); return; }
	if(val === "intent_feedback") { view.setUint8(offset, 8); return; }
}

export function deserializeRuntimeRefKind(view: DataView, offset: number): RuntimeRefKind {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "none";
		case 1: return "entity";
		case 2: return "value";
		case 3: return "memory";
		case 4: return "event";
		case 5: return "goal";
		case 6: return "intent";
		case 7: return "snapshot";
		case 8: return "controller";
		case 9: return "topic";
		default: throw new Error("Unknown Enum value for RuntimeRefKind: " + v);
	}
}

export function serializeRuntimeRefKind(val: RuntimeRefKind, view: DataView, offset: number): void {
	if(val === "none") { view.setUint8(offset, 0); return; }
	if(val === "entity") { view.setUint8(offset, 1); return; }
	if(val === "value") { view.setUint8(offset, 2); return; }
	if(val === "memory") { view.setUint8(offset, 3); return; }
	if(val === "event") { view.setUint8(offset, 4); return; }
	if(val === "goal") { view.setUint8(offset, 5); return; }
	if(val === "intent") { view.setUint8(offset, 6); return; }
	if(val === "snapshot") { view.setUint8(offset, 7); return; }
	if(val === "controller") { view.setUint8(offset, 8); return; }
	if(val === "topic") { view.setUint8(offset, 9); return; }
}

export function deserializeRuntimeRefStatus(view: DataView, offset: number): RuntimeRefStatus {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "invalid";
		case 1: return "live";
		case 2: return "stale";
		case 3: return "historical";
		case 4: return "destroyed";
		default: throw new Error("Unknown Enum value for RuntimeRefStatus: " + v);
	}
}

export function serializeRuntimeRefStatus(val: RuntimeRefStatus, view: DataView, offset: number): void {
	if(val === "invalid") { view.setUint8(offset, 0); return; }
	if(val === "live") { view.setUint8(offset, 1); return; }
	if(val === "stale") { view.setUint8(offset, 2); return; }
	if(val === "historical") { view.setUint8(offset, 3); return; }
	if(val === "destroyed") { view.setUint8(offset, 4); return; }
}

export function deserializeRuntimeRefHandle(view: DataView, offset: number, outObj?: any): RuntimeRefHandle {
	if (!outObj) {
		return {
			tokenId: view.getUint32(offset, true),
			generation: view.getUint32(offset + 4, true),
			kind: deserializeRuntimeRefKind(view, offset + 8),
			status: deserializeRuntimeRefStatus(view, offset + 9),
		} as any;
	}
	outObj.tokenId = view.getUint32(offset, true);
	outObj.generation = view.getUint32(offset + 4, true);
	outObj.kind = deserializeRuntimeRefKind(view, offset + 8);
	outObj.status = deserializeRuntimeRefStatus(view, offset + 9);
	return outObj;
}

export function serializeRuntimeRefHandle(val: RuntimeRefHandle, view: DataView, offset: number): void {
	view.setUint32(offset, val.tokenId, true);
	view.setUint32(offset + 4, val.generation, true);
	serializeRuntimeRefKind(val.kind, view, offset + 8);
	serializeRuntimeRefStatus(val.status, view, offset + 9);
}

export function deserializeVocabManifestHeader(view: DataView, offset: number, outObj?: any): VocabManifestHeader {
	if (!outObj) {
		return {
			tokenAbiVersion: view.getUint32(offset, true),
			manifestVersion: view.getUint32(offset + 4, true),
			vocabSize: view.getUint32(offset + 8, true),
			activeTokenCount: view.getUint32(offset + 12, true),
			manifestHashLo: view.getUint32(offset + 16, true),
			manifestHashHi: view.getUint32(offset + 20, true),
			reserved0: view.getUint32(offset + 24, true),
			reserved1: view.getUint32(offset + 28, true),
		} as any;
	}
	outObj.tokenAbiVersion = view.getUint32(offset, true);
	outObj.manifestVersion = view.getUint32(offset + 4, true);
	outObj.vocabSize = view.getUint32(offset + 8, true);
	outObj.activeTokenCount = view.getUint32(offset + 12, true);
	outObj.manifestHashLo = view.getUint32(offset + 16, true);
	outObj.manifestHashHi = view.getUint32(offset + 20, true);
	outObj.reserved0 = view.getUint32(offset + 24, true);
	outObj.reserved1 = view.getUint32(offset + 28, true);
	return outObj;
}

export function serializeVocabManifestHeader(val: VocabManifestHeader, view: DataView, offset: number): void {
	view.setUint32(offset, val.tokenAbiVersion, true);
	view.setUint32(offset + 4, val.manifestVersion, true);
	view.setUint32(offset + 8, val.vocabSize, true);
	view.setUint32(offset + 12, val.activeTokenCount, true);
	view.setUint32(offset + 16, val.manifestHashLo, true);
	view.setUint32(offset + 20, val.manifestHashHi, true);
	view.setUint32(offset + 24, val.reserved0, true);
	view.setUint32(offset + 28, val.reserved1, true);
}

export function deserializeVocabManifestEntry(view: DataView, offset: number, outObj?: any): VocabManifestEntry {
	if (!outObj) {
		return {
			tokenId: view.getUint32(offset, true),
			tokenClass: deserializeKrystalTokenClass(view, offset + 4),
			flags: view.getUint32(offset + 8, true),
			arity: view.getUint32(offset + 12, true),
			semanticTypeToken: view.getUint32(offset + 16, true),
			inverseToken: view.getUint32(offset + 20, true),
			reserved0: view.getUint32(offset + 24, true),
			reserved1: view.getUint32(offset + 28, true),
		} as any;
	}
	outObj.tokenId = view.getUint32(offset, true);
	outObj.tokenClass = deserializeKrystalTokenClass(view, offset + 4);
	outObj.flags = view.getUint32(offset + 8, true);
	outObj.arity = view.getUint32(offset + 12, true);
	outObj.semanticTypeToken = view.getUint32(offset + 16, true);
	outObj.inverseToken = view.getUint32(offset + 20, true);
	outObj.reserved0 = view.getUint32(offset + 24, true);
	outObj.reserved1 = view.getUint32(offset + 28, true);
	return outObj;
}

export function serializeVocabManifestEntry(val: VocabManifestEntry, view: DataView, offset: number): void {
	view.setUint32(offset, val.tokenId, true);
	serializeKrystalTokenClass(val.tokenClass, view, offset + 4);
	view.setUint32(offset + 8, val.flags, true);
	view.setUint32(offset + 12, val.arity, true);
	view.setUint32(offset + 16, val.semanticTypeToken, true);
	view.setUint32(offset + 20, val.inverseToken, true);
	view.setUint32(offset + 24, val.reserved0, true);
	view.setUint32(offset + 28, val.reserved1, true);
}

export function deserializeBrainValueKind(view: DataView, offset: number): BrainValueKind {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "none";
		case 1: return "token";
		case 2: return "context_ref";
		case 3: return "record_ref";
		case 4: return "boolean_class";
		case 5: return "scalar_band";
		case 6: return "quantity_projection";
		case 7: return "opaque_payload";
		default: throw new Error("Unknown Enum value for BrainValueKind: " + v);
	}
}

export function serializeBrainValueKind(val: BrainValueKind, view: DataView, offset: number): void {
	if(val === "none") { view.setUint8(offset, 0); return; }
	if(val === "token") { view.setUint8(offset, 1); return; }
	if(val === "context_ref") { view.setUint8(offset, 2); return; }
	if(val === "record_ref") { view.setUint8(offset, 3); return; }
	if(val === "boolean_class") { view.setUint8(offset, 4); return; }
	if(val === "scalar_band") { view.setUint8(offset, 5); return; }
	if(val === "quantity_projection") { view.setUint8(offset, 6); return; }
	if(val === "opaque_payload") { view.setUint8(offset, 7); return; }
}

export function deserializeRecordSchemaManifestHeader(view: DataView, offset: number, outObj?: any): RecordSchemaManifestHeader {
	if (!outObj) {
		return {
			version: view.getUint32(offset, true),
			schemaCount: view.getUint32(offset + 4, true),
			fieldCount: view.getUint32(offset + 8, true),
			maxRecordTokens: view.getUint32(offset + 12, true),
			schemaHashLo: view.getUint32(offset + 16, true),
			schemaHashHi: view.getUint32(offset + 20, true),
			reserved0: view.getUint32(offset + 24, true),
			reserved1: view.getUint32(offset + 28, true),
		} as any;
	}
	outObj.version = view.getUint32(offset, true);
	outObj.schemaCount = view.getUint32(offset + 4, true);
	outObj.fieldCount = view.getUint32(offset + 8, true);
	outObj.maxRecordTokens = view.getUint32(offset + 12, true);
	outObj.schemaHashLo = view.getUint32(offset + 16, true);
	outObj.schemaHashHi = view.getUint32(offset + 20, true);
	outObj.reserved0 = view.getUint32(offset + 24, true);
	outObj.reserved1 = view.getUint32(offset + 28, true);
	return outObj;
}

export function serializeRecordSchemaManifestHeader(val: RecordSchemaManifestHeader, view: DataView, offset: number): void {
	view.setUint32(offset, val.version, true);
	view.setUint32(offset + 4, val.schemaCount, true);
	view.setUint32(offset + 8, val.fieldCount, true);
	view.setUint32(offset + 12, val.maxRecordTokens, true);
	view.setUint32(offset + 16, val.schemaHashLo, true);
	view.setUint32(offset + 20, val.schemaHashHi, true);
	view.setUint32(offset + 24, val.reserved0, true);
	view.setUint32(offset + 28, val.reserved1, true);
}

export function deserializeRecordSchemaEntry(view: DataView, offset: number, outObj?: any): RecordSchemaEntry {
	if (!outObj) {
		return {
			schemaId: view.getUint32(offset, true),
			familyToken: view.getUint32(offset + 4, true),
			defaultBand: deserializeBrainBandKind(view, offset + 8),
			tokenCount: view.getUint32(offset + 12, true),
			fieldOffset: view.getUint32(offset + 16, true),
			fieldCount: view.getUint32(offset + 20, true),
			flags: view.getUint32(offset + 24, true),
			reserved0: view.getUint32(offset + 28, true),
		} as any;
	}
	outObj.schemaId = view.getUint32(offset, true);
	outObj.familyToken = view.getUint32(offset + 4, true);
	outObj.defaultBand = deserializeBrainBandKind(view, offset + 8);
	outObj.tokenCount = view.getUint32(offset + 12, true);
	outObj.fieldOffset = view.getUint32(offset + 16, true);
	outObj.fieldCount = view.getUint32(offset + 20, true);
	outObj.flags = view.getUint32(offset + 24, true);
	outObj.reserved0 = view.getUint32(offset + 28, true);
	return outObj;
}

export function serializeRecordSchemaEntry(val: RecordSchemaEntry, view: DataView, offset: number): void {
	view.setUint32(offset, val.schemaId, true);
	view.setUint32(offset + 4, val.familyToken, true);
	serializeBrainBandKind(val.defaultBand, view, offset + 8);
	view.setUint32(offset + 12, val.tokenCount, true);
	view.setUint32(offset + 16, val.fieldOffset, true);
	view.setUint32(offset + 20, val.fieldCount, true);
	view.setUint32(offset + 24, val.flags, true);
	view.setUint32(offset + 28, val.reserved0, true);
}

export function deserializeRecordFieldEntry(view: DataView, offset: number, outObj?: any): RecordFieldEntry {
	if (!outObj) {
		return {
			schemaId: view.getUint32(offset, true),
			fieldId: view.getUint32(offset + 4, true),
			localTokenIndex: view.getUint32(offset + 8, true),
			tokenWidth: view.getUint32(offset + 12, true),
			roleToken: view.getUint32(offset + 16, true),
			valueKind: deserializeBrainValueKind(view, offset + 20),
			acceptedSchemaId: view.getUint32(offset + 24, true),
			allowedBandMask: view.getUint32(offset + 28, true),
			flags: view.getUint32(offset + 32, true),
			reserved0: view.getUint32(offset + 36, true),
		} as any;
	}
	outObj.schemaId = view.getUint32(offset, true);
	outObj.fieldId = view.getUint32(offset + 4, true);
	outObj.localTokenIndex = view.getUint32(offset + 8, true);
	outObj.tokenWidth = view.getUint32(offset + 12, true);
	outObj.roleToken = view.getUint32(offset + 16, true);
	outObj.valueKind = deserializeBrainValueKind(view, offset + 20);
	outObj.acceptedSchemaId = view.getUint32(offset + 24, true);
	outObj.allowedBandMask = view.getUint32(offset + 28, true);
	outObj.flags = view.getUint32(offset + 32, true);
	outObj.reserved0 = view.getUint32(offset + 36, true);
	return outObj;
}

export function serializeRecordFieldEntry(val: RecordFieldEntry, view: DataView, offset: number): void {
	view.setUint32(offset, val.schemaId, true);
	view.setUint32(offset + 4, val.fieldId, true);
	view.setUint32(offset + 8, val.localTokenIndex, true);
	view.setUint32(offset + 12, val.tokenWidth, true);
	view.setUint32(offset + 16, val.roleToken, true);
	serializeBrainValueKind(val.valueKind, view, offset + 20);
	view.setUint32(offset + 24, val.acceptedSchemaId, true);
	view.setUint32(offset + 28, val.allowedBandMask, true);
	view.setUint32(offset + 32, val.flags, true);
	view.setUint32(offset + 36, val.reserved0, true);
}

export function deserializeTokenAuthoringSpec(view: DataView, offset: number, outObj?: any): TokenAuthoringSpec {
	if (!outObj) {
		return {
			id: view.getFloat64(offset, true),
			symbol: ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 8),
			tokenClass: deserializeKrystalTokenClass(view, offset + 12),
			semanticType: (view.getUint8(offset + 16) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 17) : undefined),
			arity: (view.getUint8(offset + 24) === 1 ? view.getFloat64(offset + 25, true) : undefined),
			doc: (view.getUint8(offset + 36) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 37) : undefined),
		} as any;
	}
	outObj.id = view.getFloat64(offset, true);
	outObj.symbol = ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 8);
	outObj.tokenClass = deserializeKrystalTokenClass(view, offset + 12);
	outObj.semanticType = (view.getUint8(offset + 16) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 17) : undefined);
	outObj.arity = (view.getUint8(offset + 24) === 1 ? view.getFloat64(offset + 25, true) : undefined);
	outObj.doc = (view.getUint8(offset + 36) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 37) : undefined);
	return outObj;
}

export function serializeTokenAuthoringSpec(val: TokenAuthoringSpec, view: DataView, offset: number): void {
	view.setFloat64(offset, val.id, true);
	{ const bytes = __textEncoder!.encode(val.symbol); const len = Math.min(bytes.length, 255); view.setUint32(offset + 8, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 8 + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 8 + 4, len).set(bytes.subarray(0, len)); }
	serializeKrystalTokenClass(val.tokenClass, view, offset + 12);
	if (val.semanticType !== undefined) { view.setUint8(offset + 16, 1); { const bytes = __textEncoder!.encode(val.semanticType); const len = Math.min(bytes.length, 255); view.setUint32(offset + 17, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 17 + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 17 + 4, len).set(bytes.subarray(0, len)); } } else { view.setUint8(offset + 16, 0); }
	if (val.arity !== undefined) { view.setUint8(offset + 24, 1); view.setFloat64(offset + 25, val.arity, true); } else { view.setUint8(offset + 24, 0); }
	if (val.doc !== undefined) { view.setUint8(offset + 36, 1); { const bytes = __textEncoder!.encode(val.doc); const len = Math.min(bytes.length, 255); view.setUint32(offset + 37, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 37 + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 37 + 4, len).set(bytes.subarray(0, len)); } } else { view.setUint8(offset + 36, 0); }
}

export function deserializeRecordFieldAuthoringSpec(view: DataView, offset: number, outObj?: any): RecordFieldAuthoringSpec {
	if (!outObj) {
		return {
			name: ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset),
			localTokenIndex: view.getFloat64(offset + 4, true),
			roleToken: view.getFloat64(offset + 12, true),
			valueKind: deserializeBrainValueKind(view, offset + 20),
			acceptedSchema: (view.getUint8(offset + 24) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 25) : undefined),
			allowedBands: (view.getUint8(offset + 32) === 1 ? ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(deserializeBrainBandKind(view, o + (i * 1))); } return a; })(offset + 33) : undefined),
			required: (view.getUint8(offset + 40) === 1 ? (view.getUint8(offset + 41) !== 0) : undefined),
			exactRuntime: (view.getUint8(offset + 48) === 1 ? (view.getUint8(offset + 49) !== 0) : undefined),
			doc: (view.getUint8(offset + 56) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 57) : undefined),
		} as any;
	}
	outObj.name = ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset);
	outObj.localTokenIndex = view.getFloat64(offset + 4, true);
	outObj.roleToken = view.getFloat64(offset + 12, true);
	outObj.valueKind = deserializeBrainValueKind(view, offset + 20);
	outObj.acceptedSchema = (view.getUint8(offset + 24) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 25) : undefined);
	outObj.allowedBands = (view.getUint8(offset + 32) === 1 ? ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(deserializeBrainBandKind(view, o + (i * 1))); } return a; })(offset + 33) : undefined);
	outObj.required = (view.getUint8(offset + 40) === 1 ? (view.getUint8(offset + 41) !== 0) : undefined);
	outObj.exactRuntime = (view.getUint8(offset + 48) === 1 ? (view.getUint8(offset + 49) !== 0) : undefined);
	outObj.doc = (view.getUint8(offset + 56) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 57) : undefined);
	return outObj;
}

export function serializeRecordFieldAuthoringSpec(val: RecordFieldAuthoringSpec, view: DataView, offset: number): void {
	{ const bytes = __textEncoder!.encode(val.name); const len = Math.min(bytes.length, 255); view.setUint32(offset, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 4, len).set(bytes.subarray(0, len)); }
	view.setFloat64(offset + 4, val.localTokenIndex, true);
	view.setFloat64(offset + 12, val.roleToken, true);
	serializeBrainValueKind(val.valueKind, view, offset + 20);
	if (val.acceptedSchema !== undefined) { view.setUint8(offset + 24, 1); { const bytes = __textEncoder!.encode(val.acceptedSchema); const len = Math.min(bytes.length, 255); view.setUint32(offset + 25, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 25 + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 25 + 4, len).set(bytes.subarray(0, len)); } } else { view.setUint8(offset + 24, 0); }
	if (val.allowedBands !== undefined) { view.setUint8(offset + 32, 1); { view.setUint32(offset + 33, val.allowedBands.length, true); let o = offset + 33 + 4; for(let i=0; i<val.allowedBands.length; i++) { serializeBrainBandKind(val.allowedBands[i]!, view, o + (i * 1)); } } } else { view.setUint8(offset + 32, 0); }
	if (val.required !== undefined) { view.setUint8(offset + 40, 1); view.setUint8(offset + 41, (val.required ? 1 : 0)); } else { view.setUint8(offset + 40, 0); }
	if (val.exactRuntime !== undefined) { view.setUint8(offset + 48, 1); view.setUint8(offset + 49, (val.exactRuntime ? 1 : 0)); } else { view.setUint8(offset + 48, 0); }
	if (val.doc !== undefined) { view.setUint8(offset + 56, 1); { const bytes = __textEncoder!.encode(val.doc); const len = Math.min(bytes.length, 255); view.setUint32(offset + 57, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 57 + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 57 + 4, len).set(bytes.subarray(0, len)); } } else { view.setUint8(offset + 56, 0); }
}

export function deserializeRecordSchemaAuthoringSpec(view: DataView, offset: number, outObj?: any): RecordSchemaAuthoringSpec {
	if (!outObj) {
		return {
			name: ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset),
			familyToken: view.getFloat64(offset + 4, true),
			defaultBand: deserializeBrainBandKind(view, offset + 12),
			fields: ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(deserializeRecordFieldAuthoringSpec(view, o + (i * 64))); } return a; })(offset + 16),
			doc: (view.getUint8(offset + 20) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 21) : undefined),
		} as any;
	}
	outObj.name = ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset);
	outObj.familyToken = view.getFloat64(offset + 4, true);
	outObj.defaultBand = deserializeBrainBandKind(view, offset + 12);
	outObj.fields = ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(deserializeRecordFieldAuthoringSpec(view, o + (i * 64))); } return a; })(offset + 16);
	outObj.doc = (view.getUint8(offset + 20) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 21) : undefined);
	return outObj;
}

export function serializeRecordSchemaAuthoringSpec(val: RecordSchemaAuthoringSpec, view: DataView, offset: number): void {
	{ const bytes = __textEncoder!.encode(val.name); const len = Math.min(bytes.length, 255); view.setUint32(offset, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 4, len).set(bytes.subarray(0, len)); }
	view.setFloat64(offset + 4, val.familyToken, true);
	serializeBrainBandKind(val.defaultBand, view, offset + 12);
	{ view.setUint32(offset + 16, val.fields.length, true); let o = offset + 16 + 4; for(let i=0; i<val.fields.length; i++) { serializeRecordFieldAuthoringSpec(val.fields[i]!, view, o + (i * 64)); } }
	if (val.doc !== undefined) { view.setUint8(offset + 20, 1); { const bytes = __textEncoder!.encode(val.doc); const len = Math.min(bytes.length, 255); view.setUint32(offset + 21, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 21 + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 21 + 4, len).set(bytes.subarray(0, len)); } } else { view.setUint8(offset + 20, 0); }
}

export function deserializeBrainBandLayout(view: DataView, offset: number, outObj?: any): BrainBandLayout {
	if (!outObj) {
		return {
			kind: deserializeBrainBandKind(view, offset),
			recordOffset: view.getUint32(offset + 4, true),
			recordCapacity: view.getUint32(offset + 8, true),
			tokenOffset: view.getUint32(offset + 12, true),
			tokenCapacity: view.getUint32(offset + 16, true),
			placement: deserializeBandPlacementPolicy(view, offset + 20),
			overflow: deserializeBandOverflowPolicy(view, offset + 21),
			flags: view.getUint32(offset + 24, true),
			reserved0: view.getUint32(offset + 28, true),
		} as any;
	}
	outObj.kind = deserializeBrainBandKind(view, offset);
	outObj.recordOffset = view.getUint32(offset + 4, true);
	outObj.recordCapacity = view.getUint32(offset + 8, true);
	outObj.tokenOffset = view.getUint32(offset + 12, true);
	outObj.tokenCapacity = view.getUint32(offset + 16, true);
	outObj.placement = deserializeBandPlacementPolicy(view, offset + 20);
	outObj.overflow = deserializeBandOverflowPolicy(view, offset + 21);
	outObj.flags = view.getUint32(offset + 24, true);
	outObj.reserved0 = view.getUint32(offset + 28, true);
	return outObj;
}

export function serializeBrainBandLayout(val: BrainBandLayout, view: DataView, offset: number): void {
	serializeBrainBandKind(val.kind, view, offset);
	view.setUint32(offset + 4, val.recordOffset, true);
	view.setUint32(offset + 8, val.recordCapacity, true);
	view.setUint32(offset + 12, val.tokenOffset, true);
	view.setUint32(offset + 16, val.tokenCapacity, true);
	serializeBandPlacementPolicy(val.placement, view, offset + 20);
	serializeBandOverflowPolicy(val.overflow, view, offset + 21);
	view.setUint32(offset + 24, val.flags, true);
	view.setUint32(offset + 28, val.reserved0, true);
}

export function deserializeFixedRecordBinding(view: DataView, offset: number, outObj?: any): FixedRecordBinding {
	if (!outObj) {
		return {
			roleToken: view.getUint32(offset, true),
			recordIndex: view.getUint32(offset + 4, true),
			expectedSchemaId: view.getUint32(offset + 8, true),
			flags: view.getUint32(offset + 12, true),
		} as any;
	}
	outObj.roleToken = view.getUint32(offset, true);
	outObj.recordIndex = view.getUint32(offset + 4, true);
	outObj.expectedSchemaId = view.getUint32(offset + 8, true);
	outObj.flags = view.getUint32(offset + 12, true);
	return outObj;
}

export function serializeFixedRecordBinding(val: FixedRecordBinding, view: DataView, offset: number): void {
	view.setUint32(offset, val.roleToken, true);
	view.setUint32(offset + 4, val.recordIndex, true);
	view.setUint32(offset + 8, val.expectedSchemaId, true);
	view.setUint32(offset + 12, val.flags, true);
}

export function deserializeBrainFrameLayoutHeader(view: DataView, offset: number, outObj?: any): BrainFrameLayoutHeader {
	if (!outObj) {
		return {
			tokenAbiVersion: view.getUint32(offset, true),
			architectureVersion: view.getUint32(offset + 4, true),
			layoutVersion: view.getUint32(offset + 8, true),
			recordWidth: view.getUint32(offset + 12, true),
			recordSlots: view.getUint32(offset + 16, true),
			tokenCapacity: view.getUint32(offset + 20, true),
			bandCount: view.getUint32(offset + 24, true),
			fixedRecordCount: view.getUint32(offset + 28, true),
			flags: view.getUint32(offset + 32, true),
			layoutHashLo: view.getUint32(offset + 36, true),
			layoutHashHi: view.getUint32(offset + 40, true),
		} as any;
	}
	outObj.tokenAbiVersion = view.getUint32(offset, true);
	outObj.architectureVersion = view.getUint32(offset + 4, true);
	outObj.layoutVersion = view.getUint32(offset + 8, true);
	outObj.recordWidth = view.getUint32(offset + 12, true);
	outObj.recordSlots = view.getUint32(offset + 16, true);
	outObj.tokenCapacity = view.getUint32(offset + 20, true);
	outObj.bandCount = view.getUint32(offset + 24, true);
	outObj.fixedRecordCount = view.getUint32(offset + 28, true);
	outObj.flags = view.getUint32(offset + 32, true);
	outObj.layoutHashLo = view.getUint32(offset + 36, true);
	outObj.layoutHashHi = view.getUint32(offset + 40, true);
	return outObj;
}

export function serializeBrainFrameLayoutHeader(val: BrainFrameLayoutHeader, view: DataView, offset: number): void {
	view.setUint32(offset, val.tokenAbiVersion, true);
	view.setUint32(offset + 4, val.architectureVersion, true);
	view.setUint32(offset + 8, val.layoutVersion, true);
	view.setUint32(offset + 12, val.recordWidth, true);
	view.setUint32(offset + 16, val.recordSlots, true);
	view.setUint32(offset + 20, val.tokenCapacity, true);
	view.setUint32(offset + 24, val.bandCount, true);
	view.setUint32(offset + 28, val.fixedRecordCount, true);
	view.setUint32(offset + 32, val.flags, true);
	view.setUint32(offset + 36, val.layoutHashLo, true);
	view.setUint32(offset + 40, val.layoutHashHi, true);
}

export function deserializeBrainFrameLayout(view: DataView, offset: number, outObj?: any): BrainFrameLayout {
	if (!outObj) {
		const _arr_bands = new Array(11);
		for (let i = 0, _off_bands = offset + 44; i < 11; i++, _off_bands += 32) {
			_arr_bands[i] = ({ kind: deserializeBrainBandKind(view, _off_bands), recordOffset: view.getUint32(_off_bands + 4, true), recordCapacity: view.getUint32(_off_bands + 8, true), tokenOffset: view.getUint32(_off_bands + 12, true), tokenCapacity: view.getUint32(_off_bands + 16, true), placement: deserializeBandPlacementPolicy(view, _off_bands + 20), overflow: deserializeBandOverflowPolicy(view, _off_bands + 21), flags: view.getUint32(_off_bands + 24, true), reserved0: view.getUint32(_off_bands + 28, true) });
		}
		const _arr_fixedRecords = new Array(32);
		for (let i = 0, _off_fixedRecords = offset + 396; i < 32; i++, _off_fixedRecords += 16) {
			_arr_fixedRecords[i] = ({ roleToken: view.getUint32(_off_fixedRecords, true), recordIndex: view.getUint32(_off_fixedRecords + 4, true), expectedSchemaId: view.getUint32(_off_fixedRecords + 8, true), flags: view.getUint32(_off_fixedRecords + 12, true) });
		}
		return {
			header: deserializeBrainFrameLayoutHeader(view, offset),
			bands: _arr_bands,
			fixedRecords: _arr_fixedRecords,
		} as any;
	}
	const _arr_bands = new Array(11);
	for (let i = 0, _off_bands = offset + 44; i < 11; i++, _off_bands += 32) {
		_arr_bands[i] = ({ kind: deserializeBrainBandKind(view, _off_bands), recordOffset: view.getUint32(_off_bands + 4, true), recordCapacity: view.getUint32(_off_bands + 8, true), tokenOffset: view.getUint32(_off_bands + 12, true), tokenCapacity: view.getUint32(_off_bands + 16, true), placement: deserializeBandPlacementPolicy(view, _off_bands + 20), overflow: deserializeBandOverflowPolicy(view, _off_bands + 21), flags: view.getUint32(_off_bands + 24, true), reserved0: view.getUint32(_off_bands + 28, true) });
	}
	const _arr_fixedRecords = new Array(32);
	for (let i = 0, _off_fixedRecords = offset + 396; i < 32; i++, _off_fixedRecords += 16) {
		_arr_fixedRecords[i] = ({ roleToken: view.getUint32(_off_fixedRecords, true), recordIndex: view.getUint32(_off_fixedRecords + 4, true), expectedSchemaId: view.getUint32(_off_fixedRecords + 8, true), flags: view.getUint32(_off_fixedRecords + 12, true) });
	}
	outObj.header = deserializeBrainFrameLayoutHeader(view, offset);
	outObj.bands = _arr_bands;
	outObj.fixedRecords = _arr_fixedRecords;
	return outObj;
}

export function serializeBrainFrameLayout(val: BrainFrameLayout, view: DataView, offset: number): void {
	serializeBrainFrameLayoutHeader(val.header, view, offset);
	{ for (let i = 0, __o = offset + 44; i < 11; i++, __o += 32) { const __e = val.bands[i]!; { serializeBrainBandKind(__e.kind, view, __o); view.setUint32(__o + 4, __e.recordOffset, true); view.setUint32(__o + 8, __e.recordCapacity, true); view.setUint32(__o + 12, __e.tokenOffset, true); view.setUint32(__o + 16, __e.tokenCapacity, true); serializeBandPlacementPolicy(__e.placement, view, __o + 20); serializeBandOverflowPolicy(__e.overflow, view, __o + 21); view.setUint32(__o + 24, __e.flags, true); view.setUint32(__o + 28, __e.reserved0, true); } } }
	{ for (let i = 0, __o = offset + 396; i < 32; i++, __o += 16) { const __e = val.fixedRecords[i]!; { view.setUint32(__o, __e.roleToken, true); view.setUint32(__o + 4, __e.recordIndex, true); view.setUint32(__o + 8, __e.expectedSchemaId, true); view.setUint32(__o + 12, __e.flags, true); } } }
}

export function deserializeBrainTokenMeta(view: DataView, offset: number, outObj?: any): BrainTokenMeta {
	if (!outObj) {
		return {
			fieldId: view.getUint32(offset, true),
			roleToken: view.getUint32(offset + 4, true),
			flags: view.getUint32(offset + 8, true),
			referenceBinding: view.getUint32(offset + 12, true),
		} as any;
	}
	outObj.fieldId = view.getUint32(offset, true);
	outObj.roleToken = view.getUint32(offset + 4, true);
	outObj.flags = view.getUint32(offset + 8, true);
	outObj.referenceBinding = view.getUint32(offset + 12, true);
	return outObj;
}

export function serializeBrainTokenMeta(val: BrainTokenMeta, view: DataView, offset: number): void {
	view.setUint32(offset, val.fieldId, true);
	view.setUint32(offset + 4, val.roleToken, true);
	view.setUint32(offset + 8, val.flags, true);
	view.setUint32(offset + 12, val.referenceBinding, true);
}

export function deserializeBrainReferenceBinding(view: DataView, offset: number, outObj?: any): BrainReferenceBinding {
	if (!outObj) {
		return {
			localTokenIndex: view.getUint32(offset, true),
			fieldId: view.getUint32(offset + 4, true),
			flags: view.getUint32(offset + 8, true),
			reserved0: view.getUint32(offset + 12, true),
			handle: deserializeRuntimeRefHandle(view, offset + 16),
		} as any;
	}
	outObj.localTokenIndex = view.getUint32(offset, true);
	outObj.fieldId = view.getUint32(offset + 4, true);
	outObj.flags = view.getUint32(offset + 8, true);
	outObj.reserved0 = view.getUint32(offset + 12, true);
	outObj.handle = deserializeRuntimeRefHandle(view, offset + 16);
	return outObj;
}

export function serializeBrainReferenceBinding(val: BrainReferenceBinding, view: DataView, offset: number): void {
	view.setUint32(offset, val.localTokenIndex, true);
	view.setUint32(offset + 4, val.fieldId, true);
	view.setUint32(offset + 8, val.flags, true);
	view.setUint32(offset + 12, val.reserved0, true);
	serializeRuntimeRefHandle(val.handle, view, offset + 16);
}

export function deserializeBrainRecordHeader(view: DataView, offset: number, outObj?: any): BrainRecordHeader {
	if (!outObj) {
		return {
			schemaId: view.getUint32(offset, true),
			band: deserializeBrainBandKind(view, offset + 4),
			source: deserializeRecordSource(view, offset + 5),
			flags: view.getUint32(offset + 8, true),
			tokenCount: view.getUint32(offset + 12, true),
			referenceCount: view.getUint32(offset + 16, true),
			observedAt: view.getUint32(offset + 20, true),
			revision: view.getUint32(offset + 24, true),
			primaryReference: view.getUint32(offset + 28, true),
			continuationRecord: view.getUint32(offset + 32, true),
			salience: view.getFloat32(offset + 36, true),
			freshness: view.getFloat32(offset + 40, true),
		} as any;
	}
	outObj.schemaId = view.getUint32(offset, true);
	outObj.band = deserializeBrainBandKind(view, offset + 4);
	outObj.source = deserializeRecordSource(view, offset + 5);
	outObj.flags = view.getUint32(offset + 8, true);
	outObj.tokenCount = view.getUint32(offset + 12, true);
	outObj.referenceCount = view.getUint32(offset + 16, true);
	outObj.observedAt = view.getUint32(offset + 20, true);
	outObj.revision = view.getUint32(offset + 24, true);
	outObj.primaryReference = view.getUint32(offset + 28, true);
	outObj.continuationRecord = view.getUint32(offset + 32, true);
	outObj.salience = view.getFloat32(offset + 36, true);
	outObj.freshness = view.getFloat32(offset + 40, true);
	return outObj;
}

export function serializeBrainRecordHeader(val: BrainRecordHeader, view: DataView, offset: number): void {
	view.setUint32(offset, val.schemaId, true);
	serializeBrainBandKind(val.band, view, offset + 4);
	serializeRecordSource(val.source, view, offset + 5);
	view.setUint32(offset + 8, val.flags, true);
	view.setUint32(offset + 12, val.tokenCount, true);
	view.setUint32(offset + 16, val.referenceCount, true);
	view.setUint32(offset + 20, val.observedAt, true);
	view.setUint32(offset + 24, val.revision, true);
	view.setUint32(offset + 28, val.primaryReference, true);
	view.setUint32(offset + 32, val.continuationRecord, true);
	view.setFloat32(offset + 36, val.salience, true);
	view.setFloat32(offset + 40, val.freshness, true);
}

export function deserializeBrainRecordSlot(view: DataView, offset: number, outObj?: any): BrainRecordSlot {
	if (!outObj) {
		const _arr_tokens = new Array(8);
		for (let i = 0, _off_tokens = offset + 44; i < 8; i++, _off_tokens += 4) {
			_arr_tokens[i] = view.getUint32(_off_tokens, true);
		}
		const _arr_tokenMeta = new Array(8);
		for (let i = 0, _off_tokenMeta = offset + 76; i < 8; i++, _off_tokenMeta += 16) {
			_arr_tokenMeta[i] = ({ fieldId: view.getUint32(_off_tokenMeta, true), roleToken: view.getUint32(_off_tokenMeta + 4, true), flags: view.getUint32(_off_tokenMeta + 8, true), referenceBinding: view.getUint32(_off_tokenMeta + 12, true) });
		}
		const _arr_references = new Array(8);
		for (let i = 0, _off_references = offset + 204; i < 8; i++, _off_references += 28) {
			_arr_references[i] = ({ localTokenIndex: view.getUint32(_off_references, true), fieldId: view.getUint32(_off_references + 4, true), flags: view.getUint32(_off_references + 8, true), reserved0: view.getUint32(_off_references + 12, true), handle: deserializeRuntimeRefHandle(view, _off_references + 16) });
		}
		return {
			header: deserializeBrainRecordHeader(view, offset),
			tokens: _arr_tokens,
			tokenMeta: _arr_tokenMeta,
			references: _arr_references,
		} as any;
	}
	const _arr_tokens = new Array(8);
	for (let i = 0, _off_tokens = offset + 44; i < 8; i++, _off_tokens += 4) {
		_arr_tokens[i] = view.getUint32(_off_tokens, true);
	}
	const _arr_tokenMeta = new Array(8);
	for (let i = 0, _off_tokenMeta = offset + 76; i < 8; i++, _off_tokenMeta += 16) {
		_arr_tokenMeta[i] = ({ fieldId: view.getUint32(_off_tokenMeta, true), roleToken: view.getUint32(_off_tokenMeta + 4, true), flags: view.getUint32(_off_tokenMeta + 8, true), referenceBinding: view.getUint32(_off_tokenMeta + 12, true) });
	}
	const _arr_references = new Array(8);
	for (let i = 0, _off_references = offset + 204; i < 8; i++, _off_references += 28) {
		_arr_references[i] = ({ localTokenIndex: view.getUint32(_off_references, true), fieldId: view.getUint32(_off_references + 4, true), flags: view.getUint32(_off_references + 8, true), reserved0: view.getUint32(_off_references + 12, true), handle: deserializeRuntimeRefHandle(view, _off_references + 16) });
	}
	outObj.header = deserializeBrainRecordHeader(view, offset);
	outObj.tokens = _arr_tokens;
	outObj.tokenMeta = _arr_tokenMeta;
	outObj.references = _arr_references;
	return outObj;
}

export function serializeBrainRecordSlot(val: BrainRecordSlot, view: DataView, offset: number): void {
	serializeBrainRecordHeader(val.header, view, offset);
	{ for (let i = 0, __o = offset + 44; i < 8; i++, __o += 4) { const __e = val.tokens[i]!; view.setUint32(__o, __e, true); } }
	{ for (let i = 0, __o = offset + 76; i < 8; i++, __o += 16) { const __e = val.tokenMeta[i]!; { view.setUint32(__o, __e.fieldId, true); view.setUint32(__o + 4, __e.roleToken, true); view.setUint32(__o + 8, __e.flags, true); view.setUint32(__o + 12, __e.referenceBinding, true); } } }
	{ for (let i = 0, __o = offset + 204; i < 8; i++, __o += 28) { const __e = val.references[i]!; { view.setUint32(__o, __e.localTokenIndex, true); view.setUint32(__o + 4, __e.fieldId, true); view.setUint32(__o + 8, __e.flags, true); view.setUint32(__o + 12, __e.reserved0, true); serializeRuntimeRefHandle(__e.handle, view, __o + 16); } } }
}

export function deserializeBrainBandState(view: DataView, offset: number, outObj?: any): BrainBandState {
	if (!outObj) {
		return {
			kind: deserializeBrainBandKind(view, offset),
			activeRecords: view.getUint32(offset + 4, true),
			activeTokens: view.getUint32(offset + 8, true),
			overflowRecords: view.getUint32(offset + 12, true),
			truncatedRecords: view.getUint32(offset + 16, true),
			revision: view.getUint32(offset + 20, true),
			flags: view.getUint32(offset + 24, true),
			reserved0: view.getUint32(offset + 28, true),
		} as any;
	}
	outObj.kind = deserializeBrainBandKind(view, offset);
	outObj.activeRecords = view.getUint32(offset + 4, true);
	outObj.activeTokens = view.getUint32(offset + 8, true);
	outObj.overflowRecords = view.getUint32(offset + 12, true);
	outObj.truncatedRecords = view.getUint32(offset + 16, true);
	outObj.revision = view.getUint32(offset + 20, true);
	outObj.flags = view.getUint32(offset + 24, true);
	outObj.reserved0 = view.getUint32(offset + 28, true);
	return outObj;
}

export function serializeBrainBandState(val: BrainBandState, view: DataView, offset: number): void {
	serializeBrainBandKind(val.kind, view, offset);
	view.setUint32(offset + 4, val.activeRecords, true);
	view.setUint32(offset + 8, val.activeTokens, true);
	view.setUint32(offset + 12, val.overflowRecords, true);
	view.setUint32(offset + 16, val.truncatedRecords, true);
	view.setUint32(offset + 20, val.revision, true);
	view.setUint32(offset + 24, val.flags, true);
	view.setUint32(offset + 28, val.reserved0, true);
}

export function deserializeBrainFrameHeader(view: DataView, offset: number, outObj?: any): BrainFrameHeader {
	if (!outObj) {
		return {
			tokenAbiVersion: view.getUint32(offset, true),
			architectureVersion: view.getUint32(offset + 4, true),
			layoutVersion: view.getUint32(offset + 8, true),
			tick: view.getUint32(offset + 12, true),
			snapshot: view.getUint32(offset + 16, true),
			activeRecordCount: view.getUint32(offset + 20, true),
			activeTokenCount: view.getUint32(offset + 24, true),
			activeQueryRecord: view.getUint32(offset + 28, true),
			actorRecord: view.getUint32(offset + 32, true),
			frameRevision: view.getUint32(offset + 36, true),
			memoryRevision: view.getUint32(offset + 40, true),
			intentRevision: view.getUint32(offset + 44, true),
			flags: view.getUint32(offset + 48, true),
		} as any;
	}
	outObj.tokenAbiVersion = view.getUint32(offset, true);
	outObj.architectureVersion = view.getUint32(offset + 4, true);
	outObj.layoutVersion = view.getUint32(offset + 8, true);
	outObj.tick = view.getUint32(offset + 12, true);
	outObj.snapshot = view.getUint32(offset + 16, true);
	outObj.activeRecordCount = view.getUint32(offset + 20, true);
	outObj.activeTokenCount = view.getUint32(offset + 24, true);
	outObj.activeQueryRecord = view.getUint32(offset + 28, true);
	outObj.actorRecord = view.getUint32(offset + 32, true);
	outObj.frameRevision = view.getUint32(offset + 36, true);
	outObj.memoryRevision = view.getUint32(offset + 40, true);
	outObj.intentRevision = view.getUint32(offset + 44, true);
	outObj.flags = view.getUint32(offset + 48, true);
	return outObj;
}

export function serializeBrainFrameHeader(val: BrainFrameHeader, view: DataView, offset: number): void {
	view.setUint32(offset, val.tokenAbiVersion, true);
	view.setUint32(offset + 4, val.architectureVersion, true);
	view.setUint32(offset + 8, val.layoutVersion, true);
	view.setUint32(offset + 12, val.tick, true);
	view.setUint32(offset + 16, val.snapshot, true);
	view.setUint32(offset + 20, val.activeRecordCount, true);
	view.setUint32(offset + 24, val.activeTokenCount, true);
	view.setUint32(offset + 28, val.activeQueryRecord, true);
	view.setUint32(offset + 32, val.actorRecord, true);
	view.setUint32(offset + 36, val.frameRevision, true);
	view.setUint32(offset + 40, val.memoryRevision, true);
	view.setUint32(offset + 44, val.intentRevision, true);
	view.setUint32(offset + 48, val.flags, true);
}

export function deserializeBrainFrame(view: DataView, offset: number, outObj?: any): BrainFrame {
	if (!outObj) {
		const _arr_bands = new Array(11);
		for (let i = 0, _off_bands = offset + 52; i < 11; i++, _off_bands += 32) {
			_arr_bands[i] = ({ kind: deserializeBrainBandKind(view, _off_bands), activeRecords: view.getUint32(_off_bands + 4, true), activeTokens: view.getUint32(_off_bands + 8, true), overflowRecords: view.getUint32(_off_bands + 12, true), truncatedRecords: view.getUint32(_off_bands + 16, true), revision: view.getUint32(_off_bands + 20, true), flags: view.getUint32(_off_bands + 24, true), reserved0: view.getUint32(_off_bands + 28, true) });
		}
		const _arr_records = new Array(128);
		for (let i = 0, _off_records = offset + 404; i < 128; i++, _off_records += 428) {
			_arr_records[i] = ({ header: deserializeBrainRecordHeader(view, _off_records), tokens: ((o) => { const a: any[] = []; for(let i=0; i<8; i++) a.push(view.getUint32(o + (i * 4), true)); return a; })(_off_records + 44), tokenMeta: ((o) => { const a: any[] = []; for(let i=0; i<8; i++) a.push(deserializeBrainTokenMeta(view, o + (i * 16))); return a; })(_off_records + 76), references: ((o) => { const a: any[] = []; for(let i=0; i<8; i++) a.push(deserializeBrainReferenceBinding(view, o + (i * 28))); return a; })(_off_records + 204) });
		}
		return {
			header: deserializeBrainFrameHeader(view, offset),
			bands: _arr_bands,
			records: _arr_records,
		} as any;
	}
	const _arr_bands = new Array(11);
	for (let i = 0, _off_bands = offset + 52; i < 11; i++, _off_bands += 32) {
		_arr_bands[i] = ({ kind: deserializeBrainBandKind(view, _off_bands), activeRecords: view.getUint32(_off_bands + 4, true), activeTokens: view.getUint32(_off_bands + 8, true), overflowRecords: view.getUint32(_off_bands + 12, true), truncatedRecords: view.getUint32(_off_bands + 16, true), revision: view.getUint32(_off_bands + 20, true), flags: view.getUint32(_off_bands + 24, true), reserved0: view.getUint32(_off_bands + 28, true) });
	}
	const _arr_records = new Array(128);
	for (let i = 0, _off_records = offset + 404; i < 128; i++, _off_records += 428) {
		_arr_records[i] = ({ header: deserializeBrainRecordHeader(view, _off_records), tokens: ((o) => { const a: any[] = []; for(let i=0; i<8; i++) a.push(view.getUint32(o + (i * 4), true)); return a; })(_off_records + 44), tokenMeta: ((o) => { const a: any[] = []; for(let i=0; i<8; i++) a.push(deserializeBrainTokenMeta(view, o + (i * 16))); return a; })(_off_records + 76), references: ((o) => { const a: any[] = []; for(let i=0; i<8; i++) a.push(deserializeBrainReferenceBinding(view, o + (i * 28))); return a; })(_off_records + 204) });
	}
	outObj.header = deserializeBrainFrameHeader(view, offset);
	outObj.bands = _arr_bands;
	outObj.records = _arr_records;
	return outObj;
}

export function serializeBrainFrame(val: BrainFrame, view: DataView, offset: number): void {
	serializeBrainFrameHeader(val.header, view, offset);
	{ for (let i = 0, __o = offset + 52; i < 11; i++, __o += 32) { const __e = val.bands[i]!; { serializeBrainBandKind(__e.kind, view, __o); view.setUint32(__o + 4, __e.activeRecords, true); view.setUint32(__o + 8, __e.activeTokens, true); view.setUint32(__o + 12, __e.overflowRecords, true); view.setUint32(__o + 16, __e.truncatedRecords, true); view.setUint32(__o + 20, __e.revision, true); view.setUint32(__o + 24, __e.flags, true); view.setUint32(__o + 28, __e.reserved0, true); } } }
	{ for (let i = 0, __o = offset + 404; i < 128; i++, __o += 428) { const __e = val.records[i]!; { serializeBrainRecordHeader(__e.header, view, __o); { for (let i = 0, __o1 = __o + 44; i < 8; i++, __o1 += 4) { const __e1 = __e.tokens[i]!; view.setUint32(__o1, __e1, true); } } { for (let i = 0, __o1 = __o + 76; i < 8; i++, __o1 += 16) { const __e1 = __e.tokenMeta[i]!; { view.setUint32(__o1, __e1.fieldId, true); view.setUint32(__o1 + 4, __e1.roleToken, true); view.setUint32(__o1 + 8, __e1.flags, true); view.setUint32(__o1 + 12, __e1.referenceBinding, true); } } } { for (let i = 0, __o1 = __o + 204; i < 8; i++, __o1 += 28) { const __e1 = __e.references[i]!; { view.setUint32(__o1, __e1.localTokenIndex, true); view.setUint32(__o1 + 4, __e1.fieldId, true); view.setUint32(__o1 + 8, __e1.flags, true); view.setUint32(__o1 + 12, __e1.reserved0, true); serializeRuntimeRefHandle(__e1.handle, view, __o1 + 16); } } } } } }
}

export function deserializeHomeostasisSignal(view: DataView, offset: number, outObj?: any): HomeostasisSignal {
	if (!outObj) {
		return {
			channelToken: view.getUint32(offset, true),
			currentStateToken: view.getUint32(offset + 4, true),
			desiredStateToken: view.getUint32(offset + 8, true),
			flags: view.getUint32(offset + 12, true),
			currentValue: view.getFloat32(offset + 16, true),
			targetValue: view.getFloat32(offset + 20, true),
			urgency: view.getFloat32(offset + 24, true),
			delta: view.getFloat32(offset + 28, true),
			source: deserializeRuntimeRefHandle(view, offset + 32),
		} as any;
	}
	outObj.channelToken = view.getUint32(offset, true);
	outObj.currentStateToken = view.getUint32(offset + 4, true);
	outObj.desiredStateToken = view.getUint32(offset + 8, true);
	outObj.flags = view.getUint32(offset + 12, true);
	outObj.currentValue = view.getFloat32(offset + 16, true);
	outObj.targetValue = view.getFloat32(offset + 20, true);
	outObj.urgency = view.getFloat32(offset + 24, true);
	outObj.delta = view.getFloat32(offset + 28, true);
	outObj.source = deserializeRuntimeRefHandle(view, offset + 32);
	return outObj;
}

export function serializeHomeostasisSignal(val: HomeostasisSignal, view: DataView, offset: number): void {
	view.setUint32(offset, val.channelToken, true);
	view.setUint32(offset + 4, val.currentStateToken, true);
	view.setUint32(offset + 8, val.desiredStateToken, true);
	view.setUint32(offset + 12, val.flags, true);
	view.setFloat32(offset + 16, val.currentValue, true);
	view.setFloat32(offset + 20, val.targetValue, true);
	view.setFloat32(offset + 24, val.urgency, true);
	view.setFloat32(offset + 28, val.delta, true);
	serializeRuntimeRefHandle(val.source, view, offset + 32);
}

export function deserializeBrainQueryKind(view: DataView, offset: number): BrainQueryKind {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "none";
		case 1: return "homeostasis";
		case 2: return "tutorial";
		case 3: return "external";
		case 4: return "internal";
		case 5: return "continuation";
		case 6: return "runtime_feedback";
		default: throw new Error("Unknown Enum value for BrainQueryKind: " + v);
	}
}

export function serializeBrainQueryKind(val: BrainQueryKind, view: DataView, offset: number): void {
	if(val === "none") { view.setUint8(offset, 0); return; }
	if(val === "homeostasis") { view.setUint8(offset, 1); return; }
	if(val === "tutorial") { view.setUint8(offset, 2); return; }
	if(val === "external") { view.setUint8(offset, 3); return; }
	if(val === "internal") { view.setUint8(offset, 4); return; }
	if(val === "continuation") { view.setUint8(offset, 5); return; }
	if(val === "runtime_feedback") { view.setUint8(offset, 6); return; }
}

export function deserializeBrainQueryState(view: DataView, offset: number, outObj?: any): BrainQueryState {
	if (!outObj) {
		return {
			queryRef: deserializeRuntimeRefHandle(view, offset),
			kind: deserializeBrainQueryKind(view, offset + 12),
			routeToken: view.getUint32(offset + 16, true),
			predicateToken: view.getUint32(offset + 20, true),
			subject: deserializeRuntimeRefHandle(view, offset + 24),
			objectToken: view.getUint32(offset + 36, true),
			objectRef: deserializeRuntimeRefHandle(view, offset + 40),
			urgency: view.getFloat32(offset + 52, true),
			createdAt: view.getUint32(offset + 56, true),
			expiresAt: view.getUint32(offset + 60, true),
			flags: view.getUint32(offset + 64, true),
		} as any;
	}
	outObj.queryRef = deserializeRuntimeRefHandle(view, offset);
	outObj.kind = deserializeBrainQueryKind(view, offset + 12);
	outObj.routeToken = view.getUint32(offset + 16, true);
	outObj.predicateToken = view.getUint32(offset + 20, true);
	outObj.subject = deserializeRuntimeRefHandle(view, offset + 24);
	outObj.objectToken = view.getUint32(offset + 36, true);
	outObj.objectRef = deserializeRuntimeRefHandle(view, offset + 40);
	outObj.urgency = view.getFloat32(offset + 52, true);
	outObj.createdAt = view.getUint32(offset + 56, true);
	outObj.expiresAt = view.getUint32(offset + 60, true);
	outObj.flags = view.getUint32(offset + 64, true);
	return outObj;
}

export function serializeBrainQueryState(val: BrainQueryState, view: DataView, offset: number): void {
	serializeRuntimeRefHandle(val.queryRef, view, offset);
	serializeBrainQueryKind(val.kind, view, offset + 12);
	view.setUint32(offset + 16, val.routeToken, true);
	view.setUint32(offset + 20, val.predicateToken, true);
	serializeRuntimeRefHandle(val.subject, view, offset + 24);
	view.setUint32(offset + 36, val.objectToken, true);
	serializeRuntimeRefHandle(val.objectRef, view, offset + 40);
	view.setFloat32(offset + 52, val.urgency, true);
	view.setUint32(offset + 56, val.createdAt, true);
	view.setUint32(offset + 60, val.expiresAt, true);
	view.setUint32(offset + 64, val.flags, true);
}

export function deserializeBrainQuerySet(view: DataView, offset: number, outObj?: any): BrainQuerySet {
	if (!outObj) {
		const _arr_queries = new Array(8);
		for (let i = 0, _off_queries = offset + 16; i < 8; i++, _off_queries += 68) {
			_arr_queries[i] = ({ queryRef: deserializeRuntimeRefHandle(view, _off_queries), kind: deserializeBrainQueryKind(view, _off_queries + 12), routeToken: view.getUint32(_off_queries + 16, true), predicateToken: view.getUint32(_off_queries + 20, true), subject: deserializeRuntimeRefHandle(view, _off_queries + 24), objectToken: view.getUint32(_off_queries + 36, true), objectRef: deserializeRuntimeRefHandle(view, _off_queries + 40), urgency: view.getFloat32(_off_queries + 52, true), createdAt: view.getUint32(_off_queries + 56, true), expiresAt: view.getUint32(_off_queries + 60, true), flags: view.getUint32(_off_queries + 64, true) });
		}
		return {
			count: view.getUint32(offset, true),
			primary: view.getUint32(offset + 4, true),
			revision: view.getUint32(offset + 8, true),
			reserved0: view.getUint32(offset + 12, true),
			queries: _arr_queries,
		} as any;
	}
	const _arr_queries = new Array(8);
	for (let i = 0, _off_queries = offset + 16; i < 8; i++, _off_queries += 68) {
		_arr_queries[i] = ({ queryRef: deserializeRuntimeRefHandle(view, _off_queries), kind: deserializeBrainQueryKind(view, _off_queries + 12), routeToken: view.getUint32(_off_queries + 16, true), predicateToken: view.getUint32(_off_queries + 20, true), subject: deserializeRuntimeRefHandle(view, _off_queries + 24), objectToken: view.getUint32(_off_queries + 36, true), objectRef: deserializeRuntimeRefHandle(view, _off_queries + 40), urgency: view.getFloat32(_off_queries + 52, true), createdAt: view.getUint32(_off_queries + 56, true), expiresAt: view.getUint32(_off_queries + 60, true), flags: view.getUint32(_off_queries + 64, true) });
	}
	outObj.count = view.getUint32(offset, true);
	outObj.primary = view.getUint32(offset + 4, true);
	outObj.revision = view.getUint32(offset + 8, true);
	outObj.reserved0 = view.getUint32(offset + 12, true);
	outObj.queries = _arr_queries;
	return outObj;
}

export function serializeBrainQuerySet(val: BrainQuerySet, view: DataView, offset: number): void {
	view.setUint32(offset, val.count, true);
	view.setUint32(offset + 4, val.primary, true);
	view.setUint32(offset + 8, val.revision, true);
	view.setUint32(offset + 12, val.reserved0, true);
	{ for (let i = 0, __o = offset + 16; i < 8; i++, __o += 68) { const __e = val.queries[i]!; { serializeRuntimeRefHandle(__e.queryRef, view, __o); serializeBrainQueryKind(__e.kind, view, __o + 12); view.setUint32(__o + 16, __e.routeToken, true); view.setUint32(__o + 20, __e.predicateToken, true); serializeRuntimeRefHandle(__e.subject, view, __o + 24); view.setUint32(__o + 36, __e.objectToken, true); serializeRuntimeRefHandle(__e.objectRef, view, __o + 40); view.setFloat32(__o + 52, __e.urgency, true); view.setUint32(__o + 56, __e.createdAt, true); view.setUint32(__o + 60, __e.expiresAt, true); view.setUint32(__o + 64, __e.flags, true); } } }
}

export function deserializeMemoryTraceKind(view: DataView, offset: number): MemoryTraceKind {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "none";
		case 1: return "entity";
		case 2: return "event";
		case 3: return "goal";
		case 4: return "intent";
		case 5: return "topic";
		case 6: return "observation";
		default: throw new Error("Unknown Enum value for MemoryTraceKind: " + v);
	}
}

export function serializeMemoryTraceKind(val: MemoryTraceKind, view: DataView, offset: number): void {
	if(val === "none") { view.setUint8(offset, 0); return; }
	if(val === "entity") { view.setUint8(offset, 1); return; }
	if(val === "event") { view.setUint8(offset, 2); return; }
	if(val === "goal") { view.setUint8(offset, 3); return; }
	if(val === "intent") { view.setUint8(offset, 4); return; }
	if(val === "topic") { view.setUint8(offset, 5); return; }
	if(val === "observation") { view.setUint8(offset, 6); return; }
}

export function deserializeMemorySlotState(view: DataView, offset: number): MemorySlotState {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "empty";
		case 1: return "active";
		case 2: return "evictable";
		case 3: return "evicted";
		default: throw new Error("Unknown Enum value for MemorySlotState: " + v);
	}
}

export function serializeMemorySlotState(val: MemorySlotState, view: DataView, offset: number): void {
	if(val === "empty") { view.setUint8(offset, 0); return; }
	if(val === "active") { view.setUint8(offset, 1); return; }
	if(val === "evictable") { view.setUint8(offset, 2); return; }
	if(val === "evicted") { view.setUint8(offset, 3); return; }
}

export function deserializeMemoryUpdateReason(view: DataView, offset: number): MemoryUpdateReason {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "observation";
		case 1: return "look";
		case 2: return "interaction";
		case 3: return "comfort_delta";
		case 4: return "retrieval";
		case 5: return "rehearsal";
		case 6: return "goal";
		case 7: return "intent";
		case 8: return "decay";
		default: throw new Error("Unknown Enum value for MemoryUpdateReason: " + v);
	}
}

export function serializeMemoryUpdateReason(val: MemoryUpdateReason, view: DataView, offset: number): void {
	if(val === "observation") { view.setUint8(offset, 0); return; }
	if(val === "look") { view.setUint8(offset, 1); return; }
	if(val === "interaction") { view.setUint8(offset, 2); return; }
	if(val === "comfort_delta") { view.setUint8(offset, 3); return; }
	if(val === "retrieval") { view.setUint8(offset, 4); return; }
	if(val === "rehearsal") { view.setUint8(offset, 5); return; }
	if(val === "goal") { view.setUint8(offset, 6); return; }
	if(val === "intent") { view.setUint8(offset, 7); return; }
	if(val === "decay") { view.setUint8(offset, 8); return; }
}

export function deserializeMemoryConfig(view: DataView, offset: number, outObj?: any): MemoryConfig {
	if (!outObj) {
		return {
			slotCount: view.getUint32(offset, true),
			activationDecay: view.getFloat32(offset + 4, true),
			familiarityDecay: view.getFloat32(offset + 8, true),
			familiarityGain: view.getFloat32(offset + 12, true),
			evictionHysteresis: view.getFloat32(offset + 16, true),
			minimumResidenceTicks: view.getUint32(offset + 20, true),
			flags: view.getUint32(offset + 24, true),
			reserved0: view.getUint32(offset + 28, true),
		} as any;
	}
	outObj.slotCount = view.getUint32(offset, true);
	outObj.activationDecay = view.getFloat32(offset + 4, true);
	outObj.familiarityDecay = view.getFloat32(offset + 8, true);
	outObj.familiarityGain = view.getFloat32(offset + 12, true);
	outObj.evictionHysteresis = view.getFloat32(offset + 16, true);
	outObj.minimumResidenceTicks = view.getUint32(offset + 20, true);
	outObj.flags = view.getUint32(offset + 24, true);
	outObj.reserved0 = view.getUint32(offset + 28, true);
	return outObj;
}

export function serializeMemoryConfig(val: MemoryConfig, view: DataView, offset: number): void {
	view.setUint32(offset, val.slotCount, true);
	view.setFloat32(offset + 4, val.activationDecay, true);
	view.setFloat32(offset + 8, val.familiarityDecay, true);
	view.setFloat32(offset + 12, val.familiarityGain, true);
	view.setFloat32(offset + 16, val.evictionHysteresis, true);
	view.setUint32(offset + 20, val.minimumResidenceTicks, true);
	view.setUint32(offset + 24, val.flags, true);
	view.setUint32(offset + 28, val.reserved0, true);
}

export function deserializeMemoryTrace(view: DataView, offset: number, outObj?: any): MemoryTrace {
	if (!outObj) {
		return {
			memoryRef: deserializeRuntimeRefHandle(view, offset),
			subject: deserializeRuntimeRefHandle(view, offset + 12),
			kind: deserializeMemoryTraceKind(view, offset + 24),
			state: deserializeMemorySlotState(view, offset + 25),
			flags: view.getUint32(offset + 28, true),
			createdAt: view.getUint32(offset + 32, true),
			lastObservedAt: view.getUint32(offset + 36, true),
			lastAccessedAt: view.getUint32(offset + 40, true),
			interactionCount: view.getUint32(offset + 44, true),
			activation: view.getFloat32(offset + 48, true),
			familiarity: view.getFloat32(offset + 52, true),
			affectMagnitude: view.getFloat32(offset + 56, true),
			reserved0: view.getFloat32(offset + 60, true),
			rememberedRecord: deserializeBrainRecordSlot(view, offset + 64),
		} as any;
	}
	outObj.memoryRef = deserializeRuntimeRefHandle(view, offset);
	outObj.subject = deserializeRuntimeRefHandle(view, offset + 12);
	outObj.kind = deserializeMemoryTraceKind(view, offset + 24);
	outObj.state = deserializeMemorySlotState(view, offset + 25);
	outObj.flags = view.getUint32(offset + 28, true);
	outObj.createdAt = view.getUint32(offset + 32, true);
	outObj.lastObservedAt = view.getUint32(offset + 36, true);
	outObj.lastAccessedAt = view.getUint32(offset + 40, true);
	outObj.interactionCount = view.getUint32(offset + 44, true);
	outObj.activation = view.getFloat32(offset + 48, true);
	outObj.familiarity = view.getFloat32(offset + 52, true);
	outObj.affectMagnitude = view.getFloat32(offset + 56, true);
	outObj.reserved0 = view.getFloat32(offset + 60, true);
	outObj.rememberedRecord = deserializeBrainRecordSlot(view, offset + 64);
	return outObj;
}

export function serializeMemoryTrace(val: MemoryTrace, view: DataView, offset: number): void {
	serializeRuntimeRefHandle(val.memoryRef, view, offset);
	serializeRuntimeRefHandle(val.subject, view, offset + 12);
	serializeMemoryTraceKind(val.kind, view, offset + 24);
	serializeMemorySlotState(val.state, view, offset + 25);
	view.setUint32(offset + 28, val.flags, true);
	view.setUint32(offset + 32, val.createdAt, true);
	view.setUint32(offset + 36, val.lastObservedAt, true);
	view.setUint32(offset + 40, val.lastAccessedAt, true);
	view.setUint32(offset + 44, val.interactionCount, true);
	view.setFloat32(offset + 48, val.activation, true);
	view.setFloat32(offset + 52, val.familiarity, true);
	view.setFloat32(offset + 56, val.affectMagnitude, true);
	view.setFloat32(offset + 60, val.reserved0, true);
	serializeBrainRecordSlot(val.rememberedRecord, view, offset + 64);
}

export function deserializeMemoryUpdate(view: DataView, offset: number, outObj?: any): MemoryUpdate {
	if (!outObj) {
		return {
			subject: deserializeRuntimeRefHandle(view, offset),
			reason: deserializeMemoryUpdateReason(view, offset + 12),
			interactionToken: view.getUint32(offset + 16, true),
			flags: view.getUint32(offset + 20, true),
			activationDelta: view.getFloat32(offset + 24, true),
			familiarityDelta: view.getFloat32(offset + 28, true),
			affectMagnitude: view.getFloat32(offset + 32, true),
			tick: view.getUint32(offset + 36, true),
		} as any;
	}
	outObj.subject = deserializeRuntimeRefHandle(view, offset);
	outObj.reason = deserializeMemoryUpdateReason(view, offset + 12);
	outObj.interactionToken = view.getUint32(offset + 16, true);
	outObj.flags = view.getUint32(offset + 20, true);
	outObj.activationDelta = view.getFloat32(offset + 24, true);
	outObj.familiarityDelta = view.getFloat32(offset + 28, true);
	outObj.affectMagnitude = view.getFloat32(offset + 32, true);
	outObj.tick = view.getUint32(offset + 36, true);
	return outObj;
}

export function serializeMemoryUpdate(val: MemoryUpdate, view: DataView, offset: number): void {
	serializeRuntimeRefHandle(val.subject, view, offset);
	serializeMemoryUpdateReason(val.reason, view, offset + 12);
	view.setUint32(offset + 16, val.interactionToken, true);
	view.setUint32(offset + 20, val.flags, true);
	view.setFloat32(offset + 24, val.activationDelta, true);
	view.setFloat32(offset + 28, val.familiarityDelta, true);
	view.setFloat32(offset + 32, val.affectMagnitude, true);
	view.setUint32(offset + 36, val.tick, true);
}

export function deserializeWorkingMemoryState(view: DataView, offset: number, outObj?: any): WorkingMemoryState {
	if (!outObj) {
		const _arr_slots = new Array(32);
		for (let i = 0, _off_slots = offset + 16; i < 32; i++, _off_slots += 492) {
			_arr_slots[i] = ({ memoryRef: deserializeRuntimeRefHandle(view, _off_slots), subject: deserializeRuntimeRefHandle(view, _off_slots + 12), kind: deserializeMemoryTraceKind(view, _off_slots + 24), state: deserializeMemorySlotState(view, _off_slots + 25), flags: view.getUint32(_off_slots + 28, true), createdAt: view.getUint32(_off_slots + 32, true), lastObservedAt: view.getUint32(_off_slots + 36, true), lastAccessedAt: view.getUint32(_off_slots + 40, true), interactionCount: view.getUint32(_off_slots + 44, true), activation: view.getFloat32(_off_slots + 48, true), familiarity: view.getFloat32(_off_slots + 52, true), affectMagnitude: view.getFloat32(_off_slots + 56, true), reserved0: view.getFloat32(_off_slots + 60, true), rememberedRecord: deserializeBrainRecordSlot(view, _off_slots + 64) });
		}
		return {
			revision: view.getUint32(offset, true),
			activeCount: view.getUint32(offset + 4, true),
			evictedCount: view.getUint32(offset + 8, true),
			flags: view.getUint32(offset + 12, true),
			slots: _arr_slots,
		} as any;
	}
	const _arr_slots = new Array(32);
	for (let i = 0, _off_slots = offset + 16; i < 32; i++, _off_slots += 492) {
		_arr_slots[i] = ({ memoryRef: deserializeRuntimeRefHandle(view, _off_slots), subject: deserializeRuntimeRefHandle(view, _off_slots + 12), kind: deserializeMemoryTraceKind(view, _off_slots + 24), state: deserializeMemorySlotState(view, _off_slots + 25), flags: view.getUint32(_off_slots + 28, true), createdAt: view.getUint32(_off_slots + 32, true), lastObservedAt: view.getUint32(_off_slots + 36, true), lastAccessedAt: view.getUint32(_off_slots + 40, true), interactionCount: view.getUint32(_off_slots + 44, true), activation: view.getFloat32(_off_slots + 48, true), familiarity: view.getFloat32(_off_slots + 52, true), affectMagnitude: view.getFloat32(_off_slots + 56, true), reserved0: view.getFloat32(_off_slots + 60, true), rememberedRecord: deserializeBrainRecordSlot(view, _off_slots + 64) });
	}
	outObj.revision = view.getUint32(offset, true);
	outObj.activeCount = view.getUint32(offset + 4, true);
	outObj.evictedCount = view.getUint32(offset + 8, true);
	outObj.flags = view.getUint32(offset + 12, true);
	outObj.slots = _arr_slots;
	return outObj;
}

export function serializeWorkingMemoryState(val: WorkingMemoryState, view: DataView, offset: number): void {
	view.setUint32(offset, val.revision, true);
	view.setUint32(offset + 4, val.activeCount, true);
	view.setUint32(offset + 8, val.evictedCount, true);
	view.setUint32(offset + 12, val.flags, true);
	{ for (let i = 0, __o = offset + 16; i < 32; i++, __o += 492) { const __e = val.slots[i]!; { serializeRuntimeRefHandle(__e.memoryRef, view, __o); serializeRuntimeRefHandle(__e.subject, view, __o + 12); serializeMemoryTraceKind(__e.kind, view, __o + 24); serializeMemorySlotState(__e.state, view, __o + 25); view.setUint32(__o + 28, __e.flags, true); view.setUint32(__o + 32, __e.createdAt, true); view.setUint32(__o + 36, __e.lastObservedAt, true); view.setUint32(__o + 40, __e.lastAccessedAt, true); view.setUint32(__o + 44, __e.interactionCount, true); view.setFloat32(__o + 48, __e.activation, true); view.setFloat32(__o + 52, __e.familiarity, true); view.setFloat32(__o + 56, __e.affectMagnitude, true); view.setFloat32(__o + 60, __e.reserved0, true); serializeBrainRecordSlot(__e.rememberedRecord, view, __o + 64); } } }
}

export function deserializeActionIntentDomain(view: DataView, offset: number): ActionIntentDomain {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "external";
		case 1: return "perceptual";
		case 2: return "internal";
		case 3: return "communicative";
		case 4: return "postural";
		default: throw new Error("Unknown Enum value for ActionIntentDomain: " + v);
	}
}

export function serializeActionIntentDomain(val: ActionIntentDomain, view: DataView, offset: number): void {
	if(val === "external") { view.setUint8(offset, 0); return; }
	if(val === "perceptual") { view.setUint8(offset, 1); return; }
	if(val === "internal") { view.setUint8(offset, 2); return; }
	if(val === "communicative") { view.setUint8(offset, 3); return; }
	if(val === "postural") { view.setUint8(offset, 4); return; }
}

export function deserializeActionIntentCatalogHeader(view: DataView, offset: number, outObj?: any): ActionIntentCatalogHeader {
	if (!outObj) {
		return {
			version: view.getUint32(offset, true),
			intentCount: view.getUint32(offset + 4, true),
			argumentCount: view.getUint32(offset + 8, true),
			flags: view.getUint32(offset + 12, true),
			catalogHashLo: view.getUint32(offset + 16, true),
			catalogHashHi: view.getUint32(offset + 20, true),
			reserved0: view.getUint32(offset + 24, true),
			reserved1: view.getUint32(offset + 28, true),
		} as any;
	}
	outObj.version = view.getUint32(offset, true);
	outObj.intentCount = view.getUint32(offset + 4, true);
	outObj.argumentCount = view.getUint32(offset + 8, true);
	outObj.flags = view.getUint32(offset + 12, true);
	outObj.catalogHashLo = view.getUint32(offset + 16, true);
	outObj.catalogHashHi = view.getUint32(offset + 20, true);
	outObj.reserved0 = view.getUint32(offset + 24, true);
	outObj.reserved1 = view.getUint32(offset + 28, true);
	return outObj;
}

export function serializeActionIntentCatalogHeader(val: ActionIntentCatalogHeader, view: DataView, offset: number): void {
	view.setUint32(offset, val.version, true);
	view.setUint32(offset + 4, val.intentCount, true);
	view.setUint32(offset + 8, val.argumentCount, true);
	view.setUint32(offset + 12, val.flags, true);
	view.setUint32(offset + 16, val.catalogHashLo, true);
	view.setUint32(offset + 20, val.catalogHashHi, true);
	view.setUint32(offset + 24, val.reserved0, true);
	view.setUint32(offset + 28, val.reserved1, true);
}

export function deserializeActionIntentDescriptor(view: DataView, offset: number, outObj?: any): ActionIntentDescriptor {
	if (!outObj) {
		return {
			intentId: view.getUint32(offset, true),
			actionToken: view.getUint32(offset + 4, true),
			semanticIntentToken: view.getUint32(offset + 8, true),
			domain: deserializeActionIntentDomain(view, offset + 12),
			actorSchemaId: view.getUint32(offset + 16, true),
			argumentOffset: view.getUint32(offset + 20, true),
			argumentCount: view.getUint32(offset + 24, true),
			flags: view.getUint32(offset + 28, true),
			effectClassToken: view.getUint32(offset + 32, true),
			capabilityClassToken: view.getUint32(offset + 36, true),
			preconditionClassToken: view.getUint32(offset + 40, true),
			preferredControllerRole: view.getUint32(offset + 44, true),
		} as any;
	}
	outObj.intentId = view.getUint32(offset, true);
	outObj.actionToken = view.getUint32(offset + 4, true);
	outObj.semanticIntentToken = view.getUint32(offset + 8, true);
	outObj.domain = deserializeActionIntentDomain(view, offset + 12);
	outObj.actorSchemaId = view.getUint32(offset + 16, true);
	outObj.argumentOffset = view.getUint32(offset + 20, true);
	outObj.argumentCount = view.getUint32(offset + 24, true);
	outObj.flags = view.getUint32(offset + 28, true);
	outObj.effectClassToken = view.getUint32(offset + 32, true);
	outObj.capabilityClassToken = view.getUint32(offset + 36, true);
	outObj.preconditionClassToken = view.getUint32(offset + 40, true);
	outObj.preferredControllerRole = view.getUint32(offset + 44, true);
	return outObj;
}

export function serializeActionIntentDescriptor(val: ActionIntentDescriptor, view: DataView, offset: number): void {
	view.setUint32(offset, val.intentId, true);
	view.setUint32(offset + 4, val.actionToken, true);
	view.setUint32(offset + 8, val.semanticIntentToken, true);
	serializeActionIntentDomain(val.domain, view, offset + 12);
	view.setUint32(offset + 16, val.actorSchemaId, true);
	view.setUint32(offset + 20, val.argumentOffset, true);
	view.setUint32(offset + 24, val.argumentCount, true);
	view.setUint32(offset + 28, val.flags, true);
	view.setUint32(offset + 32, val.effectClassToken, true);
	view.setUint32(offset + 36, val.capabilityClassToken, true);
	view.setUint32(offset + 40, val.preconditionClassToken, true);
	view.setUint32(offset + 44, val.preferredControllerRole, true);
}

export function deserializeActionArgumentDescriptor(view: DataView, offset: number, outObj?: any): ActionArgumentDescriptor {
	if (!outObj) {
		return {
			intentId: view.getUint32(offset, true),
			argumentIndex: view.getUint32(offset + 4, true),
			roleToken: view.getUint32(offset + 8, true),
			valueKind: deserializeBrainValueKind(view, offset + 12),
			acceptedSchemaId: view.getUint32(offset + 16, true),
			candidateBandMask: view.getUint32(offset + 20, true),
			flags: view.getUint32(offset + 24, true),
			reserved0: view.getUint32(offset + 28, true),
		} as any;
	}
	outObj.intentId = view.getUint32(offset, true);
	outObj.argumentIndex = view.getUint32(offset + 4, true);
	outObj.roleToken = view.getUint32(offset + 8, true);
	outObj.valueKind = deserializeBrainValueKind(view, offset + 12);
	outObj.acceptedSchemaId = view.getUint32(offset + 16, true);
	outObj.candidateBandMask = view.getUint32(offset + 20, true);
	outObj.flags = view.getUint32(offset + 24, true);
	outObj.reserved0 = view.getUint32(offset + 28, true);
	return outObj;
}

export function serializeActionArgumentDescriptor(val: ActionArgumentDescriptor, view: DataView, offset: number): void {
	view.setUint32(offset, val.intentId, true);
	view.setUint32(offset + 4, val.argumentIndex, true);
	view.setUint32(offset + 8, val.roleToken, true);
	serializeBrainValueKind(val.valueKind, view, offset + 12);
	view.setUint32(offset + 16, val.acceptedSchemaId, true);
	view.setUint32(offset + 20, val.candidateBandMask, true);
	view.setUint32(offset + 24, val.flags, true);
	view.setUint32(offset + 28, val.reserved0, true);
}

export function deserializeActionArgumentAuthoringSpec(view: DataView, offset: number, outObj?: any): ActionArgumentAuthoringSpec {
	if (!outObj) {
		return {
			name: ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset),
			roleToken: view.getFloat64(offset + 4, true),
			valueKind: deserializeBrainValueKind(view, offset + 12),
			acceptedSchema: (view.getUint8(offset + 16) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 17) : undefined),
			candidateBands: (view.getUint8(offset + 24) === 1 ? ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(deserializeBrainBandKind(view, o + (i * 1))); } return a; })(offset + 25) : undefined),
			required: (view.getUint8(offset + 32) === 1 ? (view.getUint8(offset + 33) !== 0) : undefined),
			doc: (view.getUint8(offset + 40) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 41) : undefined),
		} as any;
	}
	outObj.name = ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset);
	outObj.roleToken = view.getFloat64(offset + 4, true);
	outObj.valueKind = deserializeBrainValueKind(view, offset + 12);
	outObj.acceptedSchema = (view.getUint8(offset + 16) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 17) : undefined);
	outObj.candidateBands = (view.getUint8(offset + 24) === 1 ? ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(deserializeBrainBandKind(view, o + (i * 1))); } return a; })(offset + 25) : undefined);
	outObj.required = (view.getUint8(offset + 32) === 1 ? (view.getUint8(offset + 33) !== 0) : undefined);
	outObj.doc = (view.getUint8(offset + 40) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 41) : undefined);
	return outObj;
}

export function serializeActionArgumentAuthoringSpec(val: ActionArgumentAuthoringSpec, view: DataView, offset: number): void {
	{ const bytes = __textEncoder!.encode(val.name); const len = Math.min(bytes.length, 255); view.setUint32(offset, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 4, len).set(bytes.subarray(0, len)); }
	view.setFloat64(offset + 4, val.roleToken, true);
	serializeBrainValueKind(val.valueKind, view, offset + 12);
	if (val.acceptedSchema !== undefined) { view.setUint8(offset + 16, 1); { const bytes = __textEncoder!.encode(val.acceptedSchema); const len = Math.min(bytes.length, 255); view.setUint32(offset + 17, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 17 + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 17 + 4, len).set(bytes.subarray(0, len)); } } else { view.setUint8(offset + 16, 0); }
	if (val.candidateBands !== undefined) { view.setUint8(offset + 24, 1); { view.setUint32(offset + 25, val.candidateBands.length, true); let o = offset + 25 + 4; for(let i=0; i<val.candidateBands.length; i++) { serializeBrainBandKind(val.candidateBands[i]!, view, o + (i * 1)); } } } else { view.setUint8(offset + 24, 0); }
	if (val.required !== undefined) { view.setUint8(offset + 32, 1); view.setUint8(offset + 33, (val.required ? 1 : 0)); } else { view.setUint8(offset + 32, 0); }
	if (val.doc !== undefined) { view.setUint8(offset + 40, 1); { const bytes = __textEncoder!.encode(val.doc); const len = Math.min(bytes.length, 255); view.setUint32(offset + 41, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 41 + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 41 + 4, len).set(bytes.subarray(0, len)); } } else { view.setUint8(offset + 40, 0); }
}

export function deserializeActionIntentAuthoringSpec(view: DataView, offset: number, outObj?: any): ActionIntentAuthoringSpec {
	if (!outObj) {
		return {
			name: ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset),
			actionToken: view.getFloat64(offset + 4, true),
			semanticIntentToken: view.getFloat64(offset + 12, true),
			domain: deserializeActionIntentDomain(view, offset + 20),
			arguments: ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(deserializeActionArgumentAuthoringSpec(view, o + (i * 48))); } return a; })(offset + 24),
			effectClassToken: (view.getUint8(offset + 28) === 1 ? view.getFloat64(offset + 29, true) : undefined),
			capabilityClassToken: (view.getUint8(offset + 40) === 1 ? view.getFloat64(offset + 41, true) : undefined),
			preconditionClassToken: (view.getUint8(offset + 52) === 1 ? view.getFloat64(offset + 53, true) : undefined),
			preferredControllerRole: (view.getUint8(offset + 64) === 1 ? view.getFloat64(offset + 65, true) : undefined),
			durative: (view.getUint8(offset + 76) === 1 ? (view.getUint8(offset + 77) !== 0) : undefined),
			doc: (view.getUint8(offset + 84) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 85) : undefined),
		} as any;
	}
	outObj.name = ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset);
	outObj.actionToken = view.getFloat64(offset + 4, true);
	outObj.semanticIntentToken = view.getFloat64(offset + 12, true);
	outObj.domain = deserializeActionIntentDomain(view, offset + 20);
	outObj.arguments = ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(deserializeActionArgumentAuthoringSpec(view, o + (i * 48))); } return a; })(offset + 24);
	outObj.effectClassToken = (view.getUint8(offset + 28) === 1 ? view.getFloat64(offset + 29, true) : undefined);
	outObj.capabilityClassToken = (view.getUint8(offset + 40) === 1 ? view.getFloat64(offset + 41, true) : undefined);
	outObj.preconditionClassToken = (view.getUint8(offset + 52) === 1 ? view.getFloat64(offset + 53, true) : undefined);
	outObj.preferredControllerRole = (view.getUint8(offset + 64) === 1 ? view.getFloat64(offset + 65, true) : undefined);
	outObj.durative = (view.getUint8(offset + 76) === 1 ? (view.getUint8(offset + 77) !== 0) : undefined);
	outObj.doc = (view.getUint8(offset + 84) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 85) : undefined);
	return outObj;
}

export function serializeActionIntentAuthoringSpec(val: ActionIntentAuthoringSpec, view: DataView, offset: number): void {
	{ const bytes = __textEncoder!.encode(val.name); const len = Math.min(bytes.length, 255); view.setUint32(offset, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 4, len).set(bytes.subarray(0, len)); }
	view.setFloat64(offset + 4, val.actionToken, true);
	view.setFloat64(offset + 12, val.semanticIntentToken, true);
	serializeActionIntentDomain(val.domain, view, offset + 20);
	{ view.setUint32(offset + 24, val.arguments.length, true); let o = offset + 24 + 4; for(let i=0; i<val.arguments.length; i++) { serializeActionArgumentAuthoringSpec(val.arguments[i]!, view, o + (i * 48)); } }
	if (val.effectClassToken !== undefined) { view.setUint8(offset + 28, 1); view.setFloat64(offset + 29, val.effectClassToken, true); } else { view.setUint8(offset + 28, 0); }
	if (val.capabilityClassToken !== undefined) { view.setUint8(offset + 40, 1); view.setFloat64(offset + 41, val.capabilityClassToken, true); } else { view.setUint8(offset + 40, 0); }
	if (val.preconditionClassToken !== undefined) { view.setUint8(offset + 52, 1); view.setFloat64(offset + 53, val.preconditionClassToken, true); } else { view.setUint8(offset + 52, 0); }
	if (val.preferredControllerRole !== undefined) { view.setUint8(offset + 64, 1); view.setFloat64(offset + 65, val.preferredControllerRole, true); } else { view.setUint8(offset + 64, 0); }
	if (val.durative !== undefined) { view.setUint8(offset + 76, 1); view.setUint8(offset + 77, (val.durative ? 1 : 0)); } else { view.setUint8(offset + 76, 0); }
	if (val.doc !== undefined) { view.setUint8(offset + 84, 1); { const bytes = __textEncoder!.encode(val.doc); const len = Math.min(bytes.length, 255); view.setUint32(offset + 85, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 85 + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 85 + 4, len).set(bytes.subarray(0, len)); } } else { view.setUint8(offset + 84, 0); }
}

export function deserializeSoftGatherStatus(view: DataView, offset: number): SoftGatherStatus {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "empty";
		case 1: return "selected";
		case 2: return "masked";
		case 3: return "ambiguous";
		case 4: return "error";
		default: throw new Error("Unknown Enum value for SoftGatherStatus: " + v);
	}
}

export function serializeSoftGatherStatus(val: SoftGatherStatus, view: DataView, offset: number): void {
	if(val === "empty") { view.setUint8(offset, 0); return; }
	if(val === "selected") { view.setUint8(offset, 1); return; }
	if(val === "masked") { view.setUint8(offset, 2); return; }
	if(val === "ambiguous") { view.setUint8(offset, 3); return; }
	if(val === "error") { view.setUint8(offset, 4); return; }
}

export function deserializeSoftGatherResult(view: DataView, offset: number, outObj?: any): SoftGatherResult {
	if (!outObj) {
		return {
			status: deserializeSoftGatherStatus(view, offset),
			selectedRecord: view.getUint32(offset + 4, true),
			selectedField: view.getUint32(offset + 8, true),
			selectedReference: view.getUint32(offset + 12, true),
			candidateCount: view.getUint32(offset + 16, true),
			probability: view.getFloat32(offset + 20, true),
			entropy: view.getFloat32(offset + 24, true),
			reserved0: view.getUint32(offset + 28, true),
		} as any;
	}
	outObj.status = deserializeSoftGatherStatus(view, offset);
	outObj.selectedRecord = view.getUint32(offset + 4, true);
	outObj.selectedField = view.getUint32(offset + 8, true);
	outObj.selectedReference = view.getUint32(offset + 12, true);
	outObj.candidateCount = view.getUint32(offset + 16, true);
	outObj.probability = view.getFloat32(offset + 20, true);
	outObj.entropy = view.getFloat32(offset + 24, true);
	outObj.reserved0 = view.getUint32(offset + 28, true);
	return outObj;
}

export function serializeSoftGatherResult(val: SoftGatherResult, view: DataView, offset: number): void {
	serializeSoftGatherStatus(val.status, view, offset);
	view.setUint32(offset + 4, val.selectedRecord, true);
	view.setUint32(offset + 8, val.selectedField, true);
	view.setUint32(offset + 12, val.selectedReference, true);
	view.setUint32(offset + 16, val.candidateCount, true);
	view.setFloat32(offset + 20, val.probability, true);
	view.setFloat32(offset + 24, val.entropy, true);
	view.setUint32(offset + 28, val.reserved0, true);
}

export function deserializeTypedArgumentValue(view: DataView, offset: number, outObj?: any): TypedArgumentValue {
	if (!outObj) {
		return {
			kind: deserializeBrainValueKind(view, offset),
			token: view.getUint32(offset + 4, true),
			flags: view.getUint32(offset + 8, true),
			reserved0: view.getUint32(offset + 12, true),
			handle: deserializeRuntimeRefHandle(view, offset + 16),
			selector: deserializeSoftGatherResult(view, offset + 28),
		} as any;
	}
	outObj.kind = deserializeBrainValueKind(view, offset);
	outObj.token = view.getUint32(offset + 4, true);
	outObj.flags = view.getUint32(offset + 8, true);
	outObj.reserved0 = view.getUint32(offset + 12, true);
	outObj.handle = deserializeRuntimeRefHandle(view, offset + 16);
	outObj.selector = deserializeSoftGatherResult(view, offset + 28);
	return outObj;
}

export function serializeTypedArgumentValue(val: TypedArgumentValue, view: DataView, offset: number): void {
	serializeBrainValueKind(val.kind, view, offset);
	view.setUint32(offset + 4, val.token, true);
	view.setUint32(offset + 8, val.flags, true);
	view.setUint32(offset + 12, val.reserved0, true);
	serializeRuntimeRefHandle(val.handle, view, offset + 16);
	serializeSoftGatherResult(val.selector, view, offset + 28);
}

export function deserializeIntentLifecycle(view: DataView, offset: number): IntentLifecycle {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "empty";
		case 1: return "start";
		case 2: return "maintain";
		case 3: return "stop";
		case 4: return "resume";
		default: throw new Error("Unknown Enum value for IntentLifecycle: " + v);
	}
}

export function serializeIntentLifecycle(val: IntentLifecycle, view: DataView, offset: number): void {
	if(val === "empty") { view.setUint8(offset, 0); return; }
	if(val === "start") { view.setUint8(offset, 1); return; }
	if(val === "maintain") { view.setUint8(offset, 2); return; }
	if(val === "stop") { view.setUint8(offset, 3); return; }
	if(val === "resume") { view.setUint8(offset, 4); return; }
}

export function deserializeIntentExecutionStatus(view: DataView, offset: number): IntentExecutionStatus {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "empty";
		case 1: return "proposed";
		case 2: return "accepted";
		case 3: return "active";
		case 4: return "succeeded";
		case 5: return "partial";
		case 6: return "failed";
		case 7: return "cancelled";
		case 8: return "forgotten";
		default: throw new Error("Unknown Enum value for IntentExecutionStatus: " + v);
	}
}

export function serializeIntentExecutionStatus(val: IntentExecutionStatus, view: DataView, offset: number): void {
	if(val === "empty") { view.setUint8(offset, 0); return; }
	if(val === "proposed") { view.setUint8(offset, 1); return; }
	if(val === "accepted") { view.setUint8(offset, 2); return; }
	if(val === "active") { view.setUint8(offset, 3); return; }
	if(val === "succeeded") { view.setUint8(offset, 4); return; }
	if(val === "partial") { view.setUint8(offset, 5); return; }
	if(val === "failed") { view.setUint8(offset, 6); return; }
	if(val === "cancelled") { view.setUint8(offset, 7); return; }
	if(val === "forgotten") { view.setUint8(offset, 8); return; }
}

export function deserializeIntentProposal(view: DataView, offset: number, outObj?: any): IntentProposal {
	if (!outObj) {
		const _arr_arguments = new Array(4);
		for (let i = 0, _off_arguments = offset + 80; i < 4; i++, _off_arguments += 60) {
			_arr_arguments[i] = ({ kind: deserializeBrainValueKind(view, _off_arguments), token: view.getUint32(_off_arguments + 4, true), flags: view.getUint32(_off_arguments + 8, true), reserved0: view.getUint32(_off_arguments + 12, true), handle: deserializeRuntimeRefHandle(view, _off_arguments + 16), selector: deserializeSoftGatherResult(view, _off_arguments + 28) });
		}
		return {
			proposalSlot: view.getUint32(offset, true),
			lifecycle: deserializeIntentLifecycle(view, offset + 4),
			intentId: view.getUint32(offset + 8, true),
			flags: view.getUint32(offset + 12, true),
			intentRef: deserializeRuntimeRefHandle(view, offset + 16),
			purposeGoal: deserializeRuntimeRefHandle(view, offset + 28),
			controllerHint: deserializeRuntimeRefHandle(view, offset + 40),
			topic: deserializeRuntimeRefHandle(view, offset + 52),
			activation: view.getFloat32(offset + 64, true),
			priority: view.getFloat32(offset + 68, true),
			persistence: view.getFloat32(offset + 72, true),
			confidence: view.getFloat32(offset + 76, true),
			arguments: _arr_arguments,
		} as any;
	}
	const _arr_arguments = new Array(4);
	for (let i = 0, _off_arguments = offset + 80; i < 4; i++, _off_arguments += 60) {
		_arr_arguments[i] = ({ kind: deserializeBrainValueKind(view, _off_arguments), token: view.getUint32(_off_arguments + 4, true), flags: view.getUint32(_off_arguments + 8, true), reserved0: view.getUint32(_off_arguments + 12, true), handle: deserializeRuntimeRefHandle(view, _off_arguments + 16), selector: deserializeSoftGatherResult(view, _off_arguments + 28) });
	}
	outObj.proposalSlot = view.getUint32(offset, true);
	outObj.lifecycle = deserializeIntentLifecycle(view, offset + 4);
	outObj.intentId = view.getUint32(offset + 8, true);
	outObj.flags = view.getUint32(offset + 12, true);
	outObj.intentRef = deserializeRuntimeRefHandle(view, offset + 16);
	outObj.purposeGoal = deserializeRuntimeRefHandle(view, offset + 28);
	outObj.controllerHint = deserializeRuntimeRefHandle(view, offset + 40);
	outObj.topic = deserializeRuntimeRefHandle(view, offset + 52);
	outObj.activation = view.getFloat32(offset + 64, true);
	outObj.priority = view.getFloat32(offset + 68, true);
	outObj.persistence = view.getFloat32(offset + 72, true);
	outObj.confidence = view.getFloat32(offset + 76, true);
	outObj.arguments = _arr_arguments;
	return outObj;
}

export function serializeIntentProposal(val: IntentProposal, view: DataView, offset: number): void {
	view.setUint32(offset, val.proposalSlot, true);
	serializeIntentLifecycle(val.lifecycle, view, offset + 4);
	view.setUint32(offset + 8, val.intentId, true);
	view.setUint32(offset + 12, val.flags, true);
	serializeRuntimeRefHandle(val.intentRef, view, offset + 16);
	serializeRuntimeRefHandle(val.purposeGoal, view, offset + 28);
	serializeRuntimeRefHandle(val.controllerHint, view, offset + 40);
	serializeRuntimeRefHandle(val.topic, view, offset + 52);
	view.setFloat32(offset + 64, val.activation, true);
	view.setFloat32(offset + 68, val.priority, true);
	view.setFloat32(offset + 72, val.persistence, true);
	view.setFloat32(offset + 76, val.confidence, true);
	{ for (let i = 0, __o = offset + 80; i < 4; i++, __o += 60) { const __e = val.arguments[i]!; { serializeBrainValueKind(__e.kind, view, __o); view.setUint32(__o + 4, __e.token, true); view.setUint32(__o + 8, __e.flags, true); view.setUint32(__o + 12, __e.reserved0, true); serializeRuntimeRefHandle(__e.handle, view, __o + 16); serializeSoftGatherResult(__e.selector, view, __o + 28); } } }
}

export function deserializeIntentSet(view: DataView, offset: number, outObj?: any): IntentSet {
	if (!outObj) {
		const _arr_proposals = new Array(8);
		for (let i = 0, _off_proposals = offset + 16; i < 8; i++, _off_proposals += 320) {
			_arr_proposals[i] = ({ proposalSlot: view.getUint32(_off_proposals, true), lifecycle: deserializeIntentLifecycle(view, _off_proposals + 4), intentId: view.getUint32(_off_proposals + 8, true), flags: view.getUint32(_off_proposals + 12, true), intentRef: deserializeRuntimeRefHandle(view, _off_proposals + 16), purposeGoal: deserializeRuntimeRefHandle(view, _off_proposals + 28), controllerHint: deserializeRuntimeRefHandle(view, _off_proposals + 40), topic: deserializeRuntimeRefHandle(view, _off_proposals + 52), activation: view.getFloat32(_off_proposals + 64, true), priority: view.getFloat32(_off_proposals + 68, true), persistence: view.getFloat32(_off_proposals + 72, true), confidence: view.getFloat32(_off_proposals + 76, true), arguments: ((o) => { const a: any[] = []; for(let i=0; i<4; i++) a.push(deserializeTypedArgumentValue(view, o + (i * 60))); return a; })(_off_proposals + 80) });
		}
		return {
			tick: view.getUint32(offset, true),
			count: view.getUint32(offset + 4, true),
			revision: view.getUint32(offset + 8, true),
			flags: view.getUint32(offset + 12, true),
			proposals: _arr_proposals,
		} as any;
	}
	const _arr_proposals = new Array(8);
	for (let i = 0, _off_proposals = offset + 16; i < 8; i++, _off_proposals += 320) {
		_arr_proposals[i] = ({ proposalSlot: view.getUint32(_off_proposals, true), lifecycle: deserializeIntentLifecycle(view, _off_proposals + 4), intentId: view.getUint32(_off_proposals + 8, true), flags: view.getUint32(_off_proposals + 12, true), intentRef: deserializeRuntimeRefHandle(view, _off_proposals + 16), purposeGoal: deserializeRuntimeRefHandle(view, _off_proposals + 28), controllerHint: deserializeRuntimeRefHandle(view, _off_proposals + 40), topic: deserializeRuntimeRefHandle(view, _off_proposals + 52), activation: view.getFloat32(_off_proposals + 64, true), priority: view.getFloat32(_off_proposals + 68, true), persistence: view.getFloat32(_off_proposals + 72, true), confidence: view.getFloat32(_off_proposals + 76, true), arguments: ((o) => { const a: any[] = []; for(let i=0; i<4; i++) a.push(deserializeTypedArgumentValue(view, o + (i * 60))); return a; })(_off_proposals + 80) });
	}
	outObj.tick = view.getUint32(offset, true);
	outObj.count = view.getUint32(offset + 4, true);
	outObj.revision = view.getUint32(offset + 8, true);
	outObj.flags = view.getUint32(offset + 12, true);
	outObj.proposals = _arr_proposals;
	return outObj;
}

export function serializeIntentSet(val: IntentSet, view: DataView, offset: number): void {
	view.setUint32(offset, val.tick, true);
	view.setUint32(offset + 4, val.count, true);
	view.setUint32(offset + 8, val.revision, true);
	view.setUint32(offset + 12, val.flags, true);
	{ for (let i = 0, __o = offset + 16; i < 8; i++, __o += 320) { const __e = val.proposals[i]!; { view.setUint32(__o, __e.proposalSlot, true); serializeIntentLifecycle(__e.lifecycle, view, __o + 4); view.setUint32(__o + 8, __e.intentId, true); view.setUint32(__o + 12, __e.flags, true); serializeRuntimeRefHandle(__e.intentRef, view, __o + 16); serializeRuntimeRefHandle(__e.purposeGoal, view, __o + 28); serializeRuntimeRefHandle(__e.controllerHint, view, __o + 40); serializeRuntimeRefHandle(__e.topic, view, __o + 52); view.setFloat32(__o + 64, __e.activation, true); view.setFloat32(__o + 68, __e.priority, true); view.setFloat32(__o + 72, __e.persistence, true); view.setFloat32(__o + 76, __e.confidence, true); { for (let i = 0, __o1 = __o + 80; i < 4; i++, __o1 += 60) { const __e1 = __e.arguments[i]!; { serializeBrainValueKind(__e1.kind, view, __o1); view.setUint32(__o1 + 4, __e1.token, true); view.setUint32(__o1 + 8, __e1.flags, true); view.setUint32(__o1 + 12, __e1.reserved0, true); serializeRuntimeRefHandle(__e1.handle, view, __o1 + 16); serializeSoftGatherResult(__e1.selector, view, __o1 + 28); } } } } } }
}

export function deserializeActiveIntentState(view: DataView, offset: number, outObj?: any): ActiveIntentState {
	if (!outObj) {
		return {
			intentRef: deserializeRuntimeRefHandle(view, offset),
			purposeGoal: deserializeRuntimeRefHandle(view, offset + 12),
			intentId: view.getUint32(offset + 24, true),
			status: deserializeIntentExecutionStatus(view, offset + 28),
			flags: view.getUint32(offset + 32, true),
			startedAt: view.getUint32(offset + 36, true),
			lastMaintainedAt: view.getUint32(offset + 40, true),
			completedAt: view.getUint32(offset + 44, true),
			activation: view.getFloat32(offset + 48, true),
			progress: view.getFloat32(offset + 52, true),
			outcomeMagnitude: view.getFloat32(offset + 56, true),
			reserved0: view.getFloat32(offset + 60, true),
		} as any;
	}
	outObj.intentRef = deserializeRuntimeRefHandle(view, offset);
	outObj.purposeGoal = deserializeRuntimeRefHandle(view, offset + 12);
	outObj.intentId = view.getUint32(offset + 24, true);
	outObj.status = deserializeIntentExecutionStatus(view, offset + 28);
	outObj.flags = view.getUint32(offset + 32, true);
	outObj.startedAt = view.getUint32(offset + 36, true);
	outObj.lastMaintainedAt = view.getUint32(offset + 40, true);
	outObj.completedAt = view.getUint32(offset + 44, true);
	outObj.activation = view.getFloat32(offset + 48, true);
	outObj.progress = view.getFloat32(offset + 52, true);
	outObj.outcomeMagnitude = view.getFloat32(offset + 56, true);
	outObj.reserved0 = view.getFloat32(offset + 60, true);
	return outObj;
}

export function serializeActiveIntentState(val: ActiveIntentState, view: DataView, offset: number): void {
	serializeRuntimeRefHandle(val.intentRef, view, offset);
	serializeRuntimeRefHandle(val.purposeGoal, view, offset + 12);
	view.setUint32(offset + 24, val.intentId, true);
	serializeIntentExecutionStatus(val.status, view, offset + 28);
	view.setUint32(offset + 32, val.flags, true);
	view.setUint32(offset + 36, val.startedAt, true);
	view.setUint32(offset + 40, val.lastMaintainedAt, true);
	view.setUint32(offset + 44, val.completedAt, true);
	view.setFloat32(offset + 48, val.activation, true);
	view.setFloat32(offset + 52, val.progress, true);
	view.setFloat32(offset + 56, val.outcomeMagnitude, true);
	view.setFloat32(offset + 60, val.reserved0, true);
}

export function deserializeActiveIntentTable(view: DataView, offset: number, outObj?: any): ActiveIntentTable {
	if (!outObj) {
		const _arr_intents = new Array(16);
		for (let i = 0, _off_intents = offset + 16; i < 16; i++, _off_intents += 64) {
			_arr_intents[i] = ({ intentRef: deserializeRuntimeRefHandle(view, _off_intents), purposeGoal: deserializeRuntimeRefHandle(view, _off_intents + 12), intentId: view.getUint32(_off_intents + 24, true), status: deserializeIntentExecutionStatus(view, _off_intents + 28), flags: view.getUint32(_off_intents + 32, true), startedAt: view.getUint32(_off_intents + 36, true), lastMaintainedAt: view.getUint32(_off_intents + 40, true), completedAt: view.getUint32(_off_intents + 44, true), activation: view.getFloat32(_off_intents + 48, true), progress: view.getFloat32(_off_intents + 52, true), outcomeMagnitude: view.getFloat32(_off_intents + 56, true), reserved0: view.getFloat32(_off_intents + 60, true) });
		}
		return {
			revision: view.getUint32(offset, true),
			activeCount: view.getUint32(offset + 4, true),
			completedCount: view.getUint32(offset + 8, true),
			flags: view.getUint32(offset + 12, true),
			intents: _arr_intents,
		} as any;
	}
	const _arr_intents = new Array(16);
	for (let i = 0, _off_intents = offset + 16; i < 16; i++, _off_intents += 64) {
		_arr_intents[i] = ({ intentRef: deserializeRuntimeRefHandle(view, _off_intents), purposeGoal: deserializeRuntimeRefHandle(view, _off_intents + 12), intentId: view.getUint32(_off_intents + 24, true), status: deserializeIntentExecutionStatus(view, _off_intents + 28), flags: view.getUint32(_off_intents + 32, true), startedAt: view.getUint32(_off_intents + 36, true), lastMaintainedAt: view.getUint32(_off_intents + 40, true), completedAt: view.getUint32(_off_intents + 44, true), activation: view.getFloat32(_off_intents + 48, true), progress: view.getFloat32(_off_intents + 52, true), outcomeMagnitude: view.getFloat32(_off_intents + 56, true), reserved0: view.getFloat32(_off_intents + 60, true) });
	}
	outObj.revision = view.getUint32(offset, true);
	outObj.activeCount = view.getUint32(offset + 4, true);
	outObj.completedCount = view.getUint32(offset + 8, true);
	outObj.flags = view.getUint32(offset + 12, true);
	outObj.intents = _arr_intents;
	return outObj;
}

export function serializeActiveIntentTable(val: ActiveIntentTable, view: DataView, offset: number): void {
	view.setUint32(offset, val.revision, true);
	view.setUint32(offset + 4, val.activeCount, true);
	view.setUint32(offset + 8, val.completedCount, true);
	view.setUint32(offset + 12, val.flags, true);
	{ for (let i = 0, __o = offset + 16; i < 16; i++, __o += 64) { const __e = val.intents[i]!; { serializeRuntimeRefHandle(__e.intentRef, view, __o); serializeRuntimeRefHandle(__e.purposeGoal, view, __o + 12); view.setUint32(__o + 24, __e.intentId, true); serializeIntentExecutionStatus(__e.status, view, __o + 28); view.setUint32(__o + 32, __e.flags, true); view.setUint32(__o + 36, __e.startedAt, true); view.setUint32(__o + 40, __e.lastMaintainedAt, true); view.setUint32(__o + 44, __e.completedAt, true); view.setFloat32(__o + 48, __e.activation, true); view.setFloat32(__o + 52, __e.progress, true); view.setFloat32(__o + 56, __e.outcomeMagnitude, true); view.setFloat32(__o + 60, __e.reserved0, true); } } }
}

export function deserializeIntentFeedback(view: DataView, offset: number, outObj?: any): IntentFeedback {
	if (!outObj) {
		return {
			intentRef: deserializeRuntimeRefHandle(view, offset),
			status: deserializeIntentExecutionStatus(view, offset + 12),
			effectClassToken: view.getUint32(offset + 16, true),
			resultToken: view.getUint32(offset + 20, true),
			progress: view.getFloat32(offset + 24, true),
			outcomeMagnitude: view.getFloat32(offset + 28, true),
			comfortMagnitude: view.getFloat32(offset + 32, true),
			tick: view.getUint32(offset + 36, true),
			feedbackRecord: view.getUint32(offset + 40, true),
			flags: view.getUint32(offset + 44, true),
		} as any;
	}
	outObj.intentRef = deserializeRuntimeRefHandle(view, offset);
	outObj.status = deserializeIntentExecutionStatus(view, offset + 12);
	outObj.effectClassToken = view.getUint32(offset + 16, true);
	outObj.resultToken = view.getUint32(offset + 20, true);
	outObj.progress = view.getFloat32(offset + 24, true);
	outObj.outcomeMagnitude = view.getFloat32(offset + 28, true);
	outObj.comfortMagnitude = view.getFloat32(offset + 32, true);
	outObj.tick = view.getUint32(offset + 36, true);
	outObj.feedbackRecord = view.getUint32(offset + 40, true);
	outObj.flags = view.getUint32(offset + 44, true);
	return outObj;
}

export function serializeIntentFeedback(val: IntentFeedback, view: DataView, offset: number): void {
	serializeRuntimeRefHandle(val.intentRef, view, offset);
	serializeIntentExecutionStatus(val.status, view, offset + 12);
	view.setUint32(offset + 16, val.effectClassToken, true);
	view.setUint32(offset + 20, val.resultToken, true);
	view.setFloat32(offset + 24, val.progress, true);
	view.setFloat32(offset + 28, val.outcomeMagnitude, true);
	view.setFloat32(offset + 32, val.comfortMagnitude, true);
	view.setUint32(offset + 36, val.tick, true);
	view.setUint32(offset + 40, val.feedbackRecord, true);
	view.setUint32(offset + 44, val.flags, true);
}

export function deserializeTutorialBeatKind(view: DataView, offset: number): TutorialBeatKind {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "narrate";
		case 1: return "ask";
		case 2: return "present";
		case 3: return "focus";
		case 4: return "demonstrate";
		case 5: return "show_reaction";
		case 6: return "assess";
		case 7: return "wait";
		case 8: return "reset";
		default: throw new Error("Unknown Enum value for TutorialBeatKind: " + v);
	}
}

export function serializeTutorialBeatKind(val: TutorialBeatKind, view: DataView, offset: number): void {
	if(val === "narrate") { view.setUint8(offset, 0); return; }
	if(val === "ask") { view.setUint8(offset, 1); return; }
	if(val === "present") { view.setUint8(offset, 2); return; }
	if(val === "focus") { view.setUint8(offset, 3); return; }
	if(val === "demonstrate") { view.setUint8(offset, 4); return; }
	if(val === "show_reaction") { view.setUint8(offset, 5); return; }
	if(val === "assess") { view.setUint8(offset, 6); return; }
	if(val === "wait") { view.setUint8(offset, 7); return; }
	if(val === "reset") { view.setUint8(offset, 8); return; }
}

export function deserializeTutorialProbeKind(view: DataView, offset: number): TutorialProbeKind {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "token";
		case 1: return "boolean";
		case 2: return "pointer";
		case 3: return "property";
		case 4: return "intent";
		case 5: return "consequence";
		case 6: return "counterexample";
		case 7: return "unknown";
		default: throw new Error("Unknown Enum value for TutorialProbeKind: " + v);
	}
}

export function serializeTutorialProbeKind(val: TutorialProbeKind, view: DataView, offset: number): void {
	if(val === "token") { view.setUint8(offset, 0); return; }
	if(val === "boolean") { view.setUint8(offset, 1); return; }
	if(val === "pointer") { view.setUint8(offset, 2); return; }
	if(val === "property") { view.setUint8(offset, 3); return; }
	if(val === "intent") { view.setUint8(offset, 4); return; }
	if(val === "consequence") { view.setUint8(offset, 5); return; }
	if(val === "counterexample") { view.setUint8(offset, 6); return; }
	if(val === "unknown") { view.setUint8(offset, 7); return; }
}

export function deserializeTutorialProgramHeader(view: DataView, offset: number, outObj?: any): TutorialProgramHeader {
	if (!outObj) {
		return {
			version: view.getUint32(offset, true),
			lessonToken: view.getUint32(offset + 4, true),
			beatOffset: view.getUint32(offset + 8, true),
			beatCount: view.getUint32(offset + 12, true),
			probeOffset: view.getUint32(offset + 16, true),
			probeCount: view.getUint32(offset + 20, true),
			creatorTokenOffset: view.getUint32(offset + 24, true),
			creatorTokenCount: view.getUint32(offset + 28, true),
			flags: view.getUint32(offset + 32, true),
			reserved0: view.getUint32(offset + 36, true),
		} as any;
	}
	outObj.version = view.getUint32(offset, true);
	outObj.lessonToken = view.getUint32(offset + 4, true);
	outObj.beatOffset = view.getUint32(offset + 8, true);
	outObj.beatCount = view.getUint32(offset + 12, true);
	outObj.probeOffset = view.getUint32(offset + 16, true);
	outObj.probeCount = view.getUint32(offset + 20, true);
	outObj.creatorTokenOffset = view.getUint32(offset + 24, true);
	outObj.creatorTokenCount = view.getUint32(offset + 28, true);
	outObj.flags = view.getUint32(offset + 32, true);
	outObj.reserved0 = view.getUint32(offset + 36, true);
	return outObj;
}

export function serializeTutorialProgramHeader(val: TutorialProgramHeader, view: DataView, offset: number): void {
	view.setUint32(offset, val.version, true);
	view.setUint32(offset + 4, val.lessonToken, true);
	view.setUint32(offset + 8, val.beatOffset, true);
	view.setUint32(offset + 12, val.beatCount, true);
	view.setUint32(offset + 16, val.probeOffset, true);
	view.setUint32(offset + 20, val.probeCount, true);
	view.setUint32(offset + 24, val.creatorTokenOffset, true);
	view.setUint32(offset + 28, val.creatorTokenCount, true);
	view.setUint32(offset + 32, val.flags, true);
	view.setUint32(offset + 36, val.reserved0, true);
}

export function deserializeTutorialBeat(view: DataView, offset: number, outObj?: any): TutorialBeat {
	if (!outObj) {
		return {
			kind: deserializeTutorialBeatKind(view, offset),
			sceneCue: view.getUint32(offset + 4, true),
			utteranceOffset: view.getUint32(offset + 8, true),
			utteranceCount: view.getUint32(offset + 12, true),
			holdFrames: view.getUint32(offset + 16, true),
			probeIndex: view.getUint32(offset + 20, true),
			expectedIntentId: view.getUint32(offset + 24, true),
			flags: view.getUint32(offset + 28, true),
		} as any;
	}
	outObj.kind = deserializeTutorialBeatKind(view, offset);
	outObj.sceneCue = view.getUint32(offset + 4, true);
	outObj.utteranceOffset = view.getUint32(offset + 8, true);
	outObj.utteranceCount = view.getUint32(offset + 12, true);
	outObj.holdFrames = view.getUint32(offset + 16, true);
	outObj.probeIndex = view.getUint32(offset + 20, true);
	outObj.expectedIntentId = view.getUint32(offset + 24, true);
	outObj.flags = view.getUint32(offset + 28, true);
	return outObj;
}

export function serializeTutorialBeat(val: TutorialBeat, view: DataView, offset: number): void {
	serializeTutorialBeatKind(val.kind, view, offset);
	view.setUint32(offset + 4, val.sceneCue, true);
	view.setUint32(offset + 8, val.utteranceOffset, true);
	view.setUint32(offset + 12, val.utteranceCount, true);
	view.setUint32(offset + 16, val.holdFrames, true);
	view.setUint32(offset + 20, val.probeIndex, true);
	view.setUint32(offset + 24, val.expectedIntentId, true);
	view.setUint32(offset + 28, val.flags, true);
}

export function deserializeTutorialProbe(view: DataView, offset: number, outObj?: any): TutorialProbe {
	if (!outObj) {
		return {
			kind: deserializeTutorialProbeKind(view, offset),
			querySchemaId: view.getUint32(offset + 4, true),
			expectedToken: view.getUint32(offset + 8, true),
			expectedIntentId: view.getUint32(offset + 12, true),
			expectedRecord: view.getUint32(offset + 16, true),
			expectedField: view.getUint32(offset + 20, true),
			oracleBinding: view.getUint32(offset + 24, true),
			flags: view.getUint32(offset + 28, true),
			reserved0: view.getUint32(offset + 32, true),
		} as any;
	}
	outObj.kind = deserializeTutorialProbeKind(view, offset);
	outObj.querySchemaId = view.getUint32(offset + 4, true);
	outObj.expectedToken = view.getUint32(offset + 8, true);
	outObj.expectedIntentId = view.getUint32(offset + 12, true);
	outObj.expectedRecord = view.getUint32(offset + 16, true);
	outObj.expectedField = view.getUint32(offset + 20, true);
	outObj.oracleBinding = view.getUint32(offset + 24, true);
	outObj.flags = view.getUint32(offset + 28, true);
	outObj.reserved0 = view.getUint32(offset + 32, true);
	return outObj;
}

export function serializeTutorialProbe(val: TutorialProbe, view: DataView, offset: number): void {
	serializeTutorialProbeKind(val.kind, view, offset);
	view.setUint32(offset + 4, val.querySchemaId, true);
	view.setUint32(offset + 8, val.expectedToken, true);
	view.setUint32(offset + 12, val.expectedIntentId, true);
	view.setUint32(offset + 16, val.expectedRecord, true);
	view.setUint32(offset + 20, val.expectedField, true);
	view.setUint32(offset + 24, val.oracleBinding, true);
	view.setUint32(offset + 28, val.flags, true);
	view.setUint32(offset + 32, val.reserved0, true);
}

export function deserializeTutorialRuntimeStatus(view: DataView, offset: number): TutorialRuntimeStatus {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "idle";
		case 1: return "demonstrating";
		case 2: return "waiting";
		case 3: return "assessing";
		case 4: return "passed";
		case 5: return "failed";
		case 6: return "done";
		default: throw new Error("Unknown Enum value for TutorialRuntimeStatus: " + v);
	}
}

export function serializeTutorialRuntimeStatus(val: TutorialRuntimeStatus, view: DataView, offset: number): void {
	if(val === "idle") { view.setUint8(offset, 0); return; }
	if(val === "demonstrating") { view.setUint8(offset, 1); return; }
	if(val === "waiting") { view.setUint8(offset, 2); return; }
	if(val === "assessing") { view.setUint8(offset, 3); return; }
	if(val === "passed") { view.setUint8(offset, 4); return; }
	if(val === "failed") { view.setUint8(offset, 5); return; }
	if(val === "done") { view.setUint8(offset, 6); return; }
}

export function deserializeTutorialRuntimeState(view: DataView, offset: number, outObj?: any): TutorialRuntimeState {
	if (!outObj) {
		return {
			program: view.getUint32(offset, true),
			beat: view.getUint32(offset + 4, true),
			probe: view.getUint32(offset + 8, true),
			status: deserializeTutorialRuntimeStatus(view, offset + 12),
			frameInBeat: view.getUint32(offset + 16, true),
			attempts: view.getUint32(offset + 20, true),
			correct: view.getUint32(offset + 24, true),
			incorrect: view.getUint32(offset + 28, true),
		} as any;
	}
	outObj.program = view.getUint32(offset, true);
	outObj.beat = view.getUint32(offset + 4, true);
	outObj.probe = view.getUint32(offset + 8, true);
	outObj.status = deserializeTutorialRuntimeStatus(view, offset + 12);
	outObj.frameInBeat = view.getUint32(offset + 16, true);
	outObj.attempts = view.getUint32(offset + 20, true);
	outObj.correct = view.getUint32(offset + 24, true);
	outObj.incorrect = view.getUint32(offset + 28, true);
	return outObj;
}

export function serializeTutorialRuntimeState(val: TutorialRuntimeState, view: DataView, offset: number): void {
	view.setUint32(offset, val.program, true);
	view.setUint32(offset + 4, val.beat, true);
	view.setUint32(offset + 8, val.probe, true);
	serializeTutorialRuntimeStatus(val.status, view, offset + 12);
	view.setUint32(offset + 16, val.frameInBeat, true);
	view.setUint32(offset + 20, val.attempts, true);
	view.setUint32(offset + 24, val.correct, true);
	view.setUint32(offset + 28, val.incorrect, true);
}

export function deserializeTutorialProbeAuthoringSpec(view: DataView, offset: number, outObj?: any): TutorialProbeAuthoringSpec {
	if (!outObj) {
		return {
			kind: deserializeTutorialProbeKind(view, offset),
			querySchema: (view.getUint8(offset + 4) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 5) : undefined),
			expectedToken: (view.getUint8(offset + 12) === 1 ? view.getFloat64(offset + 13, true) : undefined),
			expectedIntent: (view.getUint8(offset + 24) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 25) : undefined),
			oracleBinding: (view.getUint8(offset + 32) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 33) : undefined),
			doc: (view.getUint8(offset + 40) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 41) : undefined),
		} as any;
	}
	outObj.kind = deserializeTutorialProbeKind(view, offset);
	outObj.querySchema = (view.getUint8(offset + 4) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 5) : undefined);
	outObj.expectedToken = (view.getUint8(offset + 12) === 1 ? view.getFloat64(offset + 13, true) : undefined);
	outObj.expectedIntent = (view.getUint8(offset + 24) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 25) : undefined);
	outObj.oracleBinding = (view.getUint8(offset + 32) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 33) : undefined);
	outObj.doc = (view.getUint8(offset + 40) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 41) : undefined);
	return outObj;
}

export function serializeTutorialProbeAuthoringSpec(val: TutorialProbeAuthoringSpec, view: DataView, offset: number): void {
	serializeTutorialProbeKind(val.kind, view, offset);
	if (val.querySchema !== undefined) { view.setUint8(offset + 4, 1); { const bytes = __textEncoder!.encode(val.querySchema); const len = Math.min(bytes.length, 255); view.setUint32(offset + 5, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 5 + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 5 + 4, len).set(bytes.subarray(0, len)); } } else { view.setUint8(offset + 4, 0); }
	if (val.expectedToken !== undefined) { view.setUint8(offset + 12, 1); view.setFloat64(offset + 13, val.expectedToken, true); } else { view.setUint8(offset + 12, 0); }
	if (val.expectedIntent !== undefined) { view.setUint8(offset + 24, 1); { const bytes = __textEncoder!.encode(val.expectedIntent); const len = Math.min(bytes.length, 255); view.setUint32(offset + 25, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 25 + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 25 + 4, len).set(bytes.subarray(0, len)); } } else { view.setUint8(offset + 24, 0); }
	if (val.oracleBinding !== undefined) { view.setUint8(offset + 32, 1); { const bytes = __textEncoder!.encode(val.oracleBinding); const len = Math.min(bytes.length, 255); view.setUint32(offset + 33, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 33 + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 33 + 4, len).set(bytes.subarray(0, len)); } } else { view.setUint8(offset + 32, 0); }
	if (val.doc !== undefined) { view.setUint8(offset + 40, 1); { const bytes = __textEncoder!.encode(val.doc); const len = Math.min(bytes.length, 255); view.setUint32(offset + 41, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 41 + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 41 + 4, len).set(bytes.subarray(0, len)); } } else { view.setUint8(offset + 40, 0); }
}

export function deserializeTutorialBeatAuthoringSpec(view: DataView, offset: number, outObj?: any): TutorialBeatAuthoringSpec {
	if (!outObj) {
		return {
			kind: deserializeTutorialBeatKind(view, offset),
			sceneCue: (view.getUint8(offset + 4) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 5) : undefined),
			utterance: (view.getUint8(offset + 12) === 1 ? ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(view.getFloat64(o + (i * 8), true)); } return a; })(offset + 13) : undefined),
			holdFrames: (view.getUint8(offset + 20) === 1 ? view.getFloat64(offset + 21, true) : undefined),
			expectedIntent: (view.getUint8(offset + 32) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 33) : undefined),
			probe: (view.getUint8(offset + 40) === 1 ? deserializeTutorialProbeAuthoringSpec(view, offset + 41) : undefined),
			doc: (view.getUint8(offset + 92) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 93) : undefined),
		} as any;
	}
	outObj.kind = deserializeTutorialBeatKind(view, offset);
	outObj.sceneCue = (view.getUint8(offset + 4) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 5) : undefined);
	outObj.utterance = (view.getUint8(offset + 12) === 1 ? ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(view.getFloat64(o + (i * 8), true)); } return a; })(offset + 13) : undefined);
	outObj.holdFrames = (view.getUint8(offset + 20) === 1 ? view.getFloat64(offset + 21, true) : undefined);
	outObj.expectedIntent = (view.getUint8(offset + 32) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 33) : undefined);
	outObj.probe = (view.getUint8(offset + 40) === 1 ? deserializeTutorialProbeAuthoringSpec(view, offset + 41) : undefined);
	outObj.doc = (view.getUint8(offset + 92) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 93) : undefined);
	return outObj;
}

export function serializeTutorialBeatAuthoringSpec(val: TutorialBeatAuthoringSpec, view: DataView, offset: number): void {
	serializeTutorialBeatKind(val.kind, view, offset);
	if (val.sceneCue !== undefined) { view.setUint8(offset + 4, 1); { const bytes = __textEncoder!.encode(val.sceneCue); const len = Math.min(bytes.length, 255); view.setUint32(offset + 5, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 5 + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 5 + 4, len).set(bytes.subarray(0, len)); } } else { view.setUint8(offset + 4, 0); }
	if (val.utterance !== undefined) { view.setUint8(offset + 12, 1); { view.setUint32(offset + 13, val.utterance.length, true); let o = offset + 13 + 4; for(let i=0; i<val.utterance.length; i++) { view.setFloat64(o + (i * 8), val.utterance[i]!, true); } } } else { view.setUint8(offset + 12, 0); }
	if (val.holdFrames !== undefined) { view.setUint8(offset + 20, 1); view.setFloat64(offset + 21, val.holdFrames, true); } else { view.setUint8(offset + 20, 0); }
	if (val.expectedIntent !== undefined) { view.setUint8(offset + 32, 1); { const bytes = __textEncoder!.encode(val.expectedIntent); const len = Math.min(bytes.length, 255); view.setUint32(offset + 33, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 33 + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 33 + 4, len).set(bytes.subarray(0, len)); } } else { view.setUint8(offset + 32, 0); }
	if (val.probe !== undefined) { view.setUint8(offset + 40, 1); serializeTutorialProbeAuthoringSpec(val.probe, view, offset + 41); } else { view.setUint8(offset + 40, 0); }
	if (val.doc !== undefined) { view.setUint8(offset + 92, 1); { const bytes = __textEncoder!.encode(val.doc); const len = Math.min(bytes.length, 255); view.setUint32(offset + 93, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 93 + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 93 + 4, len).set(bytes.subarray(0, len)); } } else { view.setUint8(offset + 92, 0); }
}

export function deserializeTutorialAuthoringSpec(view: DataView, offset: number, outObj?: any): TutorialAuthoringSpec {
	if (!outObj) {
		return {
			name: ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset),
			lessonTokens: ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(view.getFloat64(o + (i * 8), true)); } return a; })(offset + 4),
			prerequisites: (view.getUint8(offset + 8) === 1 ? ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(o + (i * 4))); } return a; })(offset + 9) : undefined),
			beats: ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(deserializeTutorialBeatAuthoringSpec(view, o + (i * 100))); } return a; })(offset + 16),
			doc: (view.getUint8(offset + 20) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 21) : undefined),
		} as any;
	}
	outObj.name = ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset);
	outObj.lessonTokens = ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(view.getFloat64(o + (i * 8), true)); } return a; })(offset + 4);
	outObj.prerequisites = (view.getUint8(offset + 8) === 1 ? ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(o + (i * 4))); } return a; })(offset + 9) : undefined);
	outObj.beats = ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(deserializeTutorialBeatAuthoringSpec(view, o + (i * 100))); } return a; })(offset + 16);
	outObj.doc = (view.getUint8(offset + 20) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 21) : undefined);
	return outObj;
}

export function serializeTutorialAuthoringSpec(val: TutorialAuthoringSpec, view: DataView, offset: number): void {
	{ const bytes = __textEncoder!.encode(val.name); const len = Math.min(bytes.length, 255); view.setUint32(offset, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 4, len).set(bytes.subarray(0, len)); }
	{ view.setUint32(offset + 4, val.lessonTokens.length, true); let o = offset + 4 + 4; for(let i=0; i<val.lessonTokens.length; i++) { view.setFloat64(o + (i * 8), val.lessonTokens[i]!, true); } }
	if (val.prerequisites !== undefined) { view.setUint8(offset + 8, 1); { view.setUint32(offset + 9, val.prerequisites.length, true); let o = offset + 9 + 4; for(let i=0; i<val.prerequisites.length; i++) { { const bytes = __textEncoder!.encode(val.prerequisites[i]!); const len = Math.min(bytes.length, 255); view.setUint32(o + (i * 4), len, true); new Uint8Array(view.buffer, view.byteOffset + o + (i * 4) + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + o + (i * 4) + 4, len).set(bytes.subarray(0, len)); } } } } else { view.setUint8(offset + 8, 0); }
	{ view.setUint32(offset + 16, val.beats.length, true); let o = offset + 16 + 4; for(let i=0; i<val.beats.length; i++) { serializeTutorialBeatAuthoringSpec(val.beats[i]!, view, o + (i * 100)); } }
	if (val.doc !== undefined) { view.setUint8(offset + 20, 1); { const bytes = __textEncoder!.encode(val.doc); const len = Math.min(bytes.length, 255); view.setUint32(offset + 21, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 21 + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 21 + 4, len).set(bytes.subarray(0, len)); } } else { view.setUint8(offset + 20, 0); }
}

export function deserializeBrainRuntimeStatus(view: DataView, offset: number): BrainRuntimeStatus {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "idle";
		case 1: return "assembling_frame";
		case 2: return "running";
		case 3: return "executing";
		case 4: return "done";
		case 5: return "error";
		default: throw new Error("Unknown Enum value for BrainRuntimeStatus: " + v);
	}
}

export function serializeBrainRuntimeStatus(val: BrainRuntimeStatus, view: DataView, offset: number): void {
	if(val === "idle") { view.setUint8(offset, 0); return; }
	if(val === "assembling_frame") { view.setUint8(offset, 1); return; }
	if(val === "running") { view.setUint8(offset, 2); return; }
	if(val === "executing") { view.setUint8(offset, 3); return; }
	if(val === "done") { view.setUint8(offset, 4); return; }
	if(val === "error") { view.setUint8(offset, 5); return; }
}

export function deserializeBrainModelConfig(view: DataView, offset: number, outObj?: any): BrainModelConfig {
	if (!outObj) {
		return {
			vocabSize: view.getUint32(offset, true),
			contextTokens: view.getUint32(offset + 4, true),
			recordWidth: view.getUint32(offset + 8, true),
			recordSlots: view.getUint32(offset + 12, true),
			hiddenSize: view.getUint32(offset + 16, true),
			recordSize: view.getUint32(offset + 20, true),
			layerCount: view.getUint32(offset + 24, true),
			attentionHeads: view.getUint32(offset + 28, true),
			maxIntentProposals: view.getUint32(offset + 32, true),
			flags: view.getUint32(offset + 36, true),
		} as any;
	}
	outObj.vocabSize = view.getUint32(offset, true);
	outObj.contextTokens = view.getUint32(offset + 4, true);
	outObj.recordWidth = view.getUint32(offset + 8, true);
	outObj.recordSlots = view.getUint32(offset + 12, true);
	outObj.hiddenSize = view.getUint32(offset + 16, true);
	outObj.recordSize = view.getUint32(offset + 20, true);
	outObj.layerCount = view.getUint32(offset + 24, true);
	outObj.attentionHeads = view.getUint32(offset + 28, true);
	outObj.maxIntentProposals = view.getUint32(offset + 32, true);
	outObj.flags = view.getUint32(offset + 36, true);
	return outObj;
}

export function serializeBrainModelConfig(val: BrainModelConfig, view: DataView, offset: number): void {
	view.setUint32(offset, val.vocabSize, true);
	view.setUint32(offset + 4, val.contextTokens, true);
	view.setUint32(offset + 8, val.recordWidth, true);
	view.setUint32(offset + 12, val.recordSlots, true);
	view.setUint32(offset + 16, val.hiddenSize, true);
	view.setUint32(offset + 20, val.recordSize, true);
	view.setUint32(offset + 24, val.layerCount, true);
	view.setUint32(offset + 28, val.attentionHeads, true);
	view.setUint32(offset + 32, val.maxIntentProposals, true);
	view.setUint32(offset + 36, val.flags, true);
}

export function deserializeBrainRuntimeConfig(view: DataView, offset: number, outObj?: any): BrainRuntimeConfig {
	if (!outObj) {
		return {
			tokenAbiVersion: view.getUint32(offset, true),
			architectureVersion: view.getUint32(offset + 4, true),
			frameLayoutVersion: view.getUint32(offset + 8, true),
			vocabManifestVersion: view.getUint32(offset + 12, true),
			recordManifestVersion: view.getUint32(offset + 16, true),
			actionCatalogVersion: view.getUint32(offset + 20, true),
			tutorialVersion: view.getUint32(offset + 24, true),
			flags: view.getUint32(offset + 28, true),
			reserved0: view.getUint32(offset + 32, true),
			model: deserializeBrainModelConfig(view, offset + 36),
			memory: deserializeMemoryConfig(view, offset + 76),
		} as any;
	}
	outObj.tokenAbiVersion = view.getUint32(offset, true);
	outObj.architectureVersion = view.getUint32(offset + 4, true);
	outObj.frameLayoutVersion = view.getUint32(offset + 8, true);
	outObj.vocabManifestVersion = view.getUint32(offset + 12, true);
	outObj.recordManifestVersion = view.getUint32(offset + 16, true);
	outObj.actionCatalogVersion = view.getUint32(offset + 20, true);
	outObj.tutorialVersion = view.getUint32(offset + 24, true);
	outObj.flags = view.getUint32(offset + 28, true);
	outObj.reserved0 = view.getUint32(offset + 32, true);
	outObj.model = deserializeBrainModelConfig(view, offset + 36);
	outObj.memory = deserializeMemoryConfig(view, offset + 76);
	return outObj;
}

export function serializeBrainRuntimeConfig(val: BrainRuntimeConfig, view: DataView, offset: number): void {
	view.setUint32(offset, val.tokenAbiVersion, true);
	view.setUint32(offset + 4, val.architectureVersion, true);
	view.setUint32(offset + 8, val.frameLayoutVersion, true);
	view.setUint32(offset + 12, val.vocabManifestVersion, true);
	view.setUint32(offset + 16, val.recordManifestVersion, true);
	view.setUint32(offset + 20, val.actionCatalogVersion, true);
	view.setUint32(offset + 24, val.tutorialVersion, true);
	view.setUint32(offset + 28, val.flags, true);
	view.setUint32(offset + 32, val.reserved0, true);
	serializeBrainModelConfig(val.model, view, offset + 36);
	serializeMemoryConfig(val.memory, view, offset + 76);
}

export function deserializeBrainRuntimeState(view: DataView, offset: number, outObj?: any): BrainRuntimeState {
	if (!outObj) {
		return {
			status: deserializeBrainRuntimeStatus(view, offset),
			tick: view.getUint32(offset + 4, true),
			snapshot: view.getUint32(offset + 8, true),
			frameRevision: view.getUint32(offset + 12, true),
			memoryRevision: view.getUint32(offset + 16, true),
			queryRevision: view.getUint32(offset + 20, true),
			intentRevision: view.getUint32(offset + 24, true),
			errorCode: view.getUint32(offset + 28, true),
		} as any;
	}
	outObj.status = deserializeBrainRuntimeStatus(view, offset);
	outObj.tick = view.getUint32(offset + 4, true);
	outObj.snapshot = view.getUint32(offset + 8, true);
	outObj.frameRevision = view.getUint32(offset + 12, true);
	outObj.memoryRevision = view.getUint32(offset + 16, true);
	outObj.queryRevision = view.getUint32(offset + 20, true);
	outObj.intentRevision = view.getUint32(offset + 24, true);
	outObj.errorCode = view.getUint32(offset + 28, true);
	return outObj;
}

export function serializeBrainRuntimeState(val: BrainRuntimeState, view: DataView, offset: number): void {
	serializeBrainRuntimeStatus(val.status, view, offset);
	view.setUint32(offset + 4, val.tick, true);
	view.setUint32(offset + 8, val.snapshot, true);
	view.setUint32(offset + 12, val.frameRevision, true);
	view.setUint32(offset + 16, val.memoryRevision, true);
	view.setUint32(offset + 20, val.queryRevision, true);
	view.setUint32(offset + 24, val.intentRevision, true);
	view.setUint32(offset + 28, val.errorCode, true);
}

export function deserializeBrainStepTelemetry(view: DataView, offset: number, outObj?: any): BrainStepTelemetry {
	if (!outObj) {
		return {
			tick: view.getUint32(offset, true),
			activeRecords: view.getUint32(offset + 4, true),
			activeTokens: view.getUint32(offset + 8, true),
			truncatedRecords: view.getUint32(offset + 12, true),
			intentCount: view.getUint32(offset + 16, true),
			activeIntentCount: view.getUint32(offset + 20, true),
			memoryCount: view.getUint32(offset + 24, true),
			queryCount: view.getUint32(offset + 28, true),
			frameBuildMs: view.getFloat32(offset + 32, true),
			localEncodeMs: view.getFloat32(offset + 36, true),
			recordMixMs: view.getFloat32(offset + 40, true),
			gatherMs: view.getFloat32(offset + 44, true),
			decideMs: view.getFloat32(offset + 48, true),
			runtimeMs: view.getFloat32(offset + 52, true),
			meanGatherEntropy: view.getFloat32(offset + 56, true),
			minGatherProbability: view.getFloat32(offset + 60, true),
			flags: view.getUint32(offset + 64, true),
			errorCode: view.getUint32(offset + 68, true),
		} as any;
	}
	outObj.tick = view.getUint32(offset, true);
	outObj.activeRecords = view.getUint32(offset + 4, true);
	outObj.activeTokens = view.getUint32(offset + 8, true);
	outObj.truncatedRecords = view.getUint32(offset + 12, true);
	outObj.intentCount = view.getUint32(offset + 16, true);
	outObj.activeIntentCount = view.getUint32(offset + 20, true);
	outObj.memoryCount = view.getUint32(offset + 24, true);
	outObj.queryCount = view.getUint32(offset + 28, true);
	outObj.frameBuildMs = view.getFloat32(offset + 32, true);
	outObj.localEncodeMs = view.getFloat32(offset + 36, true);
	outObj.recordMixMs = view.getFloat32(offset + 40, true);
	outObj.gatherMs = view.getFloat32(offset + 44, true);
	outObj.decideMs = view.getFloat32(offset + 48, true);
	outObj.runtimeMs = view.getFloat32(offset + 52, true);
	outObj.meanGatherEntropy = view.getFloat32(offset + 56, true);
	outObj.minGatherProbability = view.getFloat32(offset + 60, true);
	outObj.flags = view.getUint32(offset + 64, true);
	outObj.errorCode = view.getUint32(offset + 68, true);
	return outObj;
}

export function serializeBrainStepTelemetry(val: BrainStepTelemetry, view: DataView, offset: number): void {
	view.setUint32(offset, val.tick, true);
	view.setUint32(offset + 4, val.activeRecords, true);
	view.setUint32(offset + 8, val.activeTokens, true);
	view.setUint32(offset + 12, val.truncatedRecords, true);
	view.setUint32(offset + 16, val.intentCount, true);
	view.setUint32(offset + 20, val.activeIntentCount, true);
	view.setUint32(offset + 24, val.memoryCount, true);
	view.setUint32(offset + 28, val.queryCount, true);
	view.setFloat32(offset + 32, val.frameBuildMs, true);
	view.setFloat32(offset + 36, val.localEncodeMs, true);
	view.setFloat32(offset + 40, val.recordMixMs, true);
	view.setFloat32(offset + 44, val.gatherMs, true);
	view.setFloat32(offset + 48, val.decideMs, true);
	view.setFloat32(offset + 52, val.runtimeMs, true);
	view.setFloat32(offset + 56, val.meanGatherEntropy, true);
	view.setFloat32(offset + 60, val.minGatherProbability, true);
	view.setUint32(offset + 64, val.flags, true);
	view.setUint32(offset + 68, val.errorCode, true);
}

export function deserializeBrainStepResult(view: DataView, offset: number, outObj?: any): BrainStepResult {
	if (!outObj) {
		return {
			state: deserializeBrainRuntimeState(view, offset),
			intents: deserializeIntentSet(view, offset + 32),
			telemetry: deserializeBrainStepTelemetry(view, offset + 2608),
		} as any;
	}
	outObj.state = deserializeBrainRuntimeState(view, offset);
	outObj.intents = deserializeIntentSet(view, offset + 32);
	outObj.telemetry = deserializeBrainStepTelemetry(view, offset + 2608);
	return outObj;
}

export function serializeBrainStepResult(val: BrainStepResult, view: DataView, offset: number): void {
	serializeBrainRuntimeState(val.state, view, offset);
	serializeIntentSet(val.intents, view, offset + 32);
	serializeBrainStepTelemetry(val.telemetry, view, offset + 2608);
}

