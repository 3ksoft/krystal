// THIS FILE IS AUTO-GENERATED - DO NOT CHANGE

import type { v1_0_0 } from "./schema.types.ts";

type ConstraintNodeKind = v1_0_0.ConstraintNodeKind;
type ConstraintDecoderStatus = v1_0_0.ConstraintDecoderStatus;
type ConstraintProgramHeader = v1_0_0.ConstraintProgramHeader;
type ConstraintNode = v1_0_0.ConstraintNode;
type ConstraintByteEdge = v1_0_0.ConstraintByteEdge;
type ConstraintTokenizerHeader = v1_0_0.ConstraintTokenizerHeader;
type ConstraintTokenByteEntry = v1_0_0.ConstraintTokenByteEntry;
type ConstraintDecoderState = v1_0_0.ConstraintDecoderState;
type Lfm2Mode = v1_0_0.Lfm2Mode;
type OpParams = v1_0_0.OpParams;
type LlmRuntimeStatus = v1_0_0.LlmRuntimeStatus;
type LlmRuntime = v1_0_0.LlmRuntime;
type DecodeTelemetryEntry = v1_0_0.DecodeTelemetryEntry;
type GenerateOptions = v1_0_0.GenerateOptions;
type GenerateTimings = v1_0_0.GenerateTimings;
type GenerateResult = v1_0_0.GenerateResult;
type CacheBlockOptions = v1_0_0.CacheBlockOptions;
type GpuWeightFormat = v1_0_0.GpuWeightFormat;
type Lfm2LayerKind = v1_0_0.Lfm2LayerKind;
type Lfm2RuntimeConfig = v1_0_0.Lfm2RuntimeConfig;
type MatmulDispatchArgs = v1_0_0.MatmulDispatchArgs;
type GPUBuffer = v1_0_0.GPUBuffer;
type GPUDevice = v1_0_0.GPUDevice;
type GpuTensorPage = v1_0_0.GpuTensorPage;
type GpuTensor = v1_0_0.GpuTensor;
type Lfm2RuntimeModel = v1_0_0.Lfm2RuntimeModel;
type MatmulKernelSpec = v1_0_0.MatmulKernelSpec;
type Lfm2RuntimeOptions = v1_0_0.Lfm2RuntimeOptions;

const __textDecoder = typeof TextDecoder !== "undefined" ? new TextDecoder() : null;
const __textEncoder = typeof TextEncoder !== "undefined" ? new TextEncoder() : null;

export const SIZEOF_ConstraintProgramHeader = 48;
export const SIZEOF_ConstraintNode = 48;
export const SIZEOF_ConstraintByteEdge = 4;
export const SIZEOF_ConstraintTokenizerHeader = 32;
export const SIZEOF_ConstraintTokenByteEntry = 8;
export const SIZEOF_ConstraintDecoderState = 96;
export const CONSTRAINT_DECODER_STATE_NUMBERTEXT_LEN = 16;
export const SIZEOF_OpParams = 96;
export const SIZEOF_LlmRuntime = 48;
export const SIZEOF_DecodeTelemetryEntry = 4;
export const SIZEOF_GenerateOptions = 20;
export const SIZEOF_GenerateTimings = 108;
export const SIZEOF_GenerateResult = 164;
export const SIZEOF_CacheBlockOptions = 12;
export const SIZEOF_Lfm2RuntimeConfig = 100;
export const SIZEOF_MatmulDispatchArgs = 32;
export const SIZEOF_GpuTensorPage = 24;
export const SIZEOF_GpuTensor = 28;
export const SIZEOF_Lfm2RuntimeModel = 100;
export const SIZEOF_MatmulKernelSpec = 16;
export const SIZEOF_Lfm2RuntimeOptions = 88;

export function deserializeConstraintNodeKind(view: DataView, offset: number): ConstraintNodeKind {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "literal";
		case 1: return "switch";
		case 2: return "string";
		case 3: return "number";
		case 4: return "accept";
		case 5: return "jump";
		default: throw new Error("Unknown Enum value for ConstraintNodeKind: " + v);
	}
}

export function serializeConstraintNodeKind(val: ConstraintNodeKind, view: DataView, offset: number): void {
	if(val === "literal") { view.setUint8(offset, 0); return; }
	if(val === "switch") { view.setUint8(offset, 1); return; }
	if(val === "string") { view.setUint8(offset, 2); return; }
	if(val === "number") { view.setUint8(offset, 3); return; }
	if(val === "accept") { view.setUint8(offset, 4); return; }
	if(val === "jump") { view.setUint8(offset, 5); return; }
}

export function deserializeConstraintDecoderStatus(view: DataView, offset: number): ConstraintDecoderStatus {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "running";
		case 1: return "accept";
		case 2: return "dead";
		case 3: return "error";
		default: throw new Error("Unknown Enum value for ConstraintDecoderStatus: " + v);
	}
}

export function serializeConstraintDecoderStatus(val: ConstraintDecoderStatus, view: DataView, offset: number): void {
	if(val === "running") { view.setUint8(offset, 0); return; }
	if(val === "accept") { view.setUint8(offset, 1); return; }
	if(val === "dead") { view.setUint8(offset, 2); return; }
	if(val === "error") { view.setUint8(offset, 3); return; }
}

export function deserializeConstraintProgramHeader(view: DataView, offset: number, outObj?: any): ConstraintProgramHeader {
	if (!outObj) {
		return {
			version: view.getUint32(offset, true),
			flags: view.getUint32(offset + 4, true),
			entryNode: view.getUint32(offset + 8, true),
			acceptNode: view.getUint32(offset + 12, true),
			nodeWordOffset: view.getUint32(offset + 16, true),
			nodeCount: view.getUint32(offset + 20, true),
			edgeWordOffset: view.getUint32(offset + 24, true),
			edgeCount: view.getUint32(offset + 28, true),
			byteWordOffset: view.getUint32(offset + 32, true),
			byteLength: view.getUint32(offset + 36, true),
			reserved0: view.getUint32(offset + 40, true),
			reserved1: view.getUint32(offset + 44, true),
		} as any;
	}
	outObj.version = view.getUint32(offset, true);
	outObj.flags = view.getUint32(offset + 4, true);
	outObj.entryNode = view.getUint32(offset + 8, true);
	outObj.acceptNode = view.getUint32(offset + 12, true);
	outObj.nodeWordOffset = view.getUint32(offset + 16, true);
	outObj.nodeCount = view.getUint32(offset + 20, true);
	outObj.edgeWordOffset = view.getUint32(offset + 24, true);
	outObj.edgeCount = view.getUint32(offset + 28, true);
	outObj.byteWordOffset = view.getUint32(offset + 32, true);
	outObj.byteLength = view.getUint32(offset + 36, true);
	outObj.reserved0 = view.getUint32(offset + 40, true);
	outObj.reserved1 = view.getUint32(offset + 44, true);
	return outObj;
}

export function serializeConstraintProgramHeader(val: ConstraintProgramHeader, view: DataView, offset: number): void {
	view.setUint32(offset, val.version, true);
	view.setUint32(offset + 4, val.flags, true);
	view.setUint32(offset + 8, val.entryNode, true);
	view.setUint32(offset + 12, val.acceptNode, true);
	view.setUint32(offset + 16, val.nodeWordOffset, true);
	view.setUint32(offset + 20, val.nodeCount, true);
	view.setUint32(offset + 24, val.edgeWordOffset, true);
	view.setUint32(offset + 28, val.edgeCount, true);
	view.setUint32(offset + 32, val.byteWordOffset, true);
	view.setUint32(offset + 36, val.byteLength, true);
	view.setUint32(offset + 40, val.reserved0, true);
	view.setUint32(offset + 44, val.reserved1, true);
}

export function deserializeConstraintNode(view: DataView, offset: number, outObj?: any): ConstraintNode {
	if (!outObj) {
		return {
			kind: deserializeConstraintNodeKind(view, offset),
			next: view.getUint32(offset + 4, true),
			dataOffset: view.getUint32(offset + 8, true),
			dataCount: view.getUint32(offset + 12, true),
			arg0: view.getUint32(offset + 16, true),
			arg1: view.getUint32(offset + 20, true),
			arg2: view.getUint32(offset + 24, true),
			arg3: view.getUint32(offset + 28, true),
			arg4: view.getUint32(offset + 32, true),
			arg5: view.getUint32(offset + 36, true),
			arg6: view.getUint32(offset + 40, true),
			arg7: view.getUint32(offset + 44, true),
		} as any;
	}
	outObj.kind = deserializeConstraintNodeKind(view, offset);
	outObj.next = view.getUint32(offset + 4, true);
	outObj.dataOffset = view.getUint32(offset + 8, true);
	outObj.dataCount = view.getUint32(offset + 12, true);
	outObj.arg0 = view.getUint32(offset + 16, true);
	outObj.arg1 = view.getUint32(offset + 20, true);
	outObj.arg2 = view.getUint32(offset + 24, true);
	outObj.arg3 = view.getUint32(offset + 28, true);
	outObj.arg4 = view.getUint32(offset + 32, true);
	outObj.arg5 = view.getUint32(offset + 36, true);
	outObj.arg6 = view.getUint32(offset + 40, true);
	outObj.arg7 = view.getUint32(offset + 44, true);
	return outObj;
}

export function serializeConstraintNode(val: ConstraintNode, view: DataView, offset: number): void {
	serializeConstraintNodeKind(val.kind, view, offset);
	view.setUint32(offset + 4, val.next, true);
	view.setUint32(offset + 8, val.dataOffset, true);
	view.setUint32(offset + 12, val.dataCount, true);
	view.setUint32(offset + 16, val.arg0, true);
	view.setUint32(offset + 20, val.arg1, true);
	view.setUint32(offset + 24, val.arg2, true);
	view.setUint32(offset + 28, val.arg3, true);
	view.setUint32(offset + 32, val.arg4, true);
	view.setUint32(offset + 36, val.arg5, true);
	view.setUint32(offset + 40, val.arg6, true);
	view.setUint32(offset + 44, val.arg7, true);
}

export function deserializeConstraintByteEdge(view: DataView, offset: number, outObj?: any): ConstraintByteEdge {
	if (!outObj) {
		return {
			word: view.getUint32(offset, true),
		} as any;
	}
	outObj.word = view.getUint32(offset, true);
	return outObj;
}

export function serializeConstraintByteEdge(val: ConstraintByteEdge, view: DataView, offset: number): void {
	view.setUint32(offset, val.word, true);
}

export function deserializeConstraintTokenizerHeader(view: DataView, offset: number, outObj?: any): ConstraintTokenizerHeader {
	if (!outObj) {
		return {
			tokenCount: view.getUint32(offset, true),
			eosToken: view.getUint32(offset + 4, true),
			entryWordOffset: view.getUint32(offset + 8, true),
			byteWordOffset: view.getUint32(offset + 12, true),
			byteLength: view.getUint32(offset + 16, true),
			reserved0: view.getUint32(offset + 20, true),
			reserved1: view.getUint32(offset + 24, true),
			reserved2: view.getUint32(offset + 28, true),
		} as any;
	}
	outObj.tokenCount = view.getUint32(offset, true);
	outObj.eosToken = view.getUint32(offset + 4, true);
	outObj.entryWordOffset = view.getUint32(offset + 8, true);
	outObj.byteWordOffset = view.getUint32(offset + 12, true);
	outObj.byteLength = view.getUint32(offset + 16, true);
	outObj.reserved0 = view.getUint32(offset + 20, true);
	outObj.reserved1 = view.getUint32(offset + 24, true);
	outObj.reserved2 = view.getUint32(offset + 28, true);
	return outObj;
}

export function serializeConstraintTokenizerHeader(val: ConstraintTokenizerHeader, view: DataView, offset: number): void {
	view.setUint32(offset, val.tokenCount, true);
	view.setUint32(offset + 4, val.eosToken, true);
	view.setUint32(offset + 8, val.entryWordOffset, true);
	view.setUint32(offset + 12, val.byteWordOffset, true);
	view.setUint32(offset + 16, val.byteLength, true);
	view.setUint32(offset + 20, val.reserved0, true);
	view.setUint32(offset + 24, val.reserved1, true);
	view.setUint32(offset + 28, val.reserved2, true);
}

export function deserializeConstraintTokenByteEntry(view: DataView, offset: number, outObj?: any): ConstraintTokenByteEntry {
	if (!outObj) {
		return {
			byteOffset: view.getUint32(offset, true),
			meta: view.getUint32(offset + 4, true),
		} as any;
	}
	outObj.byteOffset = view.getUint32(offset, true);
	outObj.meta = view.getUint32(offset + 4, true);
	return outObj;
}

export function serializeConstraintTokenByteEntry(val: ConstraintTokenByteEntry, view: DataView, offset: number): void {
	view.setUint32(offset, val.byteOffset, true);
	view.setUint32(offset + 4, val.meta, true);
}

export function deserializeConstraintDecoderState(view: DataView, offset: number, outObj?: any): ConstraintDecoderState {
	if (!outObj) {
		const _arr_numberText = new Array(16);
		for (let i = 0, _off_numberText = offset + 32; i < 16; i++, _off_numberText += 4) {
			_arr_numberText[i] = view.getUint32(_off_numberText, true);
		}
		return {
			node: view.getUint32(offset, true),
			local0: view.getUint32(offset + 4, true),
			local1: view.getUint32(offset + 8, true),
			local2: view.getUint32(offset + 12, true),
			status: deserializeConstraintDecoderStatus(view, offset + 16),
			errorCode: view.getUint32(offset + 20, true),
			reserved0: view.getUint32(offset + 24, true),
			reserved1: view.getUint32(offset + 28, true),
			numberText: _arr_numberText,
		} as any;
	}
	const _arr_numberText = new Array(16);
	for (let i = 0, _off_numberText = offset + 32; i < 16; i++, _off_numberText += 4) {
		_arr_numberText[i] = view.getUint32(_off_numberText, true);
	}
	outObj.node = view.getUint32(offset, true);
	outObj.local0 = view.getUint32(offset + 4, true);
	outObj.local1 = view.getUint32(offset + 8, true);
	outObj.local2 = view.getUint32(offset + 12, true);
	outObj.status = deserializeConstraintDecoderStatus(view, offset + 16);
	outObj.errorCode = view.getUint32(offset + 20, true);
	outObj.reserved0 = view.getUint32(offset + 24, true);
	outObj.reserved1 = view.getUint32(offset + 28, true);
	outObj.numberText = _arr_numberText;
	return outObj;
}

export function serializeConstraintDecoderState(val: ConstraintDecoderState, view: DataView, offset: number): void {
	view.setUint32(offset, val.node, true);
	view.setUint32(offset + 4, val.local0, true);
	view.setUint32(offset + 8, val.local1, true);
	view.setUint32(offset + 12, val.local2, true);
	serializeConstraintDecoderStatus(val.status, view, offset + 16);
	view.setUint32(offset + 20, val.errorCode, true);
	view.setUint32(offset + 24, val.reserved0, true);
	view.setUint32(offset + 28, val.reserved1, true);
	{ view.setUint32(offset + 32, val.numberText[0]!, true); view.setUint32(offset + 36, val.numberText[1]!, true); view.setUint32(offset + 40, val.numberText[2]!, true); view.setUint32(offset + 44, val.numberText[3]!, true); view.setUint32(offset + 48, val.numberText[4]!, true); view.setUint32(offset + 52, val.numberText[5]!, true); view.setUint32(offset + 56, val.numberText[6]!, true); view.setUint32(offset + 60, val.numberText[7]!, true); view.setUint32(offset + 64, val.numberText[8]!, true); view.setUint32(offset + 68, val.numberText[9]!, true); view.setUint32(offset + 72, val.numberText[10]!, true); view.setUint32(offset + 76, val.numberText[11]!, true); view.setUint32(offset + 80, val.numberText[12]!, true); view.setUint32(offset + 84, val.numberText[13]!, true); view.setUint32(offset + 88, val.numberText[14]!, true); view.setUint32(offset + 92, val.numberText[15]!, true); }
}

export function deserializeLfm2Mode(view: DataView, offset: number): Lfm2Mode {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "prefill";
		case 1: return "decode";
		case 2: return "continuation";
		default: throw new Error("Unknown Enum value for Lfm2Mode: " + v);
	}
}

export function serializeLfm2Mode(val: Lfm2Mode, view: DataView, offset: number): void {
	if(val === "prefill") { view.setUint8(offset, 0); return; }
	if(val === "decode") { view.setUint8(offset, 1); return; }
	if(val === "continuation") { view.setUint8(offset, 2); return; }
}

export function deserializeOpParams(view: DataView, offset: number, outObj?: any): OpParams {
	if (!outObj) {
		return {
			inputOffset: view.getUint32(offset, true),
			outputOffset: view.getUint32(offset + 4, true),
			auxOffset: view.getUint32(offset + 8, true),
			aux2Offset: view.getUint32(offset + 12, true),
			aux3Offset: view.getUint32(offset + 16, true),
			aux4Offset: view.getUint32(offset + 20, true),
			aux5Offset: view.getUint32(offset + 24, true),
			aux6Offset: view.getUint32(offset + 28, true),
			tokenCount: view.getUint32(offset + 32, true),
			inputDim: view.getUint32(offset + 36, true),
			outputDim: view.getUint32(offset + 40, true),
			rowStart: view.getUint32(offset + 44, true),
			rowCount: view.getUint32(offset + 48, true),
			layerIndex: view.getUint32(offset + 52, true),
			attentionSlot: view.getUint32(offset + 56, true),
			mode: deserializeLfm2Mode(view, offset + 60),
			f0: view.getFloat32(offset + 64, true),
			f1: view.getFloat32(offset + 68, true),
			u0: view.getUint32(offset + 72, true),
			u1: view.getUint32(offset + 76, true),
			u2: view.getUint32(offset + 80, true),
			u3: view.getUint32(offset + 84, true),
			u4: view.getUint32(offset + 88, true),
			u5: view.getUint32(offset + 92, true),
		} as any;
	}
	outObj.inputOffset = view.getUint32(offset, true);
	outObj.outputOffset = view.getUint32(offset + 4, true);
	outObj.auxOffset = view.getUint32(offset + 8, true);
	outObj.aux2Offset = view.getUint32(offset + 12, true);
	outObj.aux3Offset = view.getUint32(offset + 16, true);
	outObj.aux4Offset = view.getUint32(offset + 20, true);
	outObj.aux5Offset = view.getUint32(offset + 24, true);
	outObj.aux6Offset = view.getUint32(offset + 28, true);
	outObj.tokenCount = view.getUint32(offset + 32, true);
	outObj.inputDim = view.getUint32(offset + 36, true);
	outObj.outputDim = view.getUint32(offset + 40, true);
	outObj.rowStart = view.getUint32(offset + 44, true);
	outObj.rowCount = view.getUint32(offset + 48, true);
	outObj.layerIndex = view.getUint32(offset + 52, true);
	outObj.attentionSlot = view.getUint32(offset + 56, true);
	outObj.mode = deserializeLfm2Mode(view, offset + 60);
	outObj.f0 = view.getFloat32(offset + 64, true);
	outObj.f1 = view.getFloat32(offset + 68, true);
	outObj.u0 = view.getUint32(offset + 72, true);
	outObj.u1 = view.getUint32(offset + 76, true);
	outObj.u2 = view.getUint32(offset + 80, true);
	outObj.u3 = view.getUint32(offset + 84, true);
	outObj.u4 = view.getUint32(offset + 88, true);
	outObj.u5 = view.getUint32(offset + 92, true);
	return outObj;
}

export function serializeOpParams(val: OpParams, view: DataView, offset: number): void {
	view.setUint32(offset, val.inputOffset, true);
	view.setUint32(offset + 4, val.outputOffset, true);
	view.setUint32(offset + 8, val.auxOffset, true);
	view.setUint32(offset + 12, val.aux2Offset, true);
	view.setUint32(offset + 16, val.aux3Offset, true);
	view.setUint32(offset + 20, val.aux4Offset, true);
	view.setUint32(offset + 24, val.aux5Offset, true);
	view.setUint32(offset + 28, val.aux6Offset, true);
	view.setUint32(offset + 32, val.tokenCount, true);
	view.setUint32(offset + 36, val.inputDim, true);
	view.setUint32(offset + 40, val.outputDim, true);
	view.setUint32(offset + 44, val.rowStart, true);
	view.setUint32(offset + 48, val.rowCount, true);
	view.setUint32(offset + 52, val.layerIndex, true);
	view.setUint32(offset + 56, val.attentionSlot, true);
	serializeLfm2Mode(val.mode, view, offset + 60);
	view.setFloat32(offset + 64, val.f0, true);
	view.setFloat32(offset + 68, val.f1, true);
	view.setUint32(offset + 72, val.u0, true);
	view.setUint32(offset + 76, val.u1, true);
	view.setUint32(offset + 80, val.u2, true);
	view.setUint32(offset + 84, val.u3, true);
	view.setUint32(offset + 88, val.u4, true);
	view.setUint32(offset + 92, val.u5, true);
}

export function deserializeLlmRuntimeStatus(view: DataView, offset: number): LlmRuntimeStatus {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "idle";
		case 1: return "running";
		case 2: return "eos";
		case 3: return "done";
		case 4: return "error";
		default: throw new Error("Unknown Enum value for LlmRuntimeStatus: " + v);
	}
}

export function serializeLlmRuntimeStatus(val: LlmRuntimeStatus, view: DataView, offset: number): void {
	if(val === "idle") { view.setUint8(offset, 0); return; }
	if(val === "running") { view.setUint8(offset, 1); return; }
	if(val === "eos") { view.setUint8(offset, 2); return; }
	if(val === "done") { view.setUint8(offset, 3); return; }
	if(val === "error") { view.setUint8(offset, 4); return; }
}

export function deserializeLlmRuntime(view: DataView, offset: number, outObj?: any): LlmRuntime {
	if (!outObj) {
		return {
			contextCapacity: view.getUint32(offset, true),
			maxNewTokens: view.getUint32(offset + 4, true),
			eosToken: view.getUint32(offset + 8, true),
			promptTokenCount: view.getUint32(offset + 12, true),
			position: view.getUint32(offset + 16, true),
			generatedCount: view.getUint32(offset + 20, true),
			currentToken: view.getUint32(offset + 24, true),
			status: deserializeLlmRuntimeStatus(view, offset + 28),
			telemetryRevision: view.getUint32(offset + 32, true),
			lastToken: view.getUint32(offset + 36, true),
			errorCode: view.getUint32(offset + 40, true),
			pad0: view.getUint32(offset + 44, true),
		} as any;
	}
	outObj.contextCapacity = view.getUint32(offset, true);
	outObj.maxNewTokens = view.getUint32(offset + 4, true);
	outObj.eosToken = view.getUint32(offset + 8, true);
	outObj.promptTokenCount = view.getUint32(offset + 12, true);
	outObj.position = view.getUint32(offset + 16, true);
	outObj.generatedCount = view.getUint32(offset + 20, true);
	outObj.currentToken = view.getUint32(offset + 24, true);
	outObj.status = deserializeLlmRuntimeStatus(view, offset + 28);
	outObj.telemetryRevision = view.getUint32(offset + 32, true);
	outObj.lastToken = view.getUint32(offset + 36, true);
	outObj.errorCode = view.getUint32(offset + 40, true);
	outObj.pad0 = view.getUint32(offset + 44, true);
	return outObj;
}

export function serializeLlmRuntime(val: LlmRuntime, view: DataView, offset: number): void {
	view.setUint32(offset, val.contextCapacity, true);
	view.setUint32(offset + 4, val.maxNewTokens, true);
	view.setUint32(offset + 8, val.eosToken, true);
	view.setUint32(offset + 12, val.promptTokenCount, true);
	view.setUint32(offset + 16, val.position, true);
	view.setUint32(offset + 20, val.generatedCount, true);
	view.setUint32(offset + 24, val.currentToken, true);
	serializeLlmRuntimeStatus(val.status, view, offset + 28);
	view.setUint32(offset + 32, val.telemetryRevision, true);
	view.setUint32(offset + 36, val.lastToken, true);
	view.setUint32(offset + 40, val.errorCode, true);
	view.setUint32(offset + 44, val.pad0, true);
}

export function deserializeDecodeTelemetryEntry(view: DataView, offset: number, outObj?: any): DecodeTelemetryEntry {
	if (!outObj) {
		return {
			position: view.getUint8(offset),
			status: (view.getUint8(offset + 1) >> 0) & 15,
			tokenId: view.getUint16(offset + 2, true),
		} as any;
	}
	outObj.position = view.getUint8(offset);
	outObj.status = (view.getUint8(offset + 1) >> 0) & 15;
	outObj.tokenId = view.getUint16(offset + 2, true);
	return outObj;
}

export function serializeDecodeTelemetryEntry(val: DecodeTelemetryEntry, view: DataView, offset: number): void {
	view.setUint8(offset, val.position);
	{ let _b1 = 0; _b1 |= ((val.status & 15) << 0); view.setUint8(offset + 1, _b1); }
	view.setUint16(offset + 2, val.tokenId, true);
}

export function deserializeGenerateOptions(view: DataView, offset: number, outObj?: any): GenerateOptions {
	if (!outObj) {
		return {
			maxNewTokens: (view.getUint8(offset) === 1 ? view.getFloat64(offset + 1, true) : undefined),
			profile: (view.getUint8(offset + 12) === 1 ? (view.getUint8(offset + 13) !== 0) : undefined),
		} as any;
	}
	outObj.maxNewTokens = (view.getUint8(offset) === 1 ? view.getFloat64(offset + 1, true) : undefined);
	outObj.profile = (view.getUint8(offset + 12) === 1 ? (view.getUint8(offset + 13) !== 0) : undefined);
	return outObj;
}

export function serializeGenerateOptions(val: GenerateOptions, view: DataView, offset: number): void {
	if (val.maxNewTokens !== undefined) { view.setUint8(offset, 1); view.setFloat64(offset + 1, val.maxNewTokens, true); } else { view.setUint8(offset, 0); }
	if (val.profile !== undefined) { view.setUint8(offset + 12, 1); view.setUint8(offset + 13, (val.profile ? 1 : 0)); } else { view.setUint8(offset + 12, 0); }
}

export function deserializeGenerateTimings(view: DataView, offset: number, outObj?: any): GenerateTimings {
	if (!outObj) {
		return {
			prefillMs: view.getFloat64(offset, true),
			decodeMs: view.getFloat64(offset + 8, true),
			readbackMs: view.getFloat64(offset + 16, true),
			totalMs: view.getFloat64(offset + 24, true),
			promptTokens: view.getFloat64(offset + 32, true),
			scheduledDecodeSteps: view.getFloat64(offset + 40, true),
			cacheDepth: (view.getUint8(offset + 48) === 1 ? view.getFloat64(offset + 49, true) : undefined),
			cachedBlocks: (view.getUint8(offset + 60) === 1 ? view.getFloat64(offset + 61, true) : undefined),
			cachedTokens: (view.getUint8(offset + 72) === 1 ? view.getFloat64(offset + 73, true) : undefined),
			liveQueryTokens: (view.getUint8(offset + 84) === 1 ? view.getFloat64(offset + 85, true) : undefined),
			repairedTokens: (view.getUint8(offset + 96) === 1 ? view.getFloat64(offset + 97, true) : undefined),
		} as any;
	}
	outObj.prefillMs = view.getFloat64(offset, true);
	outObj.decodeMs = view.getFloat64(offset + 8, true);
	outObj.readbackMs = view.getFloat64(offset + 16, true);
	outObj.totalMs = view.getFloat64(offset + 24, true);
	outObj.promptTokens = view.getFloat64(offset + 32, true);
	outObj.scheduledDecodeSteps = view.getFloat64(offset + 40, true);
	outObj.cacheDepth = (view.getUint8(offset + 48) === 1 ? view.getFloat64(offset + 49, true) : undefined);
	outObj.cachedBlocks = (view.getUint8(offset + 60) === 1 ? view.getFloat64(offset + 61, true) : undefined);
	outObj.cachedTokens = (view.getUint8(offset + 72) === 1 ? view.getFloat64(offset + 73, true) : undefined);
	outObj.liveQueryTokens = (view.getUint8(offset + 84) === 1 ? view.getFloat64(offset + 85, true) : undefined);
	outObj.repairedTokens = (view.getUint8(offset + 96) === 1 ? view.getFloat64(offset + 97, true) : undefined);
	return outObj;
}

export function serializeGenerateTimings(val: GenerateTimings, view: DataView, offset: number): void {
	view.setFloat64(offset, val.prefillMs, true);
	view.setFloat64(offset + 8, val.decodeMs, true);
	view.setFloat64(offset + 16, val.readbackMs, true);
	view.setFloat64(offset + 24, val.totalMs, true);
	view.setFloat64(offset + 32, val.promptTokens, true);
	view.setFloat64(offset + 40, val.scheduledDecodeSteps, true);
	if (val.cacheDepth !== undefined) { view.setUint8(offset + 48, 1); view.setFloat64(offset + 49, val.cacheDepth, true); } else { view.setUint8(offset + 48, 0); }
	if (val.cachedBlocks !== undefined) { view.setUint8(offset + 60, 1); view.setFloat64(offset + 61, val.cachedBlocks, true); } else { view.setUint8(offset + 60, 0); }
	if (val.cachedTokens !== undefined) { view.setUint8(offset + 72, 1); view.setFloat64(offset + 73, val.cachedTokens, true); } else { view.setUint8(offset + 72, 0); }
	if (val.liveQueryTokens !== undefined) { view.setUint8(offset + 84, 1); view.setFloat64(offset + 85, val.liveQueryTokens, true); } else { view.setUint8(offset + 84, 0); }
	if (val.repairedTokens !== undefined) { view.setUint8(offset + 96, 1); view.setFloat64(offset + 97, val.repairedTokens, true); } else { view.setUint8(offset + 96, 0); }
}

export function deserializeGenerateResult(view: DataView, offset: number, outObj?: any): GenerateResult {
	if (!outObj) {
		return {
			tokenIds: ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(view.getFloat64(o + (i * 8), true)); } return a; })(offset),
			state: deserializeLlmRuntime(view, offset + 4),
			timings: (view.getUint8(offset + 52) === 1 ? deserializeGenerateTimings(view, offset + 53) : undefined),
		} as any;
	}
	outObj.tokenIds = ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(view.getFloat64(o + (i * 8), true)); } return a; })(offset);
	outObj.state = deserializeLlmRuntime(view, offset + 4);
	outObj.timings = (view.getUint8(offset + 52) === 1 ? deserializeGenerateTimings(view, offset + 53) : undefined);
	return outObj;
}

export function serializeGenerateResult(val: GenerateResult, view: DataView, offset: number): void {
	{ view.setUint32(offset, val.tokenIds.length, true); let o = offset + 4; for(let i=0; i<val.tokenIds.length; i++) { view.setFloat64(o + (i * 8), val.tokenIds[i]!, true); } }
	serializeLlmRuntime(val.state, view, offset + 4);
	if (val.timings !== undefined) { view.setUint8(offset + 52, 1); serializeGenerateTimings(val.timings, view, offset + 53); } else { view.setUint8(offset + 52, 0); }
}

export function deserializeCacheBlockOptions(view: DataView, offset: number, outObj?: any): CacheBlockOptions {
	if (!outObj) {
		return {
			depth: (view.getUint8(offset) === 1 ? view.getFloat64(offset + 1, true) : undefined),
		} as any;
	}
	outObj.depth = (view.getUint8(offset) === 1 ? view.getFloat64(offset + 1, true) : undefined);
	return outObj;
}

export function serializeCacheBlockOptions(val: CacheBlockOptions, view: DataView, offset: number): void {
	if (val.depth !== undefined) { view.setUint8(offset, 1); view.setFloat64(offset + 1, val.depth, true); } else { view.setUint8(offset, 0); }
}

export function deserializeGpuWeightFormat(view: DataView, offset: number): GpuWeightFormat {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "f16";
		case 1: return "f32";
		case 2: return "wq4";
		default: throw new Error("Unknown Enum value for GpuWeightFormat: " + v);
	}
}

export function serializeGpuWeightFormat(val: GpuWeightFormat, view: DataView, offset: number): void {
	if(val === "f16") { view.setUint8(offset, 0); return; }
	if(val === "f32") { view.setUint8(offset, 1); return; }
	if(val === "wq4") { view.setUint8(offset, 2); return; }
}

export function deserializeLfm2LayerKind(view: DataView, offset: number): Lfm2LayerKind {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "conv";
		case 1: return "attention";
		default: throw new Error("Unknown Enum value for Lfm2LayerKind: " + v);
	}
}

export function serializeLfm2LayerKind(val: Lfm2LayerKind, view: DataView, offset: number): void {
	if(val === "conv") { view.setUint8(offset, 0); return; }
	if(val === "attention") { view.setUint8(offset, 1); return; }
}

export function deserializeLfm2RuntimeConfig(view: DataView, offset: number, outObj?: any): Lfm2RuntimeConfig {
	if (!outObj) {
		return {
			contextLength: view.getFloat64(offset, true),
			hiddenSize: view.getFloat64(offset + 8, true),
			feedForwardSize: view.getFloat64(offset + 16, true),
			attentionHeads: view.getFloat64(offset + 24, true),
			kvHeadsByLayer: ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(view.getFloat64(o + (i * 8), true)); } return a; })(offset + 32),
			headDim: view.getFloat64(offset + 36, true),
			ropeTheta: view.getFloat64(offset + 44, true),
			vocabSize: view.getFloat64(offset + 52, true),
			convCacheLength: view.getFloat64(offset + 60, true),
			normEpsilon: view.getFloat64(offset + 68, true),
			eosToken: view.getFloat64(offset + 76, true),
			blockCount: view.getFloat64(offset + 84, true),
			layers: ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(deserializeLfm2LayerKind(view, o + (i * 1))); } return a; })(offset + 92),
			attentionLayerSlots: ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(view.getFloat64(o + (i * 8), true)); } return a; })(offset + 96),
		} as any;
	}
	outObj.contextLength = view.getFloat64(offset, true);
	outObj.hiddenSize = view.getFloat64(offset + 8, true);
	outObj.feedForwardSize = view.getFloat64(offset + 16, true);
	outObj.attentionHeads = view.getFloat64(offset + 24, true);
	outObj.kvHeadsByLayer = ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(view.getFloat64(o + (i * 8), true)); } return a; })(offset + 32);
	outObj.headDim = view.getFloat64(offset + 36, true);
	outObj.ropeTheta = view.getFloat64(offset + 44, true);
	outObj.vocabSize = view.getFloat64(offset + 52, true);
	outObj.convCacheLength = view.getFloat64(offset + 60, true);
	outObj.normEpsilon = view.getFloat64(offset + 68, true);
	outObj.eosToken = view.getFloat64(offset + 76, true);
	outObj.blockCount = view.getFloat64(offset + 84, true);
	outObj.layers = ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(deserializeLfm2LayerKind(view, o + (i * 1))); } return a; })(offset + 92);
	outObj.attentionLayerSlots = ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(view.getFloat64(o + (i * 8), true)); } return a; })(offset + 96);
	return outObj;
}

export function serializeLfm2RuntimeConfig(val: Lfm2RuntimeConfig, view: DataView, offset: number): void {
	view.setFloat64(offset, val.contextLength, true);
	view.setFloat64(offset + 8, val.hiddenSize, true);
	view.setFloat64(offset + 16, val.feedForwardSize, true);
	view.setFloat64(offset + 24, val.attentionHeads, true);
	{ view.setUint32(offset + 32, val.kvHeadsByLayer.length, true); let o = offset + 32 + 4; for(let i=0; i<val.kvHeadsByLayer.length; i++) { view.setFloat64(o + (i * 8), val.kvHeadsByLayer[i]!, true); } }
	view.setFloat64(offset + 36, val.headDim, true);
	view.setFloat64(offset + 44, val.ropeTheta, true);
	view.setFloat64(offset + 52, val.vocabSize, true);
	view.setFloat64(offset + 60, val.convCacheLength, true);
	view.setFloat64(offset + 68, val.normEpsilon, true);
	view.setFloat64(offset + 76, val.eosToken, true);
	view.setFloat64(offset + 84, val.blockCount, true);
	{ view.setUint32(offset + 92, val.layers.length, true); let o = offset + 92 + 4; for(let i=0; i<val.layers.length; i++) { serializeLfm2LayerKind(val.layers[i]!, view, o + (i * 1)); } }
	{ view.setUint32(offset + 96, val.attentionLayerSlots.length, true); let o = offset + 96 + 4; for(let i=0; i<val.attentionLayerSlots.length; i++) { view.setFloat64(o + (i * 8), val.attentionLayerSlots[i]!, true); } }
}

export function deserializeMatmulDispatchArgs(view: DataView, offset: number, outObj?: any): MatmulDispatchArgs {
	if (!outObj) {
		return {
			rowCount: view.getFloat64(offset, true),
			tokenCount: view.getFloat64(offset + 8, true),
			inputDim: view.getFloat64(offset + 16, true),
			outputDim: view.getFloat64(offset + 24, true),
		} as any;
	}
	outObj.rowCount = view.getFloat64(offset, true);
	outObj.tokenCount = view.getFloat64(offset + 8, true);
	outObj.inputDim = view.getFloat64(offset + 16, true);
	outObj.outputDim = view.getFloat64(offset + 24, true);
	return outObj;
}

export function serializeMatmulDispatchArgs(val: MatmulDispatchArgs, view: DataView, offset: number): void {
	view.setFloat64(offset, val.rowCount, true);
	view.setFloat64(offset + 8, val.tokenCount, true);
	view.setFloat64(offset + 16, val.inputDim, true);
	view.setFloat64(offset + 24, val.outputDim, true);
}

export function deserializeGPUBuffer(view: DataView, offset: number): GPUBuffer {
	return undefined as any;
}

export function serializeGPUBuffer(val: GPUBuffer, view: DataView, offset: number): void {
	
}

export function deserializeGPUDevice(view: DataView, offset: number): GPUDevice {
	return undefined as any;
}

export function serializeGPUDevice(val: GPUDevice, view: DataView, offset: number): void {
	
}

export function deserializeGpuTensorPage(view: DataView, offset: number, outObj?: any): GpuTensorPage {
	if (!outObj) {
		return {
			buffer: deserializeGPUBuffer(view, offset),
			rowStart: view.getFloat64(offset, true),
			rowCount: view.getFloat64(offset + 8, true),
			byteLength: view.getFloat64(offset + 16, true),
		} as any;
	}
	outObj.buffer = deserializeGPUBuffer(view, offset);
	outObj.rowStart = view.getFloat64(offset, true);
	outObj.rowCount = view.getFloat64(offset + 8, true);
	outObj.byteLength = view.getFloat64(offset + 16, true);
	return outObj;
}

export function serializeGpuTensorPage(val: GpuTensorPage, view: DataView, offset: number): void {
	serializeGPUBuffer(val.buffer, view, offset);
	view.setFloat64(offset, val.rowStart, true);
	view.setFloat64(offset + 8, val.rowCount, true);
	view.setFloat64(offset + 16, val.byteLength, true);
}

export function deserializeGpuTensor(view: DataView, offset: number, outObj?: any): GpuTensor {
	if (!outObj) {
		return {
			name: ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset),
			format: deserializeGpuWeightFormat(view, offset + 4),
			dimensions: ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(view.getFloat64(o + (i * 8), true)); } return a; })(offset + 8),
			pages: ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(deserializeGpuTensorPage(view, o + (i * 24))); } return a; })(offset + 12),
			byteLength: (view.getUint8(offset + 16) === 1 ? view.getFloat64(offset + 17, true) : undefined),
		} as any;
	}
	outObj.name = ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset);
	outObj.format = deserializeGpuWeightFormat(view, offset + 4);
	outObj.dimensions = ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(view.getFloat64(o + (i * 8), true)); } return a; })(offset + 8);
	outObj.pages = ((o) => { const l = view.getUint32(o, true); const a: any[] = []; const start = o + 4; for(let i=0; i<l; i++) { const o = start; a.push(deserializeGpuTensorPage(view, o + (i * 24))); } return a; })(offset + 12);
	outObj.byteLength = (view.getUint8(offset + 16) === 1 ? view.getFloat64(offset + 17, true) : undefined);
	return outObj;
}

export function serializeGpuTensor(val: GpuTensor, view: DataView, offset: number): void {
	{ const bytes = __textEncoder!.encode(val.name); const len = Math.min(bytes.length, 255); view.setUint32(offset, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 4, len).set(bytes.subarray(0, len)); }
	serializeGpuWeightFormat(val.format, view, offset + 4);
	{ view.setUint32(offset + 8, val.dimensions.length, true); let o = offset + 8 + 4; for(let i=0; i<val.dimensions.length; i++) { view.setFloat64(o + (i * 8), val.dimensions[i]!, true); } }
	{ view.setUint32(offset + 12, val.pages.length, true); let o = offset + 12 + 4; for(let i=0; i<val.pages.length; i++) { serializeGpuTensorPage(val.pages[i]!, view, o + (i * 24)); } }
	if (val.byteLength !== undefined) { view.setUint8(offset + 16, 1); view.setFloat64(offset + 17, val.byteLength, true); } else { view.setUint8(offset + 16, 0); }
}

export function deserializeLfm2RuntimeModel(view: DataView, offset: number, outObj?: any): Lfm2RuntimeModel {
	if (!outObj) {
		return {
			device: deserializeGPUDevice(view, offset),
			config: deserializeLfm2RuntimeConfig(view, offset),
			tensor: undefined,
		} as any;
	}
	outObj.device = deserializeGPUDevice(view, offset);
	outObj.config = deserializeLfm2RuntimeConfig(view, offset);
	outObj.tensor = undefined;
	return outObj;
}

export function serializeLfm2RuntimeModel(val: Lfm2RuntimeModel, view: DataView, offset: number): void {
	serializeGPUDevice(val.device, view, offset);
	serializeLfm2RuntimeConfig(val.config, view, offset);
	
}

export function deserializeMatmulKernelSpec(view: DataView, offset: number, outObj?: any): MatmulKernelSpec {
	if (!outObj) {
		return {
			entryPoint: ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset),
			wgsl: (view.getUint8(offset + 4) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 5) : undefined),
			workgroups: (view.getUint8(offset + 12) === 1 ? undefined : undefined),
		} as any;
	}
	outObj.entryPoint = ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset);
	outObj.wgsl = (view.getUint8(offset + 4) === 1 ? ((o) => { const l = view.getUint32(o, true); return __textDecoder!.decode(new Uint8Array(view.buffer, view.byteOffset + o + 4, l)); })(offset + 5) : undefined);
	outObj.workgroups = (view.getUint8(offset + 12) === 1 ? undefined : undefined);
	return outObj;
}

export function serializeMatmulKernelSpec(val: MatmulKernelSpec, view: DataView, offset: number): void {
	{ const bytes = __textEncoder!.encode(val.entryPoint); const len = Math.min(bytes.length, 255); view.setUint32(offset, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 4, len).set(bytes.subarray(0, len)); }
	if (val.wgsl !== undefined) { view.setUint8(offset + 4, 1); { const bytes = __textEncoder!.encode(val.wgsl); const len = Math.min(bytes.length, 255); view.setUint32(offset + 5, len, true); new Uint8Array(view.buffer, view.byteOffset + offset + 5 + 4, 255).fill(0); new Uint8Array(view.buffer, view.byteOffset + offset + 5 + 4, len).set(bytes.subarray(0, len)); } } else { view.setUint8(offset + 4, 0); }
	if (val.workgroups !== undefined) { view.setUint8(offset + 12, 1);  } else { view.setUint8(offset + 12, 0); }
}

export function deserializeLfm2RuntimeOptions(view: DataView, offset: number, outObj?: any): Lfm2RuntimeOptions {
	if (!outObj) {
		return {
			contextCapacity: (view.getUint8(offset) === 1 ? view.getFloat64(offset + 1, true) : undefined),
			maxNewTokens: (view.getUint8(offset + 12) === 1 ? view.getFloat64(offset + 13, true) : undefined),
			matmulKernels: (view.getUint8(offset + 24) === 1 ? ({ f16: (view.getUint8(offset + 25) === 1 ? deserializeMatmulKernelSpec(view, offset + 26) : undefined), f32: (view.getUint8(offset + 45) === 1 ? deserializeMatmulKernelSpec(view, offset + 46) : undefined), wq4: (view.getUint8(offset + 65) === 1 ? deserializeMatmulKernelSpec(view, offset + 66) : undefined) }) : undefined),
		} as any;
	}
	outObj.contextCapacity = (view.getUint8(offset) === 1 ? view.getFloat64(offset + 1, true) : undefined);
	outObj.maxNewTokens = (view.getUint8(offset + 12) === 1 ? view.getFloat64(offset + 13, true) : undefined);
	outObj.matmulKernels = (view.getUint8(offset + 24) === 1 ? ({ f16: (view.getUint8(offset + 25) === 1 ? deserializeMatmulKernelSpec(view, offset + 26) : undefined), f32: (view.getUint8(offset + 45) === 1 ? deserializeMatmulKernelSpec(view, offset + 46) : undefined), wq4: (view.getUint8(offset + 65) === 1 ? deserializeMatmulKernelSpec(view, offset + 66) : undefined) }) : undefined);
	return outObj;
}

export function serializeLfm2RuntimeOptions(val: Lfm2RuntimeOptions, view: DataView, offset: number): void {
	if (val.contextCapacity !== undefined) { view.setUint8(offset, 1); view.setFloat64(offset + 1, val.contextCapacity, true); } else { view.setUint8(offset, 0); }
	if (val.maxNewTokens !== undefined) { view.setUint8(offset + 12, 1); view.setFloat64(offset + 13, val.maxNewTokens, true); } else { view.setUint8(offset + 12, 0); }
	if (val.matmulKernels !== undefined) { view.setUint8(offset + 24, 1); if (val.matmulKernels.f16 !== undefined) { view.setUint8(offset + 25, 1); serializeMatmulKernelSpec(val.matmulKernels.f16, view, offset + 26); } else { view.setUint8(offset + 25, 0); } if (val.matmulKernels.f32 !== undefined) { view.setUint8(offset + 45, 1); serializeMatmulKernelSpec(val.matmulKernels.f32, view, offset + 46); } else { view.setUint8(offset + 45, 0); } if (val.matmulKernels.wq4 !== undefined) { view.setUint8(offset + 65, 1); serializeMatmulKernelSpec(val.matmulKernels.wq4, view, offset + 66); } else { view.setUint8(offset + 65, 0); } } else { view.setUint8(offset + 24, 0); }
}

